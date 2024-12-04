import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
from typing import Optional, Sequence
import torchaudio
from pqmf import PQMF
from ae import Quantize,RVQ

warnings.filterwarnings("ignore")
class MultiScaleSTFT(nn.Module):

    def __init__(self,
                 scales: Sequence[int],
                 sample_rate: int,
                 magnitude: bool = True,
                 normalized: bool = False,
                 num_mels: Optional[int] = None) -> None:
        super().__init__()
        self.scales = scales
        self.magnitude = magnitude
        self.num_mels = num_mels

        self.stfts = []
        for scale in scales:
            self.stfts.append(
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=scale,
                    win_length=scale,
                    hop_length=scale // 4,
                    normalized=normalized,
                    n_mels=num_mels,
                    power=1,
                ))

        self.stfts = nn.ModuleList(self.stfts)

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        b,c,t = x.shape
        x = torch.reshape(x,(b*c,t))
        stfts = []
        for stft in self.stfts:
            y = stft(x)
            stfts.append(y.abs())

        return stfts
 

class AudioDistance(nn.Module):

    def __init__(self, params,
                 log_epsilon: float) -> None:
        super().__init__()
        self.multiscale_stft = MultiScaleSTFT(params.stft_scales,params.sample_rate,num_mels=params.n_mels)
        self.log_epsilon = log_epsilon

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        stfts_x = self.multiscale_stft(x)
        stfts_y = self.multiscale_stft(y)
        distance = 0.

        for x, y in zip(stfts_x, stfts_y):
            logx = torch.log(x + self.log_epsilon)
            logy = torch.log(y + self.log_epsilon)

            lin_distance = F.mse_loss(x,y)
            log_distance = F.l1_loss(logx, logy)

            distance = distance + lin_distance + log_distance

        return distance

class AE(nn.Module):
    """Some Information about AE"""
    def __init__(self,params):
        super(AE, self).__init__()
        hidden_dim = 16
        self.params = params
        self.spec_distance = AudioDistance(params,params.log_epsilon)
        self.pqmf = PQMF(100,params.n_band)
        self.encoder = nn.Sequential(
            nn.Conv1d(params.n_band,params.n_band*4,7,padding=3),
            nn.BatchNorm1d(params.n_band*4),
            nn.LeakyReLU(.2),
            nn.Conv1d(params.n_band*4,params.n_band*8,kernel_size=4*2+1,stride=4,padding=4),
            nn.BatchNorm1d(params.n_band*8),
            nn.LeakyReLU(.2),
            nn.Conv1d(params.n_band*8,params.n_band*16,kernel_size=4*2+1,stride=4,padding=4),
            nn.BatchNorm1d(params.n_band*16),
            nn.LeakyReLU(.2),
            nn.Conv1d(params.n_band*16,params.n_band*32,kernel_size=4*2+1,stride=4,padding=4),
            nn.BatchNorm1d(params.n_band*32),
            nn.LeakyReLU(.2),
            nn.Conv1d(params.n_band*32,hidden_dim,1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(.2),
        )


        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim,params.n_band*32,1),
            nn.BatchNorm1d(params.n_band*32),
            nn.LeakyReLU(.2),
            nn.ConvTranspose1d(params.n_band*32,params.n_band*16,kernel_size=4*2,stride=4,padding=4//2),
            nn.BatchNorm1d(params.n_band*16),
            nn.LeakyReLU(.2),
            nn.ConvTranspose1d(params.n_band*16,params.n_band*8,kernel_size=4*2,stride=4,padding=4//2),
            nn.BatchNorm1d(params.n_band*8),
            nn.LeakyReLU(.2),
            nn.ConvTranspose1d(params.n_band*8,params.n_band*4,kernel_size=4*2,stride=4,padding=4//2),
            nn.BatchNorm1d(params.n_band*4),
            nn.LeakyReLU(.2),
        )
        self.wave_gen = nn.Conv1d(params.n_band*4,params.n_band,7,padding=3)
        self.loud_gen = nn.Conv1d(params.n_band*4,params.n_band,3,1,padding=1)
        
    def mod_sigmoid(self,x):
        return 2 * torch.sigmoid(x)**2.3 + 1e-7
    
    def decode(self,x):
        z_ = self.decoder(x)
        loud = self.loud_gen(z_)
        wave = self.wave_gen(z_)
        x = torch.tanh(wave) *  self.mod_sigmoid(loud)
        return x
    
    def encode(self,x):
        mb_auido = self.pqmf(x)
        z = self.encoder(mb_auido)
        return z,mb_auido
    
    def forward(self, x):
        z,mb_auido = self.encode(x)
        x_reconstruct = self.decode(z)

        b,c,t = mb_auido.shape
        x_reconstruct = x_reconstruct[:,:,:t]

        input_audio = torch.reshape(mb_auido,(b*c,1,t))
        output_audio = torch.reshape(x_reconstruct,(b*c,1,-1))
        distance = self.spec_distance(input_audio,output_audio)
        l2_distance = F.mse_loss(input_audio,output_audio)

        x_reconstruct = self.pqmf.inverse(x_reconstruct)
        return x_reconstruct,distance,l2_distance

class Layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1)  # Outputs s and t
        )
    
    def forward(self, x):
        res = x 
        x = self.net(x)
        if res.shape == x.shape:
            return x + res 
        else:
            return  x

class Block(nn.Module):
    """Some Information about Flow"""
    def __init__(self, feature_dim,hidden_dim, num_flow_layers):
        super().__init__()
        layers = [Layer(feature_dim,hidden_dim)]
        layers +=[Layer(hidden_dim,hidden_dim) for _ in range(num_flow_layers-2)]
        layers +=[Layer(hidden_dim,feature_dim) ]
        self.flows = nn.ModuleList(layers)
    
    def forward(self, x):
        for flow in self.flows:
            x =  flow(x)
        return x
    


class CouplingLayer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(feature_dim // 2, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, feature_dim, kernel_size=3, padding=1)  # Outputs s and t
        )
    
    def forward(self, x):
        # Split input into two parts
        x1, x2 = x.chunk(2, dim=1)  # Split along the channel dimension
        
        # Predict scale (s) and translation (t) using x1
        st = self.net(x1)
        s, t = st.chunk(2, dim=1)
        s = torch.tanh(s)  # Regularize scaling
        
        # Apply transformation to x2
        y2 = x2 * torch.exp(s) + t
        y = torch.cat([x1, y2], dim=1)  # Combine x1 and transformed x2
        
        # Compute log determinant of Jacobian
        log_det_jacobian = torch.sum(s, dim=(1, 2))  # Sum over channel and time dimensions
        return y, log_det_jacobian
    
    def inverse(self, y):
        # Split output into two parts
        y1, y2 = y.chunk(2, dim=1)
        
        # Predict scale (s) and translation (t) using y1
        st = self.net(y1)
        s, t = st.chunk(2, dim=1)
        s = torch.tanh(s)  # Regularize scaling
        
        # Inverse transformation to get x2
        x2 = (y2 - t) * torch.exp(-s)
        x = torch.cat([y1, x2], dim=1)  # Combine y1 and inverse-transformed x2
        return x

class FlowBlock(nn.Module):
    """Some Information about Flow"""
    def __init__(self, feature_dim, num_flow_layers):
        super().__init__()
        self.flows = nn.ModuleList([
            CouplingLayer(feature_dim) for _ in range(num_flow_layers)
        ])
    
    def forward(self, x):
        log_det_jacobian = 0
        for flow in self.flows:
            x, ldj = flow(x)
            log_det_jacobian += ldj
        return x, log_det_jacobian
    
    def inverse(self, z):
        for flow in reversed(self.flows):
            z = flow.inverse(z)
        return z
    
class RVQLayer(nn.Module):
    """Some Information about MyModule"""
    def __init__(self,feature_dim=16,hid_dim=256,num_flow_layers=12):
        super().__init__()
        self.encoder = Block(feature_dim,hid_dim, num_flow_layers)
        self.decoder = Block(feature_dim,hid_dim, num_flow_layers)
        # self.vq_layer = Quantize(feature_dim,1024)
        self.vq_layer = RVQ(num_flow_layers,1024,feature_dim)

    def quantize(self,z):
        zq = self.encoder(z)
        zq = z.permute(0,2,1) 
        zq, vq_loss, _ = self.vq_layer(zq)
        zq = zq.permute(0,2,1) 
        return zq,vq_loss
    
    def forward(self, z):
        zq,vq_loss = self.quantize(z)
        z = self.decoder(zq)
        return z,vq_loss
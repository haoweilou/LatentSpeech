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
    
class Glow(nn.Module):
    """Some Information about Glow"""
    def __init__(self,feature_dim=16,num_layer=16):
        super(Glow, self).__init__()
        self.flows = nn.ModuleList([GlowStep(feature_dim) for _ in range(num_layer)])

    def forward(self, x,x_mask=None):
        #x = b,c,t, mask = b,1,t
        b,c,t = x.shape
        if x_mask is None: x_mask = torch.ones(b, 1, t).to(device=x.device, dtype=x.dtype)

        log_det = 0
        for flow in self.flows:
            x, ld = flow(x,x_mask)
            log_det += ld
        return x, log_det

    def reverse(self,y,x_mask=None):
        b,c,t = y.shape
        if x_mask is None: x_mask = torch.ones(b, 1, t).to(device=y.device, dtype=y.dtype)

        for flow in reversed(self.flows):
            y = flow.reverse(y,x_mask)
        return y

class GlowStep(nn.Module):
    """Some Information about Glow"""
    def __init__(self,feature_dim):
        super().__init__()
        self.actnorm = ActNorm(feature_dim)
        self.inv1d = InvCov(feature_dim)
        self.affine = AffineCouple(feature_dim)

    def forward(self, x, x_mask=None):
        x,logdet0 = self.actnorm(x,x_mask)
        x,logdet1 = self.inv1d(x,x_mask)
        y,logdet2 = self.affine(x,x_mask)
        l = logdet0+logdet1+logdet2
        return y,l
    
    def reverse(self,y, x_mask=None):
        x = self.affine.reverse(y,x_mask)
        x = self.inv1d.reverse(x,x_mask)
        x = self.actnorm.reverse(x,x_mask)
        return x
    
class AffineCouple(nn.Module):
    """Some Information about AffineCouple"""
    def __init__(self,in_channels=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels // 2, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 512, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, x_mask):
        #x = B, C, T
        xa,xb = x.chunk(2, dim=1)
        logs,t  = self.net(xb).chunk(2, dim=1) #logs, t
        s = torch.sigmoid(logs + 2)
        ya = s*xa + t 
        yb = xb 
        y = torch.cat((ya,yb),dim=1)

        logdet = torch.sum(torch.log(s).view(x.shape[0], -1), 1)
        
        return y,logdet
    
    def reverse(self,y, x_mask):
        #y = B,C,T
        ya,yb = y.chunk(2,dim=1)
        logs,t = self.net(yb).chunk(2, dim=1) #logs, t
        s = torch.sigmoid(logs+2)
        xa = (ya-t) / s 
        xb = yb 

        x = torch.cat((xa,xb),dim=1)

        return x

class WN(nn.Module):
    """Some Information about WN"""
    def __init__(self,in_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
        super().__init__()
        assert(kernel_size % 2 == 1)
        assert(hidden_channels % 2 == 0)
        self.in_channels = in_channels
        self.hidden_channels =hidden_channels
        self.kernel_size = kernel_size,
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = torch.nn.Conv1d(gin_channels, 2*hidden_channels*n_layers, 1)
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')
        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(hidden_channels, 2*hidden_channels, kernel_size,
                                    dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

        res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
        res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
        self.res_skip_layers.append(res_skip_layer)


    def forward(self, x, x_mask=None, g=None, **kwargs):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])
        
        if g is not None:g = self.cond_layer(g)#g is the condition

        for i in range(self.n_layers):
          x_in = self.in_layers[i](x)
          x_in = self.drop(x_in)
          if g is not None:
            cond_offset = i * 2 * self.hidden_channels
            g_l = g[:,cond_offset:cond_offset+2*self.hidden_channels,:]
          else:
            g_l = torch.zeros_like(x_in)

        return x

class InvCov(nn.Module):
    """Some Information about Inv1DCov"""
    def __init__(self,num_channels=16, n_split=4):
        super().__init__()
        self.channels = num_channels
        self.n_split = n_split
        
        w = torch.qr(torch.FloatTensor(self.n_split, self.n_split).normal_())[0]
        if torch.det(w) < 0: w[:,0] = -1 * w[:,0]
        self.weight = nn.Parameter(w)  # 1x1 conv kernel

    def forward(self, x,x_mask):
        #B,C,T
        b,c,t = x.shape
        x_len = torch.sum(x_mask, [1, 2])

        x = x.view(b, 2, c // self.n_split, self.n_split // 2, t)
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(b, self.n_split, c // self.n_split, t)
        weight = self.weight
        logdet = torch.logdet(self.weight) * (c / self.n_split) * x_len # [b]
        weight = weight.view(self.n_split, self.n_split, 1, 1)
        y = F.conv2d(x, weight)
        y = y.view(b, 2, self.n_split // 2, c // self.n_split, t)
        y = y.permute(0, 1, 3, 2, 4).contiguous().view(b, c, t) * x_mask
        
        return y,logdet

    def reverse(self,y,x_mask):
        b,c,t = y.shape

        y = y.view(b, 2, c // self.n_split, self.n_split // 2, t)
        y = y.permute(0, 1, 3, 2, 4).contiguous().view(b, self.n_split, c // self.n_split, t)

        weight = torch.inverse(self.weight.float()).to(dtype=self.weight.dtype)

        x = F.conv2d(y, weight)
        x = x.view(b, 2, self.n_split // 2, c // self.n_split, t)
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(b, c, t) * x_mask

        return x
    
class ActNorm(nn.Module):
    """Some Information about Actnorm"""
    def __init__(self,num_channels=16):
        super().__init__()
        self.num_channel = num_channels
        self.logs = nn.Parameter(torch.zeros(1, num_channels,  1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.initialized = False


    def initialize_parameters(self, x, x_mask):
        """
        Initialize scale and bias using the first batch of data.
        """
        with torch.no_grad():
            denom = torch.sum(x_mask, [0, 2])
            m = torch.sum(x * x_mask, [0, 2]) / denom
            m_sq = torch.sum(x * x * x_mask, [0, 2]) / denom
            v = m_sq - (m ** 2)
            logs = 0.5 * torch.log(torch.clamp_min(v, 1e-6))

            bias_init = (-m * torch.exp(-logs)).view(*self.bias.shape).to(dtype=self.bias.dtype)
            logs_init = (-logs).view(*self.logs.shape).to(dtype=self.logs.dtype)

            self.bias.data.copy_(bias_init)
            self.logs.data.copy_(logs_init)
        self.initialized = True

    def forward(self, x, x_mask):
        #x = [b,c,t], x_mask = [b,1,t]
        x_len = torch.sum(x_mask, [1, 2])
        if not self.initialized:
            self.initialize_parameters(x, x_mask)

        y = (self.bias + torch.exp(self.logs) * x) * x_mask
        logdet = torch.sum(self.logs) * x_len # [b]
        return y,logdet
    
    def reverse(self,y,x_mask):
        #x = b,c,t, mask = b,1,t
        x = (y - self.bias) * torch.exp(-self.logs)
        return x
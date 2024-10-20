import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from typing import Callable, Optional, Sequence, Union
import warnings
from pqmf import PQMF
from math import sqrt


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

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
            nn.Tanh(),
            nn.Conv1d(params.n_band*4,params.n_band*8,kernel_size=5*2+1,stride=5,padding=5),
            nn.BatchNorm1d(params.n_band*8),
            nn.Tanh(),
            nn.Conv1d(params.n_band*8,params.n_band*16,kernel_size=4*2+1,stride=4,padding=4),
            nn.BatchNorm1d(params.n_band*16),
            nn.Tanh(),
            nn.Conv1d(params.n_band*16,params.n_band*32,kernel_size=3*2+1,stride=3,padding=3),
            nn.BatchNorm1d(params.n_band*32),
            nn.Tanh(),
            nn.Conv1d(params.n_band*32,hidden_dim,1),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
        )

        self.encoder.apply(weights_init)

        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim,params.n_band*32,1),
            nn.BatchNorm1d(params.n_band*32),
            nn.Tanh(),
            nn.ConvTranspose1d(params.n_band*32,params.n_band*16,kernel_size=3*2,stride=3,padding=3//2),
            nn.BatchNorm1d(params.n_band*16),
            nn.Tanh(),
            nn.ConvTranspose1d(params.n_band*16,params.n_band*8,kernel_size=4*2,stride=4,padding=4//2),
            nn.BatchNorm1d(params.n_band*8),
            nn.Tanh(),
            nn.ConvTranspose1d(params.n_band*8,params.n_band*4,kernel_size=5*2,stride=5,padding=5//2),
            nn.BatchNorm1d(params.n_band*4),
            nn.Tanh(),
        )
        self.decoder.apply(weights_init)
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

        x_reconstruct = self.pqmf.inverse(x_reconstruct)
        return x_reconstruct,distance
    
class VQEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VQEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, z):
        #Z = [batch, FeatDim, T]
        #Flat = batch*T, FeatDim
        z_flattened = z.permute(0, 2, 1).contiguous().view(-1, self.embedding_dim)
        #distance = [Z-codebook]*2 = Z^2 - 2*Z*codebook + codebook^2
        #[batch*T,num_embed]
        distances = (torch.sum(z_flattened ** 2, dim=1, keepdim=True)  + torch.sum(self.embedding.weight ** 2, dim=1)
                    - 2 * torch.matmul(z_flattened, self.embedding.weight.t()))

        #batch*T,1
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        #batch*T,num_embed, filled with one and one , one is for selected embed
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)
        # Quantize and unflatten
        #[batch*T,num_embed] * [num_embed,feat_dim] => [batch,T,feat_dim]
        quantized = torch.matmul(encodings, self.embedding.weight).view(z.size())

        # Calculate loss
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Add the gradient back to input
        quantized = z + (quantized - z).detach()
        
        return quantized, loss, encoding_indices
    
    def forward_inference(self, z):
        # Flatten the input z
        z_flattened = z.permute(0, 2, 1).contiguous().view(-1, self.embedding_dim)
        
        # Compute distances between z and embedding vectors
        distances = (torch.sum(z_flattened ** 2, dim=1, keepdim=True) +
                    torch.sum(self.embedding.weight ** 2, dim=1) -
                    2 * torch.matmul(z_flattened, self.embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1)

        # Directly index the embedding weights using encoding_indices
        quantized = self.embedding.weight[encoding_indices].view(z.size())
        # print(quantized.shape)
        # print(z_flattened[0],self.embedding.weight[encoding_indices][0],self.embedding.weight[389])

        return quantized, encoding_indices

class VQAE(nn.Module):
    """Some Information about AE"""
    def __init__(self,params):
        super(VQAE, self).__init__()
        hidden_dim = 16
        self.params = params
        self.spec_distance = AudioDistance(params,params.log_epsilon)
        self.pqmf = PQMF(100,params.n_band)
        self.encoder = nn.Sequential(
            nn.Conv1d(params.n_band,params.n_band*4,7,padding=3),
            nn.BatchNorm1d(params.n_band*4),
            nn.Tanh(),
            nn.Conv1d(params.n_band*4,params.n_band*8,kernel_size=5*2+1,stride=5,padding=5),
            nn.BatchNorm1d(params.n_band*8),
            nn.Tanh(),
            nn.Conv1d(params.n_band*8,params.n_band*16,kernel_size=4*2+1,stride=4,padding=4),
            nn.BatchNorm1d(params.n_band*16),
            nn.Tanh(),
            nn.Conv1d(params.n_band*16,params.n_band*32,kernel_size=3*2+1,stride=3,padding=3),
            nn.BatchNorm1d(params.n_band*32),
            nn.Tanh(),
            nn.Conv1d(params.n_band*32,hidden_dim,1),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
        )
        self.encoder.apply(weights_init)

        self.vq_layer = VQEmbedding(num_embeddings=512, embedding_dim=hidden_dim,commitment_cost=0.1)

        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim,params.n_band*32,1),
            nn.BatchNorm1d(params.n_band*32),
            nn.Tanh(),
            nn.ConvTranspose1d(params.n_band*32,params.n_band*16,kernel_size=3*2,stride=3,padding=3//2),
            nn.BatchNorm1d(params.n_band*16),
            nn.Tanh(),
            nn.ConvTranspose1d(params.n_band*16,params.n_band*8,kernel_size=4*2,stride=4,padding=4//2),
            nn.BatchNorm1d(params.n_band*8),
            nn.Tanh(),
            nn.ConvTranspose1d(params.n_band*8,params.n_band*4,kernel_size=5*2,stride=5,padding=5//2),
            nn.BatchNorm1d(params.n_band*4),
            nn.Tanh(),
        )
        self.decoder.apply(weights_init)
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
        mb_audio = self.pqmf(x)
        z = self.encoder(mb_audio)
        z_q, vq_loss, _ = self.vq_layer(z)
        return z_q, vq_loss, mb_audio
    
    def forward(self, x):
        z_q, vq_loss, mb_audio = self.encode(x)
        x_reconstruct = self.decode(z_q)

        b,c,t = mb_audio.shape
        x_reconstruct = x_reconstruct[:,:,:t]

        input_audio = torch.reshape(mb_audio,(b*c,1,t))
        output_audio = torch.reshape(x_reconstruct,(b*c,1,-1))
        spectral_loss = self.spec_distance(input_audio,output_audio)

        x_reconstruct = self.pqmf.inverse(x_reconstruct)
        
        return x_reconstruct,spectral_loss,vq_loss
    

class AEOld(nn.Module):
    """Some Information about AE"""
    def __init__(self,params):
        super(AEOld, self).__init__()
        hidden_dim = 16
        self.params = params
        self.spec_distance = AudioDistance(params,params.log_epsilon)
        self.pqmf = PQMF(100,params.n_band)
        self.encoder = nn.Sequential(
            nn.Conv1d(params.n_band,params.n_band*4,7,padding=3),
            nn.BatchNorm1d(params.n_band*4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(params.n_band*4,params.n_band*8,kernel_size=4*2+1,stride=4,padding=4),
            nn.BatchNorm1d(params.n_band*8),
            nn.LeakyReLU(0.2),
            nn.Conv1d(params.n_band*8,params.n_band*16,kernel_size=4*2+1,stride=4,padding=4),
            nn.BatchNorm1d(params.n_band*16),
            nn.LeakyReLU(0.2),
            nn.Conv1d(params.n_band*16,params.n_band*32,kernel_size=4*2+1,stride=4,padding=4),
            nn.BatchNorm1d(params.n_band*32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(params.n_band*32,hidden_dim,1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
        )

        self.encoder.apply(weights_init)

        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim,params.n_band*32,1),
            nn.BatchNorm1d(params.n_band*32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(params.n_band*32,params.n_band*16,kernel_size=4*2,stride=4,padding=4//2),
            nn.BatchNorm1d(params.n_band*16),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(params.n_band*16,params.n_band*8,kernel_size=4*2,stride=4,padding=4//2),
            nn.BatchNorm1d(params.n_band*8),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(params.n_band*8,params.n_band*4,kernel_size=4*2,stride=4,padding=4//2),
            nn.BatchNorm1d(params.n_band*4),
            nn.LeakyReLU(0.2),
        )
        self.decoder.apply(weights_init)
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

        x_reconstruct = self.pqmf.inverse(x_reconstruct)
        return x_reconstruct,distance
    

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input
        return out
    
class SpecEncoder(nn.Module):
    def __init__(self, channel,in_channel=1, n_res_block=4, n_res_channel=32):
        super().__init__()
        sf = 2 #2^sf will be the reduction ratio
        self.blocks = nn.ModuleList()
        for i in range(n_res_block):
            kernel_size = 2*(i+1)
            block = [
                nn.Conv2d(in_channel, channel // sf, kernel_size, stride=2, padding=i),
                nn.ReLU(),
                nn.Conv2d(channel // sf, channel, kernel_size, stride=2, padding=i),
                nn.ReLU(),
                nn.Conv2d(channel, channel, 3, padding=1),
                ResBlock(channel, n_res_channel),
                nn.ReLU()
            ]
            self.blocks.append(nn.Sequential(*block))

    def forward(self, input):
        output = self.blocks[0](input) + self.blocks[1](input) + self.blocks[2](input) + self.blocks[3](input) 
        return output


class SpecDecoder(nn.Module):
    def __init__(self, in_channel,out_channel,channel, n_res_channel=32):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channel, channel, 3, padding=1),
            ResBlock(channel, n_res_channel),
            ResBlock(channel, n_res_channel),
            ResBlock(channel, n_res_channel),
            ResBlock(channel, n_res_channel),
            nn.ReLU(),
            nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1)
        )

    def forward(self, z_q):
        return self.decoder(z_q)

class VQSpecAE(nn.Module):
    """Some Information about AE"""
    def __init__(self,embed_dim=64):
        super(VQSpecAE, self).__init__()
        self.encoder = SpecEncoder(channel=128)
        self.encoder.apply(weights_init)
        self.mapper = nn.Conv2d(128,embed_dim,1)
        self.decoder = SpecDecoder(in_channel=embed_dim,out_channel=1,channel=128)
        self.decoder.apply(weights_init)
        self.vq_layer = VQEmbedding(num_embeddings=512, embedding_dim=embed_dim,commitment_cost=0.1)


    def encode(self,x):
        z = self.encoder(x) #Batch,128,Height/4,Width/4
        z = self.mapper(z) #Batch,embed_dim,Height/4,Width/4
        b,embed,h,w = z.shape
        z = torch.reshape(z,(b,embed,-1))
        z_q, vq_loss, _ = self.vq_layer(z)#Batch size, embed_dim, 20*100
        z_q = torch.reshape(z,(b,embed,h,w))#Batch size, embed_dim, 20, 100
        return z_q, vq_loss

    def encode_inference(self,x):
        z = self.encoder(x) #Batch,128,Height/4,Width/4
        z = self.mapper(z) #Batch,embed_dim,Height/4,Width/4
        b,embed,h,w = z.shape
        z = torch.reshape(z,(b,embed,h*w))
        z_q, _ = self.vq_layer.forward_inference(z)#Batch size, embed_dim, 20*100
        z_q = torch.reshape(z,(b,embed,h,w))#Batch size, embed_dim, 20, 100
        return z_q
    
    def decode(self,x):
        return self.decoder(x)

    def forward(self, x):
        #Spectrogram: Batch,1,Height,Width
        z_q, vq_loss = self.encode(x) #Batch,embed_dim,Height/4,Width/4
        y = self.decode(z_q)
        return y, vq_loss
    

class VQAEExtend(nn.Module):
    """Some Information about AE"""
    def __init__(self,params):
        super().__init__()
        hidden_dim = 16
        self.params = params
        self.spec_distance = AudioDistance(params,params.log_epsilon)
        self.pqmf = PQMF(100,params.n_band)
        self.encoder = nn.Sequential(
            nn.Conv1d(params.n_band,params.n_band*4,7,padding=3),
            nn.BatchNorm1d(params.n_band*4),
            nn.Tanh(),
            nn.Conv1d(params.n_band*4,params.n_band*8,kernel_size=5*2+1,stride=5,padding=5),
            nn.BatchNorm1d(params.n_band*8),
            nn.Tanh(),
            nn.Conv1d(params.n_band*8,params.n_band*16,kernel_size=4*2+1,stride=4,padding=4),
            nn.BatchNorm1d(params.n_band*16),
            nn.Tanh(),
            nn.Conv1d(params.n_band*16,params.n_band*32,kernel_size=3*2+1,stride=3,padding=3),
            nn.BatchNorm1d(params.n_band*32),
            nn.Tanh(),
            nn.Conv1d(params.n_band*32,hidden_dim*4,1),
            nn.BatchNorm1d(hidden_dim*4),
            nn.Tanh(),
        )
        self.encoder.apply(weights_init)

        self.vq_layer = VQEmbedding(num_embeddings=512, embedding_dim=hidden_dim,commitment_cost=0.1)

        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim*4,params.n_band*32,1),
            nn.BatchNorm1d(params.n_band*32),
            nn.Tanh(),
            nn.ConvTranspose1d(params.n_band*32,params.n_band*16,kernel_size=3*2,stride=3,padding=3//2),
            nn.BatchNorm1d(params.n_band*16),
            nn.Tanh(),
            nn.ConvTranspose1d(params.n_band*16,params.n_band*8,kernel_size=4*2,stride=4,padding=4//2),
            nn.BatchNorm1d(params.n_band*8),
            nn.Tanh(),
            nn.ConvTranspose1d(params.n_band*8,params.n_band*4,kernel_size=5*2,stride=5,padding=5//2),
            nn.BatchNorm1d(params.n_band*4),
            nn.Tanh(),
        )
        self.decoder.apply(weights_init)
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
        mb_audio = self.pqmf(x)
        z = self.encoder(mb_audio)
        # z = torch.chunk(z,chunks=4,dim=1,)
        z_q, vq_loss, _ = self.vq_layer(z)
        return z_q, vq_loss, mb_audio
    
    def forward(self, x):
        z_q, vq_loss, mb_audio = self.encode(x)
        x_reconstruct = self.decode(z_q)

        b,c,t = mb_audio.shape
        x_reconstruct = x_reconstruct[:,:,:t]

        input_audio = torch.reshape(mb_audio,(b*c,1,t))
        output_audio = torch.reshape(x_reconstruct,(b*c,1,-1))
        spectral_loss = self.spec_distance(input_audio,output_audio)

        x_reconstruct = self.pqmf.inverse(x_reconstruct)
        
        return x_reconstruct,spectral_loss,vq_loss
    

class VQAEFuse(nn.Module):
    """Some Information about VQAEFuse"""
    def __init__(self,params):
        super().__init__()
        hidden_dim = 16
        self.params = params
        self.spec_distance = AudioDistance(params,params.log_epsilon)
        self.pqmf = PQMF(100,params.n_band)
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(params.n_band,params.n_band*4,7,padding=3),
            nn.BatchNorm1d(params.n_band*4),
            nn.Tanh(),
            nn.Conv1d(params.n_band*4,params.n_band*8,kernel_size=5*2+1,stride=5,padding=5),
            nn.BatchNorm1d(params.n_band*8),
            nn.Tanh(),
            nn.Conv1d(params.n_band*8,params.n_band*16,kernel_size=4*2+1,stride=4,padding=4),
            nn.BatchNorm1d(params.n_band*16),
            nn.Tanh(),
            nn.Conv1d(params.n_band*16,params.n_band*32,kernel_size=3*2+1,stride=3,padding=3),
            nn.BatchNorm1d(params.n_band*32),
            nn.Tanh(),
            nn.Conv1d(params.n_band*32,hidden_dim,1),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
        )
        
        self.melspec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=48*1000, 
            n_fft=400,
            win_length=400,
            hop_length=240,
            n_mels=80  
        )
        self.spec_encoder = SpecEncoder(channel=128)
        self.spec_encoder.apply(weights_init)
        self.mapper = nn.Sequential(
            nn.Conv2d(128,16,1),
            nn.BatchNorm1d(16),
            nn.Conv2d(16,16,kernel_size=(20, 1),stride=(1, 1),padding=(0, 0)),
            nn.BatchNorm1d(16)
        )


        self.audio_encoder.apply(weights_init)

        self.vq_layer = VQEmbedding(num_embeddings=512, embedding_dim=hidden_dim,commitment_cost=0.1)

        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim,params.n_band*32,1),
            nn.BatchNorm1d(params.n_band*32),
            nn.Tanh(),
            nn.ConvTranspose1d(params.n_band*32,params.n_band*16,kernel_size=3*2,stride=3,padding=3//2),
            nn.BatchNorm1d(params.n_band*16),
            nn.Tanh(),
            nn.ConvTranspose1d(params.n_band*16,params.n_band*8,kernel_size=4*2,stride=4,padding=4//2),
            nn.BatchNorm1d(params.n_band*8),
            nn.Tanh(),
            nn.ConvTranspose1d(params.n_band*8,params.n_band*4,kernel_size=5*2,stride=5,padding=5//2),
            nn.BatchNorm1d(params.n_band*4),
            nn.Tanh(),
        )
        self.decoder.apply(weights_init)
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
        melspec = self.melspec_transform(x)
        mb_audio = self.pqmf(x)
        z_audio = self.audio_encoder(mb_audio)
        z_spec = self.spec_encoder(melspec)
        z_spec = self.mapper(z_spec)
        z_spec = torch.squeeze(z_spec,2)

        min_size = min(z_audio.shape[-1],z_spec.shape[-1])
        z = z_audio[:,:,:min_size] + z_spec[:,:,:min_size]
        
        z_q, vq_loss, _ = self.vq_layer(z)
        return z_q, vq_loss, mb_audio, melspec
    
    def forward(self, x):
        z_q, vq_loss, mb_audio,mel_spec_real = self.encode(x)
        x_reconstruct = self.decode(z_q)

        b,c,t = mb_audio.shape
        x_reconstruct = x_reconstruct[:,:,:t]

        min_size = min(x_reconstruct.shape[-1],mb_audio.shape[-1])
        mb_audio = mb_audio[:,:,:min_size]
        x_reconstruct = x_reconstruct[:,:,:min_size]

        input_audio = torch.reshape(mb_audio,(b*c,1,-1))
        output_audio = torch.reshape(x_reconstruct,(b*c,1,-1))
        
        audio_loss = self.spec_distance(input_audio,output_audio)

        x_reconstruct = self.pqmf.inverse(x_reconstruct)
        mel_spec_fake = self.melspec_transform(x_reconstruct)

        min_size = min(mel_spec_fake.shape[-1],mel_spec_real.shape[-1])
        mel_spec_fake = mel_spec_fake[:,:,:,:min_size]
        mel_spec_real = mel_spec_real[:,:,:,:min_size]
        spectral_loss = F.mse_loss(mel_spec_fake,mel_spec_real)
        
        return x_reconstruct,audio_loss,vq_loss,spectral_loss
    

class VQAESeq(nn.Module):
    """Some Information about VQAEFuse"""
    def __init__(self,params,embed_dim=16):
        super().__init__()
        self.params = params
        self.spec_distance = AudioDistance(params,params.log_epsilon)
        self.pqmf = PQMF(100,params.n_band)

        
        self.melspec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=48*1000, 
            n_fft=400,
            win_length=400,
            hop_length=240,
            n_mels=80  
        )
        self.spec_ae = VQSpecAE(embed_dim)
        self.spec_ae.apply(weights_init)
        self.mapper = nn.Sequential(
            nn.Conv1d(80,16,1),
            nn.BatchNorm1d(16),
            nn.Conv1d(16,16,kernel_size=4,stride=4,padding=0),
            nn.BatchNorm1d(16)
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(params.n_band,params.n_band*32,1),
            nn.BatchNorm1d(params.n_band*32),
            nn.Tanh(),
            nn.ConvTranspose1d(params.n_band*32,params.n_band*16,kernel_size=3*2,stride=3,padding=3//2),
            nn.BatchNorm1d(params.n_band*16),
            nn.Tanh(),
            nn.ConvTranspose1d(params.n_band*16,params.n_band*8,kernel_size=4*2,stride=4,padding=4//2),
            nn.BatchNorm1d(params.n_band*8),
            nn.Tanh(),
            nn.ConvTranspose1d(params.n_band*8,params.n_band*4,kernel_size=5*2,stride=5,padding=5//2),
            nn.BatchNorm1d(params.n_band*4),
            nn.Tanh(),
        )
        self.decoder.apply(weights_init)
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
    
    def encode_inference(self,x):
        melspec = self.melspec_transform(x)
        z_q = self.spec_ae.encode_inference(melspec)
        return z_q
    
    def decode_inference(self,z_q):
        spec = self.spec_ae.decode(z_q)
        latent = self.mapper(spec.squeeze(1))
        audio = self.decode(latent)
        audio = self.pqmf.inverse(audio)
        return audio
    
    def forward(self, x):

        melspec_real = self.melspec_transform(x)
        melspec_fake, vq_loss = self.spec_ae(melspec_real)
        
        latent_temp = self.mapper(melspec_fake.squeeze(1))

        audio_fake = self.decode(latent_temp)
        audio_real = self.pqmf(x)
        b,c,t = audio_real.shape

        min_size = min(audio_fake.shape[-1],audio_real.shape[-1])
        audio_fake = audio_fake[:,:,:min_size] 
        audio_real = audio_real[:,:,:min_size] 
        audio_loss = self.spec_distance(audio_fake,audio_real)

        min_size = min(melspec_fake.shape[-1],melspec_real.shape[-1])
        melspec_fake = melspec_fake[:,:,:,:min_size]
        melspec_real = melspec_real[:,:,:,:min_size]
        spectral_loss = F.mse_loss(melspec_real,melspec_fake)

        audio_fake = self.pqmf.inverse(audio_fake)
        
        return audio_fake,audio_loss,vq_loss,spectral_loss
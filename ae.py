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
        #[batch*T,num_embed] * [num_embed,feat_dim] => [batch*T,feat_dim]
        quantized = torch.matmul(encodings, self.embedding.weight)

        b,feat_dim,t = z.shape
        #[batch*T,num_embed] => [batch,T,num_embed] => [batch,num_embed,T]
        quantized = quantized.view(b,t,feat_dim).permute(0, 2, 1).contiguous()

        # Calculate loss
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Add the gradient back to input
        quantized = z + (quantized - z).detach()
        
        return quantized, loss, encoding_indices
    
    def forward_inference(self, z):
        # Flatten the input z
        #[batch,num_embed,T]
        z_flattened = z.permute(0, 2, 1).contiguous().view(-1, self.embedding_dim)
        #distance = [Z-codebook]*2 = Z^2 - 2*Z*codebook + codebook^2
        #[batch*T,num_embed]
        distances = (torch.sum(z_flattened ** 2, dim=1, keepdim=True)  + torch.sum(self.embedding.weight ** 2, dim=1)
                    - 2 * torch.matmul(z_flattened, self.embedding.weight.t()))

        #batch*T,1
        # quantized = self.embedding.weight[encoding_indices.squeeze()]
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)
        # Quantize and unflatten
        #[batch*T,num_embed] * [num_embed,feat_dim]
        quantized = torch.matmul(encodings, self.embedding.weight)#[batch*T,feat_dim]
        b,feat_dim,t = z.shape
        #[batch*T,num_embed] => [batch,T,num_embed] => [batch,num_embed,T]
        quantized = quantized.view(b,t,feat_dim).permute(0, 2, 1).contiguous()

        return quantized, encoding_indices


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())
    def encode(self,input):
        flatten = input.reshape(-1, self.dim)
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)
        return quantize
    
    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

class VQAE(nn.Module):
    def __init__(self,params,embed_dim=64):
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

        self.spec_encoder = SpecEncoder(channel=128)
        self.spec_encoder.apply(weights_init)
        self.spec_mapper = nn.Conv2d(128,embed_dim,1)

        # self.vq_layer = VQEmbedding(512,embed_dim,0.25)
        self.vq_layer = Quantize(64,512)
        
        self.spec_decoder = SpecDecoder(in_channel=embed_dim,out_channel=1,channel=128)
        self.spec_decoder.apply(weights_init)
        # self.spec_vqae = VQSpecAE(16)
        
        self.audio_mapper = nn.Sequential(
            nn.Conv1d(80,16,1),
            nn.BatchNorm1d(16),
            nn.Conv1d(16,16,kernel_size=4,stride=4,padding=0),
            nn.BatchNorm1d(16)
        )
        self.audio_mapper.apply(weights_init)

        self.audio_decoder = nn.Sequential(
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
        self.audio_decoder.apply(weights_init)
        self.wave_gen = nn.Conv1d(params.n_band*4,params.n_band,7,padding=3)
        self.loud_gen = nn.Conv1d(params.n_band*4,params.n_band,3,1,padding=1)
        
    def mod_sigmoid(self,x):
        return 2 * torch.sigmoid(x)**2.3 + 1e-7
    
    def audio_decode(self,x):
        z_ = self.audio_decoder(x)
        loud = self.loud_gen(z_)
        wave = self.wave_gen(z_)
        x = torch.tanh(wave) *  self.mod_sigmoid(loud)
        return x
    
    
    def decode_inference(self,z_q,b,embed_dim,H,T):
        z_q = z_q.transpose(1,2) #Batch,embed_dim,Height/4*T/4
        z_q = z_q.reshape(b,embed_dim, H , T)#Batch,embed_dim,Height/4*T/4
        melspec_f = self.spec_decoder(z_q) #Batch,1,80,T

        audio_f = self.audio_mapper(melspec_f.squeeze(1))#Batch,16,A//16
        audio_f = self.audio_decode(audio_f)#Batch,16,A//16
        audio = self.pqmf.inverse(audio_f)
        return audio
    
    def equal_size(self,a:torch.Tensor,b:torch.Tensor):
        min_size = min(a.shape[-1],b.shape[-1])
        a_truncated = a[..., :min_size]  # Keep all dimensions except truncate last dimension
        b_truncated = b[..., :min_size]  # Same truncation for b
        return a_truncated, b_truncated

    def encode_inference(self,x):
        melspec_r = self.melspec_transform(x)#Batch,1,80,T
        
        z = self.spec_encoder(melspec_r) #Batch,128,Height/4,Width/4
        z = self.spec_mapper(z) #Batch,embed_dim,Height/4,Width/4
        # b,embed,h,w = z.shape
        z_q = z.permute(0, 2, 3, 1)#height, width,channel
        # z = torch.reshape(z,(b,embed,h*w))
        z_q, vq_loss, _ = self.vq_layer(z_q)#height, width, channel
        z_q = z_q.permute(0, 3, 1, 2)#channel, height, width
        return z_q

    def forward(self, x):
        #x is audio, X:[Batch,A]
        melspec_r = self.melspec_transform(x)#Batch,1,80,T

        z = self.spec_encoder(melspec_r) #Batch,128,Height/4,Width/4
        z = self.spec_mapper(z) #Batch,embed_dim,Height/4,Width/4
        # b,embed,h,w = z.shape
        z_q = z.permute(0, 2, 3, 1)#height, width,channel
        # z = torch.reshape(z,(b,embed,h*w))
        z_q, vq_loss, _ = self.vq_layer(z_q)#height, width, channel
        z_q = z_q.permute(0, 3, 1, 2)#channel, height, width
        # z_q = torch.reshape(z_q,(b,embed,h,w))#Batch size, embed_dim, 20, 100
        melspec_f = self.spec_decoder(z_q)

        # melspec_f, vq_loss = self.spec_vqae(melspec_r)

        audio_f = self.audio_mapper(melspec_f.squeeze(1))#Batch,16,A//16
        audio_f = self.audio_decode(audio_f)#Batch,16,A//16
        audio_r = self.pqmf(x)

        audio_r,audio_f = self.equal_size(audio_r,audio_f)
        melspec_r,melspec_f = self.equal_size(melspec_r,melspec_f)

        audio_loss = self.spec_distance(audio_r,audio_f)
        # audio_loss = self.spec_distance(audio_r,audio_r)
        spectral_loss = F.mse_loss(melspec_r,melspec_f)
        
        
        audio = self.pqmf.inverse(audio_f)
        return audio,audio_loss,vq_loss,spectral_loss

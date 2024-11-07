import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from ae import Quantize,PQMF,AudioDistance
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, zero_out=False, res_scale=1.0):
        super().__init__()
        padding = dilation
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(n_in, n_state, 3, 1, padding, dilation),
            nn.ReLU(),
            nn.Conv1d(n_state, n_in, 1, 1, 0),
        )
        if zero_out:
            out = self.model[-1]
            nn.init.zeros_(out.bias)
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.model(x)


class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, m_conv=1.0, dilation_growth_rate=1, dilation_cycle=None, zero_out=False, res_scale=False, reverse_dilation=False):
        super().__init__()
        def _get_depth(depth):
            if dilation_cycle is None:
                return depth
            else:
                return depth % dilation_cycle
            
        blocks = [ResConv1DBlock(n_in, int(m_conv * n_in),
                                 dilation=dilation_growth_rate ** _get_depth(depth),
                                 zero_out=zero_out,
                                 res_scale=1.0 if not res_scale else 1.0 / math.sqrt(n_depth))
                  for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
        

class EncoderBlock(nn.Module):
    def __init__(self,input_dim,output_dim, down_t,stride_t,hidden_dim, depth, m_conv,
                 dilation_growth_rate=1, dilation_cycle=None, zero_out=False,
                 res_scale=False):
        super(EncoderBlock, self).__init__()
        blocks = []
        filter_t, pad_t = stride_t *2, stride_t // 2
        if down_t > 0:
            for i in range(down_t):
                block = nn.Sequential(
                    nn.Conv1d(input_dim if i == 0 else hidden_dim,hidden_dim,kernel_size=filter_t,stride=stride_t,padding=pad_t),
                    Resnet1D(hidden_dim, depth, m_conv, dilation_growth_rate, dilation_cycle, zero_out, res_scale),
                )
                blocks.append(block)
            block = nn.Conv1d(hidden_dim, output_dim, 3, 1, 1)
            blocks.append(block)
        self.model = nn.Sequential(*blocks)


    def forward(self, x):
        return self.model(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim, output_dim, down_t,
                 stride_t, hidden_dim, depth, m_conv, dilation_growth_rate=1, dilation_cycle=None, zero_out=False, res_scale=False, reverse_decoder_dilation=False):
        super().__init__()
        blocks = []
        if down_t > 0:
            filter_t, pad_t = stride_t * 2, stride_t // 2
            block = nn.Conv1d(output_dim, hidden_dim, 3, 1, 1)
            blocks.append(block)
            for i in range(down_t):
                block = nn.Sequential(
                    Resnet1D(hidden_dim, depth, m_conv, dilation_growth_rate, dilation_cycle, zero_out=zero_out, res_scale=res_scale, reverse_dilation=reverse_decoder_dilation),
                    nn.ConvTranspose1d(output_dim, input_dim if i == (down_t - 1) else hidden_dim, filter_t, stride_t, pad_t)
                )
                blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
from ae import ResidualBlock






class Encoder(nn.Module):
    """Some Information about Encoder"""
    def __init__(self,in_channel,out_channel,hidden_channel):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channel,hidden_channel,kernel_size=4*2+1,stride=4,padding=4),
            nn.BatchNorm1d(hidden_channel),
            nn.LeakyReLU(0.2),
            ResidualBlock(hidden_channel,hidden_channel,dilation=2,kernel_size=3),
            nn.BatchNorm1d(hidden_channel),
            nn.LeakyReLU(0.2),
            ResidualBlock(hidden_channel,out_channel,dilation=2,kernel_size=3),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(0.2),
        )
    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    """Some Information about Decoder"""
    def __init__(self,in_channel,out_channel,hidden_channel):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            ResidualBlock(in_channel,hidden_channel,dilation=2,kernel_size=3),
            nn.BatchNorm1d(hidden_channel),
            nn.LeakyReLU(0.2),
            ResidualBlock(hidden_channel,hidden_channel,dilation=2,kernel_size=3),
            nn.BatchNorm1d(hidden_channel),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(hidden_channel,out_channel,kernel_size=4*2,stride=4,padding=4//2),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.model(x)


class VQAE(nn.Module):
    """Some Information about VQAE"""
    def __init__(self,params):
        super().__init__()
        self.pqmf_channel = 16
        self.hidden_dim = 64
        self.spec_distance = AudioDistance(params,params.log_epsilon)
        self.pqmf = PQMF(100,n_band=self.pqmf_channel)

        self.encoder1 = Encoder(self.pqmf_channel,self.hidden_dim,self.hidden_dim)
        self.vq_layer1 = Quantize(self.hidden_dim,2048)
        self.decoder1 = Decoder(self.hidden_dim,self.pqmf_channel,self.hidden_dim)
        self.encoder1.apply(weights_init)
        self.decoder1.apply(weights_init)

        self.encoder2 = Encoder(self.hidden_dim,self.hidden_dim,self.hidden_dim)
        self.vq_layer2 = Quantize(self.hidden_dim,2048)
        self.decoder2 = Decoder(self.hidden_dim,self.hidden_dim,self.hidden_dim)
        self.decoder2_audio = Decoder(self.hidden_dim,self.pqmf_channel,self.hidden_dim)
        self.encoder2.apply(weights_init)
        self.decoder2.apply(weights_init)
        self.decoder2_audio.apply(weights_init)
        self.upsampler = nn.Sequential(
            Decoder(self.hidden_dim,self.hidden_dim,self.hidden_dim),
            ResidualBlock(self.hidden_dim,self.hidden_dim,dilation=2,kernel_size=3),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),
            ResidualBlock(self.hidden_dim,self.hidden_dim,dilation=2,kernel_size=3),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),
            ResidualBlock(self.hidden_dim,self.hidden_dim,dilation=2,kernel_size=3),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),
            ResidualBlock(self.hidden_dim,self.hidden_dim,dilation=2,kernel_size=3),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),
        )

        self.upsampler2 = nn.Sequential(
            ResidualBlock(self.hidden_dim,self.hidden_dim,dilation=2,kernel_size=3),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),
            ResidualBlock(self.hidden_dim,self.hidden_dim,dilation=2,kernel_size=3),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),
            ResidualBlock(self.hidden_dim,self.hidden_dim,dilation=2,kernel_size=3),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),
            ResidualBlock(self.hidden_dim,self.hidden_dim,dilation=2,kernel_size=3),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),
        )

        self.wave_gen = nn.Conv1d(self.pqmf_channel,self.pqmf_channel,7,padding=3)
        self.loud_gen = nn.Conv1d(self.pqmf_channel,self.pqmf_channel,3,1,padding=1)

    def mod_sigmoid(self,x):
        return 2 * torch.sigmoid(x)**2.3 + 1e-7
    
    def equal_size(self,a:torch.Tensor,b:torch.Tensor):
        min_size = min(a.shape[-1],b.shape[-1])
        a_truncated = a[..., :min_size]  # Keep all dimensions except truncate last dimension
        b_truncated = b[..., :min_size]  # Same truncation for b
        return a_truncated, b_truncated
    
    def quant(self,z,vq_layer):
        z = z.permute(0,2,1)  
        zq,vq_loss,_ = vq_layer(z)
        return zq.permute(0,2,1), vq_loss

    def inference(self,x):
        pqmf_audio = self.pqmf(x)

        z1 = self.encoder1(pqmf_audio)
        z_q1,_ = self.quant(z1,self.vq_layer1)
        # pqmf_audio = self.decoder1(z_q1)


        z1 = z_q1.detach()
        z2 = self.encoder2(z1)
        z_q2,_ = self.quant(z2,self.vq_layer2)
        z_q1f = self.upsampler(z_q2)

        z_q1f,_ = self.quant(z_q1f,self.vq_layer1)
        
        pqmf_audio = self.decoder1(z_q1f)
        pqmf_audio = self.decode_audio(pqmf_audio)
        return self.pqmf.inverse(pqmf_audio)

    def decode_audio(self,x):
        loud = self.loud_gen(x)
        wave = self.wave_gen(x)
        return torch.tanh(wave) *  self.mod_sigmoid(loud)


    def train_sampler(self,x):
        pqmf_audio = self.pqmf(x)
        z1 = self.encoder1(pqmf_audio)
        z_q1,_ = self.quant(z1,self.vq_layer1)

        z1 = z_q1.detach()
        z2 = self.encoder2(z1)
        z_q2,_ = self.quant(z2,self.vq_layer2)

        z_q1f = self.upsampler(z_q2.detach())
        z_q1f = self.upsampler2(z_q1f)
        z_q1f,z1 = self.equal_size(z_q1f,z1)
        feature_loss2 = F.mse_loss(z_q1f,z1)
        return feature_loss2



    def forward(self, x):
        pqmf_audio = self.pqmf(x)
        z1 = self.encoder1(pqmf_audio)
        z_q1,vq_loss1 = self.quant(z1,self.vq_layer1)
        #level 1 loss
        pqmf_audio1 = self.decoder1(z_q1)
        pqmf_audio1, pqmf_audio = self.equal_size(pqmf_audio1,pqmf_audio)
        pqmf_audio1 = self.decode_audio(pqmf_audio1)
        audio_loss1 = self.spec_distance(pqmf_audio,pqmf_audio1)
        #layer1 loss = audio_loss1 + vq_loss1

        z1 = z_q1.detach()
        z2 = self.encoder2(z1)
        z_q2,vq_loss2 = self.quant(z2,self.vq_layer2)
        z_q1f = self.decoder2(z_q2)
        z_q1f,z_q1 = self.equal_size(z_q1f,z_q1)
        z_q1f,_ = self.quant(z_q1f,self.vq_layer1)

        pqmf_audio2 = self.decoder2_audio(z_q1f)
        pqmf_audio2, pqmf_audio = self.equal_size(pqmf_audio2,pqmf_audio)

        pqmf_audio2 = self.decode_audio(pqmf_audio2)
        audio_loss2 = self.spec_distance(pqmf_audio,pqmf_audio2)

        vq_loss = vq_loss1 + vq_loss2
        audio_loss = audio_loss1 + audio_loss2

        z1 = z_q1.detach()
        z2 = z_q2.detach()
        z1_f = self.upsampler(z2)
        z1_f,z1 = self.equal_size(z1_f,z1)
        feature_loss2 = F.mse_loss(z1_f,z1)
        feature_loss = feature_loss2

        return self.pqmf.inverse(pqmf_audio2),audio_loss,vq_loss,feature_loss
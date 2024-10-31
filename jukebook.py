import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from ae import Quantize,PQMF,AudioDistance

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
            nn.init.zeros_(out.weight)
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
    

class VQAE(nn.Module):
    """Some Information about VQAE"""
    def __init__(self,params):
        super().__init__()
        self.pqmf_channel = 16
        self.hidden_dim = 64
        self.spec_distance = AudioDistance(params,params.log_epsilon)
        self.pqmf = PQMF(100,n_band=self.pqmf_channel)
        self.encoder = EncoderBlock(input_dim=self.pqmf_channel
                                    ,output_dim=self.hidden_dim,
                                    down_t=1,stride_t=2,hidden_dim=self.hidden_dim,
                                    depth=4,m_conv=10,dilation_growth_rate=3)
        self.vq_layer = Quantize(self.hidden_dim,2048)
        self.decoder = DecoderBlock(input_dim=self.pqmf_channel
                                    ,output_dim=self.hidden_dim,
                                    down_t=1,stride_t=2,hidden_dim=self.hidden_dim,
                                    depth=4,m_conv=10,dilation_growth_rate=3)

    def forward(self, x):
        pqmf_audio = self.pqmf(x)
        z = self.encoder(pqmf_audio)
        z = z.permute(0,2,1)
        z_q,vq_loss,_ = self.vq_layer(z)
        z_q = z_q.permute(0,2,1)
        pqmf_audio_f = self.decoder(z_q)
        audio_loss = self.spec_distance(pqmf_audio,pqmf_audio_f)
        return self.pqmf.inverse(pqmf_audio_f),audio_loss,vq_loss
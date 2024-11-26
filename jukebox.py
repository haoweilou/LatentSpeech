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
    def __init__(self,in_channel,out_channel,hidden_channel,ratio:int=4):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channel,hidden_channel,kernel_size=ratio*2+1,stride=ratio,padding=ratio),
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
    def __init__(self,in_channel,out_channel,hidden_channel,ratio:int=4):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            ResidualBlock(in_channel,hidden_channel,dilation=2,kernel_size=3),
            nn.BatchNorm1d(hidden_channel),
            nn.LeakyReLU(0.2),
            ResidualBlock(hidden_channel,hidden_channel,dilation=2,kernel_size=3),
            nn.BatchNorm1d(hidden_channel),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(hidden_channel,out_channel,kernel_size=ratio*2,stride=ratio,padding=ratio//2),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.model(x)

class Permute(nn.Module):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

class UpSampler(nn.Module):
    """Some Information about UpSampler"""
    def __init__(self,feature_dim, hidden_dim, num_res_layer,ratio:int=4):
        super(UpSampler, self).__init__()
        model_list = [
                nn.Conv1d(feature_dim,hidden_dim,1),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2)]
        
        for _ in range(num_res_layer):
            model_list += [
                ResidualBlock(hidden_dim,hidden_dim,dilation=2,kernel_size=3),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
            ]
        model_list += [Decoder(hidden_dim,feature_dim,hidden_dim,ratio),
            Permute(0, 2, 1),
            nn.LayerNorm(feature_dim),
            Permute(0, 2, 1)
        ]
        
        self.model = nn.Sequential(
            *model_list
        )
        
    def forward(self, x):
        return self.model(x)

class ResNet(nn.Module):
    def __init__(self, in_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, dilation=dilation, padding=dilation)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, dilation=dilation, padding=dilation)
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        return self.activation(x + residual)
    
class UpSampler3(nn.Module):
    def __init__(self,feature_dim, hidden_dim, num_res_layer,ratio:int=4):
        super(UpSampler3, self).__init__()
        self.pre_upsample_conv = nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1)
        self.upsample = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=ratio, stride=ratio)
        self.residual_blocks = nn.ModuleList([
            ResNet(hidden_dim) for _ in range(num_res_layer)
        ])
        self.post_upsample_conv = nn.Conv1d(hidden_dim, feature_dim, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2)  # Add activation after pre-convolution

    
    def forward(self, x):
        x = self.activation(self.pre_upsample_conv(x))
        
        x = self.upsample(x)
        x = self.activation(x)  
        
        for block in self.residual_blocks:
            x = block(x)
        
        x = self.post_upsample_conv(x)
        return x

class VQAE(nn.Module):
    """Some Information about VQAE"""
    def __init__(self,ratios):
        super(VQAE, self).__init__()
        encoders = []
        decoders = []
        self.pqmf_channel = 16
        self.hidden_dim = 64
        for i,r in enumerate(ratios):
            if i == 0:
                encoder = Encoder(self.pqmf_channel,self.hidden_dim,self.hidden_dim,r)
                decoder = Decoder(self.hidden_dim,self.pqmf_channel,self.hidden_dim,r)
            else: 
                encoder = Encoder(self.hidden_dim,self.hidden_dim,self.hidden_dim,r)
                decoder = Decoder(self.hidden_dim,self.hidden_dim,self.hidden_dim,r)
            encoders.append(encoder)
            decoders.append(decoder)
        self.vq_layer = Quantize(self.hidden_dim,2048)
        decoders = decoders[::-1]
        self.encoder = nn.Sequential(*encoders)
        self.decoder = nn.Sequential(*decoders)

    def encode(self,x):
        z = self.encoder(x)
        zq,_ = self.quant(z)
        return zq
    
    def quant(self,z):
        z = z.permute(0,2,1)  
        zq,vq_loss,_ = self.vq_layer(z)
        return zq.permute(0,2,1), vq_loss
    
    def forward(self, x):
        z = self.encoder(x)
        zq,vq_loss = self.quant(z)
        x_f = self.decoder(zq)
        return x_f, vq_loss

class Jukebox(nn.Module):
    """Some Information about Jukebox"""
    def __init__(self,params):
        super(Jukebox, self).__init__()
        self.pqmf_channel = 16
        self.hidden_dim = 64
        self.spec_distance = AudioDistance(params,params.log_epsilon)
        self.pqmf = PQMF(100,n_band=self.pqmf_channel)
        self.vqae1 = VQAE([4])
        self.vqae1.apply(weights_init)
        self.vqae2 = VQAE([4,4])
        self.vqae2.apply(weights_init)
        self.vqae3 = VQAE([4,4,4])
        self.vqae3.apply(weights_init)

        self.upsampler1 = UpSampler(self.hidden_dim,256,num_res_layer=16,ratio=4)
        self.upsampler2 = UpSampler(self.hidden_dim,256,num_res_layer=16,ratio=4)
        
        self.wave_gen = nn.Conv1d(self.pqmf_channel,self.pqmf_channel,7,padding=3)
        self.loud_gen = nn.Conv1d(self.pqmf_channel,self.pqmf_channel,3,1,padding=1)
    
    
    def equal_size(self,a:torch.Tensor,b:torch.Tensor):
        min_size = min(a.shape[-1],b.shape[-1])
        a_truncated = a[..., :min_size]  # Keep all dimensions except truncate last dimension
        b_truncated = b[..., :min_size]  # Same truncation for b
        return a_truncated, b_truncated
    
    def mod_sigmoid(self,x):
        return 2 * torch.sigmoid(x)**2.3 + 1e-7
    
    def upsample(self,x):
        z3q = self.vqae3.encode(x)
        z2q_f = self.upsampler2(z3q.detach())#3=>2
        z2q,_ = self.vqae2.quant(z2q_f)

        # z2q = self.vqae2.encode(x)
        pqmf_audio = self.vqae2.decoder(z2q)

        z1q = self.upsampler1(z2q.detach())#2=>1
        z1q,_ = self.vqae1.quant(z1q)
        pqmf_audio = self.vqae1.decoder(z1q)
        pqmf_audio = self.decode_audio(pqmf_audio)
        return self.pqmf.inverse(pqmf_audio)
    
    def upsample1(self,x,upsampler):
        z3q = self.vqae3.encode(x)
        z1q_f = upsampler(z3q.detach())#3=>2
        z1q,_ = self.vqae1.quant(z1q_f)

        # z2q = self.vqae2.encode(x)
        # pqmf_audio = self.vqae2.decoder(z2q)

        pqmf_audio = self.vqae1.decoder(z1q)
        pqmf_audio = self.decode_audio(pqmf_audio)
        return self.pqmf.inverse(pqmf_audio)

    def train_sampler(self,x):
        pqmf_audio = self.pqmf(x)
        with torch.no_grad():
            # z1q = self.vqae1.encode(pqmf_audio)
            z2q = self.vqae2.encode(pqmf_audio)
            z3q = self.vqae3.encode(pqmf_audio)
        # z1q_f = self.upsampler1(z2q.detach())
        # z1q_f,z1q = self.equal_size(z1q_f,z1q)
        # feature_loss1 = F.mse_loss(z1q_f,z1q)

        z2q_f = self.upsampler2(z3q.detach())
        z2q_f,z2q = self.equal_size(z2q_f,z2q)
        feature_loss2 = F.mse_loss(z2q_f,z2q)

        # z1q_f = self.upsampler1(z2q_f)
        # z1q_f,z1q = self.equal_size(z1q_f,z1q)
        # feature_loss3 =  F.mse_loss(z1q_f,z1q)
        
        # feature_loss = feature_loss1 + feature_loss2 + feature_loss3
        feature_loss =  feature_loss2
        return feature_loss

    def decode_audio(self,x):
        loud = self.loud_gen(x)
        wave = self.wave_gen(x)
        audio = torch.tanh(wave) *  self.mod_sigmoid(loud)
        # decay_factors = torch.linspace(1.0, 0.01, 16).to("cuda")  # Adjust range as needed
        # audio = audio * decay_factors.view(1, -1, 1)  # Reshape for broadcasting
        return audio
    
    def forward(self, x):
        pqmf_audio = self.pqmf(x)

        pqmf_audio1,vq_loss1 = self.vqae1(pqmf_audio)
        pqmf_audio1,pqmf_audio = self.equal_size(pqmf_audio1,pqmf_audio)
        pqmf_audio1 = self.decode_audio(pqmf_audio1)
        audio_loss1 = self.spec_distance(pqmf_audio,pqmf_audio1)

        pqmf_audio2,vq_loss2 = self.vqae2(pqmf_audio)
        pqmf_audio2,pqmf_audio = self.equal_size(pqmf_audio2,pqmf_audio)
        pqmf_audio2 = self.decode_audio(pqmf_audio2)
        audio_loss2 = self.spec_distance(pqmf_audio,pqmf_audio2)

        pqmf_audio3,vq_loss3 = self.vqae3(pqmf_audio)
        pqmf_audio3,pqmf_audio = self.equal_size(pqmf_audio3,pqmf_audio)
        pqmf_audio3 = self.decode_audio(pqmf_audio3)
        audio_loss3 = self.spec_distance(pqmf_audio,pqmf_audio3)

        vq_loss = vq_loss1 + vq_loss2 + vq_loss3
        audio_loss = audio_loss1 + audio_loss2 + audio_loss3
        return self.pqmf.inverse(pqmf_audio2),audio_loss,vq_loss

# class VQAE(nn.Module):
#     """Some Information about VQAE"""
#     def __init__(self,params):
#         super().__init__()
#         self.pqmf_channel = 16
#         self.hidden_dim = 64
#         self.spec_distance = AudioDistance(params,params.log_epsilon)
#         self.pqmf = PQMF(100,n_band=self.pqmf_channel)

#         self.encoder1 = Encoder(self.pqmf_channel,self.hidden_dim,self.hidden_dim)
#         self.vq_layer1 = Quantize(self.hidden_dim,2048)
#         self.decoder1 = Decoder(self.hidden_dim,self.pqmf_channel,self.hidden_dim)
#         self.encoder1.apply(weights_init)
#         self.decoder1.apply(weights_init)

#         self.encoder2 = Encoder(self.hidden_dim,self.hidden_dim,self.hidden_dim,ratio=16)
#         self.vq_layer2 = Quantize(self.hidden_dim,2048)
#         self.decoder2 = Decoder(self.hidden_dim,self.hidden_dim,self.hidden_dim)
#         self.decoder2_audio = Decoder(self.hidden_dim,self.pqmf_channel,self.hidden_dim)
#         self.encoder2.apply(weights_init)
#         self.decoder2.apply(weights_init)
#         self.decoder2_audio.apply(weights_init)
#         self.upsampler = nn.Sequential(
#             Decoder(self.hidden_dim,self.hidden_dim,self.hidden_dim),
#             ResidualBlock(self.hidden_dim,self.hidden_dim,dilation=2,kernel_size=3),
#             nn.BatchNorm1d(self.hidden_dim),
#             nn.LeakyReLU(0.2),
#             ResidualBlock(self.hidden_dim,self.hidden_dim,dilation=2,kernel_size=3),
#             nn.BatchNorm1d(self.hidden_dim),
#             nn.LeakyReLU(0.2),
#             ResidualBlock(self.hidden_dim,self.hidden_dim,dilation=2,kernel_size=3),
#             nn.BatchNorm1d(self.hidden_dim),
#             nn.LeakyReLU(0.2),
#             ResidualBlock(self.hidden_dim,self.hidden_dim,dilation=2,kernel_size=3),
#             nn.BatchNorm1d(self.hidden_dim),
#             nn.LeakyReLU(0.2),
#         )

#         self.upsampler2 = nn.Sequential(
#             ResidualBlock(self.hidden_dim,self.hidden_dim,dilation=2,kernel_size=3),
#             nn.BatchNorm1d(self.hidden_dim),
#             nn.LeakyReLU(0.2),
#             ResidualBlock(self.hidden_dim,self.hidden_dim,dilation=2,kernel_size=3),
#             nn.BatchNorm1d(self.hidden_dim),
#             nn.LeakyReLU(0.2),
#             ResidualBlock(self.hidden_dim,self.hidden_dim,dilation=2,kernel_size=3),
#             nn.BatchNorm1d(self.hidden_dim),
#             nn.LeakyReLU(0.2),
#             ResidualBlock(self.hidden_dim,self.hidden_dim,dilation=2,kernel_size=3),
#             nn.BatchNorm1d(self.hidden_dim),
#             nn.LeakyReLU(0.2),
#         )

#         self.wave_gen = nn.Conv1d(self.pqmf_channel,self.pqmf_channel,7,padding=3)
#         self.loud_gen = nn.Conv1d(self.pqmf_channel,self.pqmf_channel,3,1,padding=1)

#     def mod_sigmoid(self,x):
#         return 2 * torch.sigmoid(x)**2.3 + 1e-7
    
#     def equal_size(self,a:torch.Tensor,b:torch.Tensor):
#         min_size = min(a.shape[-1],b.shape[-1])
#         a_truncated = a[..., :min_size]  # Keep all dimensions except truncate last dimension
#         b_truncated = b[..., :min_size]  # Same truncation for b
#         return a_truncated, b_truncated
    
#     def quant(self,z,vq_layer):
#         z = z.permute(0,2,1)  
#         zq,vq_loss,_ = vq_layer(z)
#         return zq.permute(0,2,1), vq_loss

#     def inference(self,x):
#         pqmf_audio = self.pqmf(x)

#         z1 = self.encoder1(pqmf_audio)
#         z_q1,_ = self.quant(z1,self.vq_layer1)
#         # pqmf_audio = self.decoder1(z_q1)


#         z1 = z_q1.detach()
#         z2 = self.encoder2(z1)
#         z_q2,_ = self.quant(z2,self.vq_layer2)
#         z_q1f = self.upsampler(z_q2)
#         z_q1f = self.upsampler2(z_q1f)

#         z_q1f,_ = self.quant(z_q1f,self.vq_layer1)
        
#         pqmf_audio = self.decoder1(z_q1f)
#         pqmf_audio = self.decode_audio(pqmf_audio)
#         return self.pqmf.inverse(pqmf_audio)

#     def decode_audio(self,x):
#         loud = self.loud_gen(x)
#         wave = self.wave_gen(x)
#         return torch.tanh(wave) *  self.mod_sigmoid(loud)


#     def train_sampler(self,x):
#         pqmf_audio = self.pqmf(x)
#         z1 = self.encoder1(pqmf_audio)
#         z_q1,_ = self.quant(z1,self.vq_layer1)

#         z1 = z_q1.detach()
#         z2 = self.encoder2(z1)
#         z_q2,_ = self.quant(z2,self.vq_layer2)

#         z_q1f = self.upsampler(z_q2.detach())
#         z_q1f = self.upsampler2(z_q1f)
#         z_q1f,z1 = self.equal_size(z_q1f,z1)
#         feature_loss2 = F.mse_loss(z_q1f,z1)
#         return feature_loss2



#     def forward(self, x):
#         pqmf_audio = self.pqmf(x)
#         z1 = self.encoder1(pqmf_audio)
#         z_q1,vq_loss1 = self.quant(z1,self.vq_layer1)
#         #level 1 loss
#         pqmf_audio1 = self.decoder1(z_q1)
#         pqmf_audio1, pqmf_audio = self.equal_size(pqmf_audio1,pqmf_audio)
#         pqmf_audio1 = self.decode_audio(pqmf_audio1)
#         audio_loss1 = self.spec_distance(pqmf_audio,pqmf_audio1)
#         #layer1 loss = audio_loss1 + vq_loss1

#         z1 = z_q1.detach()
#         z2 = self.encoder2(z1)
#         z_q2,vq_loss2 = self.quant(z2,self.vq_layer2)
#         z_q1f = self.decoder2(z_q2)
#         z_q1f,z_q1 = self.equal_size(z_q1f,z_q1)
#         z_q1f,_ = self.quant(z_q1f,self.vq_layer1)

#         pqmf_audio2 = self.decoder2_audio(z_q1f)
#         pqmf_audio2, pqmf_audio = self.equal_size(pqmf_audio2,pqmf_audio)

#         pqmf_audio2 = self.decode_audio(pqmf_audio2)
#         audio_loss2 = self.spec_distance(pqmf_audio,pqmf_audio2)

#         vq_loss = vq_loss1 + vq_loss2
#         audio_loss = audio_loss1 + audio_loss2

#         z1 = z_q1.detach()
#         z2 = z_q2.detach()
#         z1_f = self.upsampler(z2)
#         z1_f,z1 = self.equal_size(z1_f,z1)
#         feature_loss2 = F.mse_loss(z1_f,z1)
#         feature_loss = feature_loss2

#         return self.pqmf.inverse(pqmf_audio2),audio_loss,vq_loss,feature_loss
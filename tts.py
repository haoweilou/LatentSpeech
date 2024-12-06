import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#Transformer
def sinusoid_encoding_table(seq_len, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(seq_len)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)

class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, config):
        super().__init__()

        self.n_head = config['head_num']
        self.d_model = config['hidden_dim']
        self.dropout = config["dropout"]
        self.d_k = self.d_model // self.n_head
        self.d_v = self.d_k

        self.w_qs = nn.Linear(self.d_model, self.n_head * self.d_k)
        self.w_ks = nn.Linear(self.d_model, self.n_head * self.d_k)
        self.w_vs = nn.Linear(self.d_model, self.n_head * self.d_v)

        self.attention = ScaledDotProductAttention(temperature=np.power(self.d_k, 0.5))
        self.layer_norm = nn.LayerNorm(self.d_model)

        self.fc = nn.Linear(self.n_head * self.d_v, self.d_model)

        self.dropout = nn.Dropout(self.dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = (
            output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        )

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn
  

class FFT(nn.Module):
    """Some Information about MyModule"""
    def __init__(self,config):
        super(FFT, self).__init__()
        self.self_attention = MultiHeadAttention(config)
        self.conv1d = Conv1DLayer(config)

    def forward(self, x, mask=None, slf_attn_mask=None):
        enc_x, enc_slf_attn = self.self_attention(x,x,x,mask=slf_attn_mask) 
        enc_x = enc_x.masked_fill(mask.unsqueeze(-1), 0)
        enc_x = self.conv1d(enc_x)
        enc_x = enc_x.masked_fill(mask.unsqueeze(-1), 0)

        return enc_x, enc_slf_attn
 

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        output = torch.bmm(attn, v)

        return output, attn
    
class Conv1DLayer(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, config):
        super().__init__()
        self.d_in = config["hidden_dim"]
        self.d_hid = config["filter_num"]
        self.kernel_size = config["kernel_size"]
        self.dropout = config["dropout"]
        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            self.d_in,
            self.d_hid,
            kernel_size=self.kernel_size,
            padding=(self.kernel_size - 1) // 2,
        )
        # position-wise
        self.w_2 = nn.Conv1d(
            self.d_hid,
            self.d_in,
            kernel_size=self.kernel_size,
            padding=(self.kernel_size - 1) // 2,
        )

        self.layer_norm = nn.LayerNorm(self.d_in)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output



class Encoder(nn.Module):
    """Some Information about Encoder"""
    def __init__(self,config,max_word=512):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(
            config["word_num"],config["word_dim"],padding_idx=config["padding_idx"]
        )
        self.pos_enc = nn.Parameter(
            sinusoid_encoding_table(max_word,config["word_dim"]).unsqueeze(0),
            requires_grad=False
        )
        
        self.fft_layers = nn.ModuleList(
            [FFT(config["FFT"]) for _ in range(config["n_layers"])]
        )

    def forward(self, x, mask):
        #x = BATCH_SIZE, MAX_LEN
        batch_size, max_len =  x.shape[0],  x.shape[1]
        #mask = 1, MAX_LEN, MAX_SEQ_LEN
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
        #pos_enc = BATCH_SIZE, MAX_LEN, WORD_DIM
        pos_enc = self.pos_enc[:,:max_len,:].expand(batch_size, -1, -1)
        #enc_x = word_embed + pos_enc = BATCH_SIZE, MAX_SEQ_LEN, , WORD_DIM
        enc_x = self.embed(x) + pos_enc
        for fft in self.fft_layers:
            enc_x, enc_slf_attn = fft(enc_x, mask=mask, slf_attn_mask=slf_attn_mask)
        return enc_x
    
class CONV(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(CONV, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x
#Length adaptor
class LengthPredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, config, word_dim=256):
        super(LengthPredictor, self).__init__()

        self.word_dim = word_dim
        self.filter_size = config["filter_num"]
        self.kernel = config["kernel_size"]
        self.conv_output_size = config["filter_num"]
        self.dropout = config["dropout"]

        self.conv = nn.Sequential(
            CONV(self.word_dim,self.filter_size,self.kernel,padding=(self.kernel - 1) // 2),
            nn.ReLU(),
            nn.LayerNorm(self.filter_size),
            nn.Dropout(self.dropout),

            CONV(self.filter_size,self.filter_size,self.kernel,padding=1),
            nn.ReLU(),
            nn.LayerNorm(self.filter_size),
            nn.Dropout(self.dropout),
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, x, mask):
        x = self.conv(x)

        x = self.linear_layer(x)
        x = x.squeeze(-1)

        if mask is not None:
            x = x.masked_fill(mask, 0.0)

        return x

def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask

def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded

class LengthAdaptor(nn.Module):
    """Length Adaptor"""

    def __init__(self, model_config,word_dim=256):
        super(LengthAdaptor, self).__init__()
        self.duration_predictor = LengthPredictor(model_config,word_dim=word_dim)
        self.length_regulator = LengthRegulator()

       
    def forward(self, x, mask, mel_mask=None, max_len=None, duration_target=None, d_control=1.0):
        #BATCH_SIZE, SEQ_LEN
        log_duration_prediction = self.duration_predictor(x, mask)
        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=5,
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len,max_len=max_len)

        return ( x, log_duration_prediction, duration_rounded, mel_len, mel_mask,)

class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len

class Decoder(nn.Module):
    """Some Information about Decoder"""
    def __init__(self,config,max_word=512):
        super(Decoder, self).__init__()
        self.pos_enc = nn.Parameter(
            sinusoid_encoding_table(max_word,config["word_dim"]).unsqueeze(0),
            requires_grad=False
        )

        self.fft_layers = nn.ModuleList(
            [FFT(config["FFT"]) for _ in range(config["n_layers"])]
        )


    def forward(self, x,mask):
        batch_size, max_len = x.shape[0], x.shape[1]
        dec_output = x[:, :max_len, :] + self.pos_enc[:, :max_len, :].expand(batch_size, -1, -1)
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
        slf_attn_mask = slf_attn_mask[:, :, :max_len]

        for dec_layer in self.fft_layers:
            dec_output, dec_slf_attn = dec_layer(dec_output, mask=mask, slf_attn_mask=slf_attn_mask)
        return dec_output,mask

class SpecAdapter(nn.Module):
    """Some Information about SpecAdapter"""
    def __init__(self,embed_dim=64):
        super(SpecAdapter, self).__init__()
        self.upsampler = model = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),  # Changed kernel size to 3, padding 1
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),  # Changed kernel size to 3, padding 1
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),  # Changed kernel size to 3, padding 1
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.upsampler(x)

class StyleSpeech(nn.Module):
    """Some Information about StyleSpeech"""
    def __init__(self,config,embed_dim=64,output_channel=1):
        super(StyleSpeech, self).__init__()
        self.max_word = config["max_seq_len"] + 1
        self.pho_encoder = Encoder(config["pho_config"],max_word=self.max_word)
        self.style_encoder = Encoder(config["style_config"],max_word=self.max_word)
        self.length_adaptor = LengthAdaptor(config["len_config"],word_dim=config["pho_config"]['word_dim'])
        self.fuse_decoder = Decoder(config["fuse_config"],max_word=self.max_word)
        self.output_channel = output_channel
        if output_channel != 1:
            self.channel = nn.Sequential(
                nn.Conv2d(1,output_channel,kernel_size=(3, 1), padding=(1, 0))
            )


        self.mel_linear = nn.Sequential(
            nn.Linear(
                config["fuse_config"]["word_dim"],
                embed_dim
            ),
            nn.LeakyReLU(.2)
        )
        

    def forward(self, x, s, src_lens,duration_target=None,mel_lens=None,max_mel_len=None):
        batch_size, max_src_len = x.shape[0],x.shape[1]
        src_mask = get_mask_from_lengths(src_lens,max_len=max_src_len)
        mel_mask = get_mask_from_lengths(mel_lens, max_len=max_mel_len)
        pho_embed = self.pho_encoder(x,src_mask)
        style_embed = self.style_encoder(s,src_mask)
        
        fused = pho_embed + style_embed
        fused,log_duration_prediction, duration_rounded, _, mel_mask = self.length_adaptor(fused,src_mask, mel_mask=mel_mask, max_len=max_mel_len, duration_target=duration_target)
        fused,mel_mask = self.fuse_decoder(fused,mel_mask)
        if self.output_channel != 1:
            fused = fused.unsqueeze(1)
            fused = self.channel(fused)

        mel = self.mel_linear(fused)       
        return mel,log_duration_prediction,mel_mask


class ContextEncoder(nn.Module):
    """Some Information about StyleSpeech"""
    def __init__(self,config):
        super(ContextEncoder, self).__init__()
        self.max_word = config["max_seq_len"] + 1
        self.pho_encoder = Encoder(config["pho_config"],max_word=self.max_word)
        self.style_encoder = Encoder(config["style_config"],max_word=self.max_word)
        self.duration_predictor = LengthPredictor(config["len_config"],word_dim=config["pho_config"]['word_dim'])

        self.fc = nn.Sequential(
            nn.Linear(
                2*config["pho_config"]["word_dim"],
                config["pho_config"]["word_dim"]
            ),
            nn.LeakyReLU(.2),

        )
        
    def forward(self, x, s, x_lens):
        #in: [b,t] => out [b,t,c]
        batch_size, max_src_len = x.shape[0],x.shape[1]
        x_mask = get_mask_from_lengths(x_lens,max_len=max_src_len)
        pho_embed = self.pho_encoder(x,x_mask)
        style_embed = self.style_encoder(s,x_mask)
        fused = torch.cat([pho_embed, style_embed],dim=2) # [b, t, c]
        fused = self.fc(fused)
        return fused
    
import math
from glow import GlowDecoder
class LayerNorm(nn.Module):
  def __init__(self, channels, eps=1e-4):
      super().__init__()
      self.channels = channels
      self.eps = eps
      self.gamma = nn.Parameter(torch.ones(channels))
      self.beta = nn.Parameter(torch.zeros(channels))

  def forward(self, x):
    n_dims = len(x.shape)
    mean = torch.mean(x, 1, keepdim=True)
    variance = torch.mean((x -mean)**2, 1, keepdim=True)

    x = (x - mean) * torch.rsqrt(variance + self.eps)

    shape = [1, -1] + [1] * (n_dims - 2)
    x = x * self.gamma.view(*shape) + self.beta.view(*shape)
    return x

class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size=3, p_dropout=0.05):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask
    
def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape

def generate_path(duration, mask):
    """
    duration: [b, t_x]
    mask: [b, t_x, t_y]
    """
    device = duration.device
    
    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)
    path = torch.zeros(b, t_x, t_y, dtype=mask.dtype).to(device=device)
    
    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:,:-1]
    path = path * mask
    return path


def maximum_path(value, mask, max_neg_val=-np.inf):
    """ Numpy-friendly version. It's about 4 times faster than torch version.
    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]
    """
    value = value * mask

    device = value.device
    dtype = value.dtype
    value = value.cpu().detach().numpy()
    mask = mask.cpu().detach().numpy().astype(np.bool)
    
    b, t_x, t_y = value.shape
    direction = np.zeros(value.shape, dtype=np.int64)
    v = np.zeros((b, t_x), dtype=np.float32)
    x_range = np.arange(t_x, dtype=np.float32).reshape(1,-1)
    for j in range(t_y):
        v0 = np.pad(v, [[0,0],[1,0]], mode="constant", constant_values=max_neg_val)[:, :-1]
        v1 = v
        max_mask = (v1 >= v0)
        v_max = np.where(max_mask, v1, v0)
        direction[:, :, j] = max_mask
        
        index_mask = (x_range <= j)
        v = np.where(index_mask, v_max + value[:, :, j], max_neg_val)
    direction = np.where(mask, direction, 1)
        
    path = np.zeros(value.shape, dtype=np.float32)
    index = mask[:, :, 0].sum(1).astype(np.int64) - 1
    index_range = np.arange(b)
    for j in reversed(range(t_y)):
        path[index_range, index, j] = 1
        index = index + direction[index_range, index, j] - 1
    path = path * mask.astype(np.float32)
    path = torch.from_numpy(path).to(device=device, dtype=dtype)
    return path

class StyleSpeech2(nn.Module):
    """Some Information about StyleSpeech2"""
    def __init__(self,config,n_speakers=2):
        super().__init__()
        self.encoder = ContextEncoder(config)
        self.proj_m = nn.Conv1d(256, 16, 1)
        self.proj_s = nn.Conv1d(256, 16, 1)
        gin_channels = 256
        self.hidden_channel=256
        self.proj_w = DurationPredictor(gin_channels+self.hidden_channel,16)
        self.n_sqz = 2
        self.decoder = GlowDecoder(in_channels=16, 
            hidden_channels=256, 
            kernel_size=3, 
            dilation_rate=1, 
            n_blocks=12, 
            n_layers=4, 
            p_dropout=0.05, 
            n_split=4,
            n_sqz=self.n_sqz,
            sigmoid_scale=False,
            gin_channels=gin_channels)
        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)
            nn.init.uniform_(self.emb_g.weight, -0.1, 0.1)

    def forward(self, x, s, x_lens, y=None, y_lens=None, g=None, gen=False, noise_scale=1., length_scale=1.):
        #x [b,N], x_lens [b], y [b,c,t], y_lens b 
        if g is not None:#g is the speaker identity
            g = F.normalize(self.emb_g(g)).unsqueeze(-1) # [b, h]
        x = self.encoder(x,s,x_lens) * math.sqrt(self.hidden_channel) #b,t,c
        x = torch.permute(x,(0,2,1)) #b,c,t
        x_mask = torch.unsqueeze(sequence_mask(x_lens, x.size(2)), 1).to(x.dtype)#b,1,t
        if g is not None:
            g_exp = g.expand(-1, -1, x.size(-1))
            x_dp = torch.cat([torch.detach(x), g_exp], 1)#b,2c,t
        else:
            x_dp = torch.detach(x)
        x_m = self.proj_m(x) * x_mask
        x_logs = self.proj_s(x) * x_mask

        logw = self.proj_w(x_dp, x_mask)#duration

        if gen:
            w = torch.exp(logw) * x_mask * length_scale
            w_ceil = torch.ceil(w)
            y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
            y_max_length = None
        else:
            y_max_length = y.size(2)
            
        y, y_lengths, y_max_length = self.preprocess(y, y_lens, y_max_length)
        z_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(z_mask, 2)

        if gen:
            attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
            z_m = torch.matmul(attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
            z_logs = torch.matmul(attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
            logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask

            z = (z_m + torch.exp(z_logs) * torch.randn_like(z_m) * noise_scale) * z_mask
            y, logdet = self.decoder(z, z_mask, g=g, reverse=True)
            return (y, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_)
        else:
            z, logdet = self.decoder(y, z_mask, g=g, reverse=False)
            with torch.no_grad():
                x_s_sq_r = torch.exp(-2 * x_logs)
                logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - x_logs, [1]).unsqueeze(-1) # [b, t, 1]
                logp2 = torch.matmul(x_s_sq_r.transpose(1,2), -0.5 * (z ** 2)) # [b, t, d] x [b, d, t'] = [b, t, t']
                logp3 = torch.matmul((x_m * x_s_sq_r).transpose(1,2), z) # [b, t, d] x [b, d, t'] = [b, t, t']
                logp4 = torch.sum(-0.5 * (x_m ** 2) * x_s_sq_r, [1]).unsqueeze(-1) # [b, t, 1]
                logp = logp1 + logp2 + logp3 + logp4 # [b, t, t']

                attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()

            z_m = torch.matmul(attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
            z_logs = torch.matmul(attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
            logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask
        return (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_)
    
    def preprocess(self, y, y_lengths, y_max_length):
        if y_max_length is not None:
            y_max_length = (y_max_length // self.n_sqz) * self.n_sqz
            y = y[:,:,:y_max_length]
            y_lengths = (y_lengths // self.n_sqz) * self.n_sqz
        return y, y_lengths, y_max_length

    def store_inverse(self):
        self.decoder.store_inverse()

def MAS(S):
    #S = similarity matrix
    #N = num phone
    #M = num mel
    N,M = S.size()
    Q = torch.full((N, M), float('-inf'), device=S.device)
    A = torch.zeros((N, M), dtype=torch.long, device=S.device)

    Q[0, :] = S[0, :]
    for j in range(1, M):
        for i in range(min(j, N)):  # Ensure j >= i (monotonicity constraint)
            # Consider all previous mel frames up to j
            if i == j:
                v_cur = float('-inf')
            else:
                v_cur = Q[i, j-1]
            
            if i == 0:
                if j == 0:
                    v_prev = 0.
                else:
                    v_prev = float('-inf')
            else:
                v_prev = Q[i-1,j-1]
            Q[i][j] = max(v_prev,v_cur) + Q[i][j]

    index = N - 1
    for j in range(M - 1, -1, -1):
        A[index, j] = 1
        if index != 0 and (index == j or Q[index, j-1] < Q[index-1, j-1]):
            index = index - 1
    return A


# class DurationAligner(nn.Module):
#     def __init__(self,feature_dim=16, tar_dim=16):
#         super().__init__()
#         self.feature_dim = feature_dim
#         self.tar_dim = tar_dim
        
#     def forward(self, pho_embed, tar_embed):
#         # pho_embed: F, L, F = feat dim, L = number phone, 
#         # tar_embed: M, T, M = mel dim, T = number mel 
#         pho_norm = pho_embed / pho_embed.norm(dim=-1, keepdim=True) #F, L
#         tar_norm = tar_embed / tar_embed.norm(dim=-1, keepdim=True) #F, T
#         similarity_matrix = torch.matmul(pho_norm.transpose(1, 2), tar_norm)  # [L, T]
#         As = [MAS(s) for s in similarity_matrix]
#         durations = torch.stack([a.sum(dim=1) for a in As])
#         return durations


class FastSpeechLoss(nn.Module):
    """Some Information about FastSpeechLoss"""
    def __init__(self):
        super(FastSpeechLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, y,y_pred,log_l_pred,l,mel_masks,device="cuda"):
        log_l = torch.log(l.float() + 1).to(device)
        # mel_masks = torch.logical_not(mel_masks.unsqueeze(-1).expand_as(y_pred))
        # y = y*mel_masks
        # y_pred = y_pred*mel_masks

        y = y.to(device)
        mel_loss = self.mse_loss(y_pred, y)
        duration_loss = self.mae_loss(log_l_pred, log_l)
        total_loss = mel_loss+duration_loss
        return total_loss, mel_loss, duration_loss
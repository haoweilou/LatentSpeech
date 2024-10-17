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
    def __init__(self,config,fuse_step = 0,embed_dim=64):
        super(StyleSpeech, self).__init__()
        self.max_word = config["max_seq_len"] + 1
        self.pho_encoder = Encoder(config["pho_config"],max_word=self.max_word)
        self.style_encoder = Encoder(config["style_config"],max_word=self.max_word)
        self.length_adaptor = LengthAdaptor(config["len_config"],word_dim=config["pho_config"]['word_dim'])
        self.fuse_decoder = Decoder(config["fuse_config"],max_word=self.max_word)
        self.mel_linear = nn.Sequential(
            nn.Linear(
                config["fuse_config"]["word_dim"],
                embed_dim
            ),
            nn.LeakyReLU(.2)
        ) 
        self.fuse_step = fuse_step

    def forward(self, x, s, src_lens,duration_target=None,mel_lens=None,max_mel_len=None):
        batch_size, max_src_len = x.shape[0],x.shape[1]
        src_mask = get_mask_from_lengths(src_lens,max_len=max_src_len)
        mel_mask = get_mask_from_lengths(mel_lens, max_len=max_mel_len)
        pho_embed = self.pho_encoder(x,src_mask)
        style_embed = self.style_encoder(s,src_mask)
        
        fused = pho_embed + style_embed
        fused,log_duration_prediction, duration_rounded, _, mel_mask = self.length_adaptor(fused,src_mask, mel_mask=mel_mask, max_len=max_mel_len, duration_target=duration_target)
        fused,mel_mask = self.fuse_decoder(fused,mel_mask)
        mel = self.mel_linear(fused)       

        return mel,log_duration_prediction,mel_mask

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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from params import params
# from tts import DurationAligner
from flow import AE
from function import loadModel,saveModel,save_audio,draw_wave,draw_heatmap
import pandas as pd

from dataset import BakerAudio,BakerText
from torch.utils.data import DataLoader
from params import params
from tts import StyleSpeech2

bakertext = BakerText(normalize=False,start=0,end=100,path="D:/baker/")
bakeraudio = BakerAudio(start=0,end=100,path="D:/baker/",return_len=True)


# bakertext = BakerText(normalize=False,start=0,end=100)
# bakeraudio = BakerAudio(start=0,end=100,return_len=True)
def collate_fn(batch):
    text_batch, audio_batch = zip(*batch)
    text_batch = [torch.stack([item[i] for item in text_batch]) for i in range(len(text_batch[0]))]
    audio_batch = bakeraudio.collate(audio_batch)
    return text_batch, audio_batch
from tqdm import tqdm
from tts_config import config

loader = DataLoader(dataset=list(zip(bakertext, bakeraudio)), collate_fn=collate_fn, batch_size=16, shuffle=False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ae = AE(params).to(device)
ae = loadModel(ae,"ae20k16_1000","./model")
import math
def learning_rate(d_model=256,step=1,warmup_steps=400):
    return (1/math.sqrt(d_model)) * min(1/math.sqrt(step),step*warmup_steps**-1.5)


modelname = "StyleSpeech2"
model = StyleSpeech2(config).to(device)
model = loadModel(model,"StyleSpeech2_1000","./model/")



import torchaudio.transforms as T



for i,(text_batch,audio_batch) in enumerate(tqdm(loader)):
    x,s,_,x_lens,_ = [tensor.to(device) for tensor in text_batch]
    audio,y_lens = audio_batch[0].to(device),audio_batch[1].to(device)
    with torch.no_grad():
        y,_ = ae.encode(audio)
        y_lens = torch.ceil(y_lens/16/64)

    speaker = torch.zeros(x_lens.shape).to(dtype=x_lens.dtype,device=x_lens.device)
    noise_scale = 1
    length_scale = 1.0
    (y_gen, *_), *_, (attn_gen, *_) = model(x, s, x_lens,y_lens=y_lens, gen=True, noise_scale=noise_scale, length_scale=length_scale,g=speaker)
    # y_gen,log_l,y_mask = model(x, s, x_lens,y_lens=y_lens,max_y_len=y.shape[-1])
    print(y_gen.shape)
    draw_heatmap(y_gen[0].detach().cpu().numpy(),name="ss2_heat")
    draw_heatmap(y[0].detach().cpu().numpy(),name="real_heat")
    pqmf_audio = ae.decode(y_gen)
    audio_f = ae.pqmf.inverse(pqmf_audio)
    audio_f = audio_f.detach().cpu()
    save_audio(audio_f[0],48000,f"ss2","./sample/")
    draw_wave(audio_f[0][0].to("cpu"),f"ss2")
    audio = audio.detach().cpu()
    save_audio(audio[0],48000,f"real","./sample/")
    draw_wave(audio[0][0].to("cpu"),f"real")

    
    from function import phone_to_phone_idx,hanzi_to_pinyin
    hanzi = "老师你好我是自动语音机器人"
    pinyin = hanzi_to_pinyin(hanzi)
    print(pinyin)
    # # pinyin = ["la1","la2","la3","la4","la5"]
    phone_idx,tone = phone_to_phone_idx(pinyin)
    # print(phone_idx,tone)
    d = 10
    duration = torch.tensor([[d for _ in range(len(phone_idx))]]).to(device)
    phone_mask = torch.tensor([[0 for _ in range(len(phone_idx))]]).to(device)
    phone_idx = torch.tensor([phone_idx]).to(device)  
    tone = torch.tensor([tone]).to(device)
    hidden_mask = torch.tensor([[0 for _ in range(1024)]]).to(device)

    src_lens = torch.tensor([phone_idx.shape[-1]]).to(device)
    mel_lens = torch.tensor([d*phone_idx.shape[-1]]).to(device)
    speaker = torch.zeros(src_lens.shape).to(dtype=x_lens.dtype,device=x_lens.device)

    (y_gen, *_), *_, (attn_gen, *_) = model(phone_idx, tone, src_lens,y_lens=mel_lens, gen=True, noise_scale=noise_scale, length_scale=length_scale,g=speaker)
    # y_gen,log_l,y_mask = model(phone_idx, tone, src_lens,y_lens=mel_lens,max_y_len=500)

    pqmf_audio = ae.decode(y_gen)
    audio_f = ae.pqmf.inverse(pqmf_audio)
    audio_f = audio_f.detach().cpu()
    print(audio_f.shape)
    save_audio(audio_f[0],48000,f"custom_glow","./sample/")
    
    break


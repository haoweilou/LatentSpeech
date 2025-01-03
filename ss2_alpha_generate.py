import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from params import params
# from tts import DurationAligner
from flow import AE
from function import loadModel,saveModel,save_audio,draw_wave
import pandas as pd

from dataset import BakerAudio,BakerText,LJSpeechAudio,LJSpeechText
from torch.utils.data import DataLoader
from params import params

# bakertext = BakerText(normalize=False,start=0,end=100,path="L:/baker/",alphabet=True)
# bakeraudio = BakerAudio(start=0,end=100,path="L:/baker/",return_len=True)
from ipa import alpha_pho_dict


bakertext = LJSpeechText(start=0,end=100,path="L:/LJSpeech/",alphabet=True)
bakeraudio = LJSpeechAudio(start=0,end=100,path="L:/LJSpeech/",return_len=True)
def collate_fn(batch):
    text_batch, audio_batch = zip(*batch)
    text_batch = [torch.stack([item[i] for item in text_batch]) for i in range(len(text_batch[0]))]
    audio_batch = bakeraudio.collate(audio_batch)
    return text_batch, audio_batch
from tts import StyleSpeech2_FF
from tqdm import tqdm
from tts_config import config
config["pho_config"]["word_num"] = len(alpha_pho_dict)

loader = DataLoader(dataset=list(zip(bakertext, bakeraudio)), collate_fn=collate_fn, batch_size=16, shuffle=False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = StyleSpeech2_FF(config,embed_dim=16).to(device)

ae = AE(params).to(device)
ae = loadModel(ae,"ae20k16_1000","./model")


modelname = "StyleSpeech2_FF_18K_ALPHA"
model = loadModel(model,f"{modelname}_100","./model/")


for i,(text_batch,audio_batch) in enumerate(tqdm(loader)):
    x,s,_,x_lens,mel_lens,language = [tensor.to(device) for tensor in text_batch]
    audio,y_lens = audio_batch[0].to(device),audio_batch[1].to(device)
    with torch.no_grad():
        y,_ = ae.encode(audio) 
        y_lens = torch.ceil(y_lens/16/64)
    speaker = torch.zeros(x_lens.shape).to(dtype=x_lens.dtype,device=x_lens.device)
    noise_scale = 1
    length_scale = 1.0
    
    s = s+5
    y_pred,log_l,y_mask = model(x, s, x_lens,y_lens=mel_lens,max_y_len=max(mel_lens).item(),language=language)
    print(y_pred.shape)
    pqmf_audio = ae.decode(y_pred)
    audio_f = ae.pqmf.inverse(pqmf_audio)
    audio_f = audio_f.detach().cpu()
    save_audio(audio_f[0],48000,f"ss2_alpha","./sample/")
    draw_wave(audio_f[0][0].to("cpu"),f"ss2")
    audio = audio.detach().cpu()
    save_audio(audio[0],48000,f"real","./sample/")
    draw_wave(audio[0][0].to("cpu"),f"real")
    break


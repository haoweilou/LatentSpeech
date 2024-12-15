import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from params import params
from flow import AE
from function import loadModel,saveModel,save_audio,draw_wave,draw_heatmap
import pandas as pd

from dataset import LJSpeechText,LJSpeechAudio
from torch.utils.data import DataLoader
from params import params
from tts import StyleSpeech2,StyleSpeech2_FF

bakertext = LJSpeechText(start=0,end=100,path="L:/LJSpeech/")
bakeraudio = LJSpeechAudio(start=0,end=100,path="L:/LJSpeech/",return_len=True)


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


from ipa import ipa_pho_dict
config["pho_config"]["word_num"] = len(ipa_pho_dict)
modelname = "StyleSpeech2_FF"
model = StyleSpeech2_FF(config,embed_dim=16).to(device)
model = loadModel(model,f"{modelname}_250","./model/")


import torchaudio.transforms as T



# from function import phone_to_phone_idx,hanzi_to_pinyin
hanzi = "中华人民共和国今天成立了"
from ipa import mandarin_chinese_to_ipa, ipa_to_idx
pinyin,tone = mandarin_chinese_to_ipa(hanzi)
print(pinyin)
# # pinyin = ["la1","la2","la3","la4","la5"]
phone_idx = ipa_to_idx(pinyin)
# print(phone_idx,tone)
phone_mask = torch.tensor([[0 for _ in range(len(phone_idx))]]).to(device)
phone_idx = torch.tensor([phone_idx]).to(device)  
tone = torch.tensor([tone]).to(device)
hidden_mask = torch.tensor([[0 for _ in range(1024)]]).to(device)

src_lens = torch.tensor([phone_idx.shape[-1]]).to(device)
d = 10
mel_lens = torch.tensor([d*phone_idx.shape[-1]]).to(device)

# (y_gen, *_), *_, (attn_gen, *_) = model(phone_idx, tone, src_lens,y_lens=mel_lens, gen=True, noise_scale=noise_scale, length_scale=length_scale,g=speaker)
language = torch.zeros_like(phone_idx).to(dtype=torch.long,device=phone_idx.device)
y_gen,log_l,y_mask = model(phone_idx, tone, src_lens,y_lens=mel_lens,max_y_len=500,language=language)

pqmf_audio = ae.decode(y_gen)
audio_f = ae.pqmf.inverse(pqmf_audio)
audio_f = audio_f.detach().cpu()
print(audio_f.shape)
save_audio(audio_f[0],48000,f"custom_cn","./sample/")


english = "hello world how are you"
from ipa import english_sentence_to_ipa, ipa_to_idx
english,tone = english_sentence_to_ipa(english)
print(english)
# # pinyin = ["la1","la2","la3","la4","la5"]
phone_idx = ipa_to_idx(english)
# print(phone_idx,tone)
phone_mask = torch.tensor([[0 for _ in range(len(phone_idx))]]).to(device)
phone_idx = torch.tensor([phone_idx]).to(device)  
# tone = torch.tensor([tone]).to(device)
tone = torch.zeros_like(phone_idx)
hidden_mask = torch.tensor([[0 for _ in range(1024)]]).to(device)

src_lens = torch.tensor([phone_idx.shape[-1]]).to(device)
mel_lens = torch.tensor([d*phone_idx.shape[-1]]).to(device)
language = torch.ones_like(phone_idx).to(dtype=torch.long,device=phone_idx.device)
y_gen,log_l,y_mask = model(phone_idx, tone, src_lens,y_lens=mel_lens,max_y_len=500,language=language)

pqmf_audio = ae.decode(y_gen)
audio_f = ae.pqmf.inverse(pqmf_audio)
audio_f = audio_f.detach().cpu()
print(audio_f.shape)
save_audio(audio_f[0],48000,f"custom_en","./sample/")
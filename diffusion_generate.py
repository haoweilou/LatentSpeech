from tts import StyleSpeech2_FF,FastSpeechLoss,StyleSpeech2_Diff
from diffusion import DiffusionBlock
from dataset import BakerAudio,BakerText,LJSpeechAudio,LJSpeechText
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tts_config import config
from function import saveModel,saveLog
import math
import os
from tqdm import tqdm
from params import params
import numpy as np
from torch.utils.data import DataLoader
from logger import Log
def learning_rate(d_model=256,step=1,warmup_steps=400):
    return (1/math.sqrt(d_model)) * min(1/math.sqrt(step),step*warmup_steps**-1.5)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from tts import ContextEncoder
# tts_model = StyleSpeech2_FF(config,embed_dim=16).to(device)

# context_model = ContextEncoder(config).to(device)
context_model = StyleSpeech2_Diff(config,embed_dim=16).to(device)
diffusion_block = DiffusionBlock(params).to(device)

lr = learning_rate()
optimizer = optim.Adam(list(context_model.parameters())+list(diffusion_block.parameters()), betas=(0.9,0.98),eps=1e-9,lr=lr)
loss_func = FastSpeechLoss()
l1_loss = nn.MSELoss()
losses = []
losses2 = []
losses3 = []
num_epoch = 1001
print("Initial learnign rate: ",lr)
print("Load Dataset: ")

root = "L:/"
bakertext = BakerText(normalize=False,start=0,end=500,path=f"{root}baker/",ipa=True)
bakeraudio = BakerAudio(start=0,end=500,path=f"{root}baker/",return_len=True)

ljspeechtext = LJSpeechText(start=0,end=500,path=f"{root}LJSpeech/")
ljspeechaudio = LJSpeechAudio(start=0,end=500,path=f"{root}LJSpeech/",return_len=True)

from dataset import CombinedTextDataset,CombinedAudioDataset
textdataset = CombinedTextDataset(bakertext,ljspeechtext)
audiodataset = CombinedAudioDataset(bakeraudio,ljspeechaudio)

from flow import AE
from function import loadModel

ae = AE(params).to(device)
ae = loadModel(ae,"ae20k16_1000","./model")

def collate_fn(batch):
    text_batch, audio_batch = zip(*batch)
    text_batch = [torch.stack([item[i] for item in text_batch]) for i in range(len(text_batch[0]))]
    audio_batch = bakeraudio.collate(audio_batch)
    return text_batch, audio_batch

loader = DataLoader(dataset=list(zip(textdataset, audiodataset)), collate_fn=collate_fn, batch_size=32, shuffle=False)

beta = np.array(params.noise_schedule)
noise_level = np.cumprod(1 - beta)
noise_level = torch.tensor(noise_level.astype(np.float32))
model_name = "StyleSpeechDiff"
from function import loadModel,save_audio,draw_wave

epoch = 900
context_model = loadModel(context_model,f"{model_name}_content_{epoch}","./model/")
diffusion_block = loadModel(diffusion_block,f"{model_name}_diff_{epoch}","./model/")


def noise_schedule(tts_embed,training_noise_schedule,inference_noise_schedule):
    talpha = 1 - training_noise_schedule
    talpha_cum = np.cumprod(talpha)

    beta = inference_noise_schedule
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)
    T = []
    for s in range(len(inference_noise_schedule)):
        for t in range(len(training_noise_schedule) - 1):
            if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
                twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
                T.append(t + twiddle)
                break
    T = np.array(T, dtype=np.float32)

    tts_embed = tts_embed.to(device)#B,T,C
    hidden = torch.rand_like(tts_embed, device=device)
    hidden = torch.transpose(hidden,1,2) #B,C,T
    with torch.no_grad():
        for n in range(len(alpha) - 1, -1, -1):
            c1 = 1 / alpha[n]**0.5
            c2 = beta[n] / (1 - alpha_cum[n])**0.5
            #hidden B,C,T
            predict_noise = diffusion_block(hidden,torch.tensor([T[n]]).to(device),tts_embed) #B,C,T
            print(torch.mean(predict_noise),torch.std(predict_noise))
            hidden = c1 *(hidden - c2 * predict_noise)
            if n > 0:
                noise = torch.randn_like(hidden)
                sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
                hidden += sigma * noise

    return hidden



for epoch in range(num_epoch):
    for i,(text_batch,audio_batch) in enumerate(tqdm(loader)):
        optimizer.zero_grad()
        x,s,_,x_lens,_,language = [tensor.to(device) for tensor in text_batch]
        audio,y_lens = audio_batch[0].to(device),audio_batch[1].to(device)
        with torch.no_grad():
            #B,C,T
            y,_ = ae.encode(audio) 
            y_lens = torch.ceil(y_lens/16/64)
      
        #B,T,C
        c_embedd,x_mask = context_model(x, s, x_lens,language=language)
        N,C,T = y.shape
        t = torch.randint(0, len(params.noise_schedule), [N])
        noise_scale = noise_level[t].unsqueeze(1).to(device).unsqueeze(1)
        noise_scale_sqrt = noise_scale**0.5
#         y_ = torch.transpose(y,1,2) #B,C,T
        noise = torch.randn_like(y).to(device)
        noise_y = noise_scale_sqrt * y + (1.0 - noise_scale)**0.5 * noise #B,C,T

        tts_embed = torch.zeros_like(y) #B,C,T
        c_embedd = c_embedd.permute(0,2,1)
        c_embedd = F.pad(c_embedd,(0,T-c_embedd.shape[-1]),value=0)
        language_embed = context_model.language_encoder(language[:,0:1]).detach().permute(0,2,1) #B,C,1
        language_embed_broadcasted = language_embed.expand(-1, -1, T)  # Shape: [b, c, t]
        c_embedd = torch.where(c_embedd == 0, language_embed_broadcasted, c_embedd)
        training_noise_schedule = np.array(params.noise_schedule)
        inference_noise_schedule = np.array(params.noise_schedule)
        print(tts_embed.shape)
        # hidden = noise_schedule(torch.permute(tts_embed,(0,2,1)),training_noise_schedule,inference_noise_schedule)
        hidden = noise_schedule(torch.permute(tts_embed,(0,2,1)),training_noise_schedule,inference_noise_schedule)

        pqmf_audio1 = ae.decode(hidden)

        a = ae.pqmf.inverse(pqmf_audio1)
        draw_wave(a[0][0].detach().to("cpu"),f"diffusion")
        save_audio(a[0].detach().to("cpu"),48000,f"diffusion")
        break
    break

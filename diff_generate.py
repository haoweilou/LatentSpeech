from tts import StyleSpeech
from diffusion import DiffusionBlock

from dataset import BakerAudio,BakerText,LJSpeechAudio,LJSpeechText


import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from tts_config import config
from function import loadModel,save_audio,draw_heatmap
from function import hidden_to_audio,draw_wave
import math
import os
from tqdm import tqdm
from params import params
import numpy as np
def learning_rate(d_model=256,step=1,warmup_steps=400):
    return (1/math.sqrt(d_model)) * min(1/math.sqrt(step),step*warmup_steps**-1.5)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epoch = "S9k240" 
# epoch = "S4k240" 
# tts_model = HiddenSpeech(config).to(device)
tts_model = StyleSpeech(config).to(device)
tts_model = loadModel(tts_model,f"{epoch}_tts","./model")



diffusion_block = DiffusionBlock(params).to(device)
diffusion_block = loadModel(diffusion_block,f"{epoch}_diff","./model")


root = "L:/"
bakertext = BakerText(normalize=False,start=0,end=500,path=f"{root}baker/",ipa=True)
bakeraudio = BakerAudio(start=0,end=500,path=f"{root}baker/",return_len=True)

ljspeechtext = LJSpeechText(start=0,end=500,path=f"{root}LJSpeech/")
ljspeechaudio = LJSpeechAudio(start=0,end=500,path=f"{root}LJSpeech/",return_len=True)

from dataset import CombinedTextDataset,CombinedAudioDataset
textdataset = CombinedTextDataset(bakertext,ljspeechtext)
audiodataset = CombinedAudioDataset(bakeraudio,ljspeechaudio)

loader = DataLoader(dataset=list(zip(textdataset, audiodataset)), collate_fn=collate_fn, batch_size=32, shuffle=True)

print("Load Dataset: ")

# dataset = Baker(normalize=False,path="D:/baker/",start=0,end=10)
# dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=64, shuffle=False)

beta = np.array(params.noise_schedule)
noise_level = np.cumprod(1 - beta)
noise_level = torch.tensor(noise_level.astype(np.float32))


model_name = "StyleSpeechDiff"
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
    print(T,len(T))
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

from flow import AE
from function import loadModel

ae = AE(params).to(device)
ae = loadModel(ae,"ae20k16_1000","./model")



training_noise_schedule = np.array(params.noise_schedule)
inference_noise_schedule = np.array(params.noise_schedule)
total_loss = 0
diff_losss = 0
duration_losss = 0




for i,(text_batch,audio_batch) in enumerate(tqdm(loader)):
    x,s,_,x_lens,_,language = [tensor.to(device) for tensor in text_batch]
    audio,y_lens = audio_batch[0].to(device),audio_batch[1].to(device)
    with torch.no_grad():
        #B,C,T
        y,_ = ae.encode(audio) 
        y_lens = torch.ceil(y_lens/16/64)
    # x = batch[0].to(device)
    # s = batch[1].to(device)
    # l = batch[2].to(device)
    # src_lens = batch[4].to(device)
    # mel_lens = batch[5].to(device)
    # max_src_len = x.shape[1]
    # max_y_len = y.shape[1]

    #B,T,C
    c_embedd = context_model(x, s, x_lens,language)
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

    # N,T,C = y.shape
    # t = torch.randint(0, len(params.noise_schedule), [N])
    # noise_scale = noise_level[t].unsqueeze(1).to(device).unsqueeze(1)
    # noise_scale_sqrt = noise_scale**0.5
    # y_ = torch.transpose(y,1,2) #B,C,T
    # noise = torch.randn_like(y_).to(device)
    # noise_y = noise_scale_sqrt * y_ + (1.0 - noise_scale)**0.5 * noise #B,C,T
    # t = t.to(device)
    # predict_noise = diffusion_block(noise_y,t,tts_embedd) #B,C,T

    # _,_,duration_loss = loss_func(y,tts_embedd,log_l_pred,l,mel_masks,device=device)
    # diff_loss = l1_loss(predict_noise,noise)
    hidden = noise_schedule(tts_embed,training_noise_schedule,inference_noise_schedule)
    # loss = diff_loss + duration_loss
    # total_loss += loss.item()
    # diff_losss += diff_loss.item()
    # duration_losss += duration_loss.item()
    vmin,vmax = torch.min(y[0]).item(),torch.max(y[0]).item()
    idx = 1
    draw_heatmap(hidden[idx].detach().cpu().numpy(),vmin=vmin,vmax=vmax,name=f"h_fake_{idx}")
    draw_heatmap(y[idx].detach().cpu().numpy().T,vmin=vmin,vmax=vmax,name=f"h_real_{idx}")
    audio = hidden_to_audio(torch.transpose(hidden[idx:idx+1,:,:],1,2)).detach().cpu()[0]
    print("         ")
    print(torch.mean(hidden),torch.std(hidden))
    print(torch.mean(y),torch.std(y))
    draw_wave(audio[0],"diff_fake")
    save_audio(audio,48000,f"diffusion_fake","./sample/")

    audio = hidden_to_audio(y[idx:idx+1,:,:]).detach().cpu()[0]
    draw_wave(audio[0])
    save_audio(audio,48000,f"diffusion_real","./sample/")
    draw_wave(audio[0],"diff_real")
    break

# Backward pass and optimization
#Record

# print(f"loss: {total_loss:.4f} diff: {diff_losss:.4f} duration: {duration_losss:.4f}")      

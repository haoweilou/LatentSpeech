import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from params import params
# from tts import DurationAligner
from flow import AE
from function import loadModel,saveModel, agd_duration,fl_duration,force_alignment,duration_calculate
import pandas as pd

from dataset import BakerAudio,BakerText,LJSpeechAudio,LJSpeechText
from torch.utils.data import DataLoader
from params import params
import json
from torchaudio.transforms import MelSpectrogram

from tts import StyleSpeech2_FF,FastSpeechLoss
from tqdm import tqdm
from model import ASR
from tts_config import config
from logger import Log

import math
is_ipa = True
from ipa import alpha_pho_dict
config["pho_config"]["word_num"] = len(alpha_pho_dict)

# root = "/home/haoweilou/scratch/"
# root = "L:/"
root = "/scratch/ey69/hl6114/"

bakeraudio = BakerAudio(start=0,end=9000,path=f"{root}baker/",return_len=True)
bakertext = BakerText(normalize=False,start=0,end=9000,path=f"{root}baker/",ipa=True,alphabet=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ljspeechaudio = LJSpeechAudio(start=0,end=9000,path=f"{root}LJSpeech/",return_len=True)
ljspeechtext = LJSpeechText(start=0,end=9000,path=f"{root}LJSpeech/",alphabet=True)



from dataset import CombinedTextDataset,CombinedAudioDataset
textdataset = CombinedTextDataset(bakertext,ljspeechtext)
audiodataset = CombinedAudioDataset(bakeraudio,ljspeechaudio)

def collate_fn(batch):
    text_batch, audio_batch = zip(*batch)
    text_batch = [torch.stack([item[i] for item in text_batch]) for i in range(len(text_batch[0]))]
    audio_batch = bakeraudio.collate(audio_batch)
    return text_batch, audio_batch

# loader = DataLoader(dataset=list(zip(bakertext, bakeraudio)), collate_fn=collate_fn, batch_size=16, shuffle=True)
loader = DataLoader(dataset=list(zip(textdataset, audiodataset)), collate_fn=collate_fn, batch_size=32, shuffle=True)

ae = AE(params).to(device)
ae = loadModel(ae,"ae20k16_1000","./model")

def learning_rate(d_model=256,step=1,warmup_steps=4000):
    return (1/math.sqrt(d_model)) * min(1/math.sqrt(step),step*warmup_steps**-1.5)

lr = learning_rate()
print(f"Initial LR: {lr}")
model = StyleSpeech2_FF(config,embed_dim=16).to(device)
optimizer = optim.Adam(model.parameters(), betas=(0.9,0.98),eps=1e-9,lr=lr)

modelname = "StyleSpeech2_FF_18K_ALPHA"

fastloss = FastSpeechLoss().to(device)

log = Log(tts_loss=0,duration_loss=0)

melspec_transform = MelSpectrogram(sample_rate=48000,n_fft=1024,hop_length=1024,n_mels=80).to(device)

EPOCH = 501

step = 1
print(modelname,EPOCH)
for epoch in range(0,EPOCH):
    total_loss = 0
    fastLoss_ = 0
    duration_loss_ = 0
    for i,(text_batch,audio_batch) in enumerate(tqdm(loader)):
        x,s,l,x_lens,mel_lens,language = [tensor.to(device) for tensor in text_batch]
        audio,y_lens = audio_batch[0].to(device),audio_batch[1].to(device)

        with torch.no_grad():
            y,_ = ae.encode(audio) 
            y_lens = torch.ceil(y_lens/16/64)
        y_shape = y.shape[-1]
        max_mel_lens = max(mel_lens).item()
        diff = max_mel_lens-y_shape
        if diff > 0:
            y = F.pad(y,[0,diff],value=0)
        
        optimizer.zero_grad()
        #use aligner to predict l 
        
        #in here since there is duration difference, use the max l as the target y lens
        y_pred,log_l,y_mask = model(x, s, x_lens,l=l,y_lens=mel_lens,max_y_len=y.shape[-1],language=language)

        loss,tts_loss,duration_loss = fastloss(y,y_pred,log_l,l,y_mask)
        loss.backward()

        optimizer.step()
        fastLoss_ += tts_loss.item()
        duration_loss_ += duration_loss.item()
        total_loss += loss.item()

        step += 1

        new_lr = learning_rate(step=step)
        for param_group in optimizer.param_groups: param_group['lr'] = new_lr
        log.update(tts_loss=tts_loss.item(),duration_loss=duration_loss.item())

    print(f"Epoch: {epoch} Duration Loss: {duration_loss_/len(loader):.03f} TTS Loss: {fastLoss_/len(loader):.03f} Total: {total_loss/len(loader):.03f}")
    if epoch % 50 == 0: 
        saveModel(model,f"{modelname}_{epoch}","./model/")
    log.save(f"./log/loss_{modelname}")
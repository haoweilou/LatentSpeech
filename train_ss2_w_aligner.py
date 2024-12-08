import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from params import params
# from tts import DurationAligner
from flow import AE
from function import loadModel,saveModel, agd_duration
import pandas as pd

from dataset import BakerAudio,BakerText
from torch.utils.data import DataLoader
from params import params
import json
from torchaudio.transforms import MelSpectrogram

from tts import StyleSpeech2_FF,FastSpeechLoss
from tqdm import tqdm
from model import ASR
from tts_config import config

import math
bakertext = BakerText(normalize=False,start=0,end=100,path="L:/baker/")
bakeraudio = BakerAudio(start=0,end=100,path="L:/baker/",return_len=True)
def collate_fn(batch):
    text_batch, audio_batch = zip(*batch)
    text_batch = [torch.stack([item[i] for item in text_batch]) for i in range(len(text_batch[0]))]
    audio_batch = bakeraudio.collate(audio_batch)
    return text_batch, audio_batch

loader = DataLoader(dataset=list(zip(bakertext, bakeraudio)), collate_fn=collate_fn, batch_size=16, shuffle=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ae = AE(params).to(device)
ae = loadModel(ae,"ae20k16_1000","./model")
def learning_rate(d_model=256,step=1,warmup_steps=400):
    return (1/math.sqrt(d_model)) * min(1/math.sqrt(step),step*warmup_steps**-1.5)

model = StyleSpeech2_FF(config,embed_dim=16).to(device)
optimizer = optim.Adam(model.parameters(), betas=(0.9,0.98),eps=1e-9,lr=0.0001)

modelname = "StyleSpeech2_FF"
loss_log = pd.DataFrame({"total_loss":[],"tts_loss":[],"duration_loss":[]})
loss_log_name =  f"./log/loss_{modelname}"

with open("./save/cache/phoneme.json","r") as f: 
    phoneme_set = json.loads(f.read())["phoneme"]
aligner = ASR(80,len(phoneme_set)+1).to(device)
aligner = loadModel(aligner,"aligner_500","./model/")

fastloss = FastSpeechLoss().to(device)

melspec_transform = MelSpectrogram(sample_rate=48000,n_fft=1024,hop_length=1024,n_mels=80).to(device)


for epoch in range(0,1001):
    total_loss = 0
    fastLoss_ = 0
    duration_loss_ = 0
    for i,(text_batch,audio_batch) in enumerate(tqdm(loader)):
        x,s,_,x_lens,_ = [tensor.to(device) for tensor in text_batch]
        audio,y_lens = audio_batch[0].to(device),audio_batch[1].to(device)
        with torch.no_grad():
            y,_ = ae.encode(audio) 
            y_lens = torch.ceil(y_lens/16/64)

            melspec = melspec_transform(audio).squeeze(1) #B,T,80
            melspec = melspec.permute(0,2,1)#B,80,T

        speaker = torch.zeros(x_lens.shape).to(dtype=x_lens.dtype,device=x_lens.device)
        optimizer.zero_grad()

        #use aligner to predict l 
        prob_matrix = aligner(melspec)  # [batch_size, y_len, num_phonemes], probability 
        l = agd_duration(prob_matrix,x_max_len=x.shape[-1])
        
        y_pred,log_l,y_mask = model(x, s, x_lens,l=l,y_lens=y_lens,max_y_len=y.shape[-1])

        loss,tts_loss,duration_loss = fastloss(y,y_pred,log_l,l,y_mask)
        loss.backward()

        optimizer.step()
        fastLoss_ += tts_loss.item()
        duration_loss_ += duration_loss.item()
        total_loss += loss.item()
        

    print(f"Epoch: {epoch} Duration Loss: {duration_loss_/len(loader):.03f} TTS Loss: {fastLoss_/len(loader):.03f} Total: {total_loss/len(loader):.03f}")

    if epoch % 100 == 0:
        saveModel(model,f"{modelname}_{epoch}","./model/")
    loss_log.loc[len(loss_log.index)] = [total_loss/len(loader),fastLoss_/len(loader),duration_loss_/len(loader)]
    loss_log.to_csv(loss_log_name)

    # if epoch > 0:
    #     new_lr = learning_rate(step=epoch)
    #     for param_group in optimizer.param_groups: param_group['lr'] = new_lr
        
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

import math
is_ipa = True
from ipa import ipa_pho_dict
if is_ipa: config["pho_config"]["word_num"] = len(ipa_pho_dict)

root = "/home/haoweilou/scratch/"
# root = "L:/"
loss_log = pd.DataFrame({"total_loss":[],"ctc_loss":[]})
# bakertext = BakerText(normalize=False,start=0,end=500,path=f"{root}baker/",ipa=True)
bakeraudio = BakerAudio(start=0,end=500,path=f"{root}baker/",return_len=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

aligner = ASR(80,len(ipa_pho_dict)+1).to(device)
aligner = loadModel(aligner,"aligner_en_600","./model/")

ljspeechaudio = LJSpeechAudio(start=0,end=2000,path=f"{root}LJSpeech/",return_len=True)
ljspeechtext = LJSpeechText(start=0,end=2000,path=f"{root}LJSpeech/")
# ljspeechtext.calculate_l(aligner,ys=ljspeechaudio.audios,y_lens=ljspeechaudio.audio_lens)



from dataset import CombinedTextDataset,CombinedAudioDataset
# textdataset = CombinedTextDataset(bakertext,ljspeechtext)
# audiodataset = CombinedAudioDataset(bakeraudio,ljspeechaudio)

textdataset = CombinedTextDataset(ljspeechtext,ljspeechtext)
audiodataset = CombinedAudioDataset(ljspeechaudio,ljspeechaudio)

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

modelname = "StyleSpeech2_FF"
loss_log = pd.DataFrame({"total_loss":[],"tts_loss":[],"duration_loss":[]})
loss_log_name =  f"./log/loss_{modelname}"
# model = loadModel(model, f"{modelname}_200","./model/")

# aligner = ASR(80,len(phoneme_set)+1).to(device)


fastloss = FastSpeechLoss().to(device)

melspec_transform = MelSpectrogram(sample_rate=48000,n_fft=1024,hop_length=1024,n_mels=80).to(device)


step = 1
for epoch in range(0,501):
    total_loss = 0
    fastLoss_ = 0
    duration_loss_ = 0
    for i,(text_batch,audio_batch) in enumerate(tqdm(loader)):
        x,s,l,x_lens,_,language = [tensor.to(device) for tensor in text_batch]
        # print(language)
        audio,y_lens = audio_batch[0].to(device),audio_batch[1].to(device)

        with torch.no_grad():
            y,_ = ae.encode(audio) 
            y_lens = torch.ceil(y_lens/16/64)

            # melspec = melspec_transform(audio).squeeze(1) #B,T,80
            # melspec = melspec.permute(0,2,1)#B,80,T

            # prob_matrix = aligner(melspec,language)  # [batch_size, y_len, num_phonemes], probability 
            # emission = torch.log_softmax(prob_matrix,dim=-1) # [seq_len, batch_size, num_phonemes]
            # l = duration_calculate(emission.cpu(),x.cpu(),x_lens.cpu(),y_lens.cpu(),max_x_len = x.shape[-1])
            # l = agd_duration(prob_matrix,x_max_len=x.shape[-1])
            # l = fl_duration(prob_matrix,x,x_max_len=x.shape[-1])
        # print(x.shape,s.shape,l.shape)
        optimizer.zero_grad()
        #use aligner to predict l 
        
        y_pred,log_l,y_mask = model(x, s, x_lens,l=l,y_lens=y_lens,max_y_len=y.shape[-1],language=language)

        loss,tts_loss,duration_loss = fastloss(y,y_pred,log_l,l,y_mask)
        loss.backward()

        optimizer.step()
        fastLoss_ += tts_loss.item()
        duration_loss_ += duration_loss.item()
        total_loss += loss.item()

        step += 1

        new_lr = learning_rate(step=step)
        for param_group in optimizer.param_groups: param_group['lr'] = new_lr

    print(f"Epoch: {epoch} Duration Loss: {duration_loss_/len(loader):.03f} TTS Loss: {fastLoss_/len(loader):.03f} Total: {total_loss/len(loader):.03f}")
    if epoch % 50 == 0: 
        saveModel(model,f"{modelname}_{epoch}","./model/")
    
    loss_log.loc[len(loss_log.index)] = [total_loss/len(loader),fastLoss_/len(loader),duration_loss_/len(loader)]
    loss_log.to_csv(loss_log_name)
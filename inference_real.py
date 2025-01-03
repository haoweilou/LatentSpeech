import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from params import params
# from tts import DurationAligner
from flow import AE
from function import loadModel,saveModel, agd_duration,fl_duration,force_alignment,duration_calculate

from dataset import BakerAudio,BakerText,LJSpeechAudio,LJSpeechText
from torch.utils.data import DataLoader
from params import params

from tqdm import tqdm

import math
is_ipa = True

# root = "/home/haoweilou/scratch/"
root = "L:/"
no_sil = False
sil_duration = None

bakeraudio = BakerAudio(start=9000,end=10000,path=f"{root}baker/",return_len=True)
bakertext = BakerText(normalize=False,start=9000,end=10000,path=f"{root}baker/",ipa=True,no_sil=no_sil,sil_duration=sil_duration)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ljspeechaudio = LJSpeechAudio(start=9000,end=10000,path=f"{root}LJSpeech/",return_len=True)
ljspeechtext = LJSpeechText(start=9000,end=10000,path=f"{root}LJSpeech/",no_sil=no_sil,sil_duration=sil_duration)


from dataset import CombinedTextDataset,CombinedAudioDataset
textdataset = CombinedTextDataset(bakertext,ljspeechtext)
audiodataset = CombinedAudioDataset(bakeraudio,ljspeechaudio)

def collate_fn(batch):
    text_batch, audio_batch = zip(*batch)
    text_batch = [torch.stack([item[i] for item in text_batch]) for i in range(len(text_batch[0]))]
    audio_batch = bakeraudio.collate(audio_batch)
    return text_batch, audio_batch

# loader = DataLoader(dataset=list(zip(bakertext, bakeraudio)), collate_fn=collate_fn, batch_size=16, shuffle=True)
loader = DataLoader(dataset=list(zip(textdataset, audiodataset)), collate_fn=collate_fn, batch_size=32, shuffle=False)

ae = AE(params).to(device)
ae = loadModel(ae,"ae20k16_1000","./model")

def learning_rate(d_model=256,step=1,warmup_steps=4000):
    return (1/math.sqrt(d_model)) * min(1/math.sqrt(step),step*warmup_steps**-1.5)

lr = learning_rate()
print(f"Initial LR: {lr}")
modelname = "real_18k"


import os 
from function import save_audio
audio_path = f"L:/evaluate/{modelname}"
if os.path.exists(audio_path) == False:
    os.makedirs(audio_path)

for file in os.listdir(audio_path):
    os.remove(f"{audio_path}/{file}")

step = 0
print(modelname)
audio_num = 0
for epoch in range(0,1):
    total_loss = 0
    fastLoss_ = 0
    duration_loss_ = 0
    for i,(text_batch,audio_batch) in enumerate(tqdm(loader)):
        x,s,l,x_lens,_,language = [tensor.to(device) for tensor in text_batch]
        # print(language)
        audios,y_lens = audio_batch[0].to(device),audio_batch[1].to(device)

        for j,audio in enumerate(audios):
            print(j,audio.shape)
            audio = audio[:,:y_lens[j]].cpu()
            if audio_num < 1000: audio_name = f"ch_{audio_num}"
            else: audio_name = f"en_{audio_num%1000}"
            save_audio(audio,48000,audio_name,audio_path)
            audio_num += 1
import torch
import torch.optim as optim
from params import params
# from tts import DurationAligner
from flow import AE
from function import loadModel,saveModel

from dataset import BakerAudio,BakerText,LJSpeechAudio,LJSpeechText
from torch.utils.data import DataLoader
from params import params
from torchaudio.transforms import MelSpectrogram

from tts import FastSpeech,FastSpeechLoss
from tqdm import tqdm
from tts_config import config
from logger import Log

import math
is_ipa = True
from ipa import ipa_pho_dict
if is_ipa: config["pho_config"]["word_num"] = len(ipa_pho_dict)

root = "/home/haoweilou/scratch/"
root = "/scratch/ey69/hl6114/"
# root = "L:/"
no_sil = False
sil_duration = None

bakeraudio = BakerAudio(start=0,end=9000,path=f"{root}baker/",return_len=True)
bakertext = BakerText(normalize=False,start=0,end=9000,path=f"{root}baker/",ipa=True,no_sil=no_sil,sil_duration=sil_duration)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


ljspeechaudio = LJSpeechAudio(start=0,end=9000,path=f"{root}LJSpeech/",return_len=True)
ljspeechtext = LJSpeechText(start=0,end=9000,path=f"{root}LJSpeech/",no_sil=no_sil,sil_duration=sil_duration)

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
model = FastSpeech(config,embed_dim=16).to(device)
optimizer = optim.Adam(model.parameters(), betas=(0.9,0.98),eps=1e-9,lr=lr)

modelname = "FastSpeech_18K"

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
        x,s,l,x_lens,_,language = [tensor.to(device) for tensor in text_batch]
        # print(language)
        audio,y_lens = audio_batch[0].to(device),audio_batch[1].to(device)

        with torch.no_grad():
            y,_ = ae.encode(audio) 
            y_lens = torch.ceil(y_lens/16/64)

        optimizer.zero_grad()
        #use aligner to predict l 
        
        y_pred,log_l,y_mask = model(x, x_lens,duration_target=l,mel_lens=y_lens,max_mel_len=y.shape[-1])
        y_pred = torch.transpose(y_pred,1,2)#ypred = B,T,C => B,C,T
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
from tts import StyleSpeech2_FF,FastSpeechLoss
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
from tts import StyleSpeech2_Diff
# tts_model = StyleSpeech2_FF(config,embed_dim=16).to(device)

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

# root = "L:/"
root = "/home/haoweilou/scratch/"

bakertext = BakerText(normalize=False,start=0,end=2000,path=f"{root}baker/",ipa=True)
bakeraudio = BakerAudio(start=0,end=2000,path=f"{root}baker/",return_len=True)

ljspeechtext = LJSpeechText(start=0,end=2000,path=f"{root}LJSpeech/")
ljspeechaudio = LJSpeechAudio(start=0,end=2000,path=f"{root}LJSpeech/",return_len=True)

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

loader = DataLoader(dataset=list(zip(textdataset, audiodataset)), collate_fn=collate_fn, batch_size=32, shuffle=True)

beta = np.array(params.noise_schedule)
noise_level = np.cumprod(1 - beta)
noise_level = torch.tensor(noise_level.astype(np.float32))
model_name = "StyleSpeechDiff"

log = Log(l2_loss=0)
log.load(f"./log/loss_{model_name}")
from tts import sequence_mask
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

        # print(c_embedd,c_embedd[:,:,x_lens[0].item():])

        # tts_embed = tts_embed + 
        # print(tts_embed.shape,c_embedd.shape)
        # print(language_embed.shape)
        t = t.to(device)
        c_embedd = c_embedd.permute(0,2,1)
        # print(c_embedd.shape)
        y_mask = sequence_mask(y_lens, y.shape[-1]).unsqueeze(1).to(y.dtype) #[B,1,L]
        predict_noise = diffusion_block(noise_y,t,c_embedd) #B,C,T
        # print(y.shape,noise_y.shape)

        # print(tts_embedd.shape,x.shape)

        diff_loss = l1_loss(predict_noise*y_mask,noise*y_mask)
        loss = diff_loss
        log.update(l2_loss=diff_loss.item())
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch} {log}")
        


    log.save(f"./log/loss_{model_name}")

    if epoch % 100 == 0:
        saveModel(context_model,f"{model_name}_content_{epoch}","./model/")
        saveModel(diffusion_block,f"{model_name}_diff_{epoch}","./model/")
    if epoch > 0:
        new_lr = learning_rate(step=epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
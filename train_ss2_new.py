import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from params import params
# from tts import DurationAligner
from flow import AE
from function import loadModel,saveModel
import pandas as pd

from dataset import BakerAudio,BakerText
from torch.utils.data import DataLoader
from params import params

bakertext = BakerText(normalize=False,start=0,end=50,path="C:/baker/")
bakeraudio = BakerAudio(start=0,end=50,path="C:/baker/",return_len=True)
# bakertext = BakerText(normalize=False,start=0,end=3000)
# bakeraudio = BakerAudio(start=0,end=3000,return_len=True)
def collate_fn(batch):
    text_batch, audio_batch = zip(*batch)
    text_batch = [torch.stack([item[i] for item in text_batch]) for i in range(len(text_batch[0]))]
    audio_batch = bakeraudio.collate(audio_batch)
    return text_batch, audio_batch
from tts import StyleSpeech2_New
from tqdm import tqdm
from tts_config import config

loader = DataLoader(dataset=list(zip(bakertext, bakeraudio)), collate_fn=collate_fn, batch_size=16, shuffle=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = StyleSpeech2_New(config).to(device)

ae = AE(params).to(device)
ae = loadModel(ae,"ae20k16_1000","./model")
import math
def learning_rate(d_model=256,step=1,warmup_steps=400):
    return (1/math.sqrt(d_model)) * min(1/math.sqrt(step),step*warmup_steps**-1.5)

optimizer = optim.Adam(model.parameters(), betas=(0.9,0.98),eps=1e-9,lr=learning_rate())

def mle_loss(z, m, logs, logdet, mask):
    #z: latent data, m: mean, logs: log std, logdet
    #sum(logs) + 1/2*sum((z-m)^2/s^2) - sum(logdet)
    l = torch.sum(logs) + 0.5 * torch.sum(torch.exp(-2 * logs) * ((z - m)**2)) # neg normal likelihood w/o the constant term
    l = l - torch.sum(logdet) # log jacobian determinant
    #averging
    l = l / torch.sum(torch.ones_like(z) * mask) # averaging across batch, channel and time axes
    #l + 1/2*log(2pi)
    l = l + 0.5 * math.log(2 * math.pi) # add the remaining constant term
    return l

def duration_loss(logw, logw_, lengths):
    l = torch.sum((logw - logw_)**2) / torch.sum(lengths)
    return l
modelname = "StyleSpeech2_New"
loss_log = pd.DataFrame({"total_loss":[],"mse_loss":[],"duration_loss":[]})
loss_log_name =  f"./log/loss_{modelname}"

for epoch in range(0,1001):
    total_loss = 0
    mle_loss_ = 0
    duration_loss_ = 0
    for i,(text_batch,audio_batch) in enumerate(tqdm(loader)):
        x,s,_,x_lens,_ = [tensor.to(device) for tensor in text_batch]
        audio,y_lens = audio_batch[0].to(device),audio_batch[1].to(device)
        with torch.no_grad():
            y,_ = ae.encode(audio) 
            y_lens = torch.ceil(y_lens/16/64)
        speaker = torch.zeros(x_lens.shape).to(dtype=x_lens.dtype,device=x_lens.device)
        optimizer.zero_grad()
        (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_)= model(x,s,x_lens,y,y_lens,speaker)
        l_mle = mle_loss(z, z_m, z_logs, logdet, z_mask)
        l_length = duration_loss(logw, logw_, x_lens)
        loss = l_mle + l_length
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        mle_loss_ += l_mle.item()
        duration_loss_ += l_length.item()
    print(f"Epoch: {epoch} MLE Loss: {mle_loss_/len(loader):.03f} Duration Loss: {duration_loss_/len(loader):.03f} Total: {total_loss/len(loader):.03f}")
    if epoch % 200 == 0:
        saveModel(model,f"{modelname}_{epoch}","./model/")
    loss_log.loc[len(loss_log.index)] = [total_loss/len(loader),mle_loss_/len(loader),duration_loss_/len(loader)]
    loss_log.to_csv(loss_log_name)

    if epoch > 0:
        new_lr = learning_rate(step=epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        
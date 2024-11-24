import torch
import torch.nn as nn
import torchaudio
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,ConcatDataset
from params import params
from dataset import BakerAudio,LJSpeechAudio
# from model import VQAESeq
from jukebox import Jukebox,UpSampler
from tqdm import tqdm
from function import saveModel,loadModel
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import pandas as pd
torch.autograd.set_detect_anomaly(True)

model = Jukebox(params).to(device)
# model = loadModel(model,f"jukebox_upsampler","./model/",strict=False)
# model = loadModel(model,f"jukebox_upsampler1_2000","./model/",strict=True)
# model = loadModel(model,f"jukebox_upsampler2_3000","./model/",strict=True)


upsampler =  nn.Sequential(
    UpSampler(64,256,num_res_layer=12,ratio=4),
    UpSampler(64,256,num_res_layer=12,ratio=4)
).to(device)
# upsampler = loadModel(upsampler,"upsampler3_500","./model/")

optimizer = optim.Adam(upsampler.parameters(),lr=0.0003)
loss_log = pd.DataFrame({"total_loss":[], "feature_loss":[]})
dataset1 = BakerAudio(0,1000,path="/scratch/ey69/hl6114/baker/")
# dataset = ConcatDataset([dataset1])
# dataset2 = LJSpeechAudio(0,10000,path="/g/data/ey69/haowei/LJSpeech/")
# dataset = ConcatDataset([dataset1, dataset2])
dataset = ConcatDataset([dataset1])
model_name = "upsampler3"

batch_size = 32
loader = DataLoader(dataset,batch_size=batch_size,collate_fn=dataset1.collate,drop_last=True,shuffle=True)
epochs = 2001

for epoch in range(epochs):
    loss_val = 0
    feature_loss_ = 0
    for audio in tqdm(loader):
        optimizer.zero_grad()
        audio = audio.to(device)
        time_steps = audio.shape[-1]
        pad_amount = (16 - (time_steps % 16)) % 16
        if pad_amount > 0:audio = F.pad(audio, (0, pad_amount))
        pqmf_audio = model.pqmf(audio)
        with torch.no_grad():
            z1q = model.vqae1.encode(pqmf_audio)
            z3q = model.vqae3.encode(pqmf_audio)
        z1q_f = upsampler(z3q.detach())
        z1q_f,z1q = model.equal_size(z1q_f,z1q)
        feature_loss = F.mse_loss(z1q_f,z1q)
        loss =  feature_loss
        loss_val += loss.item()
        feature_loss_ += feature_loss.item()

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    print(f"Epoch: {epoch} Feature Loss: {feature_loss_/len(loader):.03f} Total: {loss_val/len(loader):.03f}")
    
    if epoch % 100 == 0:
        saveModel(upsampler,f"{model_name}_{epoch}","./model/")

    loss_log.loc[len(loss_log.index)] = [loss_val/len(loader),feature_loss_/len(loader)]
    loss_log.to_csv(f"./log/loss_{model_name}")
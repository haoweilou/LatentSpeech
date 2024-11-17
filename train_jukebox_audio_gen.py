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
from jukebox import Jukebox
from tqdm import tqdm
from function import saveModel,loadModel
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import pandas as pd
torch.autograd.set_detect_anomaly(True)

model = Jukebox(params).to(device)
# model = loadModel(model,f"jukebox_upsampler","./model/",strict=False)
# model = loadModel(model,f"jukebox_upsampler1_2000","./model/",strict=True)
model = loadModel(model,f"jukebox_upsampler2_1400","./model/",strict=True)
wave_gen = model.wave_gen
loud_gen = model.loud_gen


optimizer = optim.Adam(list(loud_gen.parameters())+list(wave_gen.parameters()),lr=0.001)

loss_log = pd.DataFrame({"total_loss":[], "feature_loss":[]})
dataset1 = BakerAudio(0,10000)
dataset2 = LJSpeechAudio(0,10000)
dataset = ConcatDataset([dataset1, dataset2])

batch_size = 32
loader = DataLoader(dataset,batch_size=batch_size,collate_fn=dataset1.collate,drop_last=True,shuffle=True)
epochs = 3001
model_name = "wavegen"

for epoch in range(epochs):
    loss_val = 0
    audio_loss_ = 0
    for audio in tqdm(loader):
        optimizer.zero_grad()
        audio = audio.to(device)
        time_steps = audio.shape[-1]
        pad_amount = (16 - (time_steps % 16)) % 16
        if pad_amount > 0:audio = F.pad(audio, (0, pad_amount))
        with torch.no_grad():
            pqmf_audio = model.pqmf(audio)
            audio_embed,_ = model.vqae1(pqmf_audio)
            audio_embed,pqmf_audio = model.equal_size(audio_embed,pqmf_audio)
            
        loud = loud_gen(audio_embed)
        wave = wave_gen(audio_embed)
        pqmf_audio1 = torch.tanh(wave) *  model.mod_sigmoid(loud)
        audio_loss =    model.spec_distance(pqmf_audio,pqmf_audio1)

        loss =  audio_loss
        loss_val += loss.item()
        audio_loss_ += audio_loss.item()

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    print(f"Epoch: {epoch} Audio Loss: {audio_loss_/len(loader):.03f} Total: {loss_val/len(loader):.03f}")
    
    if epoch % 100 == 0:
        saveModel(loud_gen,f"loud_{epoch}","./model/")
        saveModel(wave_gen,f"wave_{epoch}","./model/")


    loss_log.loc[len(loss_log.index)] = [loss_val/len(loader),audio_loss_/len(loader)]
    loss_log.to_csv(f"./log/loss_{model_name}")
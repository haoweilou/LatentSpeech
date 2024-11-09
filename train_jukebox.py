import torch
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
optimizer = optim.Adam(model.parameters(),lr=0.0001)

loss_log = pd.DataFrame({"total_loss":[],"vq_loss":[],"audio_loss":[]})
dataset1 = BakerAudio(0,10000)
dataset2 = LJSpeechAudio(0,10000)
dataset = ConcatDataset([dataset1, dataset2])
# dataset = ConcatDataset([dataset1])

model_name = "jukebox"
model = loadModel(model,f"{model_name}_{600}","./model/",strict=True)
batch_size = 16
loader = DataLoader(dataset,batch_size=batch_size,collate_fn=dataset1.collate,drop_last=True,shuffle=True)
epochs = 801

for epoch in range(601,epochs):
    loss_val = 0
    vq_loss_ = 0
    audio_loss_ = 0
    for audio in tqdm(loader):
        optimizer.zero_grad()
        audio = audio.to(device)
        time_steps = audio.shape[-1]
        pad_amount = (16 - (time_steps % 16)) % 16
        if pad_amount > 0:audio = F.pad(audio, (0, pad_amount))
        y,audio_loss,vq_loss = model(audio)

        loss = vq_loss + audio_loss
        loss_val += loss.item()
        vq_loss_ += vq_loss.item()
        audio_loss_ += audio_loss.item()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    print(f"Epoch: {epoch} Audio Loss: {audio_loss_/len(loader):.03f} VQ Loss: {vq_loss_/len(loader):.03f} Total: {loss_val/len(loader):.03f}")
    
    if epoch % 10 == 0:
        saveModel(model,f"{model_name}_{epoch}","./model/")

    loss_log.loc[len(loss_log.index)] = [loss_val/len(loader),vq_loss_/len(loader),audio_loss_/len(loader)]
    loss_log.to_csv(f"./log/loss_{model_name}")
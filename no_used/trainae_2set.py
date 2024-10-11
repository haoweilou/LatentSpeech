import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,ConcatDataset
from params import params
from dataset import BakerAudio,LJSpeechAudio
from model import AE
from tqdm import tqdm
from function import saveModel,loadModel
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import pandas as pd
num_hiddens=80
num_residual_layers=30
num_residual_hiddens=80
num_embeddings=256
embedding_dim=80
commitment_cost=0.25

# model = AE(params).to(device)
model = AE(params).to(device)
model = loadModel(model,"ae9k_400","./model/")
optimizer = optim.Adam(model.parameters(),lr=params.learning_rate)
loss_log = pd.DataFrame({"total_loss":[],"spectral_loss":[]})
dataset1 = BakerAudio(0,10000)
dataset2 = LJSpeechAudio(0,10000)
dataset = ConcatDataset([dataset1, dataset2])

loader = DataLoader(dataset,batch_size=params.batch_size,collate_fn=dataset1.collate,drop_last=True,shuffle=True)
epochs = 501
model_name = "ae9k2set"
for epoch in range(0,epochs):
    loss_val = 0
    spectral_loss_ = 0
    for audio in tqdm(loader):
        optimizer.zero_grad()
        audio = audio.to(device)
        audio_reconstruct,spectral_loss = model(audio)
        loss = spectral_loss
        loss_val += loss.item()
        spectral_loss_ += spectral_loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    print(f"Epoch: {epoch} Spectral Loss: {spectral_loss_/len(loader):.03f} Total: {loss_val/len(loader):.03f}")
    
    if epoch % 10 == 0:
        saveModel(model,f"{model_name}_{epoch}","./model/")

    loss_log.loc[len(loss_log.index)] = [loss_val/len(loader),spectral_loss_/len(loader)]
    loss_log.to_csv(f"./log/loss_{model_name}")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from params import params
from dataset import BakerAudio
from model import VQAE,AE
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

ae = AE(params).to(device)
ae = loadModel(ae,"ae","./model/")
codebook = torch.load("codebook").to(device)
model = VQAE(params).to(device)
model.encoder = ae.encoder
model.decoder = ae.decoder
model.vq_layer.embedding.weight = torch.nn.Parameter(codebook)


for param in model.encoder.parameters():
    param.requires_grad = False
for param in model.decoder.parameters():
    param.requires_grad = True
for param in model.vq_layer.parameters():
    param.requires_grad = True

# optimizer = optim.Adam(model.parameters(),lr=params.learning_rate)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params.learning_rate)

loss_log = pd.DataFrame({"total_loss":[],"spectral_loss":[],"vq_loss":[]})
dataset = BakerAudio(0,1000)
loader = DataLoader(dataset,batch_size=params.batch_size,collate_fn=dataset.collate,drop_last=True,shuffle=True)
epochs = 2001
model_name = "vqaeinit"
for epoch in range(0,epochs):
    loss_val = 0
    spectral_loss_ = 0
    vq_loss_ = 0
    for audio in tqdm(loader):
        optimizer.zero_grad()
        audio = audio.to(device)
        audio_reconstruct,spectral_loss,vq_loss = model(audio)
        loss = spectral_loss+vq_loss
        loss_val += loss.item()
        spectral_loss_ += spectral_loss.item()
        vq_loss_ += vq_loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    print(f"Epoch: {epoch} Spectral Loss: {spectral_loss_/len(loader):.03f} VQ Loss: {vq_loss_/len(loader):.03f} Total: {loss_val/len(loader):.03f}")
    
    if epoch % 100 == 0:
        saveModel(model,f"{model_name}_{epoch}","./model/")

    loss_log.loc[len(loss_log.index)] = [loss_val/len(loader),spectral_loss_/len(loader),vq_loss_/len(loader)]
    loss_log.to_csv(f"./log/loss_{model_name}")
import torch
import torchaudio
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from params import params
from dataset import BakerAudio
from model import VQAESeq
from tqdm import tqdm
from function import saveModel
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import pandas as pd
torch.autograd.set_detect_anomaly(True)
num_hiddens=80
num_residual_layers=30
num_residual_hiddens=80
num_embeddings=256
embedding_dim=80
commitment_cost=0.25

model = VQAESeq(params).to(device)

optimizer = optim.Adam(model.parameters(),lr=params.learning_rate)
loss_log = pd.DataFrame({"total_loss":[],"spectral_loss":[],"vq_loss":[],"audio_loss":[]})
dataset = BakerAudio(0,10000)
batch_size = 16
# loader = DataLoader(dataset,batch_size=params.batch_size,collate_fn=dataset.collate,drop_last=True,shuffle=True)
loader = DataLoader(dataset,batch_size=batch_size,collate_fn=dataset.collate,drop_last=True,shuffle=True)
epochs = 501
model_name = "vqae"
for epoch in range(0,epochs):
    loss_val = 0
    spectral_loss_ = 0
    vq_loss_ = 0
    audio_loss_ = 0
    for audio in tqdm(loader):
        optimizer.zero_grad()
        audio = audio.to(device)
        time_steps = audio.shape[-1]
        pad_amount = (16 - (time_steps % 16)) % 16
        if pad_amount > 0:audio = F.pad(audio, (0, pad_amount))
        y,audio_loss,vq_loss,spectral_loss= model(audio)
        loss = spectral_loss+vq_loss + audio_loss
        loss_val += loss.item()
        spectral_loss_ += spectral_loss.item()
        vq_loss_ += vq_loss.item()
        audio_loss_ += audio_loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    print(f"Epoch: {epoch} Audio Loss: {audio_loss_/len(loader):.03f} Spectral Loss: {spectral_loss_/len(loader):.03f} VQ Loss: {vq_loss_/len(loader):.03f} Total: {loss_val/len(loader):.03f}")
    
    if epoch % 10 == 0:
        saveModel(model,f"{model_name}_{epoch}","./model/")

    loss_log.loc[len(loss_log.index)] = [loss_val/len(loader),spectral_loss_/len(loader),vq_loss_/len(loader),audio_loss_/len(loader)]
    loss_log.to_csv(f"./log/loss_{model_name}")
import torch
import torchaudio
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from params import params
from dataset import BakerAudio
from model import VQSpecAE
from tqdm import tqdm
from function import saveModel,loadModel
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import pandas as pd
torch.autograd.set_detect_anomaly(True)
num_hiddens=80
num_residual_layers=30
num_residual_hiddens=80
num_embeddings=256
embedding_dim=80
commitment_cost=0.25

model = VQSpecAE().to(device)
optimizer = optim.Adam(model.parameters(),lr=params.learning_rate)
loss_log = pd.DataFrame({"total_loss":[],"spectral_loss":[],"vq_loss":[]})
dataset = BakerAudio(0,1000)
loader = DataLoader(dataset,batch_size=params.batch_size,collate_fn=dataset.collate,drop_last=True,shuffle=True)
epochs = 501
model_name = "vqspecae"

melspec_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=48*1000,      # Adjust the sample rate to your audio's sample rate
    n_fft=400,              # Size of FFT, you can adjust this
    win_length=400,         # Window size
    hop_length=160,         # Hop length between frames
    n_mels=80               # Set number of mel bins to 80
).to(device)

for epoch in range(0,epochs):
    loss_val = 0
    spectral_loss_ = 0
    vq_loss_ = 0
    for audio in tqdm(loader):
        optimizer.zero_grad()
        audio = audio.to(device)
        melspec = melspec_transform(audio)

        time_steps = melspec.shape[-1]
        pad_amount = (4 - (time_steps % 4)) % 4
        if pad_amount > 0:melspec = F.pad(melspec, (0, pad_amount))

        melspec_reconstruct,vq_loss = model(melspec)
        spectral_loss = F.mse_loss(melspec_reconstruct,melspec)
        loss = spectral_loss+vq_loss
        loss_val += loss.item()
        spectral_loss_ += spectral_loss.item()
        vq_loss_ += vq_loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    print(f"Epoch: {epoch} Spectral Loss: {spectral_loss_/len(loader):.03f} VQ Loss: {vq_loss_/len(loader):.03f} Total: {loss_val/len(loader):.03f}")
    
    if epoch % 50 == 0:
        saveModel(model,f"{model_name}_{epoch}","./model/")

    loss_log.loc[len(loss_log.index)] = [loss_val/len(loader),spectral_loss_/len(loader),vq_loss_/len(loader)]
    loss_log.to_csv(f"./log/loss_{model_name}")
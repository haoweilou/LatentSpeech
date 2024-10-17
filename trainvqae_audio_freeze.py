import torch
import torchaudio
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,ConcatDataset
from params import params
from dataset import BakerAudio,LJSpeechAudio
from ae import VQAE_Audio
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

model = VQAE_Audio(params,embed_dim=128,num_embeddings=1024*4).to(device)

loss_log = pd.DataFrame({"total_loss":[],"vq_loss":[],"audio_loss":[]})
dataset1 = BakerAudio(0,10000)
# dataset2 = LJSpeechAudio(0,1000)
# dataset = ConcatDataset([dataset1, dataset2])
dataset = ConcatDataset([dataset1])

batch_size = 16
# loader = DataLoader(dataset,batch_size=params.batch_size,collate_fn=dataset.collate,drop_last=True,shuffle=True)
loader = DataLoader(dataset,batch_size=batch_size,collate_fn=dataset1.collate,drop_last=True,shuffle=True)
epochs = 201
model_name = "vqae_audio_2T"
# model = loadModel(model,f"vqae_audio","./model/")
# for param in model.encoder.parameters():
#     param.requires_grad = False
# # for param in model.decoder.parameters():
# #     param.requires_grad = True

# optimizer = optim.Adam(model.parameters(),lr=params.learning_rate)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params.learning_rate)
for epoch in range(epochs):
    loss_val = 0
    vq_loss_ = 0
    audio_loss_ = 0
    for audio in tqdm(loader):
        optimizer.zero_grad()
        audio = audio.to(device)
        time_steps = audio.shape[-1]
        pad_amount = (16 - (time_steps % 16)) % 16
        if pad_amount > 0:audio = F.pad(audio, (0, pad_amount))
        y,audio_loss,vq_loss= model(audio)
        loss = vq_loss + audio_loss
        loss_val += loss.item()
        vq_loss_ += vq_loss.item()
        audio_loss_ += audio_loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    print(f"Epoch: {epoch} Audio Loss: {audio_loss_/len(loader):.03f} VQ Loss: {vq_loss_/len(loader):.03f} Total: {loss_val/len(loader):.03f}")
    
    if epoch % 20 == 0:
        saveModel(model,f"{model_name}_{epoch}","./model/")

    loss_log.loc[len(loss_log.index)] = [loss_val/len(loader),vq_loss_/len(loader),audio_loss_/len(loader)]
    loss_log.to_csv(f"./log/loss_{model_name}")
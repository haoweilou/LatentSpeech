import torch
import torchaudio
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,ConcatDataset
from params import params
from dataset import BakerAudio,LJSpeechAudio
# from model import VQAESeq
from ae import VQAE_Audio2,Upsampler
from tqdm import tqdm
from function import saveModel,loadModel
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import pandas as pd
torch.autograd.set_detect_anomaly(True)

model = VQAE_Audio2(params).to(device)
model = loadModel(model,"vqae_audio2_200","./model")
upsample = Upsampler(embed_dim=64,num_bands=64).to(device)

optimizer = optim.Adam(upsample.parameters(),lr=0.001)
loss_log = pd.DataFrame({"total_loss":[],"vq_loss":[],"audio_loss":[]})
dataset1 = BakerAudio(0,1000)
dataset2 = LJSpeechAudio(0,1000)
dataset = ConcatDataset([dataset1, dataset2])
# dataset = ConcatDataset([dataset1])

batch_size = 15
# loader = DataLoader(dataset,batch_size=params.batch_size,collate_fn=dataset.collate,drop_last=True,shuffle=True)
loader = DataLoader(dataset,batch_size=batch_size,collate_fn=dataset1.collate,drop_last=True,shuffle=True)
epochs = 501
model_name = "upsampler"

for epoch in range(epochs):
    loss_val = 0
    level1_loss_ = 0
    level2_loss_ = 0
    for audio in tqdm(loader):
        optimizer.zero_grad()
        audio = audio.to(device)
        time_steps = audio.shape[-1]
        pad_amount = (16 - (time_steps % 16)) % 16
        if pad_amount > 0:audio = F.pad(audio, (0, pad_amount))
        with torch.no_grad():
            pqmf_audio = model.pqmf(audio)#[1,A]=>[Channel,A]
            level1_embed,_ = model.level1.encode(pqmf_audio)
            level2_embed,_ = model.level2.encode(pqmf_audio)
            level3_embed,_ = model.level3.encode(pqmf_audio)

        level1_embed_f,level2_embed_f = upsample(level1_embed,level2_embed,level3_embed)
        
        level1_loss = F.mse_loss(level1_embed,level1_embed_f)
        level2_loss = F.mse_loss(level2_embed,level2_embed_f)
        loss = level1_loss + level2_loss
        loss_val += loss.item()
        level1_loss_ += level1_loss.item()
        level2_loss_ += level2_loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    print(f"Epoch: {epoch} Level1 Loss: {level1_loss_/len(loader):.03f} Level2 Loss: {level2_loss_/len(loader):.03f} Total: {loss_val/len(loader):.03f}")
    
    if epoch % 50 == 0:
        saveModel(upsample,f"{model_name}_{epoch}","./model/")

    loss_log.loc[len(loss_log.index)] = [loss_val/len(loader),level1_loss_/len(loader),level2_loss_/len(loader)]
    loss_log.to_csv(f"./log/loss_{model_name}")
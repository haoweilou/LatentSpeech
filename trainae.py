import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from params import params
from dataset import Baker,BakerOlder
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
# model = VQVAE(num_hiddens,num_residual_layers,num_residual_hiddens,num_embeddings,embedding_dim,commitment_cost).to(device)
# model = RVAE(params).to(device)
model = AE(params).to(device)
print(model)

optimizer = optim.Adam(model.parameters(),lr=params.learning_rate)
loss_log = pd.DataFrame({"loss":[]})
dataset = BakerOlder(1,9000,root="/home/haoweilou/scratch/baker/Wave/")
loader = DataLoader(dataset,batch_size=params.batch_size,collate_fn=dataset.padding,drop_last=True,shuffle=True)
epochs = 501
model_name = "ae9k_dim16"
# model = loadModel(model,"rvaev7_100",root="/home/haoweilou/scratch/model/ae")
for epoch in range(0,epochs):
    loss_val = 0
    loss1 = 0
    loss2 = 0
    for audio in tqdm(loader):
        optimizer.zero_grad()
        audio = audio.to(device)
        audio_reconstruct,spectral_loss,l2_loss = model(audio)
        loss = spectral_loss
        loss_val += loss.item()
        loss1 += l2_loss.item()
        loss2 += spectral_loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
    print(f"Epoch: {epoch} L2 Loss: {loss1/len(loader):.03f} Spec Loss: {loss2/len(loader):.03f} Total: {loss_val/len(loader):.03f}")
    

    if epoch % 5 == 0:
        saveModel(model,f"{model_name}_{epoch}","/home/haoweilou/scratch/model/ae")
        # saveModel(model,f"rvae_{epoch}","./model")
    loss_log.loc[len(loss_log.index)] = [loss_val/len(loader)]
    loss_log.to_csv(f"./log/loss_{model_name}")


        


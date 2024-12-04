import torch
import torchaudio
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,ConcatDataset
from params import params
from dataset import BakerAudio,LJSpeechAudio
# from model import VQAESeq
from flow import AE,FlowBlock,Block,RVQLayer
from ae import RVQ
from tqdm import tqdm
from function import saveModel,loadModel
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import pandas as pd
torch.autograd.set_detect_anomaly(True)

ae = AE(params).to(device)
ae = loadModel(ae,"ae20k16_1000","./model")

# feature_dim = 16
# hid_dim = 256
# num_flow_layers  = 12
# # flow_module = FlowBlock(feature_dim, num_flow_layers).to(device)
# encoder = Block(feature_dim,hid_dim, num_flow_layers).to(device)
# decoder = Block(feature_dim,hid_dim, num_flow_layers).to(device)

# vq_layer = RVQ(12,1024,feature_dim).to(device)
rvq = RVQLayer().to(device)
# rvq = loadModel(rvq,"rvq_1000","./model/")
optimizer = optim.Adam(rvq.parameters(),lr=0.0001)
# optimizer = optim.Adam(vq_layer.parameters(),lr=0.0001)
# model_name = "rvq20k"
model_name = "rvq_rvq"

loss_log = pd.DataFrame({"total_loss":[],"vq_loss":[],"feature_loss":[]})
dataset1 = BakerAudio(0,1000)
dataset2 = LJSpeechAudio(0,1000)
# dataset1 = BakerAudio(0,10000,"/scratch/ey69/hl6114/baker/")
# dataset2 = LJSpeechAudio(0,10000,"/scratch/ey69/hl6114/baker/")
dataset = ConcatDataset([dataset1,dataset2])

batch_size = 32
loader = DataLoader(dataset,batch_size=batch_size,collate_fn=dataset1.collate,drop_last=True,shuffle=True)
epochs = 1001

for epoch in range(epochs):
    loss_val = 0
    vq_loss_ = 0
    feature_loss_ = 0
    for audio in tqdm(loader):
        optimizer.zero_grad()
        audio = audio.to(device)
        time_steps = audio.shape[-1]
        pad_amount = (16 - (time_steps % 16)) % 16
        if pad_amount > 0:audio = F.pad(audio, (0, pad_amount))
        with torch.no_grad():
            z,_ = ae.encode(audio)
            
        zq,vq_loss = rvq(z)
        feature_loss = F.mse_loss(z, zq)

        loss = feature_loss + vq_loss

        loss_val += loss.item()
        vq_loss_ += vq_loss.item()
        feature_loss_ += feature_loss.item()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    print(f"Epoch: {epoch} VQ Loss: {vq_loss_/len(loader):.03f} Feature Loss: {feature_loss_/len(loader):.03f} Total: {loss_val/len(loader):.03f}")
    
    if epoch % 50 == 0:
        saveModel(rvq,f"{model_name}_{epoch}","./model/")

    loss_log.loc[len(loss_log.index)] = [loss_val/len(loader),vq_loss_/len(loader),feature_loss_/len(loader)]
    loss_log.to_csv(f"./log/loss_{model_name}")
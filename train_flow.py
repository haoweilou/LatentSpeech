import torch
import torchaudio
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,ConcatDataset
from params import params
from dataset import BakerAudio,LJSpeechAudio
# from model import VQAESeq
from flow import AE,FlowBlock,Block
from ae import Quantize
from tqdm import tqdm
from function import saveModel,loadModel
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import pandas as pd
torch.autograd.set_detect_anomaly(True)

ae = AE(params).to(device)
ae = loadModel(ae,"ae9k16","./model")

feature_dim = 16
hid_dim = 256
num_flow_layers  = 12
# flow_module = FlowBlock(feature_dim, num_flow_layers).to(device)
encoder = Block(feature_dim,hid_dim, num_flow_layers).to(device)
decoder = Block(feature_dim,hid_dim, num_flow_layers).to(device)

vq_layer = Quantize(feature_dim,1024).to(device)
print(vq_layer)

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(vq_layer.parameters()),lr=0.0001)
# optimizer = optim.Adam(vq_layer.parameters(),lr=0.0001)

loss_log = pd.DataFrame({"total_loss":[],"vq_loss":[],"det_loss":[],"feature_loss":[]})
dataset1 = BakerAudio(0,1000)
dataset = ConcatDataset([dataset1])

batch_size = 32
loader = DataLoader(dataset,batch_size=batch_size,collate_fn=dataset1.collate,drop_last=True,shuffle=True)
epochs = 1001
model_name = "flow"

for epoch in range(epochs):
    loss_val = 0
    vq_loss_ = 0
    det_loss_ = 0
    feature_loss_ = 0
    for audio in tqdm(loader):
        optimizer.zero_grad()
        audio = audio.to(device)
        time_steps = audio.shape[-1]
        pad_amount = (16 - (time_steps % 16)) % 16
        if pad_amount > 0:audio = F.pad(audio, (0, pad_amount))
        z,_ = ae.encode(audio)

        zq = encoder(z)
        zq = z.permute(0,2,1) 
        zq, vq_loss, _ = vq_layer(zq)
        zq = zq.permute(0,2,1) 
        zq = decoder(zq)

        # zq, log_det_jacobian = flow_module(z)
        # z_recons =  flow_module.inverse(zq)

        feature_loss = F.mse_loss(z, zq)

        loss = feature_loss + vq_loss - 0

        loss_val += loss.item()
        vq_loss_ += vq_loss.item()
        det_loss_ += 0
        feature_loss_ += feature_loss.item()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    print(f"Epoch: {epoch} Det Loss: {det_loss_/len(loader):.03f} VQ Loss: {vq_loss_/len(loader):.03f} Feature Loss: {feature_loss_/len(loader):.03f} Total: {loss_val/len(loader):.03f}")
    
    if epoch % 100 == 0:
        saveModel(encoder,f"{model_name}_encoder_{epoch}","./model/")
        saveModel(decoder,f"{model_name}_decoder_{epoch}","./model/")
        saveModel(vq_layer,f"{model_name}_vq_{epoch}","./model/")

    loss_log.loc[len(loss_log.index)] = [loss_val/len(loader),vq_loss_/len(loader),det_loss_/len(loader),feature_loss_/len(loader)]
    loss_log.to_csv(f"./log/loss_{model_name}")
import torch
import torchaudio
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,ConcatDataset
from params import params
from dataset import BakerAudio,LJSpeechAudio
# from model import VQAESeq
from flow import AE,Glow
from ae import Quantize
from tqdm import tqdm
from function import saveModel,loadModel
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import pandas as pd
torch.autograd.set_detect_anomaly(True)
from torch.distributions import Normal

ae = AE(params).to(device)
ae = loadModel(ae,"ae20k16_1000","./model")

feature_dim = 16
hid_dim = 256
num_flow_layers  = 12
# flow_module = FlowBlock(feature_dim, num_flow_layers).to(device)
glow = Glow(feature_dim=feature_dim,num_layer=num_flow_layers).to(device)

gaussian = Normal(loc=0.0, scale=1.0)

optimizer = optim.Adam(glow.parameters(),lr=0.0001)
# optimizer = optim.Adam(vq_layer.parameters(),lr=0.0001)

loss_log = pd.DataFrame({"total_loss":[],"det_loss":[],"glow_loss":[]})
dataset1 = BakerAudio(0,1000)
dataset = ConcatDataset([dataset1])

batch_size = 32
loader = DataLoader(dataset,batch_size=batch_size,collate_fn=dataset1.collate,drop_last=True,shuffle=True)
epochs = 1001
model_name = "glow"

for epoch in range(epochs):
    loss_val = 0
    det_loss_ = 0
    glow_loss_ = 0
    for audio in tqdm(loader):
        optimizer.zero_grad()
        audio = audio.to(device)
        time_steps = audio.shape[-1]
        pad_amount = (16 - (time_steps % 16)) % 16
        if pad_amount > 0:audio = F.pad(audio, (0, pad_amount))
        z,_ = ae.encode(audio)

        hidden, log_det = glow(z)
        log_pz = gaussian.log_prob(z).sum(dim=[1, 2]) 
        glow_loss = -(log_pz + log_det).mean()


        loss = glow_loss

        loss_val += loss.item()
        det_loss_ += log_det.mean().item()
        glow_loss_ += glow_loss.item()

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    print(f"Epoch: {epoch} Det Loss: {det_loss_/len(loader):.03f} Glow Loss: {glow_loss_/len(loader):.03f} Total: {loss_val/len(loader):.03f}")
    
    if epoch % 100 == 0:
        saveModel(glow,f"{model_name}_{epoch}","./model/")


    loss_log.loc[len(loss_log.index)] = [loss_val/len(loader),det_loss_/len(loader),glow_loss_/len(loader)]
    loss_log.to_csv(f"./log/loss_{model_name}")
import torch
import torchaudio
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,ConcatDataset
from params import params
from dataset import BakerAudio,LJSpeechAudio
# from model import VQAESeq
from jukebox import Jukebox
from tqdm import tqdm
from function import saveModel,loadModel
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import pandas as pd
torch.autograd.set_detect_anomaly(True)

model = Jukebox(params).to(device)
# model = loadModel(model,f"jukebox_upsampler","./model/",strict=False)
model = loadModel(model,f"jukebox_upsampler_1000","./model/",strict=True)

# model = loadModel(model,"jukebox_700",root="./model/",strict=False)

# model2 = Jukebox(params).to(device)
# model2 = loadModel(model2,"jukbox_upsampler_2000",root="./model/",strict=False)

# model.upsampler1 = model2.upsampler1
# model.upsampler2 = model2.upsampler2

optimizer = optim.Adam(list(model.upsampler1.parameters())+list(model.upsampler2.parameters()),lr=0.0003)

loss_log = pd.DataFrame({"total_loss":[], "feature_loss":[]})
dataset1 = BakerAudio(0,10000)
dataset2 = LJSpeechAudio(0,10000)
dataset = ConcatDataset([dataset1, dataset2])
# dataset = ConcatDataset([dataset1])

batch_size = 32
loader = DataLoader(dataset,batch_size=batch_size,collate_fn=dataset1.collate,drop_last=True,shuffle=True)
epochs = 2001
model_name = "jukebox_upsampler"

for epoch in range(1001,epochs):
    loss_val = 0
    feature_loss_ = 0
    for audio in tqdm(loader):
        optimizer.zero_grad()
        audio = audio.to(device)
        time_steps = audio.shape[-1]
        pad_amount = (16 - (time_steps % 16)) % 16
        if pad_amount > 0:audio = F.pad(audio, (0, pad_amount))
        feature_loss = model.train_sampler(audio)

        loss =  feature_loss
        loss_val += loss.item()
        feature_loss_ += feature_loss.item()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    print(f"Epoch: {epoch} Feature Loss: {feature_loss_/len(loader):.03f} Total: {loss_val/len(loader):.03f}")
    
    if epoch % 50 == 0:
        saveModel(model,f"{model_name}_{epoch}","./model/")

    loss_log.loc[len(loss_log.index)] = [loss_val/len(loader),feature_loss_/len(loader)]
    loss_log.to_csv(f"./log/loss_{model_name}")
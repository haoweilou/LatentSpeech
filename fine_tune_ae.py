import torch
import torchaudio
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,ConcatDataset
from params import params
from dataset import BakerAudio,LJSpeechAudio
from ae import VQAE_Audio,WaveNet
from tqdm import tqdm
from function import saveModel,loadModel
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import pandas as pd
from ae import VQAE_Audio
torch.autograd.set_detect_anomaly(True)
num_hiddens=80
num_residual_layers=30
num_residual_hiddens=80
num_embeddings=256
embedding_dim=80
commitment_cost=0.25

model = VQAE_Audio(params,embed_dim=64,num_embeddings=2048).to(device)
model = loadModel(model,"vqae_audio","./model/")
for param in model.parameters():
    param.requires_grad = False
    
loss_log = pd.DataFrame({"total_loss":[],"audio_loss":[]})
dataset1 = BakerAudio(0,500)
dataset2 = LJSpeechAudio(0,500)
dataset = ConcatDataset([dataset1, dataset2])
batch_size = 16
loader = DataLoader(dataset,batch_size=batch_size,collate_fn=dataset1.collate,drop_last=True,shuffle=True)
epochs = 501
model_name = "vqae_audio_finetune"
finetune = WaveNet(num_layers=20).to(device)
optimizer = optim.Adam(finetune.parameters(),lr=params.learning_rate)
spec_transform = torchaudio.transforms.MelSpectrogram(sample_rate=48*1000, n_fft=2048 ,win_length=2048 ,hop_length=960,n_mels=80).to(device)

for epoch in range(epochs):
    loss_val = 0
    audio_loss_ = 0
    for audio in tqdm(loader):
        optimizer.zero_grad()
        audio = audio.to(device)
        time_steps = audio.shape[-1]
        pad_amount = (16 - (time_steps % 16)) % 16
        if pad_amount > 0:audio = F.pad(audio, (0, pad_amount))
        with torch.no_grad():
            y = model.pqmf_output(audio)
        y_finetuned = finetune(y)
        y_finetuned = model.pqmf.inverse(y_finetuned)
        spec_r, spec_f = spec_transform(audio),spec_transform(y_finetuned)
        spec_r, spec_f = model.equal_size(spec_r,spec_f)
        audio_loss =F.mse_loss(spec_r, spec_f)

        loss = audio_loss
        loss_val += loss.item()
        audio_loss_ += audio_loss.item()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch: {epoch} Audio Loss: {audio_loss_/len(loader):.03f} Total: {loss_val/len(loader):.03f}")
    
    if epoch % 50 == 0:
        saveModel(finetune,f"{model_name}_{epoch}","./model/")

    loss_log.loc[len(loss_log.index)] = [loss_val/len(loader),audio_loss_/len(loader)]
    loss_log.to_csv(f"./log/loss_{model_name}")
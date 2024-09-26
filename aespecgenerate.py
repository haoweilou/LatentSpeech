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
from function import saveModel,loadModel,draw_heatmap
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
loss_log = pd.DataFrame({"total_loss":[],"spectral_loss":[],"vq_loss":[]})
dataset = BakerAudio(0,1000)
loader = DataLoader(dataset,batch_size=params.batch_size,collate_fn=dataset.collate,drop_last=True,shuffle=False)
epochs = 501
model_name = "vqspecae"
model = loadModel(model,f"{model_name}_{50}","./model/")

melspec_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=48*1000,      # Adjust the sample rate to your audio's sample rate
    n_fft=400,              # Size of FFT, you can adjust this
    win_length=400,         # Window size
    hop_length=160,         # Hop length between frames
    n_mels=80               # Set number of mel bins to 80
).to(device)
with torch.no_grad():
    for audio in tqdm(loader):
        audio = audio.to(device)
        melspec = melspec_transform(audio)

        time_steps = melspec.shape[-1]
        pad_amount = (4 - (time_steps % 4)) % 4
        if pad_amount > 0:melspec = F.pad(melspec, (0, pad_amount))

        melspec_reconstruct,vq_loss = model(melspec)
        draw_heatmap(melspec[0][0].cpu(),name="raw")
        draw_heatmap(melspec_reconstruct[0][0].cpu(),name="reconstruct")
        break
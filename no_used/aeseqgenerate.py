import torch
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from params import params
from dataset import BakerAudio
from model import VQAESeq,VQAESeqTanh
from tqdm import tqdm
from function import saveModel,loadModel,draw_heatmap,save_audio
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import pandas as pd
torch.autograd.set_detect_anomaly(True)
num_hiddens=80
num_residual_layers=30
num_residual_hiddens=80
num_embeddings=256
embedding_dim=80
commitment_cost=0.25
# model = VQAESeq(params).to(device)
model = VQAESeqTanh(params).to(device)


dataset = BakerAudio(0,1000,path="L:/baker/")
loader = DataLoader(dataset,batch_size=params.batch_size,collate_fn=dataset.collate,drop_last=True,shuffle=False)
epochs = 501
# model_name = "vqaeseq"
# model = loadModel(model,f"{model_name}_{500}","./model/")
model_name = "vqaeseqtanh"
model = loadModel(model,f"{model_name}_{200}","./model/")


with torch.no_grad():
    for audio in tqdm(loader):
        audio = audio.to(device)
        time_steps = audio.shape[-1]
        pad_amount = (16 - (time_steps % 16)) % 16
        if pad_amount > 0:audio = F.pad(audio, (0, pad_amount))
        y,audio_loss,vq_loss,spectral_loss= model(audio)

        melspec_real = model.melspec_transform(audio)
        melspec_fake, vq_loss = model.spec_ae(melspec_real)
        draw_heatmap(melspec_real[0][0].cpu(),name="real")
        draw_heatmap(melspec_fake[0][0].cpu(),name="fake")
        print(melspec_real.shape,y.shape)
        save_audio(y[0].cpu(),sample_rate=48*1000,name="fake")
        break
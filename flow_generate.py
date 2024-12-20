from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from params import params
from function import loadModel,save_audio,draw_wave,draw_heatmap,draw_dot,load_audio
from dataset import BakerAudio,pad16,LJSpeechAudio
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import DataLoader
from ae import PQMF
from params import params
import torch
import torchaudio
import torch
import torch.nn.functional as F
import torch.optim as optim
from params import params
# from model import VQAESeq
from flow import AE,RVQLayer
from ae import Quantize
from tqdm import tqdm
from function import saveModel,loadModel
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import pandas as pd
torch.autograd.set_detect_anomaly(True)

ae = AE(params).to(device)
ae = loadModel(ae,"ae20k16_1000","./model")

feature_dim = 16
hid_dim = 256
num_flow_layers  = 12
num = 600
# flow_module = FlowBlock(feature_dim, num_flow_layers).to(device)
# encoder = Block(feature_dim,hid_dim, num_flow_layers).to(device)
# decoder = Block(feature_dim,hid_dim, num_flow_layers).to(device)
rvq = RVQLayer().to(device)
rvq = loadModel(rvq,"rvq_rvq_1000","./model/")
# embed_size = 1024
# vq_layer = Quantize(feature_dim,1024).to(device)

# encoder = loadModel(encoder,f"flow_encoder_{num}","./model/",strict=True)
# decoder = loadModel(decoder,f"flow_decoder_{num}","./model/",strict=True)
# vq_layer = loadModel(vq_layer,f"flow_vq_{num}","./model/",strict=True)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
# finetune = WaveNet(num_layers=20).to(device)
import random

base = random.randint(2000,4000)
dataset = BakerAudio(base+0,base+10,"C:/baker/")
# # dataset = LJSpeechAudio(base+0,base+10,"L:/LJSpeech/")
loader = DataLoader(dataset,batch_size=32,collate_fn=dataset.collate,drop_last=False,shuffle=False)

n_bands = 16
pqmf = PQMF(100,n_bands).to(device)
# audio = audio.unsqueeze(1).to(device)
# print(audio.shape)
with torch.no_grad():
    for audio in tqdm(loader):
        audio = audio.to(device)
        time_steps = audio.shape[-1]
        pad_amount = (16 - (time_steps % 16)) % 16
        if pad_amount > 0:audio = F.pad(audio, (0, pad_amount))
        draw_wave(audio[0][0].to("cpu"),f"real")
        save_audio(audio[0].to("cpu"),48000,f"real")

        z,_ = ae.encode(audio)
        zq,vq_loss = rvq(z)
        pqmf_audio1 = ae.decode(zq)
        a = pqmf.inverse(pqmf_audio1)
        draw_wave(a[0][0].to("cpu"),f"rvq")
        save_audio(a[0].to("cpu"),48000,f"rvq")

        pqmf_audio1 = ae.decode(z)
        a = pqmf.inverse(pqmf_audio1)
        draw_wave(a[0][0].to("cpu"),f"ae")
        save_audio(a[0].to("cpu"),48000,f"ae")

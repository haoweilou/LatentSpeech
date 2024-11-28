from jukebox import Jukebox,UpSampler,UpSampler3
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
from function import plot_pqmf_bands,spectral_denoise
from ae import PQMF
from params import params
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
hid_dim = 64
num_flow_layers  = 24
num = 100
# flow_module = FlowBlock(feature_dim, num_flow_layers).to(device)
encoder = Block(feature_dim,hid_dim, num_flow_layers).to(device)
decoder = Block(feature_dim,hid_dim, num_flow_layers).to(device)

vq_layer = Quantize(feature_dim,2048).to(device)

encoder = loadModel(encoder,f"flow_encoder_{num}","./model/",strict=True)
decoder = loadModel(decoder,f"flow_decoder_{num}","./model/",strict=True)
vq_layer = loadModel(vq_layer,f"flow_vq_{num}","./model/",strict=True)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
# finetune = WaveNet(num_layers=20).to(device)
import random

base = random.randint(1,10000)
# dataset = BakerAudio(base+0,base+10,"D:/baker/")
# # dataset = LJSpeechAudio(base+0,base+10,"L:/LJSpeech/")
# loader = DataLoader(dataset,batch_size=32,collate_fn=dataset.collate,drop_last=False,shuffle=False)

n_bands = 16
pqmf = PQMF(100,n_bands).to(device)
audio,sr = load_audio("./sample/real.wav")
audio = audio.unsqueeze(1).to(device)
print(audio.shape)
with torch.no_grad():
    time_steps = audio.shape[-1]
    pad_amount = (16 - (time_steps % 16)) % 16
    if pad_amount > 0:audio = F.pad(audio, (0, pad_amount))
    draw_wave(audio[0][0].to("cpu"),f"real")

    z,_ = ae.encode(audio)
    zq = encoder(z)
    zq = z.permute(0,2,1) 
    zq, vq_loss, _ = vq_layer(zq)
    zq = zq.permute(0,2,1) 
    zq = decoder(zq)

    # a,_,_,_ = model(audio)
    pqmf_audio1 = ae.decode(zq)

    a = pqmf.inverse(pqmf_audio1)
    # plot_pqmf_bands(a,48000,pqmf,n_bands)
    draw_wave(a[0][0].to("cpu"),f"flow")
    save_audio(a[0].to("cpu"),48000,f"flow")

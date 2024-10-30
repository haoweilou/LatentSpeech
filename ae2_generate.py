from ae import VQAE_Audio,VQAE,WaveNet,AE,VQAE_Audio2,Upsampler
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from params import params
from function import loadModel,save_audio,draw_wave,draw_heatmap,draw_dot
from dataset import BakerAudio,pad16,LJSpeechAudio
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import DataLoader

from params import params
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = VQAE_Audio2(params).to(device)
model_name = "vqae_audio2"
model = loadModel(model,f"{model_name}_200","./model/")

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
# model = loadModel(model,f"{model_name}_{num}","./model/")

upsampler = Upsampler(64,64).to(device)
upsampler = loadModel(upsampler,f"upsampler_50","./model/")

# finetune = WaveNet(num_layers=20).to(device)
dataset = BakerAudio(0,10,"L:/baker/")
# dataset = LJSpeechAudio(0,10,"L:/LJSpeech/")
loader = DataLoader(dataset,batch_size=32,collate_fn=dataset.collate,drop_last=False,shuffle=False)

with torch.no_grad():
    for audio in tqdm(loader):
        audio = audio.to(device)
        pqmf_audio = model.pqmf(audio)
        time_steps = audio.shape[-1]
        pad_amount = (16 - (time_steps % 16)) % 16
        if pad_amount > 0:audio = F.pad(audio, (0, pad_amount))
        
        draw_wave(audio[0][0].to("cpu"),"real")
        save_audio(audio[0].to("cpu"),48000,"real")
        
        pqmf_audio = model.pqmf(audio)#[1,A]=>[Channel,A]

        level3_embed,_ = model.level3.encode(pqmf_audio)

        level2_embed = upsampler.upsample3(level3_embed)
        level2_embed = level2_embed.permute(0,2,1)#C,T=>T,C
        level2_embed,_,_ = model.level2.vq_layer(level2_embed)
        level2_embed = level2_embed.permute(0,2,1)#T,C=>C,T
        # level2_embed,_ = model.level2.encode(pqmf_audio)
        
        level2_embed,_ = model.level2.encode(pqmf_audio)
        level1_embed = upsampler.upsample2(level2_embed)
        # level1_embed = level1_embed.permute(0,2,1)#C,T=>T,C
        # level1_embed,_,_ = model.level1.vq_layer(level1_embed)
        # level1_embed = level1_embed.permute(0,2,1)#T,C=>C,T

        pqmf_audio_f = model.level1.decode(level1_embed)
        a = model.pqmf.inverse(pqmf_audio_f)
        draw_wave(a[0][0].to("cpu"),f"fake_audio_upsamp")
        save_audio(a[0].to("cpu"),48000,f"fake_audio_upsamp")

        # a = model.pqmf.inverse(pqmf_audio_f)
        break
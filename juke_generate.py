from jukebox import Jukebox
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
from function import plot_pqmf_bands
from ae import PQMF
from params import params
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# num = 0
model = Jukebox(params).to(device)
num = 700
# model_name = "jukebox"
num = 2000
model_name = "jukebox_upsampler1"
num = 1800
model_name = "jukebox_upsampler2"

model = loadModel(model,f"{model_name}_{num}","./model/",strict=False)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
# finetune = WaveNet(num_layers=20).to(device)
base = 0
dataset = BakerAudio(base+0,base+10,"L:/baker/")
# dataset = LJSpeechAudio(base+0,base+10,"L:/LJSpeech/")
loader = DataLoader(dataset,batch_size=32,collate_fn=dataset.collate,drop_last=False,shuffle=False)
# wave_gen = nn.Conv1d(16,16,7,padding=3).to(device)
# loud_gen = nn.Conv1d(16,16,3,1,padding=1).to(device)
# wave_gen = loadModel(wave_gen,"wave_100","./model/")
# loud_gen = loadModel(loud_gen,"loud_100","./model/")
# model.loud_gen = loud_gen 
# model.wave_gen = wave_gen
# model.vqae1 = loadModel(model.vqae1,"vqae1_100","./model/")
n_bands = 4
pqmf = PQMF(100,n_bands).to(device)
with torch.no_grad():
    for audio in tqdm(loader):
        audio = audio.to(device)
        pqmf_audio = model.pqmf(audio)
        time_steps = audio.shape[-1]
        pad_amount = (16 - (time_steps % 16)) % 16
        if pad_amount > 0:audio = F.pad(audio, (0, pad_amount))
        # plot_pqmf_bands(audio,48000,pqmf,n_bands)
        
        draw_wave(audio[0][0].to("cpu"),"real")
        save_audio(audio[0].to("cpu"),48000,"real")

        # a,_,_,_ = model(audio)
        pqmf_audio1,vq_loss1 = model.vqae1(pqmf_audio)

        pqmf_audio1 = model.decode_audio(pqmf_audio1)
        a = model.pqmf.inverse(pqmf_audio1)
        # plot_pqmf_bands(a,48000,pqmf,n_bands)
        draw_wave(a[0][0].to("cpu"),f"fake_audio_layer1")
        save_audio(a[0].to("cpu"),48000,f"fake_audio_layer1")

        pqmf_audio1,vq_loss1 = model.vqae2(pqmf_audio)
        pqmf_audio1 = model.decode_audio(pqmf_audio1)
        a = model.pqmf.inverse(pqmf_audio1)
        # plot_pqmf_bands(a,48000,pqmf,n_bands)
        draw_wave(a[0][0].to("cpu"),f"fake_audio_layer2")
        save_audio(a[0].to("cpu"),48000,f"fake_audio_layer2")

        pqmf_audio1,vq_loss1 = model.vqae3(pqmf_audio)
        pqmf_audio1 = model.decode_audio(pqmf_audio1)
        a = model.pqmf.inverse(pqmf_audio1)
        # plot_pqmf_bands(a,48000,pqmf,n_bands)
        draw_wave(a[0][0].to("cpu"),f"fake_audio_layer3")
        save_audio(a[0].to("cpu"),48000,f"fake_audio_layer3")

        a = model.upsample(pqmf_audio)
        # plot_pqmf_bands(a,48000,pqmf,n_bands)
        draw_wave(a[0][0].to("cpu"),f"fake_audio_upsample")
        save_audio(a[0].to("cpu"),48000,f"fake_audio_upsample")

        # z = model.encoder1(pqmf_audio)
        # z = z.permute(0,2,1).reshape(-1,64)
        # z_q,vq_loss,_ = model.vq_layer1(z)
        # codebook = model.vq_layer1.embed.permute(1,0)

        # z1 = model.encoder1(pqmf_audio)
        # z1 = z1.permute(0,2,1)
        # z_q1,vq_loss1,_ = model.vq_layer1(z1)
        # z_q1 = z_q1.permute(0,2,1)
        
        # z2 = z_q1 + z1.permute(0,2,1)
        # z2 = model.encoder2(z2)
        # z2 = z2.permute(0,2,1).reshape(-1,64)
        # z_q2,vq_loss2,_ = model.vq_layer2(z2)
        # z_q = z_q2
        # codebook = model.vq_layer2.embed.permute(1,0)
        # combined = torch.concat((z_q,codebook),dim=0)
        # print(z_q.shape,codebook.shape,combined.shape)
        # combined = pca.fit_transform(combined.cpu())
        # draw_dot([combined[:-2048],combined[-2048:]],["z_q","codebook"],name=f"z_q and codebook")
       
        break
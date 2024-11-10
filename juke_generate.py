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

from params import params
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num = 50
model = Jukebox(params).to(device)
# model_name = "jukebox"
# model_name = "jukbox_upsampler"
model_name = "jukebox_upsampler"

# model_name = "juke_vqae_upsampler"
model = loadModel(model,f"{model_name}_{num}","./model/")
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
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

        # a,_,_,_ = model(audio)
        pqmf_audio1,vq_loss1 = model.vqae1(pqmf_audio)
        pqmf_audio1 = model.decode_audio(pqmf_audio1)
        a = model.pqmf.inverse(pqmf_audio1)
        draw_wave(a[0][0].to("cpu"),f"fake_audio_layer1")
        save_audio(a[0].to("cpu"),48000,f"fake_audio_layer1")

        pqmf_audio1,vq_loss1 = model.vqae2(pqmf_audio)
        pqmf_audio1 = model.decode_audio(pqmf_audio1)
        a = model.pqmf.inverse(pqmf_audio1)
        draw_wave(a[0][0].to("cpu"),f"fake_audio_layer2")
        save_audio(a[0].to("cpu"),48000,f"fake_audio_layer2")

        pqmf_audio1,vq_loss1 = model.vqae3(pqmf_audio)
        pqmf_audio1 = model.decode_audio(pqmf_audio1)
        a = model.pqmf.inverse(pqmf_audio1)
        draw_wave(a[0][0].to("cpu"),f"fake_audio_layer3")
        save_audio(a[0].to("cpu"),48000,f"fake_audio_layer3")

        a = model.upsample(pqmf_audio)
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
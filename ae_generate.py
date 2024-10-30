from ae import VQAE_Audio,VQAE,WaveNet,AE,VQAE_Audio2
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
# model = VQAE_Audio2(params).to(device)
# model_name = "vqae_audio2"

model = AE(params).to(device)
model_name = "qae"
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
# model = loadModel(model,f"{model_name}_{num}","./model/")
model = loadModel(model,f"{model_name}_200","./model/")
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

        z_q,b,t= model.encode_inference(audio)
        a = model.decode_inference(z_q,b,t)

        draw_wave(a[0][0].to("cpu"),f"fake_audio")
        save_audio(a[0].to("cpu"),48000,f"fake_audio")
    
        print(a.shape)
        codebook = model.vq_layer.embed.permute(1,0)
        print(z_q.shape)
        combined = torch.concat((z_q,codebook),dim=0)
        print(z_q.shape,codebook.shape,combined.shape)
        combined = pca.fit_transform(combined.cpu())
        draw_dot([combined[:-2048],combined[-2048:]],["z_q","codebook"],name=f"z_q and codebook")
        # models = [model.level1,model.level2,model.level3]
        # for idx,m in enumerate(models):
        #     z_q,_= m.encode(pqmf_audio)
        #     pqmf_audio_f = m.decode(z_q)
        #     a = model.pqmf.inverse(pqmf_audio_f)

        #     draw_wave(a[0][0].to("cpu"),f"fake_audio_{idx}")
        #     save_audio(a[0].to("cpu"),48000,f"fake_audio_{idx}")
        
        #     print(a.shape)
        #     codebook = m.vq_layer.embed.permute(1,0)
        #     print(z_q.shape)
        #     z_q = z_q.permute(0, 2, 1).reshape(-1,64) 
        #     combined = torch.concat((z_q,codebook),dim=0)
        #     print(z_q.shape,codebook.shape,combined.shape)
        #     combined = pca.fit_transform(combined.cpu())
        #     draw_dot([combined[:-2048],combined[-2048:]],["z_q","codebook"],name=f"z_q and codebook_{idx}")
            
        break
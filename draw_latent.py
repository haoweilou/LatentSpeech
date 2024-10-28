from ae import VQAE_Audio,VQAE,WaveNet,AE
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

is_audio = True
if is_audio:
    embed_dim = 64
    num_embeddings=2048
    num = 120
    model_name = "vqae_audio"
    model = VQAE_Audio(params,embed_dim=embed_dim,num_embeddings=num_embeddings).to(device)
    

    # model = VQAE_Audio(params,embed_dim,num_embeddings).to(device)
else:
    num = 200
    embed_dim = 20
    model_name = f"qae_{num}"
    # model = VQAE(params,embed_dim=64).to(device)
    model = AE(params).to(device)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
# model = loadModel(model,f"{model_name}_{num}","./model/")
model = loadModel(model,f"{model_name}","./model/")
# finetune = WaveNet(num_layers=20).to(device)
dataset = BakerAudio(0,10,"L:/baker/")
# dataset = LJSpeechAudio(0,10,"L:/LJSpeech/")
loader = DataLoader(dataset,batch_size=32,collate_fn=dataset.collate,drop_last=False,shuffle=False)
for epoch in range(180,200,50):
    # finetune = loadModel(finetune,f"vqae_audio_finetune_{epoch}","./model/")
    with torch.no_grad():
        for audio in tqdm(loader):
            audio = audio.to(device)
            time_steps = audio.shape[-1]
            pad_amount = (16 - (time_steps % 16)) % 16
            if pad_amount > 0:audio = F.pad(audio, (0, pad_amount))
            
            draw_wave(audio[0][0].to("cpu"),"real")
            save_audio(audio[0].to("cpu"),48000,"real")
            if is_audio:
                z_q = model.encode_inference(audio)
                z_q = z_q.permute(0, 2, 1).view(-1,embed_dim) 
                pqmf_audio = model.pqmf_output(audio)
                a = model.pqmf.inverse(pqmf_audio)
                draw_wave(a[0][0].to("cpu"),f"no_finetune")
                save_audio(a[0].to("cpu"),48000,f"no_finetune")

                # pqmf_audio = finetune(pqmf_audio)
                a = model.pqmf.inverse(pqmf_audio)
                draw_wave(a[0][0].to("cpu"),f"fake_audio_{epoch}")
                save_audio(a[0].to("cpu"),48000,f"fake_audio_{epoch}")
            else:
                a,_,_,_ = model(audio)
                draw_wave(a[0][0].to("cpu"),"fake_spec")
                save_audio(a[0].to("cpu"),48000,f"fake_spec")
                z_q,b,t = model.encode_inference(a)
                print(z_q.shape)
            print(a.shape)
            codebook = model.vq_layer.embed.permute(1,0)
            combined = torch.concat((z_q,codebook),dim=0)
            print(z_q.shape,codebook.shape,combined.shape)
            combined = pca.fit_transform(combined.cpu())
            if is_audio:
                draw_dot([combined[:-num_embeddings],combined[-num_embeddings:]],["z_q","codebook"],name="z_q and codebook-audio")
            else:
                draw_dot([combined[:-512],combined[-512:]],["z_q","codebook"],name="z_q and codebook-spec")
            break
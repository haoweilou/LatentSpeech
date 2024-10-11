from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from model import VQAESeq,AE
from ae import VQAE,VQAE_Audio
from params import params
from function import loadModel,save_audio,draw_wave,draw_heatmap,draw_dot
from dataset import BakerAudio,pad16,LJSpeechAudio
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import DataLoader

# num = 10
# model_name = "vqae"
# vqae = VQAE(params).to(device)

num = 500
model_name = "vqae_audio"
embed_dim = 32
num_embeddings=1024
vqae = VQAE_Audio(params,embed_dim,num_embeddings=num_embeddings).to(device)
vqae = loadModel(vqae,f"{model_name}_{num}","./model/")

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
# model = VQAESeq(params,embed_dim=16).to(device)
dataset = BakerAudio(0,10,"D:/baker/")
# dataset = LJSpeechAudio(0,13100,"L:/LJSpeech/")
loader = DataLoader(dataset,batch_size=32,collate_fn=dataset.collate,drop_last=False,shuffle=False)
i = 0
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
with torch.no_grad():
    for audio in tqdm(loader):
        audio = audio.to(device)
        time_steps = audio.shape[-1]
        pad_amount = (16 - (time_steps % 16)) % 16
        if pad_amount > 0:audio = F.pad(audio, (0, pad_amount))
        audio_num = 0
        a,audio_loss,vq_loss = vqae(audio)

        z_q = vqae.encode_inference(audio)
        z_q = z_q.permute(0, 2, 1)#for audio
        z_q = z_q.permute(0, 3, 1, 2) #for melspec
        
        batch_size = z_q.shape[0]
        for j in range(batch_size):
            data = z_q[j].cpu()
            torch.save(data,f"L:/LJSpeech/Latent/{i}")
            i += 1

#test the generated audio can reconstruct audio
# with torch.no_grad():
#     for i in range(10):
#         z_q = torch.load(f"L:/Baker/Latent/{i}").to("cuda")
#         z_q = torch.unsqueeze(z_q,0)
#         # z_q = model.encode(audio)
#         print(z_q.shape)
        
#         # draw_heatmap(codebook[:64],name="codebook")
#         break
#         latent_temp = model.mapper(melspec.squeeze(1))

#         audio = model.decode(latent_temp)
#         audio = model.pqmf.inverse(audio)[0]
#         print(audio.shape)
#         save_audio(audio.to("cpu"),48000,f"{i}")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ae import VQAE,VQAE_Audio
from params import params
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
audio = torch.randn(2,1,48000).to(device)
# vqae = VQAE(params).to(device)
vqae = VQAE_Audio(params).to(device)

from function import loadModel,save_audio,draw_wave,draw_heatmap,draw_dot
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
audio = torch.randn(2,1,48000).to(device)

# out, latent_loss  = vqae(mel)
# print(out.shape)
audio_fake,audio_loss,vq_loss = vqae(audio)
print(audio_fake.shape)
z_q,_,_ = vqae.encode(audio)
# z_q = z_q.permute(0, 2, 3, 1)
z_q = z_q.permute(0, 2, 1)
z_q_flatten = z_q.reshape(-1, 16).detach().cpu()
codebook = vqae.vq_layer.embed.detach().cpu().T


# # z_q = z_q.reshape(2,16,20*50)
print(z_q.shape,z_q_flatten.shape,codebook.shape)

# # b,embed,h,w = z_q.shape
# # z_q =  torch.reshape(z_q,(b,embed,-1))
# codebook = vqae.vq_layer.embedding.weight.detach().cpu()
# print(z_q.shape,codebook.shape)
# z_q_flatten = z_q.detach().cpu()
combined = torch.concat((z_q_flatten,codebook),dim=0)
combined = pca.fit_transform(combined)
draw_dot([combined[:-512],combined[-512:]],["z_q","codebook"],name="z_q and codebook")

# audio_fake = vqae.decode_inference(z_q,b,embed_dim,H,T)

# print(z_q.shape,audio_fake.shape)

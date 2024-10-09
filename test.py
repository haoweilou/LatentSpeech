import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ae import VQAE
from params import params
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
audio = torch.randn(1,1,48000).to(device)
vqae = VQAE(params).to(device)
from function import loadModel,save_audio,draw_wave,draw_heatmap,draw_dot
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

audio_fake,audio_loss,vq_loss,spectral_loss = vqae(audio)
z_q = vqae.encode_inference(audio)
# b,embed,h,w = z_q.shape
# z_q =  torch.reshape(z_q,(b,embed,-1))
z_q = z_q[0].T
codebook = vqae.vq_layer.embedding.weight.detach().cpu()
print(z_q.shape,codebook.shape)
z_q_flatten = z_q.detach().cpu()
combined = torch.concat((z_q_flatten,codebook),dim=0)
combined = pca.fit_transform(combined)
draw_dot([combined[:-512],combined[-512:]],["z_q","codebook"],name="z_q and codebook")

# audio_fake = vqae.decode_inference(z_q,b,embed_dim,H,T)

# print(z_q.shape,audio_fake.shape)

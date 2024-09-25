import torch
import torch.nn as nn
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from vqae import VQVAE,Encoder
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
spectrogram = torch.randn((12,1,80,400)).to(device)
# encoder = Encoder(in_channel=1,channel=128).to(device)
# z = encoder(spectrogram)
# reducer = nn.Conv2d(128, 64, 1).to(device)
# print(z.shape)#Batch size, channel, 20, 100
# z = reducer(z)
# print(z.shape)#Batch size, embed_dim, 20, 100
# b,embed,h,w = z.shape
# z = z.view(z.size(0),z.size(1),-1)
# print(z.shape)#Batch size, embed_dim, 20*100

# from model import VQEmbedding
# vq_layer = VQEmbedding(512,64).to(device)

# z_q, vq_loss, _ = vq_layer(z)
# print(z_q.shape,vq_loss)
# z_q = torch.reshape(z,(b,embed,h,w))
# print(z_q.shape,vq_loss)

# from vqae import Decoder
# decoder = Decoder(64,1,128,2,32,4).to(device)
# y = decoder(z_q)
from model import VQSpecAE
model = VQSpecAE().to(device)
y,vq_loss = model(spectrogram)
print(spectrogram.shape,y.shape)
print(vq_loss)


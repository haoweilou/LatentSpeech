import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ae import VQAE,VQAE_Audio,VQAE_Audio2
from params import params
from sklearn.manifold import TSNE
import torchaudio
# tsne = TSNE(n_components=2, random_state=42)
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# audio = torch.randn(2,1,48000).to(device)
# # vqae = VQAE(params).to(device)
# vqae = VQAE_Audio(params).to(device)

# from function import loadModel,save_audio,draw_wave,draw_heatmap,draw_dot
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# audio = torch.randn(16,1,48000).to(device)
# mb_audio = torch.randn(16,16,3000).to(device)

# from ae import AE,VQAE_Module,Upsampler

# model = AE(params).to(device)
# audio_f,audio_loss,vq_loss,spectral_loss = model(audio)
# model1 = VQAE_Audio_2(params,embed_dim=64,num_embeddings=2048).to(device)
# model = VQAE_Module(channel=16,embed_dim=64).to(device)
# model = VQAE_Audio2(params,embed_dim=64).to(device)

# pqmf_audio = model.pqmf(audio)#[1,A]=>[Channel,A]
# level1_embed,_ = model.level1.encode(pqmf_audio)
# level2_embed,_ = model.level2.encode(pqmf_audio)
# level3_embed,_ = model.level3.encode(pqmf_audio)
# print(level1_embed.shape,level2_embed.shape,level3_embed.shape)
# upsample = Upsampler(embed_dim=64,num_bands=64).to(device)
# level1_embed_f,level2_embed_f = upsample(level1_embed,level2_embed,level3_embed)

# from tts import StyleSpeech
# from tts_config import config
# tts = StyleSpeech(config)
# x = torch.tensor([[1,1,1,1]]).to(device)
# s = torch.tensor([[1,1,1,1]]).to(device)
# src_lens = torch.tensor([[4]]).to(device)
# mel_lens = torch.tensor([[16]]).to(device)
# l = torch.tensor([[4,4,4,4]]).to(device)
# max_mel_len = 20



# latent_f,log_l_pred,mel_masks = tts(x,s,src_lens=src_lens,mel_lens=mel_lens,duration_target=l,max_mel_len=max_mel_len)

# print(level1_embed_f.shape,level2_embed_f.shape)
# print(audio_f.shape,vq_loss,audio_loss)
# audio_x = model.inerence(audio)
# print(audio_x.shape)

# spec_transform = torchaudio.transforms.MelSpectrogram(sample_rate=48*1000, n_fft=2048 ,win_length=2048 ,hop_length=960,n_mels=80).to(device)
# melspec = spec_transform(audio)

# print(melspec.shape)
# out, latent_loss  = vqae(mel)
# print(out.shape)
# audio_fake,audio_loss,vq_loss = vqae(audio)
# print(audio_fake.shape)
# z_q,_,_ = vqae.encode(audio)
# # z_q = z_q.permute(0, 2, 3, 1)
# print(z_q.shape)
# z_q = z_q.permute(0, 2, 1)
# z_q_flatten = z_q.reshape(-1, 16).detach().cpu()
# codebook = vqae.vq_layer.embed.detach().cpu().T


# # # z_q = z_q.reshape(2,16,20*50)
# print(z_q.shape,z_q_flatten.shape,codebook.shape)

# # # b,embed,h,w = z_q.shape
# # # z_q =  torch.reshape(z_q,(b,embed,-1))
# # codebook = vqae.vq_layer.embedding.weight.detach().cpu()
# # print(z_q.shape,codebook.shape)
# # z_q_flatten = z_q.detach().cpu()
# combined = torch.concat((z_q_flatten,codebook),dim=0)
# combined = pca.fit_transform(combined)
# draw_dot([combined[:-512],combined[-512:]],["z_q","codebook"],name="z_q and codebook")

# # audio_fake = vqae.decode_inference(z_q,b,embed_dim,H,T)

# # print(z_q.shape,audio_fake.shape)
from params import params
from jukebook import *
pqmf_channel = 8
t = 48*1000 // pqmf_channel
model = VQAE(params)

audio = torch.rand(8,1,48000)
o = model(audio)
# encoder = EncoderBlock(pqmf_channel,64,hidden_dim=64,down_t=1,stride_t=2,depth=4,m_conv=10,dilation_growth_rate=3)
# o = encoder(pqmf_audio)
# # print(o.shape)
# decoder = DecoderBlock(pqmf_channel,64,hidden_dim=64,down_t=1,stride_t=2,depth=4,m_conv=10,dilation_growth_rate=3,dilation_cycle=3)
# a_f = decoder(o)
print(audio.shape,o.shape)
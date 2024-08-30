import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import VQAE
from params import params
from function import loadModel,save_audio
from dataset import BakerAudio,pad16
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = VQAE(params).to(device)
model = loadModel(model,"vqae_50","./model/")
dataset = BakerAudio(0,100,"L:/baker/")
audio = dataset.audios[0]
audio = pad16(audio).to(device)
audio = audio.unsqueeze(0).unsqueeze(0)
generate_audio,spec_loss,vq_loss = model(audio)
save_audio(generate_audio[0].detach().cpu(),48000,"fake",)
save_audio(audio[0].detach().cpu(),48000,"real",)

print(audio.shape,generate_audio.shape)
# print(spec_loss,vq_loss)
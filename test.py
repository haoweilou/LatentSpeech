import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import VQAE
from params import params
from function import loadModel,save_audio,draw_wave
from dataset import BakerAudio,pad16
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = VQAE(params).to(device)
num = 50
model = loadModel(model,f"vqae_{num}","./model/")
dataset = BakerAudio(0,100,"L:/baker/")
audio = dataset.audios[0]
audio = pad16(audio).to(device)
audio = audio.unsqueeze(0).unsqueeze(0)
generate_audio,spec_loss,vq_loss = model(audio)
generate_audio = generate_audio.detach().cpu()
save_audio(generate_audio[0],48000,f"fake_{num}")
save_audio(audio[0].detach().cpu(),48000,"real")

draw_wave(generate_audio[0][0],f"fake_wave_{num}")
draw_wave(audio[0][0].cpu(),"real_wave")
print(audio.shape,generate_audio.shape)
# print(spec_loss,vq_loss)
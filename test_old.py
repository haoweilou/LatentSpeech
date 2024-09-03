import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import AEOld
from params import params
from function import loadModel,save_audio,draw_wave,draw_heatmap,load_audio
from dataset import BakerAudio,pad16
import torchaudio
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num = 500
model_name = "aeold"

model = AEOld(params).to(device)
model = loadModel(model,f"{model_name}_{num}","./model/")
# model =  loadModel(model,f"ae9k16","L:/model/") 
# dataset = BakerAudio(0,100,"L:/baker/")
audio,sr = load_audio("./sample/english.wav")
# Resample the audio
resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=48000)
audio = resampler(audio)
# audio = dataset.audios[0]
audio = pad16(audio[0]).to(device)
audio = audio.unsqueeze(0).unsqueeze(0)

generate_audio,spec_loss = model(audio)
generate_audio = generate_audio.detach().cpu()
save_audio(generate_audio[0],48000,f"fake_{model_name}_{num}")
save_audio(audio[0].detach().cpu(),48000,"real")

draw_wave(generate_audio[0][0],f"fake_wave_{model_name}_{num}")
draw_wave(audio[0][0].cpu(),"real_wave")

draw_heatmap(model.encode(audio)[0].detach().cpu()[0],vmin=-1,vmax=1,name="latent feature")
print(audio.shape,generate_audio.shape)
# print(spec_loss,vq_loss)
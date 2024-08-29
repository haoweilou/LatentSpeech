import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import VQAE
from params import params
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

vqae = VQAE(params).to(device)

audio = torch.rand((64,1,48000)).to(device)
generate_audio,spec_loss,vq_loss = vqae(audio)
print(spec_loss,vq_loss)
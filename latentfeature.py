import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import VQAE,AE
from params import params
from function import loadModel,save_audio,draw_wave,draw_heatmap
from dataset import BakerAudio,pad16,LJSpeechAudio
from torch.utils.data import DataLoader,ConcatDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_name = "ae"
model = AE(params).to(device)
model = loadModel(model,f"{model_name}","./model/")

dataset1 = BakerAudio(0,10000,"L:/Baker/")
dataset2 = LJSpeechAudio(0,13200,"L:/LJSpeech/")
dataset = ConcatDataset([dataset1, dataset2])

loader = DataLoader(dataset,batch_size=params.batch_size,collate_fn=dataset1.collate,drop_last=False,shuffle=False)
from tqdm import tqdm
i = 0
with torch.no_grad():
    for audio in tqdm(loader):
        audio = audio.to(device)
        latent,mb_audio = model.encode(audio)
        torch.save(latent.cpu(),f"./latent/{i}")
        i += 1




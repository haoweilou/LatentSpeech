from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import VQAESeq,AE
from params import params
from function import loadModel,save_audio,draw_wave,draw_heatmap
from dataset import BakerAudio,pad16,LJSpeechAudio
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import DataLoader

# num = 500
# model_name = "ae9k2set"
num = 440
model_name = "vqae2set"

model = VQAESeq(params,embed_dim=64).to(device)
model = loadModel(model,f"{model_name}_{num}","./model/")
dataset = BakerAudio(0,10,"L:/baker/")
# dataset = LJSpeechAudio(0,13100,"L:/LJSpeech/")
# loader = DataLoader(dataset,batch_size=32,collate_fn=dataset.collate,drop_last=False,shuffle=False)
# i = 0
# with torch.no_grad():
#     for audio in tqdm(loader):
#         audio = audio.to(device)
#         time_steps = audio.shape[-1]
#         pad_amount = (16 - (time_steps % 16)) % 16
#         if pad_amount > 0:audio = F.pad(audio, (0, pad_amount))
#         z_q = model.encode(audio)#64,20,T
#         batch_size = z_q.shape[0]
#         for j in range(batch_size):
#             data = z_q[j].cpu()
#             torch.save(data,f"L:/LJSpeech/Latent/{i}")
#             i += 1
with torch.no_grad():
    for i in range(10):
        z_q = torch.load(f"L:/Baker/Latent/{i}").to("cuda")
        z_q = torch.unsqueeze(z_q,0)
        # z_q = model.encode(audio)
        print(z_q.shape)
        break
        latent_temp = model.mapper(melspec.squeeze(1))

        audio = model.decode(latent_temp)
        audio = model.pqmf.inverse(audio)[0]
        print(audio.shape)
        save_audio(audio.to("cpu"),48000,f"{i}")
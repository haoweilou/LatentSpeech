import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tts import StyleSpeech,FastSpeechLoss,SpecAdapter
from dataset import BakerAudio,BakerText
import math
from tts_config import config
from tqdm import tqdm
import pandas as pd
import torchaudio
from ae import VQAE_Audio,VQAE,AE
from params import params
from function import loadModel,saveModel
from alinger import SpeechRecognitionModel
import json
from function import collapse_and_duration
from torch.nn.utils.rnn import pad_sequence
import os
spec = True
def learning_rate(d_model=256,step=1,warmup_steps=400):
    return (1/math.sqrt(d_model)) * min(1/math.sqrt(step),step*warmup_steps**-1.5)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# modelname = "StyleSpeech"
modelname = "StyleSpeech4C"

if spec: modelname += "_spec"

if spec:
    # tts_model = StyleSpeech(config,embed_dim=80,output_channel=1).to(device)#old
    tts_model = StyleSpeech(config,embed_dim=20,output_channel=4).to(device)#new
    optimizer = optim.Adam(tts_model.parameters(), betas=(0.9,0.98),eps=1e-9,lr=learning_rate())
    # tts_model = loadModel(tts_model,f"{modelname}_{100}","./model")
    # modelname+="_freeze"
    # for param in tts_model.pho_encoder.parameters():
    #     param.requires_grad = False

else:
    tts_model = StyleSpeech(config,embed_dim=64).to(device)
    optimizer = optim.Adam(tts_model.parameters(), betas=(0.9,0.98),eps=1e-9,lr=learning_rate())
loss_func = FastSpeechLoss()

lr = learning_rate()
file_path = "/home/haoweilou/scratch/baker/"
# file_path = "L:/baker/"
bakertext = BakerText(normalize=False,start=0,end=9000,path=file_path)
bakeraudio = BakerAudio(start=0,end=9000,path=file_path)
def collate_fn(batch):
    text_batch, audio_batch = zip(*batch)
    text_batch = [torch.stack([item[i] for item in text_batch]) for i in range(len(text_batch[0]))]
    audio_batch = bakeraudio.collate(audio_batch)
    return text_batch, audio_batch

loader = torch.utils.data.DataLoader(dataset=list(zip(bakertext, bakeraudio)), collate_fn=collate_fn, batch_size=32, shuffle=True)

loss_log = pd.DataFrame({"total_loss":[],"mse_loss":[],"duration_loss":[]})
loss_log_name =  f"./log/loss_{modelname}"
# if os.path.exists(loss_log_name):
#     loss_log = pd.read_csv(loss_log_name, usecols=["total_loss", "mse_loss", "duration_loss"])
if spec:
    # vqae = VQAE(params,embed_dim=64).to(device)
    vqae = AE(params).to(device)
    vqae = loadModel(vqae,f"qae_200","./model")
else:
    vqae = VQAE_Audio(params,64,2048).to(device)
    vqae = loadModel(vqae,f"vqae_audio","./model/")

spec_transform = torchaudio.transforms.MelSpectrogram(sample_rate=48*1000, n_fft=2048 ,win_length=2048 ,hop_length=960,n_mels=80).to(device)

with open("./save/cache/phoneme.json","r") as f: 
    phoneme_set = json.loads(f.read())["phoneme"]
C = len(phoneme_set)+1  #Number of Phoneme Class, include blank, 87+1=88
aligner = SpeechRecognitionModel(input_dim=80,output_dim=C).to(device)
aligner = loadModel(aligner,"aligner","./model")

for epoch in range(0,201):
    total_loss = 0
    mse_loss_ = 0
    duration_loss_ = 0
    for i,(text_batch,audio_batch) in enumerate(tqdm(loader)):
        optimizer.zero_grad()
        x,s,l,src_lens,mel_lens = [tensor.to('cuda') for tensor in text_batch]
        with torch.no_grad():
            audio = audio_batch.to(device)
            if spec: 
                latent_r,b,t = vqae.encode_inference(audio)
                latent_r = latent_r.reshape(b,vqae.num_channel,t,20) #C,T,Feature_Dim
            else: 
                latent_r = vqae.encode_inference(audio).permute(0,2,1)
            melspec = spec_transform(audio).squeeze(1).permute(0,2,1)
            outputs = aligner(melspec).log_softmax(2)  # [batch_size, seq_len, num_phonemes]
            outputs = torch.argmax(outputs,dim=2) # [batch_size, melspec length
            outputs = [collapse_and_duration(i) for i in outputs] 
            l = pad_sequence([torch.tensor(i) for i in outputs],batch_first=True,padding_value=0).to(device) 
    
            padd_size = x.shape[-1] - l.shape[-1]
            if padd_size > 0: l = F.pad(l,(0,padd_size), "constant", 0)
        max_src_len = x.shape[1]
        if spec: 
            # max_mel_len = latent_r.shape[1]
            max_mel_len = t
        else:
            max_mel_len = latent_r.shape[1]
        latent_f,log_l_pred,mel_masks = tts_model(x,s,src_lens=src_lens,mel_lens=mel_lens,duration_target=l,max_mel_len=max_mel_len)
        #latent_f = [T,featDim*channel]
        # For concate channel
        # latent_f = latent_f.reshape(b,t,20,vqae.num_channel) # [T,featDim,channel]
        # latent_f = latent_f.permute(0,3,1,2) #[channel,T,featDim]

        l = l[:,:log_l_pred.shape[-1]]
        loss,mse_loss,duration_loss = loss_func(latent_r,latent_f,log_l_pred,l,mel_masks,device=device)
        total_loss += loss.item()
        mse_loss_ += mse_loss.item()
        duration_loss_ += duration_loss.item()
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(tts_model.parameters(), max_norm=1.0)

    print(f"Epoch: {epoch} MSE Loss: {mse_loss_/len(loader):.03f} Duration Loss: {duration_loss_/len(loader):.03f} Total: {total_loss/len(loader):.03f}")
    if epoch % 20 == 0:
        saveModel(tts_model,f"{modelname}_{epoch}","./model/")
    loss_log.loc[len(loss_log.index)] = [total_loss/len(loader),mse_loss_/len(loader),duration_loss_/len(loader)]
    loss_log.to_csv(loss_log_name)
    if epoch > 0:
        new_lr = learning_rate(step=epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
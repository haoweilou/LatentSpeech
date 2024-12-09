import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import json 
from model import ASR
import pandas as pd
from tqdm import tqdm
from function import saveModel, learning_rate
from dataset import BakerAudio,BakerText,LJSpeechAudio,LJSpeechText
from torchaudio.transforms import MelSpectrogram
torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
with open("./save/cache/phoneme.json","r") as f: 
    phoneme_set = json.loads(f.read())["phoneme"]
from ipa import ipa_pho_dict

T = 50                  #Input Sequence Length, melspec length
# C = len(phoneme_set)+1  #Number of Phoneme Class, include blank, 87+1=88
C = len(ipa_pho_dict)+1   #include empty already
N = 16                  #Batch size
S = 128                 #Target sequence length of the longest target in batch (zero padding) Phoneme length
S_min = 5               #Minium sequence length 5, min Phoneme length
feature_type = "Melspec"
feature_dim = 16 if feature_type != "Melspec" else 80        #feature dim

lr = learning_rate()
print("Initial learnign rate: ",lr)
print("Load Dataset: ")
model_name = "aligner"
# root = "/home/haoweilou/scratch/"
root = "L:/"
loss_log = pd.DataFrame({"total_loss":[],"ctc_loss":[]})
bakertext = BakerText(normalize=False,start=0,end=5000,path=f"{root}baker/",ipa=True)
bakeraudio = BakerAudio(start=0,end=5000,path=f"{root}baker/",return_len=True)

ljspeechtext = LJSpeechText(start=0,end=5000,path=f"{root}LJSpeech/")
ljspeechaudio = LJSpeechAudio(start=0,end=5000,path=f"{root}LJSpeech/",return_len=True)

from dataset import CombinedTextDataset,CombinedAudioDataset
textdataset = CombinedTextDataset(bakertext,ljspeechtext)
audiodataset = CombinedAudioDataset(bakeraudio,ljspeechaudio)


def collate_fn(batch):
    text_batch, audio_batch = zip(*batch)
    text_batch = [torch.stack([item[i] for item in text_batch]) for i in range(len(text_batch[0]))]
    audio_batch = bakeraudio.collate(audio_batch)
    return text_batch, audio_batch


#2k16, wo language, epoch 10 good , 2k32 wo language, epoch 10 good 
#4k32 wo language, epoch 13 good (below 1)
#8k32 wo language, epoch 6 good (below 1)
#8k32 wo language, epoch 6 good (below 1)
loader = DataLoader(dataset=list(zip(textdataset, audiodataset)), collate_fn=collate_fn, batch_size=32, shuffle=True)

aligner = ASR(input_dim=feature_dim,output_dim=C).to(device)
optimizer = optim.Adam(aligner.parameters(), betas=(0.9,0.98),eps=1e-9,lr=0.001)
CTCLoss = nn.CTCLoss()
#train aligner first 
melspec_transform = MelSpectrogram(sample_rate=48000,n_fft=1024,hop_length=1024,n_mels=80).to(device)

for epoch in range(2001):
    CTCLoss_ = 0
    for i,(text_batch,audio_batch) in enumerate(tqdm(loader)):
        optimizer.zero_grad()
        x,s,_,x_lens,_,language = [tensor.to(device) for tensor in text_batch]
        audio,y_lens = audio_batch[0].to(device),audio_batch[1].to(device)
        with torch.no_grad():
            melspec = melspec_transform(audio).squeeze(1) #B,T,80
            melspec = melspec.permute(0,2,1)#B,80,T
            y_lens = torch.ceil(y_lens/16/64).long()
        y = melspec           #batch size, phoneme length
        x_f = aligner(y,language)  # [batch_size, seq_len, num_phonemes]            
        x_f = x_f.log_softmax(2).transpose(0, 1) # [seq_len, batch_size, num_phonemes]
        loss = CTCLoss(x_f, x, y_lens, x_lens)

        CTCLoss_ += loss.item()
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch} CTC Loss: {CTCLoss_/len(loader):.03f} Total: {CTCLoss_/len(loader):.03f}")
        
    if epoch % 100 == 0:
        saveModel(aligner,f"{model_name}_{epoch}","./model/")

    loss_log.loc[len(loss_log.index)] = [CTCLoss_/len(loader),CTCLoss_/len(loader)]
    loss_log.to_csv(f"./log/loss_{model_name}")

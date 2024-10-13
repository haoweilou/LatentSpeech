import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader
import json 
from alinger import SpeechRecognitionModel
from dataset import BakerAudio,BakerText
from tqdm import tqdm
from function import saveModel, learning_rate
import pandas as pd
torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
with open("./save/cache/phoneme.json","r") as f: 
    phoneme_set = json.loads(f.read())["phoneme"]

T = 50                  #Input Sequence Length, melspec length
C = len(phoneme_set)+1  #Number of Phoneme Class, include blank, 87+1=88
N = 16                  #Batch size
S = 128                 #Target sequence length of the longest target in batch (zero padding) Phoneme length
S_min = 5               #Minium sequence length 5, min Phoneme length
feature_dim = 80        #feature dim

lr = learning_rate()
model_name = "Alinger"


loss_log = pd.DataFrame({"ctc_loss":[]})

bakertext = BakerText(normalize=False,start=0,end=1000)
bakeraudio = BakerAudio(start=0,end=1000)
def collate_fn(batch):
    text_batch, audio_batch = zip(*batch)
    text_batch = [torch.stack([item[i] for item in text_batch]) for i in range(len(text_batch[0]))]
    audio_batch = bakeraudio.collate(audio_batch)
    return text_batch, audio_batch

loader = torch.utils.data.DataLoader(dataset=list(zip(bakertext, bakeraudio)), collate_fn=collate_fn, batch_size=64, shuffle=True)

aligner = SpeechRecognitionModel(input_dim=feature_dim,output_dim=C).to(device)
optimizer = optim.Adam(aligner.parameters(), betas=(0.9,0.98),eps=1e-9,lr=0.001)
spec_transform = torchaudio.transforms.MelSpectrogram(sample_rate=48*1000, n_fft=400,win_length=400,hop_length=240,n_mels=80).to(device)

CTCLoss = nn.CTCLoss()
#train aligner first 
loss_log = pd.DataFrame({"ctc_loss":[]})

for epoch in range(3001):
    CTCLoss_ = 0
    for i,(text_batch,audio_batch) in enumerate(tqdm(loader)):
        optimizer.zero_grad()
        x,s,l,src_lens,mel_lens = [tensor.to('cuda') for tensor in text_batch]
        with torch.no_grad():
            audio = audio_batch.to(device)
            melspec = spec_transform(audio).squeeze(1).permute(0,2,1)
        real_outputs = aligner(melspec)  # [batch_size, seq_len, num_phonemes]
        real_outputs = real_outputs.log_softmax(2).transpose(0, 1) # [seq_len, batch_size, num_phonemes]
        loss = CTCLoss(real_outputs, x, mel_lens, src_lens)

        CTCLoss_ += loss.item()
        loss.backward()
        optimizer.step()
    print(f"epoch: {epoch} CTC_Loss: {CTCLoss_/len(loader):.03f}")
    loss_log.loc[len(loss_log.index)] = [CTCLoss_/len(loader)]
    loss_log.to_csv(f"./log/loss_{model_name}")
    if epoch % 200 == 0:
        saveModel(aligner,f"aligner_{epoch}",root=f"./model/")
    # if epoch > 0:
    #     new_lr = learning_rate(step=epoch)
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = new_lr
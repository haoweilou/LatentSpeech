import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader
import json 
from dataset import BakerText,BakerAudio
from tqdm import tqdm
from tts import StyleSpeech
from function import saveModel,loadModel,hidden_to_audio,save_audio
from tts_config import config
from params import params

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
with open("./save/cache/phoneme.json","r") as f: 
    phoneme_set = json.loads(f.read())["phoneme"]

T = 50                  #Input Sequence Length, melspec length
C = len(phoneme_set)+1  #Number of Phoneme Class, include blank, 87+1=88
N = 16                  #Batch size
S = 44                  #Target sequence length of the longest target in batch (zero padding) Phoneme length
S_min = 5               #Minium sequence length 5, min Phoneme length
feature_dim = 16        #feature dim


print("Load Dataset: ")
bakertext = BakerText(normalize=False,start=0,end=100,path="L:/Baker/")
bakeraudio = BakerAudio(start=0,end=100,path="L:/Baker/")
def collate_fn(batch):
    text_batch, audio_batch = zip(*batch)
    text_batch = [torch.stack([item[i] for item in text_batch]) for i in range(len(text_batch[0]))]
    audio_batch = bakeraudio.collate(audio_batch)
    return text_batch, audio_batch

loader = torch.utils.data.DataLoader(dataset=list(zip(bakertext, bakeraudio)), collate_fn=collate_fn, batch_size=64, shuffle=False)
tts_model = StyleSpeech(config,embed_dim=80).to(device)

# for epoch in range(350,400,50):
for epoch in [100]:

    tts_model = loadModel(tts_model,f"StyleSpeech_spec_{epoch}","./model/")
    index = 0
    for i,(text_batch,audio_batch) in enumerate(tqdm(loader)):
        x,s,l,src_lens,mel_lens = [tensor.to('cuda') for tensor in text_batch]
        print(x.shape,s.shape,l.shape,src_lens.shape,mel_lens.shape)

        max_mel_len = 256*2
        phonemes = x                #batch size, melspec length, feature dim
        phoneme_lengths = src_lens  #batch size, length of melspec

        hidden_lengths = mel_lens   #batch size, length of phoneme sequence

        y_pred,log_l_pred,mel_masks = tts_model(
            x,s,src_lens=src_lens,
            mel_lens=mel_lens,
            max_mel_len=max_mel_len)
        # print(y_pred.shape,log_l_pred.shape,x.shape)
        duration_rounded = torch.clamp((torch.round(torch.exp(log_l_pred) - 1) * 1),min=1,)
        print(duration_rounded)
        # print(duration_rounded[0],l[0],x[0],sep="\n")
        # save_audio(pred_audio,48000,f"{4000+index}","./sample/")
        for _ in range(64):
            print(y_pred.shape)
            pred_audio = hidden_to_audio(y_pred[i,:,:].unsqueeze(0))
            pred_audio = pred_audio.detach().cpu()
            save_audio(pred_audio[0],48000,f"{epoch}","./sample/")
            index += 1
            break
        break
        

    from function import phone_to_phone_idx,hanzi_to_pinyin
    hanzi = "如果有一天我老无所依请把我留在在这时光里娄皓维爱学习爱小宝"
    pinyin = hanzi_to_pinyin(hanzi)
    print(pinyin)
    # # pinyin = ["la1","la2","la3","la4","la5"]
    phone_idx,tone = phone_to_phone_idx(pinyin)
    # print(phone_idx,tone)
    d = 10
    duration = torch.tensor([[d for _ in range(len(phone_idx))]]).to(device)
    phone_mask = torch.tensor([[0 for _ in range(len(phone_idx))]]).to(device)
    phone_idx = torch.tensor([phone_idx]).to(device)  
    tone = torch.tensor([tone]).to(device)
    hidden_mask = torch.tensor([[0 for _ in range(1024)]]).to(device)

    src_lens = torch.tensor([phone_idx.shape[-1]]).to(device)
    mel_lens = torch.tensor([d*phone_idx.shape[-1]]).to(device)
    y_pred,log_l_pred,mel_masks = tts_model(phone_idx,tone,src_lens=src_lens,
        mel_lens=mel_lens,
        # duration_target = duration,
        max_mel_len=config["max_seq_len"])
    duration_rounded = torch.clamp((torch.round(torch.exp(log_l_pred) - 1) * 1),min=1,)
    print(duration_rounded[0],sep="\n")
    audio = hidden_to_audio(y_pred).detach().cpu()[0]
    save_audio(audio,48000,f"custom_{epoch}","./sample/")
    
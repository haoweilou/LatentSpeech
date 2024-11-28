import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from function import load_audio, slide_window, resample, save_audio
from torchvision import transforms
import torchaudio
from params import params
from tts_config import config
import os
from tqdm import tqdm
import random
import pandas as pd
from function import loadModel, load_audio
from torch.nn.utils.rnn import pad_sequence
import json
def preprocess(sample_rate,audio,filename):
    mel_args = {
      'sample_rate': sample_rate,
      'win_length': params.hop_samples * 4,
      'hop_length': params.hop_samples,
      'n_fft': params.n_fft,
      'f_min': 20.0,
      'f_max': sample_rate / 2.0,
      'n_mels': params.n_mels,
      'power': 1.0,
      'normalized': True,
    }
    mel_spec_transform = torchaudio.transforms.MelSpectrogram(**mel_args)
    spectrogram = mel_spec_transform(audio)
    spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
    spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
    np.save(f'{filename}.spec.npy', spectrogram.cpu().numpy())
    return spectrogram

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def pad16(audio):
    pad_length = (16 - audio.shape[-1] % 16) % 16
    if pad_length > 0:
        # Pad the audio signal with zeros
        return F.pad(audio, (0, pad_length), mode='constant', value=0)
    return audio

from torch.nn.utils.rnn import pad_sequence
class BakerAudio(torch.utils.data.Dataset):
    """This dataset only contain the audio file of baker dataset, it is meant to be used when train AE"""
    def __init__(self,start=0,end=10000,path="/home/haoweilou/scratch/baker/"):
        super(BakerAudio, self).__init__()
        audio_path = path
        audio_files = [f"{audio_path}Wave/{i:06d}.wav" for i in range(1,10001)][start:end]
        audios = [pad16(load_audio(f)[0][0]) for f in tqdm(audio_files)]
        self.audios = audios
        self.max_word_len = 512
        
    def collate(self, minibatch):
        output = pad_sequence(minibatch,batch_first=True) #Batch,T
        if output.shape[-1] >= 48000*10:
            output = output[:,:48000*10]
        return output.unsqueeze(1) 
        
    def __getitem__(self, index):
        return self.audios[index]

    def __len__(self):
        return len(self.audios)
    
class LJSpeechAudio(torch.utils.data.Dataset):
    """This dataset only contain the audio file of baker dataset, it is meant to be used when train AE"""
    def __init__(self,start=0,end=10000,path="/home/haoweilou/scratch/LJSpeech/"):
        super(LJSpeechAudio, self).__init__()
        audio_path = path
        audio_files = os.listdir(f"{audio_path}wavs/")
        audio_files = [f"{audio_path}wavs/{a}" for a in audio_files][start:end]
        audios = [load_audio(f)[0][0] for f in tqdm(audio_files)]
        resampler = torchaudio.transforms.Resample(orig_freq=22050, new_freq=48000)
        audios = [pad16(resampler(a)) for a in audios]
        self.audios = audios
        self.max_word_len = 512
        
    def collate(self, minibatch):
        output = pad_sequence(minibatch,batch_first=True) #Batch,T
        if output.shape[-1] >= 48000*9:
            output = output[:,:48000*9]
        return output.unsqueeze(1)

    def __getitem__(self, index):
        return self.audios[index]

    def __len__(self):
        return len(self.audios)
    
class BakerText(torch.utils.data.Dataset):
    def __init__(self,normalize=True,path="D:/baker/",start=0,end=10000):
        super(BakerText, self).__init__()
        self.max_word_len = 512
        self.path = path
        self.pho_dict = json.loads(open("./save/cache/phoneme.json","r").read())["phoneme"]
        self.phonemes = json.loads(open("./save/cache/baker_hidden.json","r").read())

        self.keys = list(self.phonemes.keys())[start:end]
        x_raw = [self.phonemes[k]["phoneme"] for k in self.keys]
        s_raw = [self.phonemes[k]["tone_list"] for k in self.keys]
        self.src_len = torch.tensor([len(self.phonemes[k]["phoneme"]) for k in self.keys])
        self.mel_len = torch.tensor([self.phonemes[k]["mel_size"] for k in self.keys])
        self.num_sentence = end - start
        
        self.x = pad_sequence(self.pho2idx(x_raw,self.pho_dict,normalize=normalize),batch_first=True,padding_value=0)
        self.s = pad_sequence([torch.tensor([int(i) for i in s]) for s in s_raw],batch_first=True,padding_value=0)
        # self.l = pad_sequence([torch.tensor(self.phonemes[k]["mel_len"]) for k in self.keys],batch_first=True,padding_value=0)
        self.l = pad_sequence([torch.ceil(torch.tensor(self.phonemes[k]["pho_len"])/4800/0.02) for k in self.keys],batch_first=True,padding_value=0)


    def pho2idx(self,phonemes,pho_list:list,normalize=True):
        output = []
        for phoneme in phonemes:
            if normalize:
                index_pho= [pho_list.index(p) for p in phoneme]
            else: 
                index_pho= [1+pho_list.index(p) for p in phoneme]
            output.append(torch.tensor(index_pho))
        return output
    
    def __getitem__(self, index):
        return self.x[index], self.s[index], self.l[index], self.src_len[index], self.mel_len[index]

    def __len__(self):
        return len(self.x)
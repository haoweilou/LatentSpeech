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
from model import AE
from function import loadModel, load_audio
from torch.nn.utils.rnn import pad_sequence

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
        if output.shape[-1] >= 48000*10:
            output = output[:,:48000*10]
        return output.unsqueeze(1)

    def __getitem__(self, index):
        return self.audios[index]

    def __len__(self):
        return len(self.audios)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.utils
import torch.utils.data
from function import load_audio, slide_window, resample, save_audio
from torchvision import transforms
import torchaudio
from params import params
from tts_config import config
import os
from tqdm import tqdm
import random
import json
import pandas as pd
from function import loadModel, load_audio
from torch.nn.utils.rnn import pad_sequence
import json
from ipa import pinyin_sentence_to_ipa,mandarin_chinese_to_ipa,ipa_pho_dict,english_sentence_to_ipa

import re
import pandas as pd

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
    def __init__(self,start=0,end=10000,path="/home/haoweilou/scratch/baker/",return_len=False):
        super(BakerAudio, self).__init__()
        audio_path = path
        audio_files = [f"{audio_path}Wave/{i:06d}.wav" for i in range(1,10001)][start:end]
        audios = [pad16(load_audio(f)[0][0]) for f in tqdm(audio_files)]
        self.audios = audios
        self.audio_lens = [len(audios[i]) for i in range(len(audios))]
        self.max_word_len = 512
        self.return_len = return_len
        
    def collate(self, minibatch):
        if self.return_len:
            minibatch_ = [a[0] for a in minibatch]
            audio_len = torch.tensor([a[1] for a in minibatch])
            minibatch = minibatch_
        output = pad_sequence(minibatch,batch_first=True) #Batch,T
        # if output.shape[-1] >= 48000*10:
        #     output = output[:,:48000*10]
        #     audio_len = torch.tensor([48000*10 for _ in minibatch])
        if self.return_len: 
            return output.unsqueeze(1), audio_len
        else:
            return output.unsqueeze(1) 
        
    def __getitem__(self, index):
        if self.return_len == True:
            return self.audios[index],self.audio_lens[index]
        else: 
            return self.audios[index]


    def __len__(self):
        return len(self.audios)
    
class LJSpeechAudio(torch.utils.data.Dataset):
    """This dataset only contain the audio file of baker dataset, it is meant to be used when train AE"""
    def __init__(self,start=0,end=10000,path="/home/haoweilou/scratch/LJSpeech/",return_len=False):
        super(LJSpeechAudio, self).__init__()
        audio_path = path
        with open(f"{path}metadata.csv","r",encoding="utf-8") as f: 
            lines = f.readlines()
        audio_names = []
        for line in lines: 
            audio_names.append(line.split("|")[0])
        audio_names = audio_names[start:end]
        audio_files = [f"{audio_path}wavs/{a}.wav" for a in audio_names]
        audios = [load_audio(f)[0][0] for f in tqdm(audio_files)]
        
        resampler = torchaudio.transforms.Resample(orig_freq=22050, new_freq=48000)
        audios = [pad16(resampler(a)) for a in audios]
        self.audio_lens = [len(audios[i]) for i in range(len(audios))]
        self.audios = audios
        self.max_word_len = 512
        self.return_len = return_len

    def collate(self, minibatch):
        if self.return_len:
            minibatch_ = [a[0] for a in minibatch]
            audio_len = torch.tensor([a[1] for a in minibatch])
            minibatch = minibatch_
        output = pad_sequence(minibatch,batch_first=True) #Batch,T
        if output.shape[-1] >= 48000*10:
            output = output[:,:48000*10]
            audio_len = torch.tensor([48000*10 for _ in minibatch])
        if self.return_len: 
            return output.unsqueeze(1), audio_len
        else:
            return output.unsqueeze(1) 

    def __getitem__(self, index):
        if self.return_len == True:
            return self.audios[index], self.audio_lens[index]
        else: 
            return self.audios[index]

    def __len__(self):
        return len(self.audios)
    
def adjust_sil_durations(phonemes,durations):
    n = len(durations)

    def no_sil(x):
        return x == 81

    i = 1
    while i < n - 1:
        if no_sil(phonemes[i]):  # Identify silence phoneme
            # Distribute the silence duration to adjacent phonemes
            if durations[i - 1] >= durations[i + 1]:
                durations[i - 1] += durations[i]
            else:
                durations[i + 1] += durations[i]

            # Remove the silence phoneme and its duration
            durations.pop(i)
            phonemes.pop(i)
            n -= 1
        else:
            i += 1
    return phonemes,durations

        
class BakerText(torch.utils.data.Dataset):
    def __init__(self,normalize=True,path="D:/baker/",start=0,end=10000,ipa=False,no_sil=False):
        super(BakerText, self).__init__()
        self.max_word_len = 512
        self.path = path
        if not ipa:
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
           

            # self.l = pad_sequence([torch.ceil(torch.tensor(self.phonemes[k]["pho_len"])/4800/0.02) for k in self.keys],batch_first=True,padding_value=0)
        else: 
            from ipa import ipa_pho_dict
            self.pho_dict = ipa_pho_dict
            with open(f"{path}ProsodyLabeling/000001-010000.txt","r",encoding="utf-8") as f: 
                lines = f.readlines()
            hanzis = []
            for i in range(start,end):
                sentence = ''.join(re.findall(r'[\u4e00-\u9fff]', lines[i*2]))
                hanzis.append(sentence)

            ipa_phonemes, tones = [],[] 
            src_lens = []

            self.hanzi = hanzis
            
            for hanzi in hanzis:
                ip,t = mandarin_chinese_to_ipa(hanzi)
                ipa_phonemes.append(ip)
                tones.append(t)
                src_lens.append(len(ip))
            self.ipa_sentences = ipa_phonemes
            self.src_len = torch.tensor(src_lens)
            self.max_len = max(self.src_len)

            ipd_idx = [[ipa_pho_dict[i] for i in row] for row in ipa_phonemes]

            with open("./save/duration/baker.json","r") as f: 
                data_str = f.read()
                data = json.loads(data_str)
                l = []
                for i in range(start,end):
                    duration = data[str(i)]
                    l.append(duration)

            if no_sil:
                tones = [
                    [tone for tone, phoneme in zip(s, p) if phoneme != 81] for s, p in zip(tones, ipd_idx)
                ]
                ipd_idx, l = zip(*[adjust_sil_durations(d, p) for d, p in zip(ipd_idx, l)])

            self.x = pad_sequence([torch.tensor([int(i) for i in x]) for x in ipd_idx],batch_first=True,padding_value=0)
            self.s = pad_sequence([torch.tensor([int(i) for i in s]) for s in tones],batch_first=True,padding_value=0)
            self.l = pad_sequence([torch.tensor(d) for d in l],batch_first=True,padding_value=0)

            # self.l = torch.ones_like(self.x)
            self.mel_len = torch.ones_like(self.src_len)
        self.language = torch.ones_like(self.src_len)

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
        return self.x[index], self.s[index], self.l[index], self.src_len[index], self.mel_len[index], self.language[index]

    def __len__(self):
        return len(self.x)
    

class LJSpeechText(torch.utils.data.Dataset):
    def __init__(self,path="D:/LJSpeech/",start=0,end=10000,no_sil=False):
        super().__init__()
        self.max_word_len = 512
        self.path = path
        
        self.pho_dict = ipa_pho_dict
        with open(f"{path}metadata.csv","r",encoding="utf-8") as f: 
            lines = f.readlines()
        text = []
        for line in lines: 
            text.append(line.split("|")[-1])
        # text = pd.read_csv(f"{path}metadata.csv",sep="|",header=None)
        english_sentence = []
        for i in range(start,end):
            sentence = ''.join(re.sub(r'[^a-zA-Z\s]', '', text[i].lower()))
            english_sentence.append(sentence.strip())


        self.english_sentence = english_sentence

        ipa_sentences, stress = [],[] 
        src_lens = []
        for sentence in english_sentence:
            ip_semtemce,t = english_sentence_to_ipa(sentence)
            ipa_sentences.append(ip_semtemce)
            stress.append(t)
            src_lens.append(len(ip_semtemce))
        self.ipa_sentences = ipa_sentences
        self.src_len = torch.tensor(src_lens)
        self.max_len = max(self.src_len)

        ipd_idx = [[ipa_pho_dict[i] for i in ip_semtemce] for ip_semtemce in ipa_sentences]
        # self.s = pad_sequence([torch.tensor([int(i) for i in s]) for s in stress],batch_first=True,padding_value=0)
        with open("./save/duration/LJSpeech.json","r") as f: 
                data_str = f.read()
                data = json.loads(data_str)
                l = []
                for i in range(start,end):
                    duration = data[str(i)]
                    l.append(duration)
        if no_sil:
            ipd_idx, l = zip(*[adjust_sil_durations(d, p) for d, p in zip(ipd_idx, l)])
            
        self.x = pad_sequence([torch.tensor([int(i) for i in x]) for x in ipd_idx],batch_first=True,padding_value=0)
        self.s = torch.zeros_like(self.x)
        self.l = pad_sequence([torch.tensor(d) for d in l],batch_first=True,padding_value=0)
        # self.l = torch.ones_like(self.x)
        # None
        self.mel_len = torch.ones_like(self.src_len)
        self.language = torch.ones_like(self.src_len)

        

    # def calculate_l(self,aligner,ys,y_lens):
    #     from torchaudio.transforms import MelSpectrogram
    #     from function import duration_calculate
    #     melspec_transform = MelSpectrogram(sample_rate=48000,n_fft=1024,hop_length=1024,n_mels=80).to(device)
    #     output = []
    #     print("Start Loading and Calculate Duration: ")
    #     for i,y in enumerate(tqdm(ys)): 
    #         y = torch.unsqueeze(y,0).to(device)
    #         melspec = melspec_transform(y).squeeze(1) #B,T,80
    #         melspec = melspec.permute(0,2,1)#B,80,T
    #         prob_matrix = aligner(melspec)  # [batch_size, y_len, num_phonemes], probability 
    #         emission = torch.log_softmax(prob_matrix,dim=-1) # [seq_len, batch_size, num_phonemes]
    #         # print(emission.shape,self.x[i].cpu().unsqueeze(0).shape)
    #         l = duration_calculate(emission.cpu(),self.x[i].cpu().unsqueeze(0),[self.src_len[i]],[y_lens[i]], max_x_len = self.src_len[i])
    #         output.append(l[0])
    #     output = pad_sequence([torch.tensor(i) for i in output],batch_first=True,padding_value=0)
    #     output = F.pad(output,pad=(0,max(self.src_len)-output.shape[-1]),value=0)
    #     self.l = output
    #     print(self.l,output.shape)

    def __getitem__(self, index):
        return self.x[index], self.s[index], self.l[index], self.src_len[index], self.mel_len[index], self.language[index]

    def __len__(self):
        return len(self.x)
    
class CombinedTextDataset(torch.utils.data.Dataset):
    def __init__(self, text_dataset1, text_dataset2):
        # Ensure the text and audio datasets have the same length
        assert len(text_dataset1) == len(text_dataset2)
        self.x = pad_sequence(list(text_dataset1.x)+list(text_dataset2.x),batch_first=True)
        self.s = pad_sequence(list(text_dataset1.s)+list(text_dataset2.s),batch_first=True)
        self.l = pad_sequence(list(text_dataset1.l)+list(text_dataset2.l),batch_first=True)
        self.src_len = torch.concat([text_dataset1.src_len,text_dataset2.src_len],dim=0)
        self.mel_len = torch.ones_like(self.src_len)
        self.language = torch.cat([torch.zeros((len(text_dataset1),self.x.shape[-1])),torch.ones((len(text_dataset2),self.x.shape[-1]))],dim=0).long()

    
    def __getitem__(self, index):
        return self.x[index], self.s[index], self.l[index], self.src_len[index], self.mel_len[index], self.language[index]

    def __len__(self):
        return len(self.x)
    

class CombinedAudioDataset(torch.utils.data.Dataset):
    def __init__(self, audio_dataset1, audio_dataset2):
        # Ensure the text and audio datasets have the same length
        assert len(audio_dataset1) == len(audio_dataset1)
        self.audio = pad_sequence(list(audio_dataset1.audios)+list(audio_dataset2.audios),batch_first=True)
        #list
        self.audio_lens = audio_dataset1.audio_lens+audio_dataset2.audio_lens

    
    def __getitem__(self, index):
        return self.audio[index],self.audio_lens[index]

    def __len__(self):
        return len(self.audio)
        

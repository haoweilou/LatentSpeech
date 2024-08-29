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
from function import loadModel
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
class BakerOlder(torch.utils.data.Dataset):
    """Some Information about Baker"""
    def __init__(self,start=0,end=10000,window=100,root="/home/haoweilou/scratch/baker/Wave32/"):
        super(BakerOlder, self).__init__()
        self.num_sentence = end - start
        self.root = root
        self.melroot = root
        
        audios = []
        melspecs = []
        phone_label = []
        hiddens = []
        self.ae = AE(params)
        self.ae = loadModel(self.ae,"ae9k16",root="./model").to(device)
        self.miss_label = [1745,2436,3007,4188,4333,4784, 5398, 5406, 6836, 7167]
        hidden_min,hidden_max = 100,-100
        target_length = params.max_time * params.sample_rate
        with open("./mfa/pho","r") as f: 
            data = f.read()
        self.phodict = data.split()
        self.phodict = {p:i for i, p in enumerate(self.phodict)}
        for i in tqdm(range(start,end+1)):
            if i in self.miss_label: continue
            file_path = f"{self.root}{i:06}.wav"
            audio,rate = load_audio(file_path)
            audio = torch.squeeze(audio,0)
            melspec_path = f"{self.melroot}{i:06}.wav.spec.npy"
            if not os.path.exists(melspec_path):
                melspec = preprocess(rate,audio,file_path)
            else:
                melspec = np.load(melspec_path)
            pad_length = 16 - (audio.size(0) % 16)
            if pad_length != 0: 
                audio = F.pad(audio, (0, pad_length), mode='constant', value=0)
            # hidden,_ = self.ae.encode(audio.unsqueeze(0).unsqueeze(0).to(device))
            hidden = torch.randn(16,400)
            hidden_min = torch.min(hidden)
            hidden_max = torch.max(hidden)
            hidden = hidden.detach().cpu()[0]
            hiddens.append(hidden)
            audios.append(audio.numpy())
            melspecs.append(melspec)
            phone_label.append(self.load_phonelabel(i))
        self.hidden_min = hidden_min.detach_().cpu()
        self.hidden_max = hidden_max.detach_().cpu()

        self.rate = rate
        self.params = params
        self.audios = audios
        self.phone = phone_label
        self.melspecs = melspecs
        self.hiddens = hiddens
    

    def collate(self, minibatch):
        samples_per_frame = self.params.hop_samples
        for record in minibatch:
            start = random.randint(0, record['spectrogram'].shape[1] - self.params.crop_mel_frames)
            end = start + self.params.crop_mel_frames
            record['spectrogram'] = record['spectrogram'][:,start:end]
            start *= samples_per_frame
            end *= samples_per_frame
            record['audio'] = record['audio'][start:end]
            record['audio'] = np.pad(record['audio'], (0, (end-start) - len(record['audio'])), mode='constant')
        audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])
        spectrogram = np.stack([record['spectrogram'] for record in minibatch if 'spectrogram' in record])
        #audio shape: [B,L], Spectrogram Shape: [B,N_MELs, L]
        return {
        'audio': torch.from_numpy(audio),
        'spectrogram': torch.from_numpy(spectrogram),
    }

    def padding(self,minibatch):
        if params.sample_rate == 3200:
            target_len = params.sample_rate * 5
        else: 
            target_len = params.sample_rate * 2
        for record in minibatch:
            audio = record["audio"]
            padding = target_len - len(audio)
            if padding > 0:
                padded_audio = np.pad(audio, (0, padding), 'constant', constant_values=(0,))
                record["audio"] = padded_audio  
            elif padding < 0:
                start = random.randint(0, audio.shape[-1] - target_len)
                end = start + target_len
                record["audio"] = audio[start:end]
            else:
                record["audio"] = audio
        audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])
        audio = torch.unsqueeze(torch.from_numpy(audio),1)
        return audio

    def load_phonelabel(self,i):
        root = "./mfa/phone/"
        if i not in self.miss_label:
            path = f"{root}{i}.csv"
            df = pd.read_csv(path)
            name = df['name'].to_list()
            phone = [i if not i[-1].isdigit() else i[:-1] for i in name]
            tone = [0 if not i[-1].isdigit() else int(i[-1]) for i in name]
            duration = df["duration"].to_list()
            output = {"phone":phone,"duration":duration,"tone":tone,"phone_idx":[self.phodict[p] for p in phone]}
            return output
        
    def phone_collate(self,minibatch):
        target_len = config["max_seq_len"]
        target_phone = config["max_phone_len"]
        phone_mask = []
        hidden_mask = []
        for i,record in enumerate(minibatch):
            audio = record["audio"]
            padding = 524288 - len(audio)
            if padding > 0:
                padded_audio = np.pad(audio, (0, padding), 'constant', constant_values=(0,))
                record["audio"] = padded_audio  
            else:
                record["audio"] = audio
            

            duration = record["phone"]["duration"]
            hidden_len = record["hidden"].shape[-1]
            hidden_padding = target_len-hidden_len
            if hidden_padding > 0:
                record["hidden"] = np.pad(record["hidden"],pad_width=((0,0),(0,hidden_padding)),constant_values=self.hidden_max)
            hidden_mask.append([0] * hidden_len + [1] * hidden_padding)
            # print(sum(duration),sum(hidden_mask[-1]))
            # print(duration,hidden_mask[-1])
            assert sum(duration) == len(hidden_mask[-1])-sum(hidden_mask[-1])

            phone = record["phone"]["phone"]
            phone_len = len(phone)
            padding = target_phone-phone_len
            if padding > 0: 
                #if dont have enough phone, create spn to fill
                for _ in range(padding):
                    record["phone"]["phone"].append("spn")
                    record["phone"]["duration"].append(0)
                    record["phone"]["tone"].append(0)
                    record["phone"]["phone_idx"].append(self.phodict["spn"])
                phone_mask.append([0 for _ in range(phone_len)]+[1 for _ in range(padding)])
            else: 
                phone_mask.append([0 for _ in range(target_phone)])

            
            # elif padding < 0: 
            #     start = random.randint(0, phone_len - target_phone)
            #     end = start + target_phone
            #     record["phone"]["phone"] = record["phone"]["phone"][start:end]
            #     record["phone"]["duration"] = record["phone"]["duration"][start:end]
            #     record["phone"]["tone"] = record["phone"]["tone"][start:end]
            #     record["phone"]["phone_idx"] = record["phone"]["phone_idx"][start:end]
            #     phone_mask.append([1 for _ in range(target_phone)])
        
        phone_idx =  np.stack([record["phone"]["phone_idx"] for record in minibatch])
        tone =  np.stack([record["phone"]["tone"] for record in minibatch])
        duration =  np.stack([record["phone"]["duration"] for record in minibatch])
        hidden = np.stack([record["hidden"] for record in minibatch])
        phone_mask = np.stack(phone_mask)
        hidden_mask = np.stack(hidden_mask)
        normalize_hidden = (torch.from_numpy(hidden) - self.hidden_min) / (self.hidden_max - self.hidden_min)
        audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])
        audio = torch.unsqueeze(torch.from_numpy(audio),1)
        return {
            'phone_idx': torch.from_numpy(phone_idx),
            'tone': torch.from_numpy(tone),
            'duration': torch.from_numpy(duration),
            "phone_mask":torch.from_numpy(phone_mask),
            "hidden":normalize_hidden,
            "hidden_mask":torch.from_numpy(hidden_mask),
            "audio": audio
        }


    def __getitem__(self, index):
        return {"audio":self.audios[index], "spectrogram":self.melspecs[index],"phone":self.phone[index],"hidden":self.hiddens[index]}

    def __len__(self):
        return len(self.audios)
    
import json

import json
from torch.nn.utils.rnn import pad_sequence
class Baker(torch.utils.data.Dataset):
    """Some Information about Baker"""
    def __init__(self,normalize=True,path="C:/Users/l/Desktop/baker/",start=0,end=10000):
        super(Baker, self).__init__()
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
        self.l = pad_sequence([torch.tensor(self.phonemes[k]["mel_len"]) for k in self.keys],batch_first=True,padding_value=0)
        # y = pad_sequence([torch.load(f"{self.path}MelSpec/{k}.pt").T for k in tqdm(self.keys)],batch_first=True,padding_value=0)
        y = pad_sequence([torch.load(f"{self.path}Hidden16/{k}.pt").T.detach() for k in tqdm(self.keys)],batch_first=True,padding_value=0)
        #normalize mel-spectrogram
        # self.hidden_max = torch.tensor(114.6849)
        # self.hidden_min = torch.tensor(-7.1979)
        # self.hidden_max = torch.tensor(91.5085)
        # self.hidden_min = torch.tensor(-7.1978)
        self.hidden_max = torch.max(y)
        self.hidden_min = torch.min(y)
        # y = (y - self.hidden_min) / (self.hidden_max - self.hidden_min)
        self.y = y


    def __getitem__(self, index):
        return self.x[index], self.s[index], self.l[index], self.y[index], self.src_len[index], self.mel_len[index]

    def __len__(self):
        return len(self.y)

    def init(self): 
        self.dict = self.load_label()
        if os.path.exists("./save/cache/baker.json"):
            self.phone = json.loads(open("./save/cache/baker.json","r").read())
        else:
            self.phone = self.load_phoneme()
            with open("./save/cache/baker.json","w") as f: 
                json_file = json.dumps(self.phone,indent=2)
                f.write(json_file)
        all_pho = []
        self.phonemes = []
        self.tones = []
        for key in self.phone.keys():
            phoneme = self.phone[key]["phoneme"]
            tone = self.phone[key]["tone_list"]
            self.tones.append(tone)
            self.phonemes.append(phoneme)
            for item in phoneme:
                if item not in all_pho: all_pho.append(item)
        phone_dict_path = "./save/cache/phoneme.json"
        if os.path.exists(phone_dict_path):
            all_pho = json.loads(open(phone_dict_path,"r").read())["phoneme"]
        else: 
            all_pho.sort()
            with open(phone_dict_path,"w") as f: 
                f.write(json.dumps({"phoneme":all_pho},indent=2))
        index_styles = [torch.tensor([int(i) for i in s]) for s in self.tones]
        index_phonemes = self.pho2idx(self.phonemes,all_pho)
        self.x1 = pad_sequence(index_phonemes, batch_first=True, padding_value=0)
        self.x2 = pad_sequence(index_styles, batch_first=True, padding_value=0)

        for key in tqdm(self.phone.keys()):
            # mel_spec = torch.load(f"{self.path}MelSpec/{key}.pt")
            mel_spec = torch.load(f"{self.path}Hidden/{key}.pt")
            mel_size = mel_spec.shape[1]
            pho_len = self.phone[key]["pho_len"]
            sum_pholen = sum(pho_len)
            mel_len = [int(i/sum_pholen*mel_size) for i in pho_len]
            try:
                assert sum(mel_len) == mel_size
            except Exception:
                mel_len[-1] += mel_size-sum(mel_len)
            assert sum(mel_len) == mel_size
            self.phone[key]["mel_len"] = mel_len
            self.phone[key]["mel_size"] = mel_size
        
        with open("./save/cache/baker_hidden.json","w") as f: 
            json_file = json.dumps(self.phone,indent=2)
            f.write(json_file)

    def load_audio(self,name):
        wave, sample_rate = torchaudio.load(filepath=f"{self.path}Wave/{name}.wav")
        return wave, sample_rate 
    
    def pho2idx(self,phonemes,pho_list:list,normalize=True):
        output = []
        for phoneme in phonemes:
            if normalize:
                index_pho= [pho_list.index(p) for p in phoneme]
            else: 
                index_pho= [1+pho_list.index(p) for p in phoneme]
            output.append(torch.tensor(index_pho))
        return output

    def load_label(self):
        output = {}
        file_path = self.path+"ProsodyLabeling/000001-010000.txt"
        with open(file_path,"r",encoding="utf-8") as f: 
            content = f.readlines()
        for i in range(len(content)//2):
            idx = content[i*2][:6]
            pinyin = content[i*2+1].strip()
            output[idx] = pinyin
        return output
    
    def load_phoneme(self):
        output = {}
        keys = self.dict.keys()
        interval_path = self.path+"PhoneLabeling/"
        for key in keys:
            filepath = interval_path+key+".interval"
            duration,pho_list,tone_list,pho_len = self.load_interval_file(filepath)
            output[key] = {
                "duration":duration,
                "phoneme":pho_list,
                "tone_list":tone_list,
                "pho_len":pho_len
            }
        return output

    def load_interval_file(self,path):
        file = open(path,"r")
        lines = file.readlines()
        wave_time = float(lines[4])
        num_pho = int(lines[11])
        pho_list = []
        tone_list = []
        aud_len_list = []
        for i in range(len(lines)-12):
            start_idx = 12+i*3
            end_idx = 12+i*3+1
            char_idx = 12+i*3+2
            if start_idx == len(lines): break

            start_time = float(lines[start_idx])
            end_time = float(lines[end_idx])
            char = lines[char_idx].strip().replace("\"","")
            if char[-1].isdigit():
                pho_list.append(char[:-1])
                tone_list.append(char[-1])
            else: 
                pho_list.append(char)
                tone_list.append("0")
            aud_len_list.append(int((end_time-start_time)*48*100))
        assert num_pho == len(pho_list)
        return wave_time,pho_list,tone_list,aud_len_list
# from fastdtw import fastdtw
import matplotlib.pyplot as plt
def plot_emission(emission):
    fig, ax = plt.subplots()
    ax.imshow(emission.cpu().T)
    ax.set_title("Frame-wise class probabilities")
    ax.set_xlabel("Time")
    ax.set_ylabel("Labels")
    fig.tight_layout()

import torch
import torch.nn as nn
# import torch.nn.functional as F
import torchaudio.functional as F
import torch.optim as optim
from params import params
# from tts import DurationAligner
from flow import AE
from function import loadModel,saveModel, agd_duration,save_audio
import pandas as pd

from dataset import BakerAudio,BakerText,LJSpeechAudio,LJSpeechText
from torch.utils.data import DataLoader
from params import params
import json
from torchaudio.transforms import MelSpectrogram

from tts import StyleSpeech2_FF,FastSpeechLoss
from tqdm import tqdm
from model import ASR
from tts_config import config

import math
is_ipa = True
from ipa import ipa_pho_dict
if is_ipa: config["pho_config"]["word_num"] = len(ipa_pho_dict)

# root = "/home/haoweilou/scratch/"
root = "L:/"
loss_log = pd.DataFrame({"total_loss":[],"ctc_loss":[]})
bakertext = BakerText(normalize=False,start=0,end=100,path=f"{root}baker/",ipa=True)
bakeraudio = BakerAudio(start=0,end=100,path=f"{root}baker/",return_len=True)

ljspeechtext = LJSpeechText(start=400,end=500,path=f"{root}LJSpeech/")
ljspeechaudio = LJSpeechAudio(start=400,end=500,path=f"{root}LJSpeech/",return_len=True)

from dataset import CombinedTextDataset,CombinedAudioDataset

textdataset = CombinedTextDataset(ljspeechtext,ljspeechtext)
audiodataset = CombinedAudioDataset(ljspeechaudio,ljspeechaudio)

def collate_fn(batch):
    text_batch, audio_batch = zip(*batch)
    text_batch = [torch.stack([item[i] for item in text_batch]) for i in range(len(text_batch[0]))]
    audio_batch = bakeraudio.collate(audio_batch)
    return text_batch, audio_batch

# loader = DataLoader(dataset=list(zip(bakertext, bakeraudio)), collate_fn=collate_fn, batch_size=16, shuffle=True)
loader = DataLoader(dataset=list(zip(textdataset, audiodataset)), collate_fn=collate_fn, batch_size=16, shuffle=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ae = AE(params).to(device)
ae = loadModel(ae,"ae20k16_1000","./model")
def learning_rate(d_model=256,step=1,warmup_steps=400):
    return (1/math.sqrt(d_model)) * min(1/math.sqrt(step),step*warmup_steps**-1.5)

lr = learning_rate()
print(f"Initial LR: {lr}")

# model = loadModel(model, f"{modelname}_200","./model/")

# aligner = ASR(80,len(phoneme_set)+1).to(device)
aligner = ASR(80,len(ipa_pho_dict)+1).to(device)
aligner = loadModel(aligner,"aligner_200","./model/")

fastloss = FastSpeechLoss().to(device)

melspec_transform = MelSpectrogram(sample_rate=48000,n_fft=1024,hop_length=1024,n_mels=80).to(device)

step = 0

def align(emission, tokens):
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    alignments, scores = F.forced_align(emission, targets, blank=0)

    alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
    scores = scores.exp()  # convert back to probability
    return alignments, scores

def unflatten(list_, lengths):
    i = 0
    ret = []
    for l in lengths:
        ret.append(list_[i : i + l])
        i += l
    return ret
from function import fl_duration
for i,(text_batch,audio_batch) in enumerate(tqdm(loader)):
    x,s,_,x_lens,_,language = [tensor.to(device) for tensor in text_batch]
    # print(language)
    audio,y_lens = audio_batch[0].to(device),audio_batch[1].to(device)
    with torch.no_grad():
        y,_ = ae.encode(audio) 
        y_lens = torch.ceil(y_lens/16/64)

        melspec = melspec_transform(audio).squeeze(1) #B,T,80
        melspec = melspec.permute(0,2,1)#B,80,T

        prob_matrix = aligner(melspec,language)  # [batch_size, y_len, num_phonemes], probability 
        l = fl_duration(prob_matrix,x,x_max_len=x.shape[-1])
        # l = agd_duration(prob_matrix,x_max_len=x.shape[-1])
    
    # print(prob_matrix[0])

    # aligned_tokens = torch.argmax(prob_matrix[0],dim=1)
    # alignment_scores = torch.max(prob_matrix[0],dim=1).values

    # # print(aligned_tokens,alignment_scores)
    # plot_emission(prob_matrix[0])
    l = l[0]
    audio = audio[0]
    LABELS = {value: key for key, value in ipa_pho_dict.items()}
    aligned_tokens = x[0]
    tokens = []
    word_spans = {}
    curr = 0
    end = 0 
    start = 0
    buffer = ""
    for i,token in enumerate(aligned_tokens):
        token = token.item()
        print(f"{LABELS[token]}\t{l[i]:.2f}")
        if LABELS[token] == "sil":
            end = curr
            word_spans[buffer]=(start,end)
            buffer = ""
            start = curr+1
        else: 
            buffer += LABELS[token]
            curr += l[i].item()
    print(word_spans)

    for word in word_spans:
        start,end = word_spans[word]
        start *= 1024
        end *=1024
        crop = audio[:,start:end]
        print(word,start,end)
        save_audio(crop.detach().cpu(),48000,word,"./sample/split/")
    
    # for i, (ali, score) in enumerate(zip(aligned_tokens, alignment_scores)):
    #     print(f"{i:3d}:\t{ali:2d} [{LABELS[ali.item()]}], {score:.2f}")
    # # plt.show()
    # token_spans = F.merge_tokens(aligned_tokens, alignment_scores)
    # print("Token\tTime\tScore")
    # tokens = []
    # word_spans = {}
    # buffer = ""
    # start = 0
    # end = 0
    # for s in token_spans:
    #     print(f"{LABELS[s.token]}\t[{s.start:3d}, {s.end:3d})\t{s.score:.2f}")
    #     tokens.append(LABELS[s.token])
    #     if LABELS[s.token] == "sil":
    #         end = s.end
    #         word_spans[buffer]=(start,end)
    #         buffer = ""
    #         start = s.start
    #     else: 
    #         buffer += LABELS[s.token]
        

    # TRANSCRIPT = "".join(tokens)
    # TRANSCRIPT = TRANSCRIPT.split("sil")
    # TRANSCRIPT.remove("")
    # for word in word_spans:
    #     start,end = word_spans[word]
    #     start *= 1024
    #     end*=1024
    #     crop = audio[0][:,start:end]
    #     print(word,start,end)
    #     save_audio(crop.detach().cpu(),48000,word,"./sample/split/")
    # save_audio(audio[0].detach().cpu(),48000,"sentence","./sample/split/")

    # audio = audio[0]
    break
    
    
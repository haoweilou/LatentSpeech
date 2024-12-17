import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from params import params
# from tts import DurationAligner
from flow import AE
from function import loadModel,saveModel, agd_duration,fl_duration,force_alignment,duration_calculate
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
# bakertext = BakerText(normalize=False,start=0,end=500,path=f"{root}baker/",ipa=True)
# bakeraudio = BakerAudio(start=0,end=500,path=f"{root}baker/",return_len=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

aligner = ASR(80,len(ipa_pho_dict)+1).to(device)
aligner = loadModel(aligner,"aligner_en_600","./model/")
from function import calculate_l
# ljspeechaudio = LJSpeechAudio(start=0,end=13100,path=f"{root}LJSpeech/",return_len=True)
# ljspeechtext = LJSpeechText(start=0,end=13100,path=f"{root}LJSpeech/")
# l = calculate_l(aligner,
#                 ys=ljspeechaudio.audios,
#                 y_lens=ljspeechaudio.audio_lens,
#                 x=ljspeechtext.x,
#                 x_len=ljspeechtext.src_len)
# import json

# data = {index:duration for index,duration in enumerate(l)}
# data_str = json.dumps(data,indent=3)
# with open("./save/duration/LJSpeech.json","w") as f: 
#     f.write(data_str)



bakeraudio = BakerAudio(start=0,end=100,path=f"{root}baker/",return_len=True)
bakertext = BakerText(start=0,end=100,path=f"{root}baker/")
l = calculate_l(aligner,
                ys=bakeraudio.audios,
                y_lens=bakeraudio.audio_lens,
                x=bakertext.x,
                x_len=bakertext.src_len)
import json

data = {index:duration for index,duration in enumerate(l)}
data_str = json.dumps(data,indent=3)
with open("./save/duration/baker.json","w") as f: 
    f.write(data_str)


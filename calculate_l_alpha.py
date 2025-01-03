import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from params import params
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
import torchaudio
from torchaudio.pipelines import MMS_FA as bundle
import torch
import torch.nn as nn
import torchaudio.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = bundle.get_model()
model.to(device)


tokenizer = bundle.get_tokenizer()
aligner = bundle.get_aligner()
data_dict = bundle.get_dict()
print(bundle.get_dict())
# root = "/home/haoweilou/scratch/"
root = "L:/"
# bakertext = BakerText(normalize=False,start=0,end=500,path=f"{root}baker/",ipa=True)
# bakeraudio = BakerAudio(start=0,end=500,path=f"{root}baker/",return_len=True)
def align(emission, tokens):
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    alignments, scores = F.forced_align(emission, targets, blank=0)

    alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
    scores = scores.exp()  # convert back to probability
    return alignments, scores

LABELS = {data_dict[k]:k for k in data_dict }
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def compute_alignments(waveform: torch.Tensor, transcript):
    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
        token_spans = aligner(emission[0], tokenizer(transcript))
    return emission, token_spans
resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)

# ljspeechaudio = LJSpeechAudio(start=0,end=13100,path=f"{root}LJSpeech/",return_len=True)
# ljspeechtext = LJSpeechText(start=0,end=13100,path=f"{root}LJSpeech/")


# durations = []
# for i in tqdm(range(len(ljspeechaudio.audios))):

#     audio = ljspeechaudio.audios[i]
#     waveform = resampler(audio).unsqueeze(0)
#     #tomorrow conver the duration to time, 1 = 0.02 s and segment to see if the segmentation is accurate
#     transcript = ljspeechtext.english_sentence[i].lower().split()
#     tokenized_transcript = [data_dict[j] for j in ljspeechtext.english_sentence[i] if j != " "  ]

#     emission, token_spans = compute_alignments(waveform, transcript)
#     aligned_tokens, alignment_scores = align(emission[0][:,:waveform.shape[-1]].unsqueeze(0), tokenized_transcript)
#     duration = duration_calculate(emission.cpu(),torch.tensor(tokenized_transcript).unsqueeze(0),[len(tokenized_transcript)],[waveform.shape[-1]], max_x_len =len(tokenized_transcript))
#     duration = duration[0].tolist() # this is the 0.02s interval, need adjust to 1/47=0.022s
#     timestep = 0
#     new_duration = []
#     for index,d in enumerate(duration):
#         start_time = timestep*0.02
#         end_time = (timestep+duration[index])*0.02
#         diff = end_time-start_time
#         diff_int = math.ceil(diff*47)
#         new_duration.append(diff_int)

#     durations.append(new_duration)
# import json

# data = {index:duration for index,duration in enumerate(durations)}
# data_str = json.dumps(data,indent=3)
# with open("./save/duration/LJSpeech_ALPHA.json","w") as f: 
#     f.write(data_str)

bakeraudio = BakerAudio(start=0,end=10000,path="L:/baker/")
bakertext = BakerText(start=0,end=10000,path="L:/baker/",ipa=True)
from ipa import mandarin_chinese_to_alpha
hanzi = bakertext.hanzi

durations = []
for i in tqdm(range(len(bakeraudio.audios))):
    audio = bakeraudio.audios[i]
    waveform = resampler(audio).unsqueeze(0)
    #tomorrow conver the duration to time, 1 = 0.02 s and segment to see if the segmentation is accurate
    transcript = bakertext.hanzi[i]
    transcript,_ = mandarin_chinese_to_alpha(transcript)  
    while "|" in transcript: transcript.remove("|")
    tokenized_transcript = [data_dict[j] for j in transcript if j != " "  ]

    emission, token_spans = compute_alignments(waveform, transcript)
    aligned_tokens, alignment_scores = align(emission[0][:,:waveform.shape[-1]].unsqueeze(0), tokenized_transcript)
    duration = duration_calculate(emission.cpu(),torch.tensor(tokenized_transcript).unsqueeze(0),[len(tokenized_transcript)],[waveform.shape[-1]], max_x_len =len(tokenized_transcript))
    duration = duration[0].tolist() # this is the 0.02s interval, need adjust to 1/47=0.022s
    timestep = 0
    new_duration = []
    for index,d in enumerate(duration):
        start_time = timestep*0.02
        end_time = (timestep+duration[index])*0.02
        diff = end_time-start_time
        diff_int = math.ceil(diff*47)
        new_duration.append(diff_int)

    durations.append(new_duration)
import json

data = {index:duration for index,duration in enumerate(durations)}
data_str = json.dumps(data,indent=3)
with open("./save/duration/Baker_ALPHA.json","w") as f: 
    f.write(data_str)
# bakeraudio = BakerAudio(start=0,end=10000,path=f"{root}baker/",return_len=True)
# bakertext = BakerText(start=0,end=10000,path=f"{root}baker/",ipa=True)
# l = calculate_l(aligner,
#                 ys=bakeraudio.audios,
#                 y_lens=bakeraudio.audio_lens,
#                 x=bakertext.x,
#                 x_len=bakertext.src_len)
# import json

# data = {index:duration for index,duration in enumerate(l)}
# data_str = json.dumps(data,indent=3)
# with open("./save/duration/baker.json","w") as f: 
#     f.write(data_str)


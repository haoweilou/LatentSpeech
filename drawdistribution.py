import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Example DataFrame with phoneme durations
# data = {
#     "Phoneme": ["a", "a", "a", "b", "b", "c", "c", "c", "c", "d"],
#     "Duration": [0.1, 0.12, 0.15, 0.2, 0.18, 0.25, 0.3, 0.28, 0.27, 0.35],
# }

# df = pd.DataFrame(data)
from dataset import BakerText,LJSpeechText
import torch
from function import draw_duration_distribution
from ipa import idx_to_ipa,ipa_pho_dict
dataset1 = BakerText(path="C:/baker/",start=0,end=10000,ipa=True,no_sil=True)
dataset2 = LJSpeechText(path="C:/LJSpeech/",start=0,end=13100,no_sil=True)
xs = dataset2.x.tolist()
ls = dataset2.l.tolist()

xs = dataset1.x.tolist()+dataset2.x.tolist()
ls = dataset1.l.tolist()+dataset2.l.tolist()


phonemes = []
durations = []
from tqdm import tqdm
for x,l in tqdm(zip(xs,ls)):
    for phoneme,duration in zip(x,l):
        if phoneme == 0: 
            assert duration == 0
            break
        phonemes.append(phoneme)
        durations.append(duration)
phonemes = [idx_to_ipa[p] for p in phonemes]
for i in idx_to_ipa:
    if idx_to_ipa[i] not in phonemes:
        phonemes.append(idx_to_ipa[i])
        durations.append(0)

data = {
    "Phoneme": phonemes,
    "Duration": durations,
}

draw_duration_distribution(data,name="ljspeech_nosil")
# print(dataset.x,dataset.l)
from function import draw_phoneme_distribution
dataset1 = BakerText(path="C:/baker/",start=0,end=10000,ipa=True,no_sil=True)
dataset2 = LJSpeechText(path="C:/LJSpeech/",start=0,end=13100,no_sil=True)
xs = dataset2.x.tolist()
count = {i:0 for i in range(82)}
for x in xs:
    for phoneme in x: 
        if phoneme == 0:
            break
        count[phoneme] += 1

print(count)
numbers = [count[i] for i in range(82)]
data = {"Phoneme":[idx_to_ipa[i] for i in range(82)],"Count":numbers}
draw_phoneme_distribution(data,name="ljspeech_count_nosil")
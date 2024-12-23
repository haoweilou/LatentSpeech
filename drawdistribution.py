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
from function import draw_distribution
dataset = BakerText(path="C:/baker/",start=0,end=10000,ipa=True)
# dataset = LJSpeechText(path="C:/LJSpeech/",start=0,end=13100)

xs = dataset.x.tolist()
ls = dataset.l.tolist()
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
from ipa import idx_to_ipa
phonemes = [idx_to_ipa[p] for p in phonemes]
data = {
    "Phoneme": phonemes,
    "Duration": durations,
}

draw_distribution(data)
# print(dataset.x,dataset.l)
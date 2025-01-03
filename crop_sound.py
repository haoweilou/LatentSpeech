import json
import torchaudio
path = "./save/duration/LJSpeech_ALPHA.json"
with open(path,"r") as f: 
    data = f.read()
data = json.loads(data)
duration = data["0"]
print(sum(duration))

path = "./save/duration/LJSpeech.json"
with open(path,"r") as f: 
    data = f.read()
data = json.loads(data)
duration = data["0"]
print(sum(duration))
from dataset import LJSpeechText,LJSpeechAudio
dataset = LJSpeechText(start=0,end=10)
sentence = dataset.english_sentence[0]
audio_dataset = LJSpeechAudio(0,10,"D:/LJSpeech/")
waveform = audio_dataset.audios[0]
print(waveform.shape)
words = sentence.split()
# i = 0
# timestep = 0
# import os 
# files = os.listdir("./sample/split/")
# for file in files: 
#     os.remove("./sample/split/"+file)
# import math
# for word_idx,word in enumerate(words):
#     start = 0
#     end = 0
#     for j,char in enumerate(word):
#         print(char,timestep,"~",timestep+duration[i],timestep*0.02,"~",(timestep+duration[i])*0.02)
#         start_time = timestep*0.02
#         end_time = (timestep+duration[i])*0.02
#         diff = end_time-start_time
#         diff_int = int(diff*47)
#         print(diff_int,diff)
#         if j == 0:
#             start = timestep*0.02
#         elif j == len(word)-1:
#             end = (timestep+duration[i])*0.02

#         timestep += duration[i]
#         i += 1
#     print(word,start,"~",end)
#     start_index  = int(start * 48000)
#     end_index = int(end * 48000)
#     audio_clip = waveform[start_index:end_index]
#     torchaudio.save(f"./sample/split/{word_idx}_{word}.wav",audio_clip.unsqueeze(0),sample_rate=48000)

# # print(sentence)
# # print(duration)
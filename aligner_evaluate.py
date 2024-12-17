import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import ASR
from torch.utils.data import DataLoader
from dataset import BakerAudio,BakerText,LJSpeechAudio,LJSpeechText
from function import loadModel
from torchaudio.transforms import MelSpectrogram

root = "L:/"
# bakertext = BakerText(normalize=False,start=0,end=100,path=f"{root}baker/",ipa=True)


from ipa import ipa_pho_dict
from logger import Log
T = 50                  #Input Sequence Length, melspec length
# C = len(phoneme_set)+1  #Number of Phoneme Class, include blank, 87+1=88
C = len(ipa_pho_dict)+1   #include empty already
N = 16                  #Batch size
S = 128                 #Target sequence length of the longest target in batch (zero padding) Phoneme length
S_min = 5               #Minium sequence length 5, min Phoneme length
feature_type = "Melspec"
feature_dim = 16 if feature_type != "Melspec" else 80        #feature dim
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

aligner = ASR(input_dim=feature_dim,output_dim=C).to(device)
epoch = 600
aligner = loadModel(aligner,f"aligner_en_{epoch}","./model/")
bakeraudio = BakerAudio(start=0,end=10,path=f"{root}baker/",return_len=True)

def collate_fn(batch):
    text_batch, audio_batch = zip(*batch)
    text_batch = [torch.stack([item[i] for item in text_batch]) for i in range(len(text_batch[0]))]
    audio_batch = bakeraudio.collate(audio_batch)
    return text_batch, audio_batch


chinese = True
number = 1000
if chinese:
    bakertext = BakerText(start=number,end=number+10,path=f"{root}baker/",ipa=True)
    bakeraudio = BakerAudio(start=number,end=number+10,path=f"{root}baker/",return_len=True)
    loader = DataLoader(dataset=list(zip(bakertext, bakeraudio)), collate_fn=collate_fn, batch_size=32, shuffle=False)
else: 
    ljspeechtext = LJSpeechText(start=number,end=number+10,path=f"{root}LJSpeech/")
    ljspeechaudio = LJSpeechAudio(start=number,end=number+10,path=f"{root}LJSpeech/",return_len=True)
    loader = DataLoader(dataset=list(zip(ljspeechtext, ljspeechaudio)), collate_fn=collate_fn, batch_size=32, shuffle=False)

melspec_transform = MelSpectrogram(sample_rate=48000,n_fft=1024,hop_length=1024,n_mels=80).to(device)
import matplotlib.pyplot as plt
from align_fig import *
from function import duration_calculate
# ys = 
# ljspeechtext.calculate_l(aligner,ys=ljspeechaudio.audios,y_lens=ljspeechaudio.audio_lens)

for i,(text_batch,audio_batch) in enumerate(loader):
    x,s,_,x_lens,_,language = [tensor.to(device) for tensor in text_batch]
    audio,y_lens = audio_batch[0].to(device),audio_batch[1].to(device)
    with torch.no_grad():
        melspec = melspec_transform(audio).squeeze(1) #B,T,80
        melspec = melspec.permute(0,2,1)#B,80,T
        y_lens = torch.ceil(y_lens/16/64).long()
    y = melspec           #batch size, phoneme length
    x_f = aligner(y,language)  # [batch_size, seq_len, num_phonemes]   
    print(x_f.shape)         
    emission = torch.log_softmax(x_f,dim=-1) # [seq_len, batch_size, num_phonemes]
    # l = duration_calculate(emission.cpu(),x.cpu(),x_lens.cpu(),y_lens.cpu(),max_x_len = x.shape[-1])
    emissions = emission.detach().cpu()
    emission = emissions[0]
    # plot(emission)
    print(emission)
    if chinese:
        transcript = bakertext.ipa_sentences[i]
    else:
        transcript = ljspeechtext.ipa_sentences[i]
    dictionary = ipa_pho_dict

    tokens = [dictionary[c] for c in transcript]
    if chinese:
        print(bakertext.hanzi[i])
    else:
        print(ljspeechtext.english_sentence[i])
    print(list(zip(transcript, tokens)))
    trellis = get_trellis(emission, tokens)
    path = backtrack(trellis, emission, tokens)
    segments = merge_repeats(path,transcript)

    from function import force_alignment
    segments = force_alignment(emission,tokens)
    for seg in segments:
        print(seg)
    
    # plot_trellis(trellis)

    print(trellis)

    
    segments = merge_repeats(path,transcript)
    for p in path:
        print(p)

    # plot_trellis_with_path(trellis,path)
    
    segments = merge_repeats(path,transcript)
    for seg in segments:
        print(seg)
    
    # plot_trellis_with_segments(trellis, segments, transcript)
    print("")
    word_segments = merge_words(segments)
    for word in word_segments:
        print(word)

    plot_alignments(
        trellis,
        segments,
        word_segments,
        audio[0].cpu(),
    )
    plt.show()
    waveform = audio[0]
    if chinese: 
        sentece_englisht = bakertext.hanzi[i]
    else:
        sentece_englisht = ljspeechtext.english_sentence[i]
    from function import save_audio
    import os 
    exist_files = os.listdir("./sample/split/")
    for f in exist_files:
        os.remove(f"./sample/split/{f}")
    def display_segment(i):
        ratio = waveform.size(1) / trellis.size(0)
        word = word_segments[i]
        x0 = int(ratio * word.start)
        x1 = int(ratio * word.end)
        print(f"{word.label} ({word.score:.2f}): {x0 /48000:.3f} - {x1 / 48000:.3f} sec")
        segment = waveform[:, x0:x1]
        if chinese:
            word = sentece_englisht[i]
        else:
            word = sentece_englisht.split(" ")[i]
        save_audio(segment.cpu(), sample_rate=48000,name=f"{i}_{word}",root="./sample/split/")

    for i in range(len(word_segments)):
        display_segment(i)
    break

# for i,sentence in enumerate(ljspeechtext.english_sentence):
#     print(sentence,"\n",ljspeechtext.ipa_sentences[i],"\n",ljspeechtext.x[i])

#     print(len(ljspeechtext.ipa_sentences[i]),len(ljspeechtext.x[i]))
#     print(list(zip(ljspeechtext.ipa_sentences[i], ljspeechtext.x[i])))
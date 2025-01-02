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
from typing import List

import IPython
import matplotlib.pyplot as plt
def compute_alignments(waveform: torch.Tensor, transcript):
    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
        token_spans = aligner(emission[0], tokenizer(transcript))
    return emission, token_spans

# Compute average score weighted by the span length
def _score(spans):
    return sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans)


def plot_alignments(waveform, token_spans, emission, transcript, sample_rate=bundle.sample_rate):
    ratio = waveform.size(1) / emission.size(1) / sample_rate

    fig, axes = plt.subplots(2, 1)
    axes[0].imshow(emission[0].detach().cpu().T, aspect="auto")
    axes[0].set_title("Emission")
    axes[0].set_xticks([])

    axes[1].specgram(waveform[0], Fs=sample_rate)
    for t_spans, chars in zip(token_spans, transcript):
        t0, t1 = t_spans[0].start, t_spans[-1].end
        axes[0].axvspan(t0 - 0.5, t1 - 0.5, facecolor="None", hatch="/", edgecolor="white")
        axes[1].axvspan(ratio * t0, ratio * t1, facecolor="None", hatch="/", edgecolor="white")
        axes[1].annotate(f"{_score(t_spans):.2f}", (ratio * t0, sample_rate * 0.51), annotation_clip=False)

        for span, char in zip(t_spans, chars):
            t0 = span.start * ratio
            axes[1].annotate(char, (t0, sample_rate * 0.55), annotation_clip=False)

    axes[1].set_xlabel("time [second]")
    fig.tight_layout()

def preview_word(waveform, spans, num_frames, transcript, sample_rate=bundle.sample_rate):
    ratio = waveform.size(1) / num_frames
    x0 = int(ratio * spans[0].start)
    x1 = int(ratio * spans[-1].end)
    print(f"{transcript} ({_score(spans):.2f}): {x0 / sample_rate:.3f} - {x1 / sample_rate:.3f} sec")
    segment = waveform[:, x0:x1]
    return IPython.display.Audio(segment.numpy(), rate=sample_rate)

from dataset import LJSpeechText,LJSpeechAudio
dataset = LJSpeechText(start=0,end=100)
audio_dataset = LJSpeechAudio(start=0,end=100,path="D:/LJSpeech/")
waveform = audio_dataset.audios[0].unsqueeze(0)
print(dataset.english_sentence[0])
print(bundle.sample_rate)
import torchaudio
resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)

resampled_waveform = resampler(waveform)

transcript = dataset.english_sentence[0].split()
tokens = tokenizer(transcript)

IPython.display.Audio(waveform, rate=bundle.sample_rate)
emission, token_spans = compute_alignments(waveform, transcript)
num_frames = emission.size(1)
def align(emission, tokens):
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    alignments, scores = F.forced_align(emission, targets, blank=0)

    alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
    scores = scores.exp()  # convert back to probability
    return alignments, scores

tokenized_transcript = [data_dict[i] for i in dataset.english_sentence[0] if i != " "  ]
aligned_tokens, alignment_scores = align(emission[0][:,:waveform.shape[-1]].unsqueeze(0), tokenized_transcript)

LABELS = {data_dict[k]:k for k in data_dict }

from function import   duration_calculate
ratio = waveform.size(1) / emission.size(1) / 16000
print(data_dict)
print(emission.shape,waveform.shape,ratio)    
l = duration_calculate(emission.cpu(),torch.tensor(tokenized_transcript).unsqueeze(0),[len(tokenized_transcript)],[waveform.shape[-1]], max_x_len =len(tokenized_transcript))
print(len(l[0]),len(tokenized_transcript))
# plot_alignments(waveform, token_spans, emission, transcript)
print(dataset.english_sentence[0],len(dataset.english_sentence[0]))
no_space = "".join(dataset.english_sentence[0].split())
print(no_space,len(no_space))
print("Raw Transcript: ", transcript)
print("Normalized Transcript: ", transcript)
# print(token_spans)

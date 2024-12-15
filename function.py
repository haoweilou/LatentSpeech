import torch
import torch.utils
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from pypinyin import pinyin, Style
def saveLog(log_record:dict,name="log",root="./save/log/"):
    log = pd.DataFrame(log_record)
    log.to_csv(f"{root}{name}")

def load_audio(file_path):
    # Load audio file
    waveform, sample_rate = torchaudio.load(file_path)
    
    return waveform, sample_rate

def melspectrogram(waveform, sample_rate, n_mels=80, n_fft=1024):
    # Create a MelSpectrogram transform
    mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft
    )
    
    # Apply the transform to the waveform
    mel_spectrogram = mel_spectrogram_transform(waveform)
    mel_spectrogram = torch.transpose(mel_spectrogram,1,2)
    return mel_spectrogram

def resample(waveform,  target_rate,start_rate = 16*1000):
    # Resample the waveform to the target sample rate
    waveform_resampled = torchaudio.transforms.Resample(
        orig_freq=start_rate,
        new_freq=target_rate
    )(waveform)
    return waveform_resampled

def slide_window(sequence,window_size=100,step=100):
    if len(sequence.shape) > 1: sequence = sequence[0]
    output = []
    for start in range(0, len(sequence) - window_size + 1, step):
        segment = sequence[start:start+step]
        output.append(segment)
    
    return np.array([output])

def saveModel(model,name,root="./save/model"):
    path = f"{root}/{name}.pth" 
    torch.save(model.state_dict(), path) 

def learning_rate(d_model=256,step=1,warmup_steps=400):
    return (1/math.sqrt(d_model)) * min(1/math.sqrt(step),step*warmup_steps**-1.5)

def loadModel(model,name,root="/home/haoweilou/scratch/model/denoise", strict=True):
    path = f"{root}/{name}.pth" 
    state_dict = torch.load(path)
    model.load_state_dict(state_dict, strict=strict)
    # filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("upsampler")}
    # model.load_state_dict(filtered_state_dict, strict=strict)
    model.eval()
    return model

def save_audio(audio,sample_rate,name="",root="./sample/"):
    torchaudio.save(f'{root}/{name}.wav', audio, sample_rate)

def draw_wave(audio,name="Test",root="./fig/"):
    #audio: t
    t = torch.linspace(0, audio.shape[-1], audio.shape[-1])
    plt.figure(figsize=(10, 8))
    # # Convert the tensors to numpy arrays for plotting
    t_np = t.numpy()
    y_np = audio.numpy()
    # # # Plot the sine wave
    plt.plot(t_np, y_np)
    plt.ylim(-1,1)

    plt.title(name)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.savefig(f"{root}{name}")
    plt.clf()


def draw_dot(datas:list,label:list,color=["blue","red","green","orange"],name="dot",root="./fig/"):
    plt.figure()
    for i,data in enumerate(datas): 
        # Plot for the 4600x64 embedding
        plt.scatter(data[:, 0], data[:, 1], s=8, color=color[i],label=label[i], alpha=0.4)

    # Plot for the 512x64 embedding

    plt.title(name)
    plt.legend()
    plt.savefig(f"{root}{name}", bbox_inches='tight', pad_inches=0)
    
    # Display the heatmap
    plt.clf()

def draw_heatmap(data, vmin=0, vmax=1, color_map='viridis',name="Test",root="./fig/"):
    
    data = np.array(data)
    vmin = data.min()
    vmax = data.max()
    # data = (data - vmin) / (vmax - vmin)
    # vmin = 0
    # vmax = 1
    plt.figure(figsize=(10, 8))
    
    heatmap = plt.imshow(data,vmin=vmin, vmax=vmax, cmap=color_map, aspect='auto')
    # cbar = plt.colorbar(heatmap)
    # cbar.ax.set_ylabel('Intensity')
    
    # Remove axis labels and ticks
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)



    # plt.title(name)

    plt.savefig(f"{root}{name}", bbox_inches='tight', pad_inches=0)
    
    # Display the heatmap
    plt.clf()

from PIL import Image

def gif_image(image_list,name,root="./fig/",duration=500):
    # Open the images
    images = [Image.open(image) for image in image_list]
    
    # Save as GIF
    images[0].save(
        f"{root}{name}.gif",
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )

import json
def phone_to_phone_idx(pinyin:list):
    with open("./save/cache/phoneme.json","r") as f: 
        data = f.read()
        data = json.loads(data)
    phoidxdict = data["phoneme"]
    phoidxdict = {p:i for i, p in enumerate(phoidxdict)}
    with open("./save/cache/pinyin.json","r") as f:
        data = f.read()
        pinyin_dict = json.loads(data)
    phone_list = []
    tone_list = []
    for py in pinyin:
        if py == "sil":
            tone = 0
        else: 
            tone = int(py[-1])
            py = py[:-1]
        phone = pinyin_dict[py]
        tone = [0]*(len(phone)-1) + [tone]
        phone_list.append(phone)
        tone_list.append(tone)

    phone = [item for sublist in phone_list for item in sublist]
    tone = [item for sublist in tone_list for item in sublist]
    phone = [i if not i[-1].isdigit() else i[:-1] for i in phone]
    phone_idx = [phoidxdict[p]+1 for p in phone]
    return phone_idx, tone


def hanzi_to_pinyin(hanzi_string):
    # Convert Hanzi to Pinyin
    pinyin_list = pinyin(hanzi_string, style=Style.TONE3)
    # Join the pinyin with spaces
    output = [item[0] if item[0][-1].isdigit() else item[0]+"5" for item in pinyin_list]
    return ["sil"]*1+output+["sil"]*1

def hidden_to_audio(hidden):
    #hidden: B,T,C
    from ae import VQAE_Audio,AE
    from params import params
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ae = AE(params).to(device)
    ae = loadModel(ae,"qae_200","./model")
    # ae = VQAE_Audio(params,64,2048).to(device)
    # ae = loadModel(ae,"vqae_audio","./model")

    hidden = hidden.reshape(1,-1,20,ae.num_channel) # [T,featDim,channel]
    hidden = hidden.permute(0,3,1,2)
    # hidden = torch.transpose(hidden,1,2)[0,:,:].unsqueeze(0)
    audio = ae.decode_inference(hidden,1,hidden.shape[2])
    audio = torch.clamp(audio, -1.0, 1.0)
    return audio

def learning_rate(d_model=256,step=1,warmup_steps=400):
    return (1/math.sqrt(d_model)) * min(1/math.sqrt(step),step*warmup_steps**-1.5)

def collapse_and_duration(phoneme_tensor):
    phoneme_array = phoneme_tensor.detach().cpu().numpy()
    # Initialize variables
    collapsed_list = []
    current_phoneme = phoneme_array[0]
    duration = 1

    # Loop through the tensor elements
    for phoneme in phoneme_array[1:]:
        if phoneme == current_phoneme or phoneme == 0:
            current_phoneme = phoneme
            duration += 1
        else:
            collapsed_list.append(duration)
            current_phoneme = phoneme
            duration = 1

    # Append the last phoneme and its duration
    collapsed_list.append(duration)

    return collapsed_list

# def collapse_and_duration(phoneme_tensor):
#     phoneme_array = phoneme_tensor.detach().cpu().numpy()
#     # Initialize variables
#     collapsed_list = []
#     current_phoneme = phoneme_array[0]
#     duration = 1

#     # Loop through the tensor elements
#     for phoneme in phoneme_array[1:]:
#         if phoneme == current_phoneme or phoneme == 0:
#             current_phoneme = phoneme
#             duration += 1
#         else:
#             collapsed_list.append(duration)
#             current_phoneme = phoneme
#             duration = 1

#     # Append the last phoneme and its duration
#     collapsed_list.append(duration)

#     return collapsed_list

def plot_pqmf_bands(audio, sr, pqmf_model, num_bands):
    # Pass the audio through the PQMF forward process
    pqmf_bands = pqmf_model(audio)  # Add batch dimension
    time = torch.linspace(0, audio.shape[-1] / sr , audio.shape[-1] // num_bands)

    # Plot each band
    fig, axs = plt.subplots(num_bands, 1, figsize=(10, 2 * num_bands))
    for i in range(num_bands):
        axs[i].plot(time, pqmf_bands[0, i, :].cpu().numpy(), label=f"Band {i + 1}")
        axs[i].set_ylabel("Amplitude")
        axs[i].legend(loc="upper right")
    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()
    plt.clf()


# Function to compute the Short-Time Fourier Transform (STFT) for batch inputs
def compute_stft_batch(audio, n_fft=1024, hop_length=None):
    if hop_length is None:
        hop_length = n_fft // 4
    batch_size, _, time_steps = audio.shape
    audio = audio.view(batch_size, time_steps)  # Remove channel dimension for STFT
    stft = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True
    )
    return stft

# Function to compute the Inverse STFT for batch inputs
def compute_istft_batch(stft, n_fft=1024, hop_length=None):
    if hop_length is None:
        hop_length = n_fft // 4
    batch_size, freq_bins, frames = stft.shape[:3]
    audio = torch.istft(
        stft,
        n_fft=n_fft,
        hop_length=hop_length
    )
    return audio.view(batch_size, 1, -1)  # Add back channel dimension

# Spectral denoising using noise gating for batch inputs
def spectral_denoise(audio, sr, noise_reduction_factor=10.0, n_fft=1024, hop_length=None):
    # Compute the STFT
    stft = compute_stft_batch(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = torch.abs(stft)
    phase = torch.angle(stft)

    # Estimate noise profile (average magnitude from the first few frames for each batch)
    noise_profile = magnitude[:, :, :10].mean(dim=2, keepdim=True)  # First 10 frames

    # Apply spectral gating (reduce noise)
    reduced_magnitude = torch.relu(magnitude - noise_reduction_factor * noise_profile)

    # Reconstruct the STFT with the reduced magnitude
    denoised_stft = reduced_magnitude * torch.exp(1j * phase)
    denoised_audio = compute_istft_batch(denoised_stft, n_fft=n_fft, hop_length=hop_length)

    return denoised_audio

def agd_duration(prob_matrix,x_max_len=None):
    #phon_prob [batch_size, y_len, num_phonemes]
    #  
    prob_matrix = torch.argmax(prob_matrix,dim=2) # [batch_size, y len]
    l = [collapse_and_duration(i) for i in prob_matrix] 
    l = pad_sequence([torch.tensor(i) for i in l],batch_first=True,padding_value=0).to(prob_matrix.device)
    if x_max_len is not None: 
        l = F.pad(l,(0, x_max_len - l.shape[-1]),value=0)
    return l


def max_path(value, mask, max_neg_val=-np.inf):
    """ Numpy-friendly version. It's about 4 times faster than torch version.
    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]
    """
    value = value * mask

    device = value.device
    dtype = value.dtype
    value = value.cpu().detach().numpy()
    mask = mask.cpu().detach().numpy().astype(np.bool)
    
    b, t_x, t_y = value.shape
    direction = np.zeros(value.shape, dtype=np.int64)
    v = np.zeros((b, t_x), dtype=np.float32)
    x_range = np.arange(t_x, dtype=np.float32).reshape(1,-1)
    for j in range(t_y):
        v0 = np.pad(v, [[0,0],[1,0]], mode="constant", constant_values=max_neg_val)[:, :-1]
        v1 = v
        max_mask = (v1 >= v0)
        v_max = np.where(max_mask, v1, v0)
        direction[:, :, j] = max_mask
        
        index_mask = (x_range <= j)
        v = np.where(index_mask, v_max + value[:, :, j], max_neg_val)
    direction = np.where(mask, direction, 1)
        
    path = np.zeros(value.shape, dtype=np.float32)
    index = mask[:, :, 0].sum(1).astype(np.int64) - 1
    index_range = np.arange(b)
    for j in reversed(range(t_y)):
        path[index_range, index, j] = 1
        index = index + direction[index_range, index, j] - 1
    path = path * mask.astype(np.float32)
    path = torch.from_numpy(path).to(device=device, dtype=dtype)
    return path

def monotonic_alignment(prob_matrix,sequence:list):
    N,T = prob_matrix.shape#N number of phoneme, T number of frame
    L = len(sequence)#L is sequence length
    output = np.zeros((L,T))
    i,j = 0,0

    for j in range(T):
        prob = prob_matrix[:,j]
        curr_p = sequence[i]
        next_p = sequence[i+1]
        curr_prob = prob[curr_p]
        next_prob = prob[next_p]
        if curr_prob >= next_prob:
            output[i][j] = 1
        else:
            i+=1
            output[i][j] = 1
        if i == L-1: 
            for j_ in range(j,T):
                output[i][j_] = 1
            break
        # print(prob, len(prob))
       
        # break
    l = np.sum(output,axis=1)
    # while i < L-1:
    #     curr = sequence[i]
    #     next = curr+1
    #     prob_curr = prob_matrix[curr][j]
    #     prob_next = prob_matrix[next][j]
    #     if prob_curr >= prob_next:
    #         output[i][j] = 1
    #     else: 
    #         output[i][j] = 1
    #         i += 1
    #     j += 1
        
        


    return l

def mas_duration(prob_matrix,x,x_max_len=None):
    #b,t_y,t_x
    prob_matrix = prob_matrix.permute(0,2,1)
    #t_x, t_y
    prob_matrix = torch.softmax(prob_matrix,dim=2)
    prob_matrix_np = prob_matrix.detach().cpu().numpy()
    # matrix_1 = monotonic_alignment(prob_matrix[0])
    ls = []
    for i in range(len(prob_matrix)):
        l = monotonic_alignment(prob_matrix_np[i],x[i].detach().cpu().numpy())
        ls.append(l)
    ls = np.array(ls)
    ls = torch.tensor(ls).long()
    print(ls.shape)
    # print(prob_matrix[0].shape,x[0].shape,l.shape)
    return ls


def greedy_decode(logits,x_max_len=None):
    # Get the most likely class at each time step
    # print(logits.shape)
    predictions = torch.argmax(logits, dim=2)  # Shape: (batch_size, time_steps)

    alignments = []
    for pred in predictions:
        alignment = []
        prev = None
        for p in pred:
            if p != prev and p != 0:  # Remove blanks and repeated tokens
                alignment.append(p.item())
            prev = p
        alignments.append(alignment)
        print(len(alignment))
    alignments = pad_sequence([F.pad(torch.tensor(a),(0,x_max_len-len(a)),value=0) for a in alignments]).long().T
    return alignments



def fl_duration(prob_matrix,target,x_max_len=None):
    import torchaudio.functional as F
    import torch.nn.functional as nnF

    from ipa import ipa_pho_dict
    #phon_prob [batch_size, y_len, num_phonemes]
    #  
    # prob_matrix = torch.argmax(prob_matrix,dim=2) # [batch_size, y len]
    xs = [seq[seq != 0] for seq in target]
    l_out = []
    for pm,x in zip(prob_matrix,xs):
        # print(pm.shape,x.shape)
        alignments, scores = F.forced_align(pm.unsqueeze(0), x.unsqueeze(0), blank=0)
        # print(alignments,scores)
        token_spans = F.merge_tokens(alignments[0], scores[0])
        # print(token_spans)
        l = []
        for i,s in enumerate(token_spans):
            if i == len(token_spans) -1:
                s.end = pm.shape[0]
            elif i != len(token_spans) -1 and s.end != token_spans[i+1].start:
                if s.token == ipa_pho_dict["sil"]:
                    s.end = token_spans[i+1].start
                elif token_spans[i+1].token == ipa_pho_dict["sil"]:
                    token_spans[i+1].start = s.end
                else: 
                    s.end = token_spans[i+1].start
            # print(s)
            l.append(s.end-s.start)
        l_out.append(torch.tensor(l))
    l_out = pad_sequence(l_out,batch_first=True)
    l_out = nnF.pad(l_out,(0,target.shape[-1]-l_out.shape[-1]),value=0)
    return l_out
        # break
            
   
    # if x_max_len is not None: 
    #     l = F.pad(l,(0, x_max_len - l.shape[-1]),value=0)
    # return l

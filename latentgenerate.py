from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import VQAESeq,AE
from params import params
from function import loadModel,save_audio,draw_wave,draw_heatmap,draw_dot
from dataset import BakerAudio,pad16,LJSpeechAudio
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import DataLoader

# num = 500
# model_name = "ae9k2set"
num = 210
model_name = "vqae16"
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
model = VQAESeq(params,embed_dim=16).to(device)
model = loadModel(model,f"{model_name}_{num}","./model/")
dataset = BakerAudio(0,10,"L:/baker/")
# dataset = LJSpeechAudio(0,13100,"L:/LJSpeech/")
loader = DataLoader(dataset,batch_size=32,collate_fn=dataset.collate,drop_last=False,shuffle=False)
i = 0
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
with torch.no_grad():
    for audio in tqdm(loader):
        audio = audio.to(device)
        time_steps = audio.shape[-1]
        pad_amount = (16 - (time_steps % 16)) % 16
        if pad_amount > 0:audio = F.pad(audio, (0, pad_amount))
        z_q = model.encode_inference(audio)#64,20,T
        a = model.decode_inference(z_q)[0]
        save_audio(a.to("cpu"),48000,f"fake")
        

        codebook = model.spec_ae.vq_layer.embedding.weight.cpu()
        b,embed,h,w = z_q.shape
        z_q = torch.reshape(z_q,(b,embed,-1)).cpu()
        z_q = z_q.permute(0, 2, 1).contiguous().view(-1, 16)
        distances = (torch.sum(z_q ** 2, dim=1, keepdim=True) +
                    torch.sum(codebook ** 2, dim=1) -
                    2 * torch.matmul(z_q, codebook.t()))
        print(distances[0][389],torch.argmin(distances,dim=1)[0])
        print(z_q.shape,codebook.shape)
        combined = torch.concat((z_q[:1024],codebook),dim=0)
        combined = tsne.fit_transform(combined)

        # draw_dot([combined[:4400],combined[4400:8800],combined[8800:]],["z_q","z","codebook"],name="all feature")
        draw_dot([combined[:1024],combined[1024:]],["z_q","codebook"],name="z_q and codebook")
        # draw_dot([combined[:4400],combined[4400:8800]],["z_q","z"],name="z_q and z")
        # draw_dot([combined[4400:8800],combined[8800:]],["z","codebook"],name="z and codebook")

        # print(combined.shape)
        break
        codebook = tsne.fit_transform(codebook)


        print(codebook.shape)
        
        break
        batch_size = z_q.shape[0]
        for j in range(batch_size):
            data = z_q[j].cpu()
            torch.save(data,f"L:/LJSpeech/Latent/{i}")
            i += 1


# with torch.no_grad():
#     for i in range(10):
#         z_q = torch.load(f"L:/Baker/Latent/{i}").to("cuda")
#         z_q = torch.unsqueeze(z_q,0)
#         # z_q = model.encode(audio)
#         print(z_q.shape)
        
#         # draw_heatmap(codebook[:64],name="codebook")
#         break
#         latent_temp = model.mapper(melspec.squeeze(1))

#         audio = model.decode(latent_temp)
#         audio = model.pqmf.inverse(audio)[0]
#         print(audio.shape)
#         save_audio(audio.to("cpu"),48000,f"{i}")
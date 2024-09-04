import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim as optim
from model import VQAE,AE
from params import params
from function import loadModel,save_audio,draw_wave,draw_heatmap
from sklearn.manifold import TSNE
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_name = "ae"
model = AE(params).to(device)
model = loadModel(model,f"{model_name}","./model/")
data_list = []
labels = []
for i in tqdm(range(0,722)):
# for i in tqdm([0,1,2,3,4,700,701,702,703,704]):
    data = torch.load(f"./latent/{i}")
    data = data.permute(0, 2, 1).contiguous().view(-1, 16)
    data_list.append(data)

    if i <= 313:
        labels.extend([0] * data.size(0))  # Label all points from Chinese data as 0
    else:
        labels.extend([1] * data.size(0))
from sklearn.decomposition import PCA

combined_data = torch.cat(data_list, dim=0)
combined_data_np = combined_data.cpu().numpy()
labels_np = np.array(labels)
# tsne = TSNE(n_components=2,n_jobs=4, random_state=42)
# results = tsne.fit_transform(combined_data_np)
from sklearn.cluster import KMeans,MiniBatchKMeans
num_clusters = 2048
mini_batch_kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=10000, random_state=42)
mini_batch_kmeans.fit(combined_data_np)

# kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10, max_iter=300)
# kmeans.fit(combined_data_np)
codebook = mini_batch_kmeans.cluster_centers_
print(codebook.shape)

pca = PCA(n_components=2)
results = pca.fit_transform(combined_data_np)
pca_codebook_results = pca.transform(codebook)  # Transform the codebook using the same PCA

plt.scatter(results[:, 0], results[:, 1], s=2, alpha=0.5, label='Chinese', c='grey')

plt.scatter(pca_codebook_results[:, 0], pca_codebook_results[:, 1], s=2, alpha=0.5, label='center', c='red')
# Separate t-SNE results by label
# tsne_chinese = results[labels_np == 0]
# tsne_english = results[labels_np == 1]

# # Plot Chinese data points in one color
# plt.scatter(tsne_chinese[:, 0], tsne_chinese[:, 1], s=2, alpha=0.5, label='Chinese', c='blue')

# # # Plot English data points in another color
# plt.scatter(tsne_english[:, 0], tsne_english[:, 1], s=2, alpha=0.5, label='English', c='blue')



# # Plot t-SNE results
plt.figure(figsize=(10, 8))
plt.title('t-SNE Visualization of Latent Space')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()
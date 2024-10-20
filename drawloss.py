import matplotlib.pyplot as plt
import pandas as pd 
dataframe = pd.read_csv("./log/loss_vqae_audio")
# Extract the 'epochs' column from the DataFrame
epochs = dataframe.iloc[:,0]

# Similarly, extract other relevant columns from the DataFrame
total_loss = dataframe['total_loss']
audio_loss = dataframe['audio_loss']

vq_loss = dataframe['vq_loss']

plt.figure(figsize=(10, 6))
plt.plot(epochs, total_loss, label='Total Loss')
plt.plot(epochs, audio_loss, label='Audio Loss')
# spectral_loss = dataframe['spectral_loss']
# plt.plot(epochs, spectral_loss, label='Spectral Loss')
plt.plot(epochs, vq_loss, label='VQ Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Graph')
plt.legend()
plt.grid(True)
plt.show()
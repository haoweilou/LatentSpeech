import torch
import torchaudio
import matplotlib.pyplot as plt
from ae import PQMF  # Replace with the actual module name where your PQMF class is implemented

# Function to plot PQMF bands


# Main function to process and plot audio
def main(audio_path, attenuation, n_band):
    # Load the audio file
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.unsqueeze(0)
    waveform = waveform.mean(0)  # Convert to mono if necessary

    print(f"Loaded audio with sample rate: {sr}")
    print(f"Audio shape: {waveform.shape}")

    # Initialize PQMF
    pqmf_model = PQMF(attenuation=attenuation, n_band=n_band, polyphase=True)
    # Plot PQMF bands
    plot_pqmf_bands(waveform, sr, pqmf_model, n_band)

# Replace 'your_audio_file.wav' with your actual audio file path
if __name__ == "__main__":
    audio_file_path = "./sample/real.wav"  # Replace with the path to your audio file
    attenuation = 100  # Example attenuation
    n_band = 16  # Number of bands
    main(audio_file_path, attenuation, n_band)

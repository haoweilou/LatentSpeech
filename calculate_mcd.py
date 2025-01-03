import os
import numpy as np
import librosa
from scipy.spatial.distance import euclidean,cosine
from tqdm import tqdm

def calculate_mcd(mfcc1, mfcc2):
    """Calculate the Mel Cepstral Distortion (MCD) between two MFCC matrices using cosine distance."""
    min_len = min(mfcc1.shape[1], mfcc2.shape[1])
    mfcc1, mfcc2 = mfcc1[:, :min_len], mfcc2[:, :min_len]

    mcd = []
    for i in range(min_len):
        mcd.append(euclidean(mfcc1[:, i], mfcc2[:, i]))

    return 10 / np.log(10) * np.mean(mcd)


def process_folders(folder1, folder2):
    """Calculate MCD for all audio files in two folders."""
    audio_files = os.listdir(folder1)
    mcd_values = []

    for file_name in tqdm(audio_files):
        file1 = os.path.join(folder1, file_name)
        file2 = os.path.join(folder2, file_name)

        if os.path.exists(file2):
            # Load audio
            y1, sr1 = librosa.load(file1, sr=None)
            y2, sr2 = librosa.load(file2, sr=None)

            # Ensure both files have the same sample rate
            if sr1 != sr2:
                raise ValueError(f"Sample rates do not match for {file_name}")

            # Extract MFCCs
            mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13)
            mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=13)

            # Calculate MCD
            mcd = calculate_mcd(mfcc1, mfcc2)
            mcd_values.append(mcd)
        else:
            print(f"File {file_name} not found in both folders.")

    return mcd_values


if __name__ == "__main__":
    folder1 = "L:/evaluate/StyleSpeech2_FF_18K_500"  # Replace with your folder path
    # folder1 = "L:/evaluate/FastSpeech_18K_500"  # Replace with your folder path
    # folder1 = "L:/evaluate/StyleSpeech_18k_500"  # Replace with your folder path
    folder2 = "L:/evaluate/real_18k"  # Replace with your folder path

    mcd_values = process_folders(folder1, folder2)

    if mcd_values:
        mean_mcd = np.mean(mcd_values)
        std_mcd = np.std(mcd_values)

        print(f"Mean MCD: {mean_mcd:.2f}")
        print(f"Standard Deviation of MCD: {std_mcd:.2f}")
    else:
        print("No MCD values calculated.")
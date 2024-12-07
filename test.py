from fastdtw import fastdtw

# Example sequences
phoneme_features = [[1, 2, 3]]  # Pseudo phoneme embeddings
audio_features = [[1, 1, 2, 2, 3, 3]]  # Pseudo audio embeddings

_, path = fastdtw(phoneme_features, audio_features)
# Compute durations from path
durations = [0] * len(phoneme_features[0])
for p, a in path:
    durations[p] += 1

print("Durations:", durations)
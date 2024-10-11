import torch.nn as nn
from typing import Optional, Sequence
import torch.nn.functional as F

class SpeechRecognitionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SpeechRecognitionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, 128, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 * 2, output_dim)  # Bidirectional LSTM doubles the hidden size

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x
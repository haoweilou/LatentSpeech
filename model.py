import torch
import torch.nn as nn

class ASR(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.language_embed = nn.Embedding(2,80)
        self.lstm = nn.LSTM(input_dim, 128, num_layers=4, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 * 2, output_dim)  # Bidirectional LSTM doubles the hidden size

    def fused_add_tanh_sigmoid_multiply(self,x,g):
        in_act = x + g
        t_act = torch.tanh(in_act)
        s_act = torch.sigmoid(in_act)
        acts = t_act * s_act
        return acts
    
    def forward(self, x,language):
        b,l,c= x.shape
        # language_embed = self.language_embed(language)[:,0,:].unsqueeze(1).expand(-1,l,-1)
        # x = self.fused_add_tanh_sigmoid_multiply(x,language_embed)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x
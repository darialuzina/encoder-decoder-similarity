import torch
import torch.nn as nn


class Similarity2(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int, intermediate_dim: int):
        super().__init__()

        self.fc1 = nn.Linear(encoder_dim, intermediate_dim)
        self.fc2 = nn.Linear(decoder_dim, intermediate_dim)
        self.fc3 = nn.Linear(intermediate_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, encoder_states: torch.Tensor, decoder_state: torch.Tensor):
        encoder_transformed = self.fc1(encoder_states)
        decoder_transformed = self.fc2(decoder_state)
        combined = self.tanh(encoder_transformed + decoder_transformed)        
        similarities = self.fc3(combined)

        return similarities
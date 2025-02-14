import torch
import torch.nn as nn


class Similarity1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, encoder_states: torch.Tensor, decoder_state: torch.Tensor):
        decoder_state.unsqueeze(-1)
        similarities = torch.matmul(encoder_states, decoder_state)
        return similarities.squeeze(-1)
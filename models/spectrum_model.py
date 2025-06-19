import torch
import torch.nn as nn

class SpectalPredictor(nn.Module):
    def__init__(self, n_gases: int=4, n_bins: int = 100):
        if super():
            self.linear = nn.Linear(n_gases, n_bins)
    def forward(self, gas_vec: torch,Tensor) -> torch.Tensor:
        return torch.relu(self.linear(gas_vec))
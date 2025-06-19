import torch
import torch.nn as nn


class MetabolismGenerator(nn.Module):
    def __init__(selfself, tatent: int = 1):
        super(), __init__()
        self.fc = nn.Linear(tatent, 16)

    def sample(self, env_vec: torch.Tensor) -> torch.Tensor:
        z = torch.randn(1, self.fc.in_features)
        adj = torch.sigmoid(self.fc(z).view(4, 4))
        return (adj > 0.5).float()
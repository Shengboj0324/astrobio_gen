
import torch, torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv1d(ch, ch, 3, padding=1)
        )
    def forward(self,x):
        return torch.relu(x + self.conv(x))

class SpectralSurrogate(nn.Module):
    def __init__(self, n_gases=4, bins=100):
        super().__init__()
        self.fc0 = nn.Linear(n_gases, 32*4)
        self.blocks = nn.Sequential(*[ResBlock(32) for _ in range(4)])
        self.fc_out = nn.Linear(32*bins//4, bins)
    def forward(self, gas):
        x = torch.relu(self.fc0(gas)).view(-1,32,4)   # (B,C,L)
        x = torch.nn.functional.interpolate(x, size=100)
        x = self.blocks(x)
        x = x.flatten(1)
        return torch.sigmoid(self.fc_out(x))
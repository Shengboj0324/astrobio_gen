import pathlib
import warnings

import torch
import torch.nn as nn


class _AE(nn.Module):
    def __init__(self, bins=100, latent=12):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(bins, 64), nn.ReLU(), nn.Linear(64, latent))
        self.dec = nn.Sequential(nn.Linear(latent, 64), nn.ReLU(), nn.Linear(64, bins))

    def forward(self, x):
        return self.dec(self.enc(x))


def get_autoencoder(bins=100):
    pt = pathlib.Path("models/spectral_autoencoder.pt")
    model = _AE(bins)
    if pt.exists():
        ckpt = torch.load(pt, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
    else:
        warnings.warn("Autoencoder weights not found; using random init")
    model.eval()
    return model

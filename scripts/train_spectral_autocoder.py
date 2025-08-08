import json
import pathlib

import numpy as np
import torch
import torch.nn as nn

DATA = pathlib.Path("data/psg_outputs")
files = list(DATA.glob("*.csv"))
if not files:
    raise SystemExit("Run pipeline with FAST_MODE=0 first to generate PSG cache")

spec = np.stack([np.loadtxt(f, delimiter=",")[:, 1] for f in files])
wave = np.loadtxt(files[0], delimiter=",")[:, 0]

x = torch.tensor(spec, dtype=torch.float32)


class AE(nn.Module):
    def __init__(self, bins=100, latent=12):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(bins, 64), nn.ReLU(), nn.Linear(64, latent))
        self.dec = nn.Sequential(nn.Linear(latent, 64), nn.ReLU(), nn.Linear(64, bins))

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)


model = AE()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(300):
    opt.zero_grad()
    out = model(x)
    loss = loss_fn(out, x)
    loss.backward()
    opt.step()
    if epoch % 50 == 0:
        print(epoch, loss.item())

path = pathlib.Path("models/spectral_autoencoder.pt")
path.parent.mkdir(exist_ok=True)
torch.save({"wave": wave, "state_dict": model.state_dict()}, path)
print("[train] saved →", path)

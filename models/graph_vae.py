"""GraphVAE for small metabolic networks (≤10 nodes).
Uses PyTorch Geometric; trains on *synthetic* random graphs so you can debug the
training loop right now, then swap in KEGG later.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool


class GVAE(nn.Module):
    def __init__(self, in_channels=1, hidden=32, z_dim=8, latent=8):
        super().__init__()
        self.gc1 = GCNConv(in_channels, hidden)
        self.fc_mu = nn.Linear(hidden, latent)
        self.fc_logvar = nn.Linear(hidden, latent)
        self.fc_dec = nn.Linear(latent, 100)  # up to 10×10 adj
        self.z_dim = latent

    # ---------- encoder ----------
    def encode(self, x, edge_index, batch):
        h = torch.relu(self.gc1(x, edge_index))
        h = global_mean_pool(h, batch)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    # ---------- decoder ----------
    def decode(self, z):
        adj_logits = self.fc_dec(z).view(-1, 10, 10)
        adj = torch.sigmoid(adj_logits)
        return (adj > 0.5).float()

    # ---------- reparameterise ----------
    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ---------- forward ----------
    def forward(self, data: Data):
        mu, logvar = self.encode(data.x, data.edge_index, data.batch)
        z = self.reparam(mu, logvar)
        adj = self.decode(z)
        return adj, mu, logvar

import torch, torch.nn as nn
from torch_geometric.nn import GCNConv

class Encoder(nn.Module):
    def __init__(self, in_dim=16, latent=8):
        super().__init__()
        self.gcn1 = GCNConv(in_dim, 32)
        self.mu = nn.Linear(32, latent)
        self.logvar = nn.Linear(32, latent)
    def forward(self, x, edge_index):
        h = torch.relu(self.gcn1(x, edge_index))
        return self.mu(h.mean(0)), self.logvar(h.mean(0))

class Decoder(nn.Module):
    def __init__(self, latent=8, num_nodes=4):
        super().__init__()
        self.fc = nn.Linear(latent, num_nodes*num_nodes)
        self.num = num_nodes
    def forward(self, z):
        adj = torch.sigmoid(self.fc(z)).view(self.num, self.num)
        return (adj>0.5).float()

class MetabolismGenerator(nn.Module):
    def __init__(self, nodes=4, latent=8):
        super().__init__()
        self.nodes=nodes
        self.enc = Encoder(in_dim=nodes, latent=latent)
        self.dec = Decoder(latent=latent, num_nodes=nodes)
    @torch.no_grad()
    def sample(self, env_vec):
        # dummy sample ignoring env for now
        z = torch.randn(1, self.dec.fc.in_features)
        return self.dec(z).squeeze(0)
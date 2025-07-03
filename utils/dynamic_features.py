
from __future__ import annotations
import torch, torch.nn as nn

class NumericEncoder(nn.Module):
    def __init__(self): super().__init__(); self.fc = nn.Linear(1, 16)
    def forward(self, x): return torch.relu(self.fc(x.unsqueeze(-1)))

class CategoricalEncoder(nn.Module):
    def __init__(self, n_cat: int): super().__init__(); self.emb = nn.Embedding(n_cat, 16)
    def forward(self, x): return self.emb(x.long())

class VectorEncoder(nn.Module):
    def __init__(self, input_dim: int): 
        super().__init__()
        self.fc = nn.Linear(input_dim, 16)
    def forward(self, x): return torch.relu(self.fc(x))

REGISTRY = {
    "numeric": lambda _: NumericEncoder(),
    "categorical": lambda n: CategoricalEncoder(n),
    "vector": lambda dim: VectorEncoder(dim),
}

def build_encoders(schema: dict):
    """
    schema = { "air_quality": ("numeric", None),
               "rock_type":   ("categorical", 12),
               "surface_vec": ("vector", 64) }
    Returns nn.ModuleDict keyed by column name.
    """
    enc = {}
    for col, (kind, arg) in schema.items():
        enc[col] = REGISTRY[kind](arg)
    return nn.ModuleDict(enc)
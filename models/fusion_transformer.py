"""
Perceiver-style Fusion Transformer.
Takes N arbitrary feature tokens → pooled latent → task-specific heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from utils.dynamic_features import build_encoders


class FusionModel(nn.Module):
    def __init__(self, schema: dict, latent_dim=128, n_heads=4, depth=4):
        super().__init__()
        self.encoders = build_encoders(schema)
        self.pos = nn.Parameter(torch.randn(1, len(schema), latent_dim))
        self.proj = nn.Linear(16, latent_dim)  # every encoder →16 dims
        self.xformers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(latent_dim, n_heads, dim_feedforward=latent_dim * 4),
            num_layers=depth,
        )
        self.cls = nn.Parameter(torch.randn(1, 1, latent_dim))  # [CLS] token
        self.reg_head = nn.Linear(latent_dim, 1)  # example regression
        self.cls_head = nn.Linear(latent_dim, 3)  # example 3-class task

    def forward(self, batch: dict[str, torch.Tensor]):
        feats = []
        for i, (col, enc) in enumerate(self.encoders.items()):
            z = enc(batch[col])  # (B, 16)
            z = self.proj(z) + self.pos[:, i]  # broadcast positional
            feats.append(z.unsqueeze(1))
        toks = torch.cat(feats, dim=1)  # (B, N_feat, dim)
        cls = self.cls.expand(toks.size(0), -1, -1)
        x = torch.cat([cls, toks], dim=1)
        x = self.xformers(x)
        pooled = x[:, 0]  # CLS output
        return {"reg": self.reg_head(pooled).squeeze(-1), "cls": self.cls_head(pooled)}  # logits

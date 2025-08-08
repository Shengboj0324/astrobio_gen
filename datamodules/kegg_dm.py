from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset


class KeggDM(pl.LightningDataModule):
    def __init__(self, root="data/kegg_graphs", batch_size=64):
        super().__init__()
        self.root = Path(root)
        self.bs = batch_size

    def setup(self, stage=None):
        files = list(self.root.glob("*.npz"))
        split = int(0.8 * len(files))
        tr, va = files[:split], files[split:]
        self.train_ds = self._load(tr)
        self.val_ds = self._load(va)

    def _load(self, paths):
        env, adj = [], []
        for f in paths:
            npz = np.load(f)
            env.append(npz["env"])
            adj.append(npz["adj"])
        env = torch.tensor(np.stack(env), dtype=torch.float32)
        adj = torch.tensor(np.stack(adj), dtype=torch.float32)
        return TensorDataset(env, adj)

    def train_dataloader(self):
        return DataLoader(self.train_ds, self.bs, True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, self.bs)

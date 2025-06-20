from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

class KeggDM(pl.LightningDataModule):
    def __init__(self, root="data/kegg_graphs", batch_size=64, num_workers=4):
        super().__init__()
        self.root, self.bs, self.nw = Path(root), batch_size, num_workers

    def setup(self, stage=None):
        files = list(self.root.glob("*.npz"))
        split = int(0.8 * len(files))
        train_npz, val_npz = files[:split], files[split:]
        self.train, self.val = self._make_ds(train_npz), self._make_ds(val_npz)

    @staticmethod
    def _make_ds(file_list):
        env, adj = [], []
        for f in file_list:
            dat = np.load(f)
            env.append(dat["env"])
            adj.append(dat["adj"])
        env_t = torch.tensor(np.stack(env), dtype=torch.float32)
        adj_t = torch.tensor(np.stack(adj), dtype=torch.float32)
        return TensorDataset(env_t, adj_t)

    def train_dataloader(self):
        return DataLoader(self.train, self.bs, True, num_workers=self.nw)

    def val_dataloader(self):
        return DataLoader(self.val, self.bs, False, num_workers=self.nw)
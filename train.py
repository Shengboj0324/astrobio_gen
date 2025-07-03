from __future__ import annotations
import torch, pytorch_lightning as pl
from utils.config import parse_cli
from models.graph_vae import GVAE             # you added earlier
from models.fusion_transformer import FusionModel
from scripts.train_gvae_dummy import random_graph
from scripts.train_fusion_dummy import schema as FUSION_SCHEMA, to_tensor
import pandas as pd, numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import DataLoader as GeometricDataLoader
import os

class LitGraphVAE(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model = GVAE(latent=cfg["model"]["graph_vae"]["latent"])
    def training_step(self, batch, _):
        adj_hat, mu, logvar = self.model(batch)
        loss = (adj_hat.sum() + mu.pow(2).mean() + logvar.exp().mean())
        self.log("loss", loss)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 1e-3)

class LitFusion(pl.LightningModule):
    def __init__(self, cfg, schema):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model = FusionModel(schema, **cfg["model"]["fusion"])
        self.loss_reg = torch.nn.MSELoss()
    def training_step(self, batch, _):
        feats, y = batch[:-1], batch[-1]
        out = self.model({k:t for k,t in zip(FUSION_SCHEMA.keys(), feats)})
        loss = self.loss_reg(out["reg"], y)
        self.log("loss", loss)
        return loss
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), 3e-4)

def main():
    cfg, _ = parse_cli()
    pl.seed_everything(0)

    if cfg["model"]["type"] == "graph_vae":
        ds = [random_graph() for _ in range(cfg["data"]["synthetic_size"])]
        dl = GeometricDataLoader(ds, batch_size=cfg["trainer"]["batch_size"], shuffle=True)
        module = LitGraphVAE(cfg)
    else:
        # fusion synthetic tabular
        N = cfg["data"]["synthetic_size"]
        df = pd.DataFrame({
            "air_quality": np.random.rand(N),
            "rock_type":   np.random.randint(0, 12, size=N),
            "surface_vec": list(np.random.randn(N, 64))
        })
        y = torch.tensor(np.random.rand(N), dtype=torch.float32)
        feat_tensors = [to_tensor(df[c]).float() if i==0 else
                        torch.tensor(df[c].values) if i==1 else
                        torch.tensor(np.stack(df[c].values)).float()
                        for i,c in enumerate(FUSION_SCHEMA.keys())]
        ds = TensorDataset(*feat_tensors, y)
        dl = DataLoader(ds, batch_size=cfg["trainer"]["batch_size"], shuffle=True)
        module = LitFusion(cfg, FUSION_SCHEMA)

    trainer = pl.Trainer(
        max_epochs=cfg["trainer"]["max_epochs"],
        accelerator=cfg["trainer"]["accelerator"],
        default_root_dir="lightning_logs",
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(module, dl)

if __name__ == "__main__":
    main()
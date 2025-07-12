from __future__ import annotations
import torch, pytorch_lightning as pl
from utils.config import parse_cli
from models.graph_vae import GVAE
from models.fusion_transformer import FusionModel
from models.surrogate_transformer import SurrogateTransformer, UncertaintyQuantification
from scripts.train_gvae_dummy import random_graph
from scripts.train_fusion_dummy import schema as FUSION_SCHEMA, to_tensor
import pandas as pd, numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import DataLoader as GeometricDataLoader
import os
from typing import Dict, Any, Optional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

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


class LitSurrogateTransformer(pl.LightningModule):
    """
    NASA-ready Lightning module for physics-informed climate modeling.
    Supports all operational modes and rigorous validation.
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(cfg)
        
        # Initialize model based on mode
        model_cfg = cfg["model"]["surrogate"]
        self.model = SurrogateTransformer(**model_cfg)
        
        # Uncertainty quantification wrapper
        self.uncertainty_model = UncertaintyQuantification(self.model)
        
        # Validation metrics storage
        self.validation_predictions = []
        self.validation_targets = []
        
        # Automatic mixed precision
        self.automatic_optimization = True
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(x)
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step with physics-informed loss"""
        planet_params, targets = batch
        
        # Forward pass
        outputs = self.model(planet_params)
        
        # Compute comprehensive loss
        losses = self.model.compute_total_loss(outputs, targets)
        
        # Log all loss components
        for loss_name, loss_value in losses.items():
            self.log(f'train/{loss_name}', loss_value, prog_bar=(loss_name == 'total_loss'))
        
        # Log physics constraint weights
        weights = torch.nn.functional.softplus(outputs['physics_weights'])
        self.log('train/radiative_weight', weights[0])
        self.log('train/mass_balance_weight', weights[1])
        
        return losses['total_loss']
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Validation step with uncertainty quantification"""
        planet_params, targets = batch
        
        # Standard prediction
        outputs = self.model(planet_params)
        losses = self.model.compute_total_loss(outputs, targets)
        
        # Log validation losses
        for loss_name, loss_value in losses.items():
            self.log(f'val/{loss_name}', loss_value, prog_bar=(loss_name == 'total_loss'))
        
        # Store for epoch-end validation
        self.validation_predictions.append(outputs)
        self.validation_targets.append(targets)
        
        # Uncertainty quantification on subset of validation data
        if batch_idx % 10 == 0:  # Sample every 10th batch
            uncertainty_outputs = self.uncertainty_model.predict_with_uncertainty(planet_params)
            
            # Log uncertainty metrics
            for key in uncertainty_outputs:
                if key.endswith('_std'):
                    mean_uncertainty = uncertainty_outputs[key].mean()
                    self.log(f'val/uncertainty_{key}', mean_uncertainty)
        
        return losses['total_loss']
    
    def on_validation_epoch_end(self):
        """Comprehensive validation metrics at epoch end"""
        if not self.validation_predictions:
            return
        
        # Aggregate predictions and targets
        all_predictions = {}
        all_targets = {}
        
        for pred_dict in self.validation_predictions:
            for key, value in pred_dict.items():
                if key not in all_predictions:
                    all_predictions[key] = []
                if isinstance(value, torch.Tensor):
                    all_predictions[key].append(value)
        
        for target_dict in self.validation_targets:
            for key, value in target_dict.items():
                if key not in all_targets:
                    all_targets[key] = []
                all_targets[key].append(value)
        
        # Compute R² scores for continuous variables
        for key in all_targets:
            if key in all_predictions and len(all_predictions[key]) > 0:
                pred_tensor = torch.cat(all_predictions[key], dim=0)
                target_tensor = torch.cat(all_targets[key], dim=0)
                
                # R² score
                ss_res = ((target_tensor - pred_tensor) ** 2).sum()
                ss_tot = ((target_tensor - target_tensor.mean()) ** 2).sum()
                r2_score = 1 - (ss_res / (ss_tot + 1e-8))
                
                self.log(f'val/r2_{key}', r2_score)
                
                # MAE
                mae = torch.abs(target_tensor - pred_tensor).mean()
                self.log(f'val/mae_{key}', mae)
        
        # Clear for next epoch
        self.validation_predictions.clear()
        self.validation_targets.clear()
    
    def configure_optimizers(self):
        """Advanced optimizer configuration with scheduling"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams["training"]["learning_rate"],
            weight_decay=self.hparams["training"]["weight_decay"],
            betas=(0.9, 0.95)  # Better for transformers
        )
        
        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=50,  # Initial restart period
            T_mult=2,  # Period multiplication factor
            eta_min=1e-6  # Minimum learning rate
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/total_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
    
    def predict_with_uncertainty(self, planet_params: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Prediction interface with uncertainty quantification"""
        return self.uncertainty_model.predict_with_uncertainty(planet_params)


def create_synthetic_climate_data(n_samples: int = 1000, mode: str = "scalar") -> tuple:
    """Create synthetic climate training data"""
    np.random.seed(42)
    
    # Planet parameters: [radius, mass, period, insolation, st_teff, st_logg, st_met, host_mass]
    planet_params = np.random.rand(n_samples, 8)
    
    # Scale to realistic ranges
    planet_params[:, 0] = planet_params[:, 0] * 2.0 + 0.5  # radius: 0.5-2.5 Earth radii
    planet_params[:, 1] = planet_params[:, 1] * 5.0 + 0.1  # mass: 0.1-5.1 Earth masses
    planet_params[:, 2] = planet_params[:, 2] * 500 + 10   # period: 10-510 days
    planet_params[:, 3] = planet_params[:, 3] * 3.0 + 0.1  # insolation: 0.1-3.1 S_earth
    planet_params[:, 4] = planet_params[:, 4] * 2000 + 3000  # stellar Teff: 3000-5000K
    planet_params[:, 5] = planet_params[:, 5] * 2.0 + 3.5   # stellar logg: 3.5-5.5
    planet_params[:, 6] = planet_params[:, 6] * 1.0 - 0.5   # stellar metallicity: -0.5 to +0.5
    planet_params[:, 7] = planet_params[:, 7] * 2.0 + 0.5   # host mass: 0.5-2.5 solar masses
    
    # Generate targets based on mode
    targets = {}
    
    if mode == "scalar":
        # Habitability score (based on insolation and radius)
        habitability = 1.0 / (1.0 + np.exp(-(planet_params[:, 3] - 1.0) * 5))  # Sigmoid around Earth-like
        habitability *= 1.0 / (1.0 + np.exp(-np.abs(planet_params[:, 0] - 1.0) * 5))  # Penalty for non-Earth-size
        
        # Surface temperature (based on insolation and stellar temperature)
        surface_temp = 255 * (planet_params[:, 3] ** 0.25) + np.random.normal(0, 10, n_samples)
        
        # Atmospheric pressure (log-normal distribution)
        atm_pressure = np.exp(np.random.normal(np.log(1.0), 0.5, n_samples))
        
        targets = {
            'habitability': torch.tensor(habitability, dtype=torch.float32).unsqueeze(1),
            'surface_temp': torch.tensor(surface_temp, dtype=torch.float32).unsqueeze(1),
            'atmospheric_pressure': torch.tensor(atm_pressure, dtype=torch.float32).unsqueeze(1)
        }
    
    elif mode == "datacube":
        # 3D temperature and humidity fields (simplified)
        temp_fields = np.random.normal(250, 50, (n_samples, 64, 32, 20))  # lat×lon×pressure
        humidity_fields = np.random.exponential(0.1, (n_samples, 64, 32, 20))
        
        targets = {
            'temperature_field': torch.tensor(temp_fields, dtype=torch.float32),
            'humidity_field': torch.tensor(humidity_fields, dtype=torch.float32)
        }
    
    planet_tensor = torch.tensor(planet_params, dtype=torch.float32)
    return planet_tensor, targets


def main():
    cfg, _ = parse_cli()
    pl.seed_everything(42)
    
    # Setup logging
    if cfg.get("logging", {}).get("use_wandb", False) and WANDB_AVAILABLE:
        logger = WandbLogger(
            project="astrobio-surrogate",
            name=f"surrogate-{cfg['model']['type']}-{cfg['model'].get('surrogate', {}).get('mode', 'scalar')}",
            config=cfg
        )
    else:
        logger = True
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            monitor="val/total_loss",
            mode="min",
            save_top_k=3,
            filename="surrogate-{epoch:02d}-{val_total_loss:.3f}",
        ),
        EarlyStopping(
            monitor="val/total_loss",
            patience=20,
            mode="min",
            verbose=True
        ),
        LearningRateMonitor(logging_interval="epoch")
    ]

    if cfg["model"]["type"] == "graph_vae":
        ds = [random_graph() for _ in range(cfg["data"]["synthetic_size"])]
        dl = GeometricDataLoader(ds, batch_size=cfg["trainer"]["batch_size"], shuffle=True)
        module = LitGraphVAE(cfg)
        
    elif cfg["model"]["type"] == "fusion":
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
        
    elif cfg["model"]["type"] == "surrogate":
        # Advanced surrogate transformer
        mode = cfg["model"]["surrogate"]["mode"]
        planet_data, targets = create_synthetic_climate_data(
            cfg["data"]["synthetic_size"], 
            mode=mode
        )
        
        # Convert targets dict to list for TensorDataset
        target_tensors = list(targets.values())
        ds = TensorDataset(planet_data, *target_tensors)
        
        # Split into train/val
        train_size = int(0.8 * len(ds))
        val_size = len(ds) - train_size
        train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])
        
        train_dl = DataLoader(train_ds, batch_size=cfg["trainer"]["batch_size"], shuffle=True, num_workers=4)
        val_dl = DataLoader(val_ds, batch_size=cfg["trainer"]["batch_size"], shuffle=False, num_workers=4)
        
        module = LitSurrogateTransformer(cfg)
        
    else:
        raise ValueError(f"Unknown model type: {cfg['model']['type']}")

    trainer = pl.Trainer(
        max_epochs=cfg["trainer"]["max_epochs"],
        accelerator=cfg["trainer"]["accelerator"],
        devices=cfg["trainer"].get("devices", "auto"),
        precision=cfg["trainer"].get("precision", "16-mixed"),  # Mixed precision for speed
        default_root_dir="lightning_logs",
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=1.0,  # Gradient clipping for stability
        accumulate_grad_batches=cfg["trainer"].get("accumulate_grad_batches", 1),
    )
    
    if cfg["model"]["type"] == "surrogate":
        trainer.fit(module, train_dl, val_dl)
    else:
        trainer.fit(module, dl)

if __name__ == "__main__":
    main()
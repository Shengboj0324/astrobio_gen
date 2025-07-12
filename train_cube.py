#!/usr/bin/env python3
"""
Climate Datacube Training CLI
============================

PyTorch Lightning CLI for training 3D U-Net models on climate datacubes.
Integrates with existing training infrastructure and supports mixed precision.

Usage:
    python train_cube.py fit --data.zarr_root data/processed/gcm_zarr --model.depth 4 --trainer.precision 16 --trainer.strategy ddp

Author: AI Assistant
Date: 2025
"""

import os
import sys
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import our modules
from datamodules.cube_dm import CubeDM
from models.datacube_unet import CubeUNet

class CubeCLI(LightningCLI):
    """
    Custom CLI for climate datacube training
    """
    
    def add_arguments_to_parser(self, parser):
        """Add additional arguments to the parser"""
        
        # Model arguments
        parser.add_argument("--model.n_input_vars", type=int, default=5, help="Number of input variables")
        parser.add_argument("--model.n_output_vars", type=int, default=5, help="Number of output variables")
        parser.add_argument("--model.base_features", type=int, default=32, help="Base number of features")
        parser.add_argument("--model.depth", type=int, default=4, help="U-Net depth")
        parser.add_argument("--model.dropout", type=float, default=0.1, help="Dropout rate")
        parser.add_argument("--model.learning_rate", type=float, default=1e-4, help="Learning rate")
        parser.add_argument("--model.weight_decay", type=float, default=1e-5, help="Weight decay")
        parser.add_argument("--model.physics_weight", type=float, default=0.1, help="Physics regularization weight")
        parser.add_argument("--model.use_physics_constraints", type=bool, default=True, help="Use physics constraints")
        
        # Data arguments
        parser.add_argument("--data.zarr_root", type=str, default="data/processed/gcm_zarr", help="Root directory for zarr data")
        parser.add_argument("--data.batch_size", type=int, default=4, help="Batch size")
        parser.add_argument("--data.num_workers", type=int, default=4, help="Number of data workers")
        parser.add_argument("--data.time_window", type=int, default=10, help="Time window for sequences")
        parser.add_argument("--data.train_fraction", type=float, default=0.8, help="Training fraction")
        parser.add_argument("--data.val_fraction", type=float, default=0.1, help="Validation fraction")
        parser.add_argument("--data.normalize", type=bool, default=True, help="Normalize data")
        
        # Training arguments  
        parser.add_argument("--trainer.max_epochs", type=int, default=100, help="Maximum training epochs")
        parser.add_argument("--trainer.precision", type=str, default="16-mixed", help="Training precision")
        parser.add_argument("--trainer.strategy", type=str, default="auto", help="Training strategy")
        parser.add_argument("--trainer.accumulate_grad_batches", type=int, default=1, help="Gradient accumulation")
        parser.add_argument("--trainer.gradient_clip_val", type=float, default=1.0, help="Gradient clipping")
        parser.add_argument("--trainer.val_check_interval", type=float, default=1.0, help="Validation check interval")
        parser.add_argument("--trainer.log_every_n_steps", type=int, default=10, help="Logging frequency")
        
        # Experiment arguments
        parser.add_argument("--experiment.name", type=str, default="datacube_unet", help="Experiment name")
        parser.add_argument("--experiment.version", type=str, default=None, help="Experiment version")
        parser.add_argument("--experiment.save_dir", type=str, default="lightning_logs", help="Save directory")
        parser.add_argument("--experiment.use_wandb", type=bool, default=False, help="Use Weights & Biases")
        parser.add_argument("--experiment.wandb_project", type=str, default="astrobio-datacube", help="W&B project name")
        
        # Checkpoint arguments
        parser.add_argument("--checkpoint.save_top_k", type=int, default=3, help="Save top k checkpoints")
        parser.add_argument("--checkpoint.monitor", type=str, default="val/total", help="Metric to monitor")
        parser.add_argument("--checkpoint.mode", type=str, default="min", help="Monitor mode")
        parser.add_argument("--checkpoint.save_last", type=bool, default=True, help="Save last checkpoint")
        parser.add_argument("--checkpoint.every_n_epochs", type=int, default=1, help="Checkpoint frequency")
        
        # Early stopping arguments
        parser.add_argument("--early_stopping.patience", type=int, default=20, help="Early stopping patience")
        parser.add_argument("--early_stopping.min_delta", type=float, default=0.001, help="Minimum delta for improvement")
        parser.add_argument("--early_stopping.mode", type=str, default="min", help="Early stopping mode")
    
    def instantiate_trainer(self, **kwargs):
        """Instantiate the trainer with custom callbacks and logger"""
        
        # Setup callbacks
        callbacks = []
        
        # Model checkpoint
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"{self.config.experiment.save_dir}/{self.config.experiment.name}/checkpoints",
            filename='{epoch:02d}-{val/total:.3f}',
            monitor=self.config.checkpoint.monitor,
            mode=self.config.checkpoint.mode,
            save_top_k=self.config.checkpoint.save_top_k,
            save_last=self.config.checkpoint.save_last,
            every_n_epochs=self.config.checkpoint.every_n_epochs,
            verbose=True
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stop_callback = EarlyStopping(
            monitor=self.config.checkpoint.monitor,
            patience=self.config.early_stopping.patience,
            min_delta=self.config.early_stopping.min_delta,
            mode=self.config.early_stopping.mode,
            verbose=True
        )
        callbacks.append(early_stop_callback)
        
        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)
        
        # Setup logger
        if self.config.experiment.use_wandb:
            logger = WandbLogger(
                project=self.config.experiment.wandb_project,
                name=self.config.experiment.name,
                version=self.config.experiment.version,
                save_dir=self.config.experiment.save_dir
            )
        else:
            logger = TensorBoardLogger(
                save_dir=self.config.experiment.save_dir,
                name=self.config.experiment.name,
                version=self.config.experiment.version
            )
        
        # Setup trainer
        trainer = pl.Trainer(
            callbacks=callbacks,
            logger=logger,
            **kwargs
        )
        
        return trainer

def main():
    """Main training function"""
    
    # Set up environment
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Set precision
    torch.set_float32_matmul_precision('medium')
    
    # Create CLI
    cli = CubeCLI(
        model_class=CubeUNet,
        datamodule_class=CubeDM,
        seed_everything_default=42,
        save_config_overwrite=True,
        parser_kwargs={
            "prog": "train_cube",
            "description": "Train 3D U-Net on climate datacubes"
        }
    )

if __name__ == "__main__":
    main() 
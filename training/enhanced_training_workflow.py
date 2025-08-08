#!/usr/bin/env python3
"""
Enhanced Training Workflow
==========================

Scalable multi-task training system with weighted loss functions and
comprehensive performance monitoring for multi-modal scientific data.

Features:
- Multi-task learning with adaptive loss weighting
- Physics-informed loss functions
- Real-time performance monitoring
- Adaptive learning rate scheduling
- Mixed precision training
- Distributed training support
- Comprehensive logging and visualization
- Model checkpointing and resuming
- Early stopping with patience
- Gradient clipping and regularization

Training Objectives:
L = α·L_climate + β·L_spectrum + γ·L_physics + δ·L_consistency

Where:
- L_climate: Climate field reconstruction loss (MSE + physics)
- L_spectrum: Spectral synthesis loss (MAE + spectral features)
- L_physics: Physics constraint violations (energy, mass, radiative)
- L_consistency: Multi-modal consistency loss
"""

import json
import logging
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

# For gradient clipping and mixed precision
from torch.cuda.amp import GradScaler, autocast
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau

from data_build.unified_dataloader_fixed import (
    DataLoaderConfig,
    MultiModalBatch,
    create_multimodal_dataloaders,
)

# Local imports
from models.domain_encoders_simple import EncoderConfig, MultiModalEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LossStrategy(Enum):
    """Loss weighting strategies"""

    FIXED = "fixed"  # Fixed weights
    ADAPTIVE = "adaptive"  # Adapt based on task performance
    UNCERTAINTY = "uncertainty"  # Uncertainty-based weighting
    GRADIENT_NORM = "gradient_norm"  # Gradient norm balancing


# OptimizationStrategy moved to enhanced_training_orchestrator.py
from .enhanced_training_orchestrator import OptimizationStrategy


@dataclass
class TrainingConfig:
    """Configuration for enhanced training"""

    # Basic training settings
    max_epochs: int = 200
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip_val: float = 1.0

    # Loss weighting
    loss_strategy: LossStrategy = LossStrategy.ADAPTIVE
    initial_loss_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "climate": 1.0,
            "spectrum": 0.3,
            "physics": 0.2,
            "consistency": 0.1,
        }
    )

    # Optimization
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.ADAMW_COSINE
    warmup_epochs: int = 10
    min_lr: float = 1e-6

    # Mixed precision and performance
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    accumulate_grad_batches: int = 1

    # Monitoring and logging
    log_every_n_steps: int = 10
    val_check_interval: float = 1.0
    save_top_k: int = 3
    patience: int = 20

    # Physics constraints
    physics_loss_schedule: Dict[str, Any] = field(
        default_factory=lambda: {"start_epoch": 0, "ramp_epochs": 50, "max_weight": 1.0}
    )

    # Data augmentation during training
    enable_training_augmentation: bool = True
    augmentation_prob: float = 0.3

    # Distributed training
    enable_distributed: bool = False
    num_gpus: int = 1

    # Wandb logging
    use_wandb: bool = True
    wandb_project: str = "astrobio-multimodal"
    wandb_tags: List[str] = field(default_factory=lambda: ["multi-modal", "astrobiology"])


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function with adaptive weighting

    Combines climate reconstruction, spectral synthesis, physics constraints,
    and multi-modal consistency losses.
    """

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config

        # Loss weights (learnable parameters for uncertainty weighting)
        if config.loss_strategy == LossStrategy.UNCERTAINTY:
            self.log_vars = nn.Parameter(torch.zeros(4))  # climate, spectrum, physics, consistency
        else:
            self.register_buffer(
                "loss_weights",
                torch.tensor(
                    [
                        config.initial_loss_weights["climate"],
                        config.initial_loss_weights["spectrum"],
                        config.initial_loss_weights["physics"],
                        config.initial_loss_weights["consistency"],
                    ]
                ),
            )

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.huber_loss = nn.HuberLoss(delta=1.0)

        # Loss history for adaptive weighting
        self.loss_history = defaultdict(deque)

        logger.info(f"🎯 Multi-task loss initialized with {config.loss_strategy.value} weighting")

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        encoder_outputs: Dict[str, Any],
        epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            encoder_outputs: Outputs from multi-modal encoder
            epoch: Current training epoch

        Returns:
            Dictionary with individual and total losses
        """
        losses = {}

        # 1. Climate field reconstruction loss
        if "climate_cubes" in targets and "climate_prediction" in predictions:
            climate_loss = self._compute_climate_loss(
                predictions["climate_prediction"], targets["climate_cubes"]
            )
            losses["climate"] = climate_loss

        # 2. Spectral synthesis loss
        if "spectra" in targets and "spectrum_prediction" in predictions:
            spectrum_loss = self._compute_spectrum_loss(
                predictions["spectrum_prediction"], targets["spectra"]
            )
            losses["spectrum"] = spectrum_loss

        # 3. Physics constraint loss
        if "physics_constraints" in encoder_outputs:
            physics_loss = self._compute_physics_loss(
                encoder_outputs["physics_constraints"], targets.get("planet_params"), epoch
            )
            losses["physics"] = physics_loss

        # 4. Multi-modal consistency loss
        if "individual_features" in encoder_outputs:
            consistency_loss = self._compute_consistency_loss(
                encoder_outputs["individual_features"]
            )
            losses["consistency"] = consistency_loss

        # Combine losses with adaptive weighting
        total_loss = self._combine_losses(losses, epoch)
        losses["total"] = total_loss

        return losses

    def _compute_climate_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute climate field reconstruction loss"""
        # Primary reconstruction loss
        recon_loss = self.mse_loss(pred, target)

        # Physics-informed loss terms
        # 1. Gradient smoothness (climate fields should be smooth)
        if pred.dim() >= 4:  # Has spatial dimensions
            grad_loss = self._compute_gradient_smoothness(pred, target)
            recon_loss = recon_loss + 0.1 * grad_loss

        # 2. Conservation constraints (if applicable)
        # This could include mass/energy conservation

        return recon_loss

    def _compute_spectrum_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute spectral synthesis loss"""
        # Use L1 loss for spectral data (more robust to outliers)
        spectral_loss = self.mae_loss(pred, target)

        # Add spectral feature loss (preserve important spectral lines)
        if pred.shape[-1] >= 2:  # Has wavelength and flux dimensions
            # Compute derivative to preserve spectral line shapes
            pred_diff = torch.diff(pred, dim=-2)
            target_diff = torch.diff(target, dim=-2)
            derivative_loss = self.mae_loss(pred_diff, target_diff)

            spectral_loss = spectral_loss + 0.2 * derivative_loss

        return spectral_loss

    def _compute_physics_loss(
        self,
        constraints: Dict[str, torch.Tensor],
        planet_params: Optional[torch.Tensor],
        epoch: int,
    ) -> torch.Tensor:
        """Compute physics constraint loss"""
        physics_loss = torch.tensor(0.0, device=next(iter(constraints.values())).device)

        # Energy balance constraint
        if "energy_violation" in constraints:
            energy_loss = torch.mean(torch.abs(constraints["energy_violation"]))
            physics_loss = physics_loss + energy_loss

        # Mass conservation constraint
        if "mass_violation" in constraints:
            mass_loss = torch.mean(torch.abs(constraints["mass_violation"]))
            physics_loss = physics_loss + mass_loss

        # Radiative equilibrium (if available)
        if "radiative_violation" in constraints:
            radiative_loss = torch.mean(torch.abs(constraints["radiative_violation"]))
            physics_loss = physics_loss + radiative_loss

        # Apply physics loss schedule (ramp up over training)
        schedule = self.config.physics_loss_schedule
        if epoch < schedule["start_epoch"]:
            physics_weight = 0.0
        elif epoch < schedule["start_epoch"] + schedule["ramp_epochs"]:
            progress = (epoch - schedule["start_epoch"]) / schedule["ramp_epochs"]
            physics_weight = progress * schedule["max_weight"]
        else:
            physics_weight = schedule["max_weight"]

        return physics_weight * physics_loss

    def _compute_consistency_loss(
        self, individual_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute multi-modal consistency loss"""
        features = list(individual_features.values())

        if len(features) < 2:
            return torch.tensor(0.0, device=features[0].device)

        # Compute pairwise consistency loss (features should be similar)
        consistency_loss = torch.tensor(0.0, device=features[0].device)
        n_pairs = 0

        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                # Cosine similarity loss (encourage similar representations)
                cos_sim = F.cosine_similarity(features[i], features[j], dim=1)
                consistency_loss = consistency_loss + (1 - cos_sim.mean())
                n_pairs += 1

        if n_pairs > 0:
            consistency_loss = consistency_loss / n_pairs

        return consistency_loss

    def _compute_gradient_smoothness(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient smoothness loss for spatial fields"""
        # Compute spatial gradients
        if pred.dim() >= 4:  # [batch, channels, height, width, ...]
            # Gradient in spatial dimensions
            grad_pred_h = torch.diff(pred, dim=-2)
            grad_pred_w = torch.diff(pred, dim=-1)
            grad_target_h = torch.diff(target, dim=-2)
            grad_target_w = torch.diff(target, dim=-1)

            grad_loss = (
                self.mse_loss(grad_pred_h, grad_target_h)
                + self.mse_loss(grad_pred_w, grad_target_w)
            ) / 2

            return grad_loss

        return torch.tensor(0.0, device=pred.device)

    def _combine_losses(self, losses: Dict[str, torch.Tensor], epoch: int) -> torch.Tensor:
        """Combine losses with adaptive weighting"""
        if not losses:
            return torch.tensor(0.0)

        loss_names = ["climate", "spectrum", "physics", "consistency"]
        available_losses = {name: losses.get(name, torch.tensor(0.0)) for name in loss_names}

        if self.config.loss_strategy == LossStrategy.UNCERTAINTY:
            # Learnable uncertainty-based weighting
            total_loss = torch.tensor(0.0, device=next(iter(losses.values())).device)

            for i, (name, loss) in enumerate(available_losses.items()):
                if loss.item() > 0:
                    precision = torch.exp(-self.log_vars[i])
                    total_loss = total_loss + precision * loss + self.log_vars[i]

            return total_loss

        elif self.config.loss_strategy == LossStrategy.ADAPTIVE:
            # Adaptive weighting based on loss magnitude
            return self._adaptive_weighting(available_losses, epoch)

        else:  # FIXED
            # Fixed weighting
            total_loss = torch.tensor(0.0, device=next(iter(losses.values())).device)

            for i, (name, loss) in enumerate(available_losses.items()):
                if loss.item() > 0:
                    weight = self.loss_weights[i]
                    total_loss = total_loss + weight * loss

            return total_loss

    def _adaptive_weighting(self, losses: Dict[str, torch.Tensor], epoch: int) -> torch.Tensor:
        """Adaptive loss weighting based on relative loss magnitudes"""
        total_loss = torch.tensor(0.0, device=next(iter(losses.values())).device)

        # Update loss history
        for name, loss in losses.items():
            if loss.item() > 0:
                self.loss_history[name].append(loss.item())
                if len(self.loss_history[name]) > 100:  # Keep last 100 values
                    self.loss_history[name].popleft()

        # Compute adaptive weights
        if epoch > 10:  # Start adapting after initial epochs
            loss_means = {}
            for name in losses.keys():
                if len(self.loss_history[name]) > 10:
                    loss_means[name] = np.mean(list(self.loss_history[name])[-10:])

            if loss_means:
                # Inverse weighting: give more weight to smaller losses
                total_mean = sum(loss_means.values())
                for name, loss in losses.items():
                    if loss.item() > 0 and name in loss_means:
                        # Inverse weighting with smoothing
                        adaptive_weight = (total_mean / (loss_means[name] + 1e-8)) ** 0.5
                        adaptive_weight = max(0.1, min(adaptive_weight, 2.0))  # Clamp weights
                        total_loss = total_loss + adaptive_weight * loss
                    else:
                        total_loss = total_loss + loss
            else:
                # Fallback to equal weighting
                for loss in losses.values():
                    if loss.item() > 0:
                        total_loss = total_loss + loss
        else:
            # Initial epochs: use fixed weights
            weights = [1.0, 0.3, 0.2, 0.1]  # climate, spectrum, physics, consistency
            for i, loss in enumerate(losses.values()):
                if loss.item() > 0:
                    total_loss = total_loss + weights[i] * loss

        return total_loss


class MultiModalTrainer(pl.LightningModule):
    """
    PyTorch Lightning trainer for multi-modal encoder

    Handles training, validation, optimization, and logging.
    """

    def __init__(self, encoder_config: EncoderConfig, training_config: TrainingConfig):
        super().__init__()
        self.save_hyperparameters()

        self.encoder_config = encoder_config
        self.training_config = training_config

        # Multi-modal encoder
        self.encoder = MultiModalEncoder(encoder_config)

        # Task-specific prediction heads
        self._build_prediction_heads()

        # Multi-task loss function
        self.loss_fn = MultiTaskLoss(training_config)

        # Mixed precision scaler
        if training_config.use_mixed_precision:
            self.scaler = GradScaler()

        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)

        # Performance monitoring
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        logger.info(f"🚀 Multi-modal trainer initialized")

    def _build_prediction_heads(self):
        """Build task-specific prediction heads"""
        latent_dim = self.encoder_config.latent_dim

        # Climate field prediction head
        self.climate_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, latent_dim * 4),
            nn.ReLU(),
            nn.Linear(latent_dim * 4, 2 * 8 * 16 * 24 * 6),  # Climate cube size
        )

        # Spectrum prediction head
        self.spectrum_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, 1000 * 2),  # Spectrum size
        )

        logger.info("🎯 Built prediction heads for climate and spectrum tasks")

    def forward(self, batch_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Multi-modal encoding
        encoder_outputs = self.encoder(batch_data)
        fused_features = encoder_outputs["fused_features"]

        predictions = {}

        # Climate prediction
        if "climate_cubes" in batch_data:
            climate_pred = self.climate_head(fused_features)
            # Reshape to match input dimensions
            batch_size = batch_data["climate_cubes"].shape[0]
            climate_pred = climate_pred.view(batch_size, 2, 8, 16, 24, 6)
            predictions["climate_prediction"] = climate_pred

        # Spectrum prediction
        if "spectra" in batch_data:
            spectrum_pred = self.spectrum_head(fused_features)
            # Reshape to match input dimensions
            batch_size = fused_features.shape[0]
            spectrum_pred = spectrum_pred.view(batch_size, 1000, 2)
            predictions["spectrum_prediction"] = spectrum_pred

        return predictions, encoder_outputs

    def training_step(self, batch: MultiModalBatch, batch_idx: int) -> torch.Tensor:
        """Training step"""
        # Convert batch to dict format
        batch_data = self._batch_to_dict(batch)

        # Forward pass
        if self.training_config.use_mixed_precision:
            with autocast():
                predictions, encoder_outputs = self(batch_data)
                losses = self.loss_fn(predictions, batch_data, encoder_outputs, self.current_epoch)
        else:
            predictions, encoder_outputs = self(batch_data)
            losses = self.loss_fn(predictions, batch_data, encoder_outputs, self.current_epoch)

        total_loss = losses["total"]

        # Log losses
        for name, loss in losses.items():
            self.log(
                f"train/{name}_loss", loss, on_step=True, on_epoch=True, prog_bar=(name == "total")
            )
            self.train_metrics[f"{name}_loss"].append(loss.item())

        # Log learning rate
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True)

        return total_loss

    def validation_step(self, batch: MultiModalBatch, batch_idx: int) -> torch.Tensor:
        """Validation step"""
        batch_data = self._batch_to_dict(batch)

        # Forward pass
        predictions, encoder_outputs = self(batch_data)
        losses = self.loss_fn(predictions, batch_data, encoder_outputs, self.current_epoch)

        total_loss = losses["total"]

        # Log losses
        for name, loss in losses.items():
            self.log(
                f"val/{name}_loss", loss, on_step=False, on_epoch=True, prog_bar=(name == "total")
            )
            self.val_metrics[f"{name}_loss"].append(loss.item())

        return total_loss

    def _batch_to_dict(self, batch: MultiModalBatch) -> Dict[str, Any]:
        """Convert MultiModalBatch to dictionary format"""
        batch_data = {"planet_params": batch.planet_params}

        if batch.climate_cubes is not None:
            batch_data["climate_cubes"] = batch.climate_cubes

        if batch.bio_graphs is not None:
            batch_data["bio_graphs"] = batch.bio_graphs

        if batch.spectra is not None:
            batch_data["spectra"] = batch.spectra

        return batch_data

    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        # Optimizer
        if self.training_config.optimization_strategy == OptimizationStrategy.ADAMW:
            optimizer = AdamW(
                self.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay,
            )
        elif self.training_config.optimization_strategy == OptimizationStrategy.SGD:
            optimizer = SGD(
                self.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay,
                momentum=0.9,
            )
        else:
            optimizer = AdamW(
                self.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay,
            )

        # Scheduler
        if self.training_config.optimization_strategy == OptimizationStrategy.ADAMW_COSINE:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.training_config.max_epochs,
                eta_min=self.training_config.min_lr,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }
        elif self.training_config.optimization_strategy == OptimizationStrategy.ADAMW_ONECYCLE:
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.training_config.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.3,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        else:
            return optimizer

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        # Get current validation loss
        if self.val_metrics["total_loss"]:
            current_val_loss = np.mean(self.val_metrics["total_loss"][-10:])

            # Early stopping logic
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Log patience info
            self.log("val/patience", self.patience_counter)
            self.log("val/best_loss", self.best_val_loss)

    def on_train_epoch_end(self):
        """Called at the end of training epoch"""
        # Log epoch metrics
        if self.train_metrics["total_loss"]:
            avg_train_loss = np.mean(self.train_metrics["total_loss"][-100:])
            self.log("train/avg_loss", avg_train_loss)


def create_trainer(
    encoder_config: EncoderConfig, training_config: TrainingConfig
) -> Tuple[MultiModalTrainer, pl.Trainer]:
    """Create trainer and PyTorch Lightning trainer"""

    # Lightning module
    model = MultiModalTrainer(encoder_config, training_config)

    # Callbacks
    callbacks = []

    # Model checkpoint
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/total_loss",
        mode="min",
        save_top_k=training_config.save_top_k,
        filename="multimodal-{epoch:02d}-{val_total_loss:.3f}",
        auto_insert_metric_name=False,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val/total_loss", mode="min", patience=training_config.patience, verbose=True
    )
    callbacks.append(early_stop_callback)

    # Learning rate monitor
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Model summary
    model_summary = pl.callbacks.ModelSummary(max_depth=2)
    callbacks.append(model_summary)

    # Logger
    logger_list = []

    # Wandb logger
    if training_config.use_wandb:
        try:
            wandb_logger = pl.loggers.WandbLogger(
                project=training_config.wandb_project,
                tags=training_config.wandb_tags,
                config={
                    "encoder_config": encoder_config.__dict__,
                    "training_config": training_config.__dict__,
                },
            )
            logger_list.append(wandb_logger)
        except Exception as e:
            logger.warning(f"Failed to initialize Wandb logger: {e}")

    # TensorBoard logger
    tb_logger = pl.loggers.TensorBoardLogger("lightning_logs", name="multimodal")
    logger_list.append(tb_logger)

    # PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=training_config.max_epochs,
        accelerator="auto",
        devices=training_config.num_gpus if training_config.num_gpus > 0 else "auto",
        precision="16-mixed" if training_config.use_mixed_precision else 32,
        gradient_clip_val=training_config.gradient_clip_val,
        accumulate_grad_batches=training_config.accumulate_grad_batches,
        log_every_n_steps=training_config.log_every_n_steps,
        val_check_interval=training_config.val_check_interval,
        callbacks=callbacks,
        logger=logger_list,
        deterministic=False,
        benchmark=True,  # Optimize for performance
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    return model, trainer


if __name__ == "__main__":
    # Test the enhanced training workflow
    def test_training_workflow():
        logger.info("🧪 Testing Enhanced Training Workflow")

        # Create configs
        encoder_config = EncoderConfig(
            latent_dim=128, use_physics_constraints=True  # Smaller for testing
        )

        training_config = TrainingConfig(
            max_epochs=3,  # Short test
            batch_size=2,
            learning_rate=1e-3,
            loss_strategy=LossStrategy.ADAPTIVE,
            optimization_strategy=OptimizationStrategy.ADAMW,
            use_mixed_precision=True,  # ✅ RE-ENABLED - Mixed precision for better performance
            use_wandb=True,  # ✅ RE-ENABLED - Weights & Biases logging for monitoring
            patience=10,
        )

        # Create model and trainer
        model, trainer = create_trainer(encoder_config, training_config)

        # Create mock data loaders
        from data_build.unified_dataloader_fixed import MockDataStorage

        mock_storage = MockDataStorage(n_runs=20)

        dataloader_config = DataLoaderConfig(
            batch_size=training_config.batch_size, num_workers=0, pin_memory=False  # For testing
        )

        train_loader, val_loader, _ = create_multimodal_dataloaders(dataloader_config, mock_storage)

        logger.info("🔄 Starting training test...")

        # Train for a few steps
        trainer.fit(model, train_loader, val_loader)

        logger.info("✅ Training workflow test completed!")

        # Show training statistics
        print("\n" + "=" * 60)
        print("🚀 ENHANCED TRAINING WORKFLOW STATISTICS")
        print("=" * 60)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Training config: {training_config.optimization_strategy.value}")
        print(f"Loss strategy: {training_config.loss_strategy.value}")
        print(f"Mixed precision: {training_config.use_mixed_precision}")
        print(f"Gradient clipping: {training_config.gradient_clip_val}")
        print("\nMulti-task losses:")
        print("  ✅ Climate reconstruction (MSE + physics)")
        print("  ✅ Spectral synthesis (MAE + features)")
        print("  ✅ Physics constraints (energy, mass)")
        print("  ✅ Multi-modal consistency")
        print("\nOptimization features:")
        print("  ✅ Adaptive loss weighting")
        print("  ✅ Learning rate scheduling")
        print("  ✅ Early stopping")
        print("  ✅ Model checkpointing")
        print("  ✅ Performance monitoring")
        print("=" * 60)

    # Run test
    test_training_workflow()

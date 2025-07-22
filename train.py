#!/usr/bin/env python3
"""
Enhanced Training Script for Astrobiology Platform
================================================

World-class training script that leverages the Enhanced Training Orchestrator to support
all advanced models, data systems, and training strategies.

Supports:
- All Enhanced Models: 5D Datacube U-Net, Multi-Modal Surrogate, Evolutionary Tracker, etc.
- Advanced Training Strategies: Meta-learning, Federated Learning, Neural Architecture Search
- Data System Integration: Advanced data management, quality systems, customer data treatment
- Performance Optimization: Mixed precision, distributed training, memory optimization
- Comprehensive Monitoring: Real-time training monitoring, diagnostics integration

Features:
- Unified training interface for all models
- Multi-modal training coordination
- Physics-informed loss functions
- Advanced optimization strategies
- Automated architecture search
- Customer data treatment training
- Federated learning capabilities
- Real-time training monitoring
- Memory-efficient training
- Distributed training support

Usage:
    # Single model training
    python train.py --model enhanced_datacube --config config/enhanced_cube.yaml
    
    # Multi-modal training
    python train.py --mode multi_modal --models enhanced_datacube,enhanced_surrogate
    
    # Meta-learning
    python train.py --mode meta_learning --episodes 1000 --support-shots 5
    
    # Federated learning
    python train.py --mode federated_learning --participants 10 --rounds 100
    
    # Neural Architecture Search
    python train.py --mode neural_architecture_search --search-epochs 50
"""

from __future__ import annotations
import os
import sys
import argparse
import asyncio
import logging
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

# Set environment variables before importing torch
os.environ.setdefault('TORCH_VISION_DISABLE', '1')

import torch
import pytorch_lightning as pl
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Disable unnecessary PyTorch extensions
try:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
except:
    pass

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Enhanced training imports
try:
    from training.enhanced_training_orchestrator import (
        EnhancedTrainingOrchestrator, 
        EnhancedTrainingConfig,
        TrainingMode,
        OptimizationStrategy,
        LossStrategy,
        train_enhanced_datacube,
        train_multimodal_system,
        create_enhanced_training_orchestrator
    )
    ENHANCED_TRAINING_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Enhanced training not available: {e}")
    ENHANCED_TRAINING_AVAILABLE = False

try:
    from training.enhanced_model_training_modules import (
        Enhanced5DDatacubeTrainingModule,
        EnhancedSurrogateTrainingModule,
        MetaLearningTrainingModule,
        CustomerDataTrainingModule,
        create_enhanced_5d_training_module,
        create_enhanced_surrogate_training_module,
        create_meta_learning_training_module,
        create_customer_data_training_module
    )
    ENHANCED_MODULES_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Enhanced training modules not available: {e}")
    ENHANCED_MODULES_AVAILABLE = False

# Legacy imports for backward compatibility
from utils.config import parse_cli
from models.graph_vae import GVAE
from models.fusion_transformer import FusionModel
from models.surrogate_transformer import SurrogateTransformer, UncertaintyQuantification
from scripts.train_gvae_dummy import random_graph
from scripts.train_fusion_dummy import schema as FUSION_SCHEMA, to_tensor
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import DataLoader as GeometricDataLoader

# Conditional imports for optional dependencies
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Conditional pytorch lightning logger import
if WANDB_AVAILABLE:
    from pytorch_lightning.loggers import WandbLogger
else:
    WandbLogger = None

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Legacy Lightning modules for backward compatibility
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
    """Enhanced Lightning module for physics-informed climate modeling"""
    
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
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            planet_params, targets = batch
        else:
            # Handle different batch formats
            planet_params = batch
            targets = batch  # Placeholder
        
        # Forward pass
        outputs = self.model(planet_params)
        
        # Compute comprehensive loss
        if hasattr(self.model, 'compute_total_loss'):
            losses = self.model.compute_total_loss(outputs, targets)
        else:
            # Fallback to simple MSE
            losses = {'total_loss': torch.nn.functional.mse_loss(outputs.get('predictions', outputs), targets)}
        
        # Log all loss components
        for loss_name, loss_value in losses.items():
            self.log(f'train/{loss_name}', loss_value, prog_bar=(loss_name == 'total_loss'))
        
        # Log physics constraint weights if available
        if 'physics_weights' in outputs:
            weights = torch.nn.functional.softplus(outputs['physics_weights'])
            for i, weight in enumerate(weights):
                self.log(f'train/physics_weight_{i}', weight)
        
        return losses['total_loss']
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Validation step with uncertainty quantification"""
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            planet_params, targets = batch
        else:
            planet_params = batch
            targets = batch
        
        # Standard prediction
        outputs = self.model(planet_params)
        
        if hasattr(self.model, 'compute_total_loss'):
            losses = self.model.compute_total_loss(outputs, targets)
        else:
            losses = {'total_loss': torch.nn.functional.mse_loss(outputs.get('predictions', outputs), targets)}
        
        # Log validation losses
        for loss_name, loss_value in losses.items():
            self.log(f'val/{loss_name}', loss_value, prog_bar=(loss_name == 'total_loss'))
        
        # Store for epoch-end validation
        self.validation_predictions.append(outputs)
        self.validation_targets.append(targets)
        
        # Uncertainty quantification on subset of validation data
        if batch_idx % 10 == 0:  # Sample every 10th batch
            try:
                uncertainty_outputs = self.uncertainty_model.predict_with_uncertainty(planet_params)
                
                # Log uncertainty metrics
                for key in uncertainty_outputs:
                    if key.endswith('_std'):
                        mean_uncertainty = uncertainty_outputs[key].mean()
                        self.log(f'val/uncertainty_{key}', mean_uncertainty)
            except Exception as e:
                logger.debug(f"Uncertainty computation failed: {e}")
        
        return losses['total_loss']
    
    def on_validation_epoch_end(self):
        """Comprehensive validation metrics at epoch end"""
        if not self.validation_predictions:
            return
        
        try:
            # Aggregate predictions and targets
            all_predictions = {}
            all_targets = {}
            
            for pred_dict in self.validation_predictions:
                if isinstance(pred_dict, dict):
                    for key, value in pred_dict.items():
                        if key not in all_predictions:
                            all_predictions[key] = []
                        if isinstance(value, torch.Tensor):
                            all_predictions[key].append(value)
            
            # Handle targets
            for target in self.validation_targets:
                if isinstance(target, torch.Tensor):
                    if 'targets' not in all_targets:
                        all_targets['targets'] = []
                    all_targets['targets'].append(target)
            
            # Compute R² scores for continuous variables
            for key in all_targets:
                if key in all_predictions and len(all_predictions[key]) > 0:
                    try:
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
                    except Exception as e:
                        logger.debug(f"Metric computation failed for {key}: {e}")
        except Exception as e:
            logger.debug(f"Validation epoch end failed: {e}")
        
        # Clear for next epoch
        self.validation_predictions.clear()
        self.validation_targets.clear()
    
    def configure_optimizers(self):
        """Advanced optimizer configuration with scheduling"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.get("training", {}).get("learning_rate", 1e-4),
            weight_decay=self.hparams.get("training", {}).get("weight_decay", 1e-5),
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

def parse_enhanced_args():
    """Parse command line arguments for enhanced training"""
    parser = argparse.ArgumentParser(description="Enhanced Training Script for Astrobiology Platform")
    
    # Training mode
    parser.add_argument("--mode", type=str, default="single_model",
                       choices=["single_model", "multi_modal", "meta_learning", 
                               "federated_learning", "neural_architecture_search", 
                               "evolutionary_training", "customer_data_training"],
                       help="Training mode")
    
    # Model selection
    parser.add_argument("--model", type=str, default="enhanced_datacube",
                       choices=["enhanced_datacube", "enhanced_surrogate", "evolutionary_tracker",
                               "uncertainty_emergence", "neural_architecture_search", "meta_learning",
                               "peft_llm", "advanced_gnn", "domain_encoders", "graph_vae", 
                               "fusion", "surrogate"],
                       help="Model to train")
    
    parser.add_argument("--models", type=str, nargs="+",
                       help="Multiple models for multi-modal training")
    
    # Configuration
    parser.add_argument("--config", type=str, default=None,
                       help="Configuration file path")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size")
    
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate")
    
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                       help="Weight decay")
    
    # Advanced training options
    parser.add_argument("--use-mixed-precision", action="store_true",
                       help="Use mixed precision training")
    
    parser.add_argument("--use-physics-constraints", action="store_true", default=True,
                       help="Use physics-informed constraints")
    
    parser.add_argument("--physics-weight", type=float, default=0.2,
                       help="Weight for physics constraints")
    
    # Meta-learning parameters
    parser.add_argument("--episodes", type=int, default=1000,
                       help="Number of episodes for meta-learning")
    
    parser.add_argument("--support-shots", type=int, default=5,
                       help="Number of support shots for meta-learning")
    
    parser.add_argument("--query-shots", type=int, default=15,
                       help="Number of query shots for meta-learning")
    
    # Federated learning parameters
    parser.add_argument("--participants", type=int, default=10,
                       help="Number of participants for federated learning")
    
    parser.add_argument("--rounds", type=int, default=100,
                       help="Number of federated rounds")
    
    # Neural Architecture Search parameters
    parser.add_argument("--search-epochs", type=int, default=50,
                       help="Number of epochs for architecture search")
    
    parser.add_argument("--search-space-size", type=int, default=1000,
                       help="Size of architecture search space")
    
    # Data parameters
    parser.add_argument("--data-path", type=str, default="data/processed",
                       help="Path to training data")
    
    parser.add_argument("--zarr-root", type=str, default=None,
                       help="Path to zarr data root")
    
    parser.add_argument("--use-customer-data", action="store_true",
                       help="Use customer data treatment")
    
    # Performance options
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of data loading workers")
    
    parser.add_argument("--distributed", action="store_true",
                       help="Use distributed training")
    
    # Monitoring
    parser.add_argument("--use-wandb", action="store_true",
                       help="Use Weights & Biases logging")
    
    parser.add_argument("--use-tensorboard", action="store_true", default=True,
                       help="Use TensorBoard logging")
    
    parser.add_argument("--use-profiler", action="store_true",
                       help="Use PyTorch profiler")
    
    # Legacy support
    parser.add_argument("--legacy", action="store_true",
                       help="Use legacy training mode")
    
    return parser.parse_args()

async def main_enhanced():
    """Main function for enhanced training"""
    args = parse_enhanced_args()
    
    logger.info("🚀 Enhanced Training Script Starting...")
    logger.info(f"   Training Mode: {args.mode}")
    logger.info(f"   Model(s): {args.model if args.model else args.models}")
    logger.info(f"   Enhanced Training Available: {ENHANCED_TRAINING_AVAILABLE}")
    logger.info(f"   Enhanced Modules Available: {ENHANCED_MODULES_AVAILABLE}")
    
    if not ENHANCED_TRAINING_AVAILABLE:
        logger.warning("Enhanced training not available, falling back to legacy training")
        return await main_legacy(args)
    
    # Create enhanced training configuration
    config = EnhancedTrainingConfig(
        training_mode=TrainingMode(args.mode),
        model_name=args.model,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_mixed_precision=args.use_mixed_precision,
        use_physics_constraints=args.use_physics_constraints,
        physics_weight=args.physics_weight,
        episodes_per_epoch=args.episodes,
        support_shots=args.support_shots,
        query_shots=args.query_shots,
        num_participants=args.participants,
        federation_rounds=args.rounds,
        search_epochs=args.search_epochs,
        search_space_size=args.search_space_size,
        data_path=args.data_path,
        zarr_root=args.zarr_root,
        use_customer_data=args.use_customer_data,
        num_workers=args.num_workers,
        use_distributed=args.distributed,
        use_wandb=args.use_wandb,
        use_tensorboard=args.use_tensorboard,
        use_profiler=args.use_profiler
    )
    
    # Create enhanced training orchestrator
    orchestrator = EnhancedTrainingOrchestrator(config)
    
    # Configure training based on mode
    if args.mode == "single_model":
        training_config = {
            'model_name': args.model,
            'model_config': get_model_config(args.model),
            'data_config': get_data_config(args),
            'training_config': config.__dict__
        }
        
        results = await orchestrator.train_model('single_model', training_config)
    
    elif args.mode == "multi_modal":
        models = args.models or ['enhanced_datacube', 'enhanced_surrogate']
        models_config = {model: get_model_config(model) for model in models}
        data_configs = {'main': get_data_config(args)}
        
        training_config = {
            'models_config': models_config,
            'data_configs': data_configs,
            'training_config': config.__dict__
        }
        
        results = await orchestrator.train_model('multi_modal', training_config)
    
    elif args.mode == "meta_learning":
        training_config = {
            'model_config': get_model_config('meta_learning'),
            'episodes_config': {
                'episodes_per_epoch': args.episodes,
                'support_shots': args.support_shots,
                'query_shots': args.query_shots
            },
            'training_config': config.__dict__
        }
        
        results = await orchestrator.train_model('meta_learning', training_config)
    
    elif args.mode == "federated_learning":
        training_config = {
            'participants_config': {
                'num_participants': args.participants,
                'federation_rounds': args.rounds,
                'fed_config': {
                    'aggregation_strategy': 'fedavg',
                    'privacy_mechanism': 'differential_privacy'
                }
            },
            'training_config': config.__dict__
        }
        
        results = await orchestrator.train_model('federated_learning', training_config)
    
    elif args.mode == "neural_architecture_search":
        training_config = {
            'search_config': {
                'search_space_size': args.search_space_size,
                'search_epochs': args.search_epochs,
                'model_config': get_model_config('neural_architecture_search')
            },
            'training_config': config.__dict__
        }
        
        results = await orchestrator.train_model('neural_architecture_search', training_config)
    
    elif args.mode == "customer_data_training":
        training_config = {
            'customer_data_config': {
                'use_federated_learning': True,
                'use_differential_privacy': True,
                'quantum_enhanced': True
            },
            'training_config': config.__dict__
        }
        
        results = await orchestrator.train_model('customer_data_training', training_config)
    
    else:
        raise ValueError(f"Unknown training mode: {args.mode}")
    
    # Save results
    results_file = f"enhanced_training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("✅ Enhanced training completed successfully!")
    logger.info(f"📄 Results saved to: {results_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("🎯 ENHANCED TRAINING COMPLETE")
    print("="*80)
    print(f"Training Mode: {args.mode}")
    print(f"Model(s): {args.model if args.model else args.models}")
    print(f"Status: {'Success' if 'error' not in results else 'Failed'}")
    if 'training_time' in results:
        print(f"Training Time: {results['training_time']:.2f} seconds")
    if 'best_loss' in results:
        print(f"Best Loss: {results['best_loss']:.6f}")
    print(f"Results File: {results_file}")
    print("="*80)
    
    return results

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get model configuration for specified model"""
    configs = {
        'enhanced_datacube': {
            'n_input_vars': 5,
            'n_output_vars': 5,
            'input_variables': ['temperature', 'pressure', 'humidity', 'velocity_u', 'velocity_v'],
            'output_variables': ['temperature', 'pressure', 'humidity', 'velocity_u', 'velocity_v'],
            'base_features': 64,
            'depth': 5,
            'use_attention': True,
            'use_transformer': True,
            'use_separable_conv': True,
            'use_gradient_checkpointing': True,
            'use_mixed_precision': True,
            'model_scaling': "efficient",
            'use_physics_constraints': True,
            'physics_weight': 0.2,
            'learning_rate': 2e-4,
            'weight_decay': 1e-4
        },
        'enhanced_surrogate': {
            'multimodal_config': {
                'use_datacube': True,
                'use_scalar_params': True,
                'use_spectral_data': True,
                'use_temporal_sequences': True,
                'fusion_strategy': "cross_attention",
                'num_attention_heads': 8,
                'hidden_dim': 256
            },
            'use_uncertainty': True,
            'use_dynamic_selection': True,
            'use_mixed_precision': True,
            'learning_rate': 1e-4
        },
        'meta_learning': {
            'input_dim': 100,
            'hidden_dim': 256,
            'output_dim': 10,
            'num_layers': 4,
            'meta_lr': 1e-3,
            'inner_lr': 1e-2,
            'adaptation_steps': 5
        },
        'neural_architecture_search': {
            'search_space': 'efficient_net',
            'population_size': 50,
            'generations': 20,
            'mutation_rate': 0.1
        },
        'graph_vae': {
            'latent': 8
        },
        'fusion': {
            'hidden_dim': 64,
            'num_layers': 3
        },
        'surrogate': {
            'dim': 256,
            'depth': 8,
            'heads': 8,
            'n_inputs': 8,
            'mode': "scalar",
            'dropout': 0.1
        }
    }
    
    return configs.get(model_name, {})

def get_data_config(args) -> Dict[str, Any]:
    """Get data configuration"""
    return {
        'data_path': args.data_path,
        'zarr_root': args.zarr_root,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'use_customer_data': args.use_customer_data,
        'synthetic_size': 1000  # For synthetic data generation
    }

async def main_legacy(args) -> Dict[str, Any]:
    """Legacy training function for backward compatibility"""
    logger.info("🔄 Running legacy training mode...")
    
    # Use original config parsing if available
    try:
        cfg, _ = parse_cli()
    except:
        # Fallback configuration
        cfg = {
            "model": {"type": args.model or "surrogate"},
            "data": {"synthetic_size": 1000},
            "trainer": {
                "max_epochs": args.epochs,
                "batch_size": args.batch_size,
                "accelerator": "auto",
                "precision": "16-mixed" if args.use_mixed_precision else 32
            },
            "logging": {"use_wandb": args.use_wandb}
        }
    
    pl.seed_everything(42)
    
    # Setup logging
    if cfg.get("logging", {}).get("use_wandb", False) and WANDB_AVAILABLE and WandbLogger is not None:
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

    start_time = time.time()

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
        precision=cfg["trainer"].get("precision", "16-mixed"),
        default_root_dir="lightning_logs",
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        accumulate_grad_batches=cfg["trainer"].get("accumulate_grad_batches", 1),
    )
    
    if cfg["model"]["type"] == "surrogate":
        trainer.fit(module, train_dl, val_dl)
    else:
        trainer.fit(module, dl)
    
    training_time = time.time() - start_time
    
    # Return results in consistent format
    results = {
        'training_mode': 'legacy',
        'model_type': cfg["model"]["type"],
        'training_time': training_time,
        'best_loss': float(trainer.callback_metrics.get('val/total_loss', trainer.callback_metrics.get('loss', 0.0))),
        'total_epochs': trainer.current_epoch,
        'status': 'completed'
    }
    
    return results

def main():
    """Main entry point"""
    args = parse_enhanced_args()
    
    if args.legacy or not ENHANCED_TRAINING_AVAILABLE:
        # Run legacy training synchronously
        results = asyncio.run(main_legacy(args))
    else:
        # Run enhanced training
        results = asyncio.run(main_enhanced())
    
    return results

if __name__ == "__main__":
    results = main()
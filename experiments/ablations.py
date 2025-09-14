#!/usr/bin/env python3
"""
Comprehensive Ablation Study Framework
=====================================

Systematic ablation studies to prove necessity of each modeling choice:
- Physics loss terms vs standard MSE
- Temporal attention mechanisms vs feedforward
- Multi-scale architectures vs single-scale
- Cross-modal fusion vs single-modal
- Uncertainty quantification vs deterministic

Essential for ISEF competition to demonstrate scientific rigor and
justify architectural choices with quantitative evidence.

Author: Astrobio Research Team
"""

import asyncio
import json
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Statistical analysis
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


class PhysicsInformedLoss(nn.Module):
    """Physics-informed loss function with conservation constraints"""
    
    def __init__(self, physics_weight: float = 1.0):
        super().__init__()
        self.physics_weight = physics_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        metadata: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute physics-informed loss
        
        Args:
            predictions: Model predictions [batch, ...]
            targets: Ground truth targets [batch, ...]
            metadata: Additional physics information
            
        Returns:
            Total loss and component losses
        """
        # Standard reconstruction loss
        reconstruction_loss = self.mse_loss(predictions, targets)
        
        loss_components = {
            'reconstruction': reconstruction_loss,
            'physics_total': torch.tensor(0.0, device=predictions.device)
        }
        
        if metadata is not None and self.physics_weight > 0:
            physics_losses = []
            
            # Energy conservation constraint
            if 'energy_input' in metadata and 'energy_output' in metadata:
                energy_conservation = torch.mean(
                    (metadata['energy_input'] - metadata['energy_output'])**2
                )
                physics_losses.append(energy_conservation)
                loss_components['energy_conservation'] = energy_conservation
            
            # Mass conservation constraint
            if 'mass_input' in metadata and 'mass_output' in metadata:
                mass_conservation = torch.mean(
                    (metadata['mass_input'] - metadata['mass_output'])**2
                )
                physics_losses.append(mass_conservation)
                loss_components['mass_conservation'] = mass_conservation
            
            # Temperature gradient constraint (lapse rate)
            if predictions.dim() >= 3:  # Assume last dim is altitude/pressure
                temp_gradients = torch.gradient(predictions, dim=-1)[0]
                # Reasonable atmospheric lapse rate: 6-10 K/km
                lapse_rate_penalty = torch.mean(
                    torch.relu(torch.abs(temp_gradients) - 0.01)**2
                )
                physics_losses.append(lapse_rate_penalty)
                loss_components['lapse_rate'] = lapse_rate_penalty
            
            # Radiative equilibrium constraint
            if 'incoming_radiation' in metadata and 'outgoing_radiation' in metadata:
                radiative_balance = torch.mean(
                    (metadata['incoming_radiation'] - metadata['outgoing_radiation'])**2
                )
                physics_losses.append(radiative_balance)
                loss_components['radiative_balance'] = radiative_balance
            
            # Combine physics losses
            if physics_losses:
                total_physics_loss = sum(physics_losses)
                loss_components['physics_total'] = total_physics_loss
            else:
                total_physics_loss = torch.tensor(0.0, device=predictions.device)
        else:
            total_physics_loss = torch.tensor(0.0, device=predictions.device)
        
        # Total loss
        total_loss = reconstruction_loss + self.physics_weight * total_physics_loss
        
        return total_loss, loss_components


class TemporalAttentionModule(nn.Module):
    """Temporal attention mechanism for time series modeling"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_heads: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, input_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal attention
        
        Args:
            x: Input tensor [batch, sequence_length, features]
            
        Returns:
            Attended output tensor
        """
        # Self-attention
        attn_output, _ = self.multihead_attn(x, x, x)
        x = self.norm1(x + attn_output)
        
        # Feedforward
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x


class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale feature extraction with different receptive fields"""
    
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        
        # Different scales of convolutions
        self.scale1 = nn.Conv2d(input_channels, output_channels // 4, kernel_size=3, padding=1)
        self.scale2 = nn.Conv2d(input_channels, output_channels // 4, kernel_size=5, padding=2)
        self.scale3 = nn.Conv2d(input_channels, output_channels // 4, kernel_size=7, padding=3)
        self.scale4 = nn.Conv2d(input_channels, output_channels // 4, kernel_size=9, padding=4)
        
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm2d(output_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract multi-scale features
        
        Args:
            x: Input tensor [batch, channels, height, width]
            
        Returns:
            Multi-scale feature tensor
        """
        # Extract features at different scales
        feat1 = self.activation(self.scale1(x))
        feat2 = self.activation(self.scale2(x))
        feat3 = self.activation(self.scale3(x))
        feat4 = self.activation(self.scale4(x))
        
        # Concatenate features
        multi_scale_features = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        
        # Normalize
        return self.norm(multi_scale_features)


class AblationTestModel(nn.Module):
    """Configurable model for ablation studies"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        use_physics_loss: bool = True,
        use_temporal_attention: bool = True,
        use_multi_scale: bool = True,
        use_uncertainty: bool = True,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.use_physics_loss = use_physics_loss
        self.use_temporal_attention = use_temporal_attention
        self.use_multi_scale = use_multi_scale
        self.use_uncertainty = use_uncertainty
        
        # Input processing
        if use_multi_scale and len(input_dim) == 4:  # 2D spatial data
            self.feature_extractor = MultiScaleFeatureExtractor(input_dim[1], hidden_dim)
            current_dim = hidden_dim
        else:
            self.feature_extractor = nn.Linear(np.prod(input_dim), hidden_dim)
            current_dim = hidden_dim
        
        # Temporal processing
        if use_temporal_attention:
            self.temporal_processor = TemporalAttentionModule(current_dim)
        else:
            self.temporal_processor = nn.Sequential(
                nn.Linear(current_dim, current_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        
        # Output layers
        if use_uncertainty:
            # Predict both mean and variance
            self.output_mean = nn.Linear(current_dim, output_dim)
            self.output_var = nn.Linear(current_dim, output_dim)
        else:
            # Deterministic output
            self.output_layer = nn.Linear(current_dim, output_dim)
        
        # Physics loss (configured externally)
        if use_physics_loss:
            self.physics_loss = PhysicsInformedLoss(physics_weight=1.0)
        else:
            self.physics_loss = PhysicsInformedLoss(physics_weight=0.0)
    
    def forward(self, x: torch.Tensor, metadata: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with configurable components
        
        Args:
            x: Input tensor
            metadata: Physics metadata for loss computation
            
        Returns:
            Dictionary with predictions and uncertainty estimates
        """
        batch_size = x.shape[0]
        
        # Feature extraction
        if self.use_multi_scale and x.dim() == 4:  # 2D spatial
            features = self.feature_extractor(x)
            features = features.mean(dim=[2, 3])  # Global average pooling
        else:
            features = self.feature_extractor(x.view(batch_size, -1))
        
        # Add sequence dimension if needed for temporal processing
        if features.dim() == 2:
            features = features.unsqueeze(1)  # [batch, 1, features]
        
        # Temporal processing
        temporal_features = self.temporal_processor(features)
        
        # Remove sequence dimension
        if temporal_features.shape[1] == 1:
            temporal_features = temporal_features.squeeze(1)
        
        # Output prediction
        if self.use_uncertainty:
            mean = self.output_mean(temporal_features)
            log_var = self.output_var(temporal_features)
            var = torch.exp(log_var)
            
            return {
                'mean': mean,
                'variance': var,
                'prediction': mean,  # For compatibility
                'uncertainty': torch.sqrt(var)
            }
        else:
            prediction = self.output_layer(temporal_features)
            return {
                'prediction': prediction,
                'mean': prediction,
                'variance': torch.zeros_like(prediction),
                'uncertainty': torch.zeros_like(prediction)
            }
    
    def compute_loss(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: torch.Tensor,
        metadata: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute loss with physics constraints"""
        
        pred_values = predictions['mean']
        
        if self.use_physics_loss:
            return self.physics_loss(pred_values, targets, metadata)
        else:
            mse_loss = nn.MSELoss()(pred_values, targets)
            return mse_loss, {'reconstruction': mse_loss, 'physics_total': torch.tensor(0.0)}


class AblationStudyFramework:
    """Comprehensive framework for ablation studies"""
    
    def __init__(
        self,
        input_dim: Union[int, Tuple[int, ...]],
        output_dim: int,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        results_path: str = "results"
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.results_path = Path(results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Ablation configurations
        self.ablation_configs = {
            'full_model': {
                'use_physics_loss': True,
                'use_temporal_attention': True,
                'use_multi_scale': True,
                'use_uncertainty': True,
                'description': 'Full model with all components'
            },
            'no_physics': {
                'use_physics_loss': False,
                'use_temporal_attention': True,
                'use_multi_scale': True,
                'use_uncertainty': True,
                'description': 'Model without physics-informed loss'
            },
            'no_temporal_attention': {
                'use_physics_loss': True,
                'use_temporal_attention': False,
                'use_multi_scale': True,
                'use_uncertainty': True,
                'description': 'Model without temporal attention mechanism'
            },
            'no_multi_scale': {
                'use_physics_loss': True,
                'use_temporal_attention': True,
                'use_multi_scale': False,
                'use_uncertainty': True,
                'description': 'Model without multi-scale feature extraction'
            },
            'no_uncertainty': {
                'use_physics_loss': True,
                'use_temporal_attention': True,
                'use_multi_scale': True,
                'use_uncertainty': False,
                'description': 'Deterministic model without uncertainty quantification'
            },
            'minimal_model': {
                'use_physics_loss': False,
                'use_temporal_attention': False,
                'use_multi_scale': False,
                'use_uncertainty': False,
                'description': 'Minimal baseline model'
            }
        }
        
        # Results storage
        self.ablation_results = {}
        
        logger.info(f"🔬 Ablation Study Framework initialized")
        logger.info(f"   Input dimension: {input_dim}")
        logger.info(f"   Output dimension: {output_dim}")
        logger.info(f"   Device: {device}")
        logger.info(f"   Ablation configurations: {len(self.ablation_configs)}")
    
    def generate_synthetic_data(
        self, 
        num_samples: int = 1000,
        add_physics_constraints: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Generate synthetic data for ablation studies
        
        Args:
            num_samples: Number of samples to generate
            add_physics_constraints: Whether to add physics metadata
            
        Returns:
            Input data, target data, and optional physics metadata
        """
        logger.info(f"🔧 Generating synthetic data: {num_samples} samples")
        
        # Generate input data
        if isinstance(self.input_dim, tuple):
            input_shape = (num_samples,) + self.input_dim
        else:
            input_shape = (num_samples, self.input_dim)
        
        # Create realistic input patterns
        np.random.seed(42)  # For reproducibility
        
        if len(input_shape) == 4:  # 2D spatial data
            # Generate spatial patterns (e.g., temperature fields)
            inputs = np.zeros(input_shape)
            for i in range(num_samples):
                # Create spatial temperature pattern
                x_coords = np.linspace(-1, 1, input_shape[2])
                y_coords = np.linspace(-1, 1, input_shape[3])
                X, Y = np.meshgrid(x_coords, y_coords)
                
                # Add multiple spatial modes
                pattern = (
                    np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) +
                    0.5 * np.sin(4 * np.pi * X) +
                    0.3 * np.random.randn(*X.shape)
                )
                
                for c in range(input_shape[1]):
                    inputs[i, c] = pattern + c * 0.1
        else:
            # Generate 1D or flattened data
            inputs = np.random.randn(*input_shape)
            
            # Add some structure (temporal correlations, etc.)
            for i in range(1, num_samples):
                inputs[i] += 0.3 * inputs[i-1]  # AR(1) process
        
        # Generate targets based on inputs with some physics
        if isinstance(self.output_dim, int):
            target_shape = (num_samples, self.output_dim)
        else:
            target_shape = (num_samples,) + self.output_dim
        
        # Create physics-based relationship
        if len(input_shape) == 4:  # Spatial data
            # Simple energy balance model
            targets = np.mean(inputs, axis=(2, 3))  # Spatial average
            targets += 0.1 * np.random.randn(*targets.shape)  # Noise
        else:
            # Linear + nonlinear relationships
            weights = np.random.randn(input_shape[-1], self.output_dim)
            targets = inputs @ weights
            targets += 0.1 * np.sin(inputs.sum(axis=1, keepdims=True))  # Nonlinearity
            targets += 0.05 * np.random.randn(*targets.shape)  # Noise
        
        # Convert to tensors
        input_tensor = torch.FloatTensor(inputs)
        target_tensor = torch.FloatTensor(targets)
        
        # Generate physics metadata if requested
        metadata = None
        if add_physics_constraints:
            metadata = {}
            
            # Energy conservation terms
            if len(input_shape) == 4:
                energy_input = torch.sum(input_tensor, dim=(2, 3))
                energy_output = torch.sum(target_tensor, dim=1, keepdim=True)
                metadata['energy_input'] = energy_input
                metadata['energy_output'] = energy_output
            
            # Mass conservation (simplified)
            mass_input = torch.sum(input_tensor, dim=-1)
            mass_output = torch.sum(target_tensor, dim=-1)
            metadata['mass_input'] = mass_input
            metadata['mass_output'] = mass_output
            
            # Radiative terms (mock)
            incoming_radiation = torch.ones_like(target_tensor) * 1361  # Solar constant
            outgoing_radiation = target_tensor ** 4 * 5.67e-8  # Stefan-Boltzmann
            metadata['incoming_radiation'] = incoming_radiation
            metadata['outgoing_radiation'] = outgoing_radiation
        
        logger.info(f"✅ Synthetic data generated: {input_tensor.shape} -> {target_tensor.shape}")
        
        return input_tensor, target_tensor, metadata
    
    def train_model(
        self,
        config_name: str,
        train_data: Tuple[torch.Tensor, torch.Tensor, Optional[Dict]],
        val_data: Tuple[torch.Tensor, torch.Tensor, Optional[Dict]],
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3
    ) -> Dict[str, Any]:
        """
        Train a model with specific ablation configuration
        
        Args:
            config_name: Name of ablation configuration
            train_data: Training data (inputs, targets, metadata)
            val_data: Validation data
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Training results and metrics
        """
        logger.info(f"🎓 Training model: {config_name}")
        
        config = self.ablation_configs[config_name]
        
        # Create model
        model = AblationTestModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            **{k: v for k, v in config.items() if k.startswith('use_')}
        ).to(self.device)
        
        # Create data loaders
        train_inputs, train_targets, train_metadata = train_data
        val_inputs, val_targets, val_metadata = val_data
        
        train_dataset = TensorDataset(train_inputs, train_targets)
        val_dataset = TensorDataset(val_inputs, val_targets)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training loop
        train_losses = []
        val_losses = []
        val_metrics = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            epoch_train_loss = 0.0
            num_train_batches = 0
            
            for batch_inputs, batch_targets in train_loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                predictions = model(batch_inputs, train_metadata)
                
                # Compute loss
                loss, loss_components = model.compute_loss(predictions, batch_targets, train_metadata)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_train_loss += loss.item()
                num_train_batches += 1
            
            avg_train_loss = epoch_train_loss / num_train_batches
            train_losses.append(avg_train_loss)
            
            # Validation phase
            model.eval()
            epoch_val_loss = 0.0
            num_val_batches = 0
            val_predictions = []
            val_true = []
            
            with torch.no_grad():
                for batch_inputs, batch_targets in val_loader:
                    batch_inputs = batch_inputs.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    predictions = model(batch_inputs, val_metadata)
                    loss, _ = model.compute_loss(predictions, batch_targets, val_metadata)
                    
                    epoch_val_loss += loss.item()
                    num_val_batches += 1
                    
                    val_predictions.append(predictions['mean'].cpu())
                    val_true.append(batch_targets.cpu())
            
            avg_val_loss = epoch_val_loss / num_val_batches
            val_losses.append(avg_val_loss)
            
            # Calculate validation metrics
            val_pred_tensor = torch.cat(val_predictions)
            val_true_tensor = torch.cat(val_true)
            
            val_mae = mean_absolute_error(val_true_tensor.numpy(), val_pred_tensor.numpy())
            val_r2 = r2_score(val_true_tensor.numpy().flatten(), val_pred_tensor.numpy().flatten())
            
            val_metrics.append({
                'mae': val_mae,
                'r2': val_r2,
                'mse': avg_val_loss
            })
            
            # Learning rate scheduling
            scheduler.step()
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), self.results_path / f"{config_name}_best_model.pth")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"   Early stopping at epoch {epoch}")
                break
            
            # Logging
            if epoch % 20 == 0:
                logger.info(f"   Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, "
                           f"Val Loss = {avg_val_loss:.6f}, Val R² = {val_r2:.4f}")
        
        training_time = time.time() - start_time
        
        # Final evaluation
        model.load_state_dict(torch.load(self.results_path / f"{config_name}_best_model.pth"))
        final_metrics = self._evaluate_model(model, val_data)
        
        results = {
            'config_name': config_name,
            'config': config,
            'training_time_seconds': training_time,
            'epochs_trained': len(train_losses),
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_metrics': val_metrics,
            'final_metrics': final_metrics,
            'model_parameters': sum(p.numel() for p in model.parameters()),
        }
        
        logger.info(f"✅ Training completed: {config_name}")
        logger.info(f"   Best validation loss: {best_val_loss:.6f}")
        logger.info(f"   Final R²: {final_metrics['r2']:.4f}")
        logger.info(f"   Training time: {training_time:.2f}s")
        
        return results
    
    def _evaluate_model(
        self,
        model: nn.Module,
        test_data: Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]
    ) -> Dict[str, float]:
        """Comprehensive model evaluation"""
        
        model.eval()
        test_inputs, test_targets, test_metadata = test_data
        
        with torch.no_grad():
            # Generate predictions
            test_inputs_device = test_inputs.to(self.device)
            predictions = model(test_inputs_device, test_metadata)
            
            pred_mean = predictions['mean'].cpu().numpy()
            pred_var = predictions['variance'].cpu().numpy()
            true_values = test_targets.numpy()
            
            # Calculate metrics
            mse = mean_squared_error(true_values, pred_mean)
            mae = mean_absolute_error(true_values, pred_mean)
            rmse = np.sqrt(mse)
            r2 = r2_score(true_values.flatten(), pred_mean.flatten())
            
            # Uncertainty metrics (if available)
            if pred_var.sum() > 0:
                # Negative log-likelihood for Gaussian
                nll = 0.5 * (np.log(2 * np.pi * pred_var) + (true_values - pred_mean)**2 / pred_var)
                mean_nll = np.mean(nll)
                
                # Coverage probability
                std_dev = np.sqrt(pred_var)
                coverage_68 = np.mean(
                    (true_values >= pred_mean - std_dev) & 
                    (true_values <= pred_mean + std_dev)
                )
                coverage_95 = np.mean(
                    (true_values >= pred_mean - 2*std_dev) & 
                    (true_values <= pred_mean + 2*std_dev)
                )
            else:
                mean_nll = 0.0
                coverage_68 = 0.0
                coverage_95 = 0.0
            
            # Physics violation metrics (if metadata available)
            physics_violations = 0.0
            if test_metadata is not None:
                # Check energy conservation
                if 'energy_input' in test_metadata and 'energy_output' in test_metadata:
                    energy_residual = torch.abs(
                        test_metadata['energy_input'] - test_metadata['energy_output']
                    )
                    physics_violations = torch.mean(energy_residual).item()
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'nll': float(mean_nll),
            'coverage_68': float(coverage_68),
            'coverage_95': float(coverage_95),
            'physics_violations': float(physics_violations),
        }
    
    async def run_comprehensive_ablation_study(
        self,
        num_samples: int = 2000,
        train_ratio: float = 0.8,
        epochs: int = 150,
        num_seeds: int = 3
    ) -> Dict[str, Any]:
        """
        Run comprehensive ablation study across all configurations
        
        Args:
            num_samples: Total number of data samples
            train_ratio: Fraction of data for training
            epochs: Training epochs per configuration
            num_seeds: Number of random seeds for statistical significance
            
        Returns:
            Complete ablation study results
        """
        logger.info("🚀 Starting comprehensive ablation study...")
        start_time = time.time()
        
        # Generate data
        inputs, targets, metadata = self.generate_synthetic_data(num_samples)
        
        # Split data
        train_size = int(num_samples * train_ratio)
        
        all_results = {}
        
        # Run ablation for each configuration and seed
        for seed in range(num_seeds):
            logger.info(f"🌱 Running ablation study with seed {seed}")
            
            # Set random seed for reproducibility
            torch.manual_seed(42 + seed)
            np.random.seed(42 + seed)
            
            # Shuffle and split data
            indices = torch.randperm(num_samples)
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            
            train_data = (
                inputs[train_indices],
                targets[train_indices],
                metadata
            )
            val_data = (
                inputs[val_indices],
                targets[val_indices],
                metadata
            )
            
            seed_results = {}
            
            # Train each configuration
            for config_name in self.ablation_configs.keys():
                logger.info(f"  🔧 Configuration: {config_name} (seed {seed})")
                
                try:
                    results = self.train_model(
                        config_name=f"{config_name}_seed{seed}",
                        train_data=train_data,
                        val_data=val_data,
                        epochs=epochs
                    )
                    seed_results[config_name] = results
                    
                except Exception as e:
                    logger.error(f"  ❌ Failed to train {config_name}: {e}")
                    seed_results[config_name] = {
                        'config_name': config_name,
                        'error': str(e),
                        'final_metrics': {'r2': 0.0, 'mse': float('inf')}
                    }
            
            all_results[f'seed_{seed}'] = seed_results
        
        # Aggregate results across seeds
        aggregated_results = self._aggregate_results_across_seeds(all_results)
        
        # Statistical analysis
        statistical_analysis = self._perform_statistical_analysis(aggregated_results)
        
        # Component importance analysis
        component_importance = self._analyze_component_importance(aggregated_results)
        
        total_time = time.time() - start_time
        
        # Final results
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'total_runtime_seconds': total_time,
            'num_seeds': num_seeds,
            'num_samples': num_samples,
            'train_ratio': train_ratio,
            'epochs_per_config': epochs,
            'configurations_tested': list(self.ablation_configs.keys()),
            'raw_results': all_results,
            'aggregated_results': aggregated_results,
            'statistical_analysis': statistical_analysis,
            'component_importance': component_importance,
            'summary': self._generate_ablation_summary(aggregated_results, statistical_analysis)
        }
        
        # Save results
        await self._save_ablation_results(final_results)
        
        logger.info(f"✅ Comprehensive ablation study completed in {total_time:.2f}s")
        
        return final_results
    
    def _aggregate_results_across_seeds(self, all_results: Dict[str, Dict]) -> Dict[str, Dict]:
        """Aggregate results across multiple random seeds"""
        
        config_names = list(self.ablation_configs.keys())
        aggregated = {}
        
        for config_name in config_names:
            # Collect metrics across seeds
            metrics_across_seeds = {
                'r2': [],
                'mse': [],
                'mae': [],
                'rmse': [],
                'nll': [],
                'coverage_68': [],
                'coverage_95': [],
                'physics_violations': [],
                'training_time_seconds': [],
                'best_val_loss': []
            }
            
            for seed_key, seed_results in all_results.items():
                if config_name in seed_results and 'final_metrics' in seed_results[config_name]:
                    final_metrics = seed_results[config_name]['final_metrics']
                    
                    for metric_name in metrics_across_seeds.keys():
                        if metric_name in final_metrics:
                            metrics_across_seeds[metric_name].append(final_metrics[metric_name])
                        elif metric_name in seed_results[config_name]:
                            metrics_across_seeds[metric_name].append(seed_results[config_name][metric_name])
            
            # Calculate statistics
            aggregated_metrics = {}
            for metric_name, values in metrics_across_seeds.items():
                if values:
                    aggregated_metrics[f'{metric_name}_mean'] = float(np.mean(values))
                    aggregated_metrics[f'{metric_name}_std'] = float(np.std(values))
                    aggregated_metrics[f'{metric_name}_min'] = float(np.min(values))
                    aggregated_metrics[f'{metric_name}_max'] = float(np.max(values))
                    aggregated_metrics[f'{metric_name}_values'] = values
            
            aggregated[config_name] = {
                'config': self.ablation_configs[config_name],
                'metrics': aggregated_metrics,
                'num_seeds': len(all_results)
            }
        
        return aggregated
    
    def _perform_statistical_analysis(self, aggregated_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Perform statistical significance testing"""
        
        statistical_tests = {}
        
        # Get full model results as baseline
        full_model_r2 = aggregated_results.get('full_model', {}).get('metrics', {}).get('r2_values', [])
        
        if not full_model_r2:
            return {'error': 'Full model results not available for statistical testing'}
        
        # Compare each ablation against full model
        for config_name, results in aggregated_results.items():
            if config_name == 'full_model':
                continue
            
            config_r2 = results.get('metrics', {}).get('r2_values', [])
            
            if len(config_r2) > 1 and len(full_model_r2) > 1:
                # Paired t-test
                try:
                    t_stat, t_p_value = ttest_rel(full_model_r2, config_r2)
                    
                    # Wilcoxon signed-rank test (non-parametric)
                    w_stat, w_p_value = wilcoxon(full_model_r2, config_r2)
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(
                        (np.var(full_model_r2) + np.var(config_r2)) / 2
                    )
                    cohens_d = (np.mean(full_model_r2) - np.mean(config_r2)) / pooled_std
                    
                    statistical_tests[config_name] = {
                        't_statistic': float(t_stat),
                        't_p_value': float(t_p_value),
                        'wilcoxon_statistic': float(w_stat),
                        'wilcoxon_p_value': float(w_p_value),
                        'cohens_d': float(cohens_d),
                        'significant_at_0.05': t_p_value < 0.05,
                        'significant_at_0.01': t_p_value < 0.01,
                        'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
                    }
                    
                except Exception as e:
                    statistical_tests[config_name] = {'error': str(e)}
        
        return statistical_tests
    
    def _analyze_component_importance(self, aggregated_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze the importance of each component"""
        
        # Get performance degradation for each ablation
        full_model_r2 = aggregated_results.get('full_model', {}).get('metrics', {}).get('r2_mean', 0.0)
        
        component_importance = {}
        
        component_mapping = {
            'no_physics': 'Physics Loss',
            'no_temporal_attention': 'Temporal Attention',
            'no_multi_scale': 'Multi-Scale Features',
            'no_uncertainty': 'Uncertainty Quantification'
        }
        
        for config_name, component_name in component_mapping.items():
            if config_name in aggregated_results:
                ablated_r2 = aggregated_results[config_name]['metrics'].get('r2_mean', 0.0)
                performance_drop = full_model_r2 - ablated_r2
                relative_importance = performance_drop / full_model_r2 if full_model_r2 > 0 else 0.0
                
                component_importance[component_name] = {
                    'performance_drop_r2': float(performance_drop),
                    'relative_importance': float(relative_importance),
                    'full_model_r2': float(full_model_r2),
                    'ablated_r2': float(ablated_r2)
                }
        
        # Rank components by importance
        ranked_components = sorted(
            component_importance.items(),
            key=lambda x: x[1]['performance_drop_r2'],
            reverse=True
        )
        
        component_importance['ranking'] = [name for name, _ in ranked_components]
        
        return component_importance
    
    def _generate_ablation_summary(
        self, 
        aggregated_results: Dict[str, Dict],
        statistical_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate high-level summary of ablation study"""
        
        summary = {
            'total_configurations_tested': len(aggregated_results),
            'configurations_with_errors': 0,
            'best_performing_config': '',
            'worst_performing_config': '',
            'significant_degradations': [],
            'component_necessity_ranking': []
        }
        
        # Find best and worst performing configurations
        config_performance = {}
        for config_name, results in aggregated_results.items():
            r2_mean = results.get('metrics', {}).get('r2_mean', 0.0)
            config_performance[config_name] = r2_mean
            
            if 'error' in results:
                summary['configurations_with_errors'] += 1
        
        if config_performance:
            summary['best_performing_config'] = max(config_performance.items(), key=lambda x: x[1])[0]
            summary['worst_performing_config'] = min(config_performance.items(), key=lambda x: x[1])[0]
        
        # Identify significant degradations
        for config_name, stats in statistical_analysis.items():
            if isinstance(stats, dict) and stats.get('significant_at_0.05', False):
                degradation_magnitude = abs(stats.get('cohens_d', 0.0))
                summary['significant_degradations'].append({
                    'configuration': config_name,
                    'p_value': stats['t_p_value'],
                    'effect_size': stats['effect_size'],
                    'cohens_d': stats['cohens_d']
                })
        
        # Sort by effect size
        summary['significant_degradations'].sort(key=lambda x: abs(x['cohens_d']), reverse=True)
        
        return summary
    
    async def _save_ablation_results(self, results: Dict[str, Any]) -> None:
        """Save ablation study results and generate visualizations"""
        
        # Save main results as JSON
        results_file = self.results_path / "ablations.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary CSV
        csv_data = []
        for config_name, config_results in results['aggregated_results'].items():
            row = {
                'configuration': config_name,
                'description': config_results['config']['description']
            }
            
            # Add metrics
            metrics = config_results.get('metrics', {})
            for metric_name, value in metrics.items():
                if not metric_name.endswith('_values'):  # Skip raw values
                    row[metric_name] = value
            
            csv_data.append(row)
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(self.results_path / "ablations.csv", index=False)
        
        # Generate figures
        await self._generate_ablation_figures(results)
        
        logger.info(f"📁 Ablation results saved to {self.results_path}")
    
    async def _generate_ablation_figures(self, results: Dict[str, Any]) -> None:
        """Generate visualization figures for ablation study"""
        
        plt.style.use('seaborn-v0_8')
        
        # Figure 1: Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Ablation Study Results - Component Necessity Analysis', fontsize=16)
        
        aggregated = results['aggregated_results']
        config_names = list(aggregated.keys())
        
        # R² comparison
        r2_means = [aggregated[name]['metrics'].get('r2_mean', 0) for name in config_names]
        r2_stds = [aggregated[name]['metrics'].get('r2_std', 0) for name in config_names]
        
        bars = axes[0, 0].bar(range(len(config_names)), r2_means, yerr=r2_stds, capsize=5)
        axes[0, 0].set_title('R² Score by Configuration')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_xticks(range(len(config_names)))
        axes[0, 0].set_xticklabels(config_names, rotation=45, ha='right')
        
        # Highlight full model
        if 'full_model' in config_names:
            full_idx = config_names.index('full_model')
            bars[full_idx].set_color('gold')
            bars[full_idx].set_edgecolor('black')
            bars[full_idx].set_linewidth(2)
        
        # MSE comparison
        mse_means = [aggregated[name]['metrics'].get('mse_mean', 0) for name in config_names]
        mse_stds = [aggregated[name]['metrics'].get('mse_std', 0) for name in config_names]
        
        axes[0, 1].bar(range(len(config_names)), mse_means, yerr=mse_stds, capsize=5)
        axes[0, 1].set_title('Mean Squared Error by Configuration')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].set_xticks(range(len(config_names)))
        axes[0, 1].set_xticklabels(config_names, rotation=45, ha='right')
        axes[0, 1].set_yscale('log')
        
        # Component importance (if available)
        if 'component_importance' in results:
            importance = results['component_importance']
            components = [k for k in importance.keys() if k != 'ranking']
            importance_values = [importance[comp]['performance_drop_r2'] for comp in components]
            
            axes[1, 0].bar(components, importance_values)
            axes[1, 0].set_title('Component Importance (Performance Drop)')
            axes[1, 0].set_ylabel('R² Performance Drop')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Statistical significance
        if 'statistical_analysis' in results:
            stats = results['statistical_analysis']
            significant_configs = []
            p_values = []
            
            for config_name, stat_result in stats.items():
                if isinstance(stat_result, dict) and 't_p_value' in stat_result:
                    significant_configs.append(config_name)
                    p_values.append(stat_result['t_p_value'])
            
            if significant_configs:
                colors = ['red' if p < 0.05 else 'orange' if p < 0.1 else 'green' for p in p_values]
                axes[1, 1].bar(significant_configs, [-np.log10(p) for p in p_values], color=colors)
                axes[1, 1].set_title('Statistical Significance (-log₁₀(p-value))')
                axes[1, 1].set_ylabel('-log₁₀(p-value)')
                axes[1, 1].axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
                axes[1, 1].axhline(y=-np.log10(0.01), color='darkred', linestyle='--', alpha=0.7, label='p=0.01')
                axes[1, 1].legend()
                axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.results_path / "fig_ablations.svg", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Training curves comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Training Dynamics Comparison', fontsize=16)
        
        # Plot training curves for each configuration (using first seed)
        if 'raw_results' in results and 'seed_0' in results['raw_results']:
            seed_0_results = results['raw_results']['seed_0']
            
            for config_name, config_results in seed_0_results.items():
                if 'train_losses' in config_results and 'val_losses' in config_results:
                    train_losses = config_results['train_losses']
                    val_losses = config_results['val_losses']
                    
                    epochs = range(len(train_losses))
                    
                    axes[0].plot(epochs, train_losses, label=f'{config_name} (train)', alpha=0.7)
                    axes[1].plot(epochs, val_losses, label=f'{config_name} (val)', alpha=0.7)
            
            axes[0].set_title('Training Loss Curves')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Training Loss')
            axes[0].set_yscale('log')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            axes[1].set_title('Validation Loss Curves')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Validation Loss')
            axes[1].set_yscale('log')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_path / "fig_training_curves.svg", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("📊 Ablation figures generated successfully")


# Example usage and testing
async def main():
    """Example usage of the ablation study framework"""
    
    # Initialize framework
    framework = AblationStudyFramework(
        input_dim=(1, 32, 32),  # Single channel 32x32 spatial data
        output_dim=10,          # 10 output features
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Run comprehensive ablation study
    results = await framework.run_comprehensive_ablation_study(
        num_samples=1000,
        epochs=50,  # Reduced for testing
        num_seeds=2  # Reduced for testing
    )
    
    # Print summary
    print("\n🔬 Ablation Study Results Summary:")
    print("=" * 60)
    
    summary = results['summary']
    print(f"Configurations tested: {summary['total_configurations_tested']}")
    print(f"Best performing: {summary['best_performing_config']}")
    print(f"Significant degradations: {len(summary['significant_degradations'])}")
    
    # Print component importance
    if 'component_importance' in results:
        importance = results['component_importance']
        print(f"\n🏆 Component Importance Ranking:")
        for i, component in enumerate(importance.get('ranking', []), 1):
            drop = importance[component]['performance_drop_r2']
            print(f"  {i}. {component}: {drop:.4f} R² drop")
    
    # Print significant results
    print(f"\n📊 Significant Performance Degradations:")
    for degradation in summary['significant_degradations']:
        config = degradation['configuration']
        p_val = degradation['p_value']
        effect = degradation['effect_size']
        print(f"  {config}: p={p_val:.4f} ({effect} effect)")


if __name__ == "__main__":
    asyncio.run(main())

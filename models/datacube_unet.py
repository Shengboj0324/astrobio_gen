#!/usr/bin/env python3
"""
3D U-Net for Climate Datacube Processing
========================================

Physics-informed 3D U-Net for climate datacube surrogate modeling.
Integrates with existing PyTorch Lightning training infrastructure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple, Any
import logging
import numpy as np
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class PhysicsConstraints:
    """Physical constraints for climate modeling"""
    mass_conservation: bool = True
    energy_conservation: bool = True
    hydrostatic_balance: bool = True
    thermodynamic_consistency: bool = True
    
    # Physical constants
    specific_heat_air: float = 1004.0  # J/kg/K
    specific_heat_water: float = 4186.0  # J/kg/K
    latent_heat_vaporization: float = 2.26e6  # J/kg
    gas_constant_dry_air: float = 287.0  # J/kg/K
    gravity: float = 9.81  # m/s^2

class Conv3DBlock(nn.Module):
    """3D Convolutional block with normalization and activation"""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        padding: int = 1,
        use_batchnorm: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
        ]
        
        if use_batchnorm:
            layers.append(nn.BatchNorm3d(out_channels))
        
        layers.extend([
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding),
        ])
        
        if use_batchnorm:
            layers.append(nn.BatchNorm3d(out_channels))
        
        layers.append(nn.ReLU(inplace=True))
        
        if dropout > 0:
            layers.append(nn.Dropout3d(dropout))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class DownSample3D(nn.Module):
    """3D downsampling block"""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv = Conv3DBlock(in_channels, out_channels, dropout=dropout)
        self.pool = nn.MaxPool3d(2, stride=2)
    
    def forward(self, x):
        conv_out = self.conv(x)
        pool_out = self.pool(conv_out)
        return conv_out, pool_out

class UpSample3D(nn.Module):
    """3D upsampling block with skip connections"""
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = Conv3DBlock(in_channels // 2 + skip_channels, out_channels, dropout=dropout)
    
    def forward(self, x, skip):
        x = self.upsample(x)
        
        # Handle size mismatches
        if x.shape != skip.shape:
            # Pad or crop to match skip connection
            diff_d = skip.shape[2] - x.shape[2]
            diff_h = skip.shape[3] - x.shape[3]
            diff_w = skip.shape[4] - x.shape[4]
            
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                         diff_h // 2, diff_h - diff_h // 2,
                         diff_d // 2, diff_d - diff_d // 2])
        
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class PhysicsRegularizer(nn.Module):
    """Physics-based regularization for climate variables"""
    
    def __init__(self, constraints: PhysicsConstraints):
        super().__init__()
        self.constraints = constraints
    
    def forward(self, predictions: torch.Tensor, variable_names: List[str]) -> Dict[str, torch.Tensor]:
        """
        Compute physics-based regularization losses
        
        Args:
            predictions: Predicted variables [batch, variables, time, lev, lat, lon]
            variable_names: Names of variables (e.g., ['temp', 'humidity', 'pressure'])
            
        Returns:
            Dictionary of physics loss terms
        """
        losses = {}
        
        # Create variable index mapping
        var_idx = {name: i for i, name in enumerate(variable_names)}
        
        # Mass conservation (continuity equation approximation)
        if self.constraints.mass_conservation and 'wind_u' in var_idx and 'wind_v' in var_idx:
            u_idx, v_idx = var_idx['wind_u'], var_idx['wind_v']
            u, v = predictions[:, u_idx], predictions[:, v_idx]
            
            # Approximate divergence using finite differences
            du_dx = torch.diff(u, dim=-1)  # longitude gradient
            dv_dy = torch.diff(v, dim=-2)  # latitude gradient
            
            # Align dimensions for divergence calculation
            min_lon = min(du_dx.shape[-1], dv_dy.shape[-1])
            min_lat = min(du_dx.shape[-2], dv_dy.shape[-2])
            
            du_dx_aligned = du_dx[..., :min_lat, :min_lon]
            dv_dy_aligned = dv_dy[..., :min_lat, :min_lon]
            
            divergence = du_dx_aligned + dv_dy_aligned
            losses['mass_conservation'] = torch.mean(divergence ** 2)
        
        # Energy conservation (simplified)
        if self.constraints.energy_conservation and 'temp' in var_idx:
            temp_idx = var_idx['temp']
            temp = predictions[:, temp_idx]
            
            # Penalize unrealistic temperature gradients
            dt_dz = torch.diff(temp, dim=-3)  # vertical gradient (lev dimension)
            dt_dt = torch.diff(temp, dim=-4)  # time gradient
            
            # Lapse rate should be reasonable (approximately 6.5 K/km)
            losses['temperature_gradient'] = torch.mean(torch.clamp(torch.abs(dt_dz) - 0.1, min=0) ** 2)
            
            # Temperature change should be smooth in time
            losses['temperature_stability'] = torch.mean(dt_dt ** 2)
        
        # Hydrostatic balance (pressure-height relationship)
        if (self.constraints.hydrostatic_balance and 
            'pressure' in var_idx and 'temp' in var_idx):
            
            pressure_idx, temp_idx = var_idx['pressure'], var_idx['temp']
            pressure, temp = predictions[:, pressure_idx], predictions[:, temp_idx]
            
            # Approximate hydrostatic equation: dp/dz = -ρg = -pg/(RT)
            dp_dz = torch.diff(pressure, dim=-3)  # vertical pressure gradient
            
            # Expected pressure gradient from hydrostatic balance
            rho_g_approx = pressure[..., 1:, :, :] * self.constraints.gravity / (
                self.constraints.gas_constant_dry_air * temp[..., 1:, :, :]
            )
            
            hydrostatic_residual = dp_dz + rho_g_approx
            losses['hydrostatic_balance'] = torch.mean(hydrostatic_residual ** 2)
        
        # Humidity constraints
        if 'humidity' in var_idx:
            humidity_idx = var_idx['humidity']
            humidity = predictions[:, humidity_idx]
            
            # Humidity should be between 0 and 1 (if relative humidity)
            losses['humidity_bounds'] = torch.mean(
                torch.clamp(humidity - 1.0, min=0) ** 2 + 
                torch.clamp(-humidity, min=0) ** 2
            )
        
        return losses

class CubeUNet(pl.LightningModule):
    """
    3D U-Net for climate datacube processing with physics constraints
    """
    
    def __init__(
        self,
        n_input_vars: int = 5,
        n_output_vars: int = 5,
        input_variables: List[str] = None,
        output_variables: List[str] = None,
        base_features: int = 32,
        depth: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        physics_weight: float = 0.1,
        use_physics_constraints: bool = True,
        **kwargs
    ):
        """
        Initialize CubeUNet
        
        Args:
            n_input_vars: Number of input variables
            n_output_vars: Number of output variables
            input_variables: Names of input variables
            output_variables: Names of output variables
            base_features: Base number of features in first layer
            depth: Depth of U-Net (number of downsampling levels)
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            physics_weight: Weight for physics regularization
            use_physics_constraints: Whether to use physics constraints
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.n_input_vars = n_input_vars
        self.n_output_vars = n_output_vars
        self.input_variables = input_variables or [f'var_{i}' for i in range(n_input_vars)]
        self.output_variables = output_variables or [f'var_{i}' for i in range(n_output_vars)]
        self.base_features = base_features
        self.depth = depth
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.physics_weight = physics_weight
        self.use_physics_constraints = use_physics_constraints
        
        # Physics constraints
        if self.use_physics_constraints:
            self.physics_regularizer = PhysicsRegularizer(PhysicsConstraints())
        
        # Build U-Net architecture
        self._build_network()
        
        # Loss tracking
        self.train_losses = []
        self.val_losses = []
        
        logger.info(f"Initialized CubeUNet with {n_input_vars} input vars, "
                   f"{n_output_vars} output vars, depth={depth}")
    
    def _build_network(self):
        """Build the U-Net architecture"""
        # Encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        
        in_channels = self.n_input_vars
        features = self.base_features
        
        # Store encoder feature sizes for proper skip connection handling
        encoder_features = []
        
        for i in range(self.depth):
            if i == 0:
                # First block - just convolution
                self.encoder_blocks.append(Conv3DBlock(in_channels, features, dropout=self.dropout))
                encoder_features.append(features)
            else:
                # Downsampling blocks
                self.downsample_blocks.append(DownSample3D(in_channels, features, dropout=self.dropout))
                encoder_features.append(features)
            
            in_channels = features
            features *= 2
        
        # Bottleneck
        self.bottleneck = Conv3DBlock(in_channels, features, dropout=self.dropout)
        
        # Decoder (upsampling path)
        self.upsample_blocks = nn.ModuleList()
        
        current_features = features
        for i in range(self.depth):
            # Skip connection channels come from corresponding encoder layer
            skip_idx = self.depth - 1 - i
            skip_channels = encoder_features[skip_idx]
            
            # Output channels for this decoder layer
            if i < self.depth - 1:
                out_channels = current_features // 4
            else:
                out_channels = self.base_features
            
            self.upsample_blocks.append(
                UpSample3D(current_features, skip_channels, out_channels, dropout=self.dropout)
            )
            
            current_features = out_channels
        
        # Output layer
        self.output_conv = nn.Conv3d(self.base_features, self.n_output_vars, 1)
        
        # Optional output activation
        self.output_activation = nn.Identity()  # Can be changed to Sigmoid, Tanh, etc.
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net
        
        Args:
            x: Input tensor [batch, variables, time, lev, lat, lon]
            
        Returns:
            Output tensor [batch, variables, time, lev, lat, lon]
        """
        # Store encoder outputs for skip connections
        encoder_outputs = []
        
        # Encoder path
        current = x
        
        # First encoder block
        enc_out = self.encoder_blocks[0](current)
        encoder_outputs.append(enc_out)
        current = F.max_pool3d(enc_out, 2, stride=2)
        
        # Remaining encoder blocks
        for downsample_block in self.downsample_blocks:
            enc_out, current = downsample_block(current)
            encoder_outputs.append(enc_out)
        
        # Bottleneck
        current = self.bottleneck(current)
        
        # Decoder path
        for i, upsample_block in enumerate(self.upsample_blocks):
            skip_idx = len(encoder_outputs) - 1 - i
            skip_connection = encoder_outputs[skip_idx]
            current = upsample_block(current, skip_connection)
        
        # Output
        output = self.output_conv(current)
        output = self.output_activation(output)
        
        return output
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute total loss including physics constraints"""
        losses = {}
        
        # Main reconstruction loss
        mse_loss = F.mse_loss(predictions, targets)
        losses['mse'] = mse_loss
        
        # Physics regularization
        if self.use_physics_constraints and self.physics_weight > 0:
            physics_losses = self.physics_regularizer(predictions, self.output_variables)
            
            total_physics_loss = torch.tensor(0.0, device=predictions.device)
            for name, loss in physics_losses.items():
                losses[f'physics_{name}'] = loss
                total_physics_loss += loss
            
            losses['physics_total'] = total_physics_loss
        else:
            losses['physics_total'] = torch.tensor(0.0, device=predictions.device)
        
        # Total loss
        total_loss = mse_loss + self.physics_weight * losses['physics_total']
        losses['total'] = total_loss
        
        return losses
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        inputs, targets = batch
        
        # Forward pass
        predictions = self(inputs)
        
        # Compute losses
        losses = self.compute_loss(predictions, targets)
        
        # Log losses
        for name, loss in losses.items():
            self.log(f'train/{name}', loss, on_step=True, on_epoch=True, prog_bar=(name == 'total'))
        
        # Additional metrics
        with torch.no_grad():
            mae = F.l1_loss(predictions, targets)
            self.log('train/mae', mae, on_step=True, on_epoch=True)
            
            # R² score approximation
            ss_res = torch.sum((targets - predictions) ** 2)
            ss_tot = torch.sum((targets - targets.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            self.log('train/r2', r2, on_step=True, on_epoch=True)
        
        return losses['total']
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step"""
        inputs, targets = batch
        
        # Forward pass
        predictions = self(inputs)
        
        # Compute losses
        losses = self.compute_loss(predictions, targets)
        
        # Log losses
        for name, loss in losses.items():
            self.log(f'val/{name}', loss, on_step=False, on_epoch=True, prog_bar=(name == 'total'))
        
        # Additional metrics
        mae = F.l1_loss(predictions, targets)
        self.log('val/mae', mae, on_step=False, on_epoch=True)
        
        # R² score
        ss_res = torch.sum((targets - predictions) ** 2)
        ss_tot = torch.sum((targets - targets.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        self.log('val/r2', r2, on_step=False, on_epoch=True)
        
        return losses['total']
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step"""
        inputs, targets = batch
        
        # Forward pass
        predictions = self(inputs)
        
        # Compute losses
        losses = self.compute_loss(predictions, targets)
        
        # Log losses
        for name, loss in losses.items():
            self.log(f'test/{name}', loss, on_step=False, on_epoch=True)
        
        # Additional metrics
        mae = F.l1_loss(predictions, targets)
        self.log('test/mae', mae, on_step=False, on_epoch=True)
        
        # R² score
        ss_res = torch.sum((targets - predictions) ** 2)
        ss_tot = torch.sum((targets - targets.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        self.log('test/r2', r2, on_step=False, on_epoch=True)
        
        return losses['total']
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/total",
                "interval": "epoch",
                "frequency": 1,
            },
        }
    
    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Prediction step"""
        inputs, _ = batch
        return self(inputs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_variables': self.input_variables,
            'output_variables': self.output_variables,
            'architecture': {
                'base_features': self.base_features,
                'depth': self.depth,
                'dropout': self.dropout
            },
            'physics_constraints': self.use_physics_constraints,
            'physics_weight': self.physics_weight
        } 
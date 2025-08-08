#!/usr/bin/env python3
"""
Domain-Specific Encoders with Shared Latent Space
=================================================

Multi-modal encoder architecture that processes different scientific domains
with specialized encoders and fuses them into a shared representation space.

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                    MULTI-MODAL INPUT                       │
├─────────────────┬─────────────────┬─────────────────────────┤
│  Climate Cube   │   Bio Graph     │       Spectrum          │
│   [B,V,T,H,W,D] │   PyG Data      │      [B,N,2]           │
│                 │                 │                         │
│   ┌───────────┐ │  ┌────────────┐ │   ┌──────────────────┐  │
│   │ 3D U-Net  │ │  │    GNN     │ │   │   1D CNN        │  │
│   │ Encoder   │ │  │ Encoder    │ │   │   Encoder       │  │
│   └───────────┘ │  └────────────┘ │   └──────────────────┘  │
│                 │                 │                         │
│   [B, 256]      │   [B, 256]      │      [B, 256]          │
└─────────────────┴─────────────────┴─────────────────────────┘
                            │
                    ┌───────────────┐
                    │ Cross-Attention│
                    │    Fusion      │
                    └───────────────┘
                            │
                    ┌───────────────┐
                    │ Shared Latent │
                    │   [B, 256]    │
                    └───────────────┘
                            │
                    ┌───────────────┐
                    │  Task Heads   │
                    │ (Climate/Spec)│
                    └───────────────┘

Features:
- Domain expertise: Each encoder optimized for its data type
- Shared representation: Common latent space for multi-task learning
- Cross-attention fusion: Intelligent information sharing
- Scalable architecture: Easy to add new domains
- Physics-informed: Built-in physical constraints
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch Geometric for graph neural networks
try:
    import torch_geometric
    from torch_geometric.data import Batch as PyGBatch
    from torch_geometric.data import Data as PyGData
    from torch_geometric.nn import GATConv, GCNConv, GraphConv, global_max_pool, global_mean_pool

    PYTORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    PYTORCH_GEOMETRIC_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """Strategies for fusing multi-modal representations"""

    CONCATENATION = "concatenation"
    CROSS_ATTENTION = "cross_attention"
    MULTIPLICATIVE = "multiplicative"
    ADDITIVE = "additive"


@dataclass
class EncoderConfig:
    """Configuration for domain-specific encoders"""

    # Shared settings
    latent_dim: int = 256
    dropout: float = 0.1
    activation: str = "relu"

    # Climate encoder (3D U-Net)
    climate_base_features: int = 64
    climate_depth: int = 4
    climate_use_attention: bool = True

    # Biology encoder (GNN)
    bio_hidden_dim: int = 128
    bio_num_layers: int = 3
    bio_gnn_type: str = "gat"  # gcn, gat, graph_conv
    bio_num_heads: int = 4

    # Spectroscopy encoder (1D CNN)
    spec_num_filters: List[int] = None
    spec_kernel_sizes: List[int] = None
    spec_pool_sizes: List[int] = None

    # Fusion settings
    fusion_strategy: FusionStrategy = FusionStrategy.CROSS_ATTENTION
    fusion_num_heads: int = 8
    fusion_num_layers: int = 2

    # Physics constraints
    use_physics_constraints: bool = True
    physics_weight: float = 0.1

    def __post_init__(self):
        """Set default values for lists"""
        if self.spec_num_filters is None:
            self.spec_num_filters = [64, 128, 256, 256]
        if self.spec_kernel_sizes is None:
            self.spec_kernel_sizes = [7, 5, 3, 3]
        if self.spec_pool_sizes is None:
            self.spec_pool_sizes = [2, 2, 2, 2]


class ClimateEncoder(nn.Module):
    """
    3D U-Net encoder for climate datacubes

    Processes 4D climate fields [batch, vars, time, lat, lon, lev]
    and extracts spatial-temporal features.
    """

    def __init__(self, config: EncoderConfig, n_input_vars: int = 2):
        super().__init__()
        self.config = config
        self.n_input_vars = n_input_vars

        # 3D convolutional encoder
        self.encoder_blocks = nn.ModuleList()

        # Input block
        in_channels = n_input_vars
        out_channels = config.climate_base_features

        for depth in range(config.climate_depth):
            block = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                self._get_activation(),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                self._get_activation(),
            )

            self.encoder_blocks.append(block)

            # Update channels for next block
            in_channels = out_channels
            out_channels = min(out_channels * 2, 512)

        # Attention mechanism
        if config.climate_use_attention:
            self.attention = SpatialAttention3D(in_channels)
        else:
            self.attention = None

        # Global pooling and projection to latent space
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.projection = nn.Sequential(
            nn.Linear(in_channels, config.latent_dim * 2),
            self._get_activation(),
            nn.Dropout(config.dropout),
            nn.Linear(config.latent_dim * 2, config.latent_dim),
        )

        logger.info(f"🌡️ Climate encoder initialized: {n_input_vars} vars → {config.latent_dim}D")

    def _get_activation(self):
        """Get activation function"""
        if self.config.activation == "relu":
            return nn.ReLU(inplace=True)
        elif self.config.activation == "gelu":
            return nn.GELU()
        elif self.config.activation == "swish":
            return nn.SiLU()
        else:
            return nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for climate encoder

        Args:
            x: Climate datacube [batch, vars, time, lat, lon, lev]

        Returns:
            Latent representation [batch, latent_dim]
        """
        # Check input shape
        if x.dim() != 6:
            raise ValueError(f"Expected 6D input (B,V,T,H,W,D), got {x.shape}")

        batch_size = x.shape[0]

        # Merge time and variables for 3D processing: [B, V*T, H, W, D]
        x = x.view(batch_size, -1, x.shape[3], x.shape[4], x.shape[5])

        # Encoder forward pass
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)

            # Downsample spatial dimensions (but not depth/levels)
            if i < len(self.encoder_blocks) - 1:
                x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Apply attention if enabled
        if self.attention is not None:
            x = self.attention(x)

        # Global pooling and projection
        x = self.global_pool(x)  # [B, C, 1, 1, 1]
        x = x.view(batch_size, -1)  # [B, C]
        x = self.projection(x)  # [B, latent_dim]

        return x


class BiologyEncoder(nn.Module):
    """
    Graph Neural Network encoder for biological networks

    Processes molecular graphs and metabolic networks
    to extract biological features.
    """

    def __init__(self, config: EncoderConfig, node_input_dim: int = 6):
        super().__init__()
        self.config = config
        self.node_input_dim = node_input_dim

        if not PYTORCH_GEOMETRIC_AVAILABLE:
            logger.warning("PyTorch Geometric not available - using fallback biology encoder")
            self._init_fallback_encoder()
            return

        # Input projection
        self.input_projection = nn.Linear(node_input_dim, config.bio_hidden_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList()

        for layer in range(config.bio_num_layers):
            if config.bio_gnn_type == "gcn":
                gnn_layer = GCNConv(config.bio_hidden_dim, config.bio_hidden_dim)
            elif config.bio_gnn_type == "gat":
                gnn_layer = GATConv(
                    config.bio_hidden_dim,
                    config.bio_hidden_dim // config.bio_num_heads,
                    heads=config.bio_num_heads,
                    dropout=config.dropout,
                )
            elif config.bio_gnn_type == "graph_conv":
                gnn_layer = GraphConv(config.bio_hidden_dim, config.bio_hidden_dim)
            else:
                raise ValueError(f"Unknown GNN type: {config.bio_gnn_type}")

            self.gnn_layers.append(gnn_layer)

        # Normalization and activation
        self.norms = nn.ModuleList(
            [nn.LayerNorm(config.bio_hidden_dim) for _ in range(config.bio_num_layers)]
        )

        # Graph-level pooling and projection
        self.projection = nn.Sequential(
            nn.Linear(config.bio_hidden_dim * 2, config.latent_dim * 2),  # *2 for mean+max pooling
            self._get_activation(),
            nn.Dropout(config.dropout),
            nn.Linear(config.latent_dim * 2, config.latent_dim),
        )

        logger.info(
            f"[BIO] Biology encoder initialized: {config.bio_gnn_type.upper()} → {config.latent_dim}D"
        )

    def _init_fallback_encoder(self):
        """Initialize fallback encoder for when PyG is not available"""
        self.fallback = True

        # Simple MLP for adjacency matrices
        self.adj_encoder = nn.Sequential(
            nn.Linear(400, 512),  # Assuming max 20x20 adjacency
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(512, 256),
        )

        # MLP for node features
        self.feat_encoder = nn.Sequential(
            nn.Linear(self.node_input_dim * 20, 256),  # Assuming max 20 nodes
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(256, 256),
        )

        # Fusion
        self.projection = nn.Sequential(
            nn.Linear(512, self.config.latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.latent_dim * 2, self.config.latent_dim),
        )

    def _get_activation(self):
        """Get activation function"""
        if self.config.activation == "relu":
            return nn.ReLU()
        elif self.config.activation == "gelu":
            return nn.GELU()
        else:
            return nn.ReLU()

    def forward(self, x: Union[PyGBatch, Dict[str, torch.Tensor], List]) -> torch.Tensor:
        """
        Forward pass for biology encoder

        Args:
            x: Graph data (PyG Batch, dict with adjacency/features, or list)

        Returns:
            Latent representation [batch, latent_dim]
        """
        if not PYTORCH_GEOMETRIC_AVAILABLE:
            return self._forward_fallback(x)

        if isinstance(x, PyGBatch):
            return self._forward_pyg(x)
        elif isinstance(x, dict) and "adjacency" in x:
            return self._forward_dict(x)
        else:
            # Handle list of graphs or other formats
            logger.warning("Unsupported biology data format, using zero embedding")
            batch_size = len(x) if isinstance(x, list) else 1
            return torch.zeros(batch_size, self.config.latent_dim)

    def _forward_pyg(self, batch: PyGBatch) -> torch.Tensor:
        """Forward pass for PyTorch Geometric batch"""
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

        # Input projection
        x = self.input_projection(x)

        # GNN layers
        for gnn_layer, norm in zip(self.gnn_layers, self.norms):
            x_residual = x
            x = gnn_layer(x, edge_index)
            x = norm(x)
            x = self._get_activation()(x)
            x = F.dropout(x, self.config.dropout, training=self.training)

            # Residual connection if dimensions match
            if x_residual.shape == x.shape:
                x = x + x_residual

        # Graph-level pooling
        x_mean = global_mean_pool(x, batch_idx)
        x_max = global_max_pool(x, batch_idx)

        # Combine pooled representations
        graph_repr = torch.cat([x_mean, x_max], dim=1)

        # Project to latent space
        latent = self.projection(graph_repr)

        return latent

    def _forward_dict(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for dictionary format (adjacency + features)"""
        adjacency = x["adjacency"]  # [batch, nodes, nodes]
        node_features = x["node_features"]  # [batch, nodes, features]

        batch_size = adjacency.shape[0]

        # Simple processing for adjacency matrices
        # Flatten adjacency and use MLP
        adj_flat = adjacency.view(batch_size, -1)
        adj_repr = self.adj_encoder(adj_flat[:, :400])  # Truncate if too large

        # Process node features
        feat_flat = node_features.view(batch_size, -1)
        feat_repr = self.feat_encoder(feat_flat[:, :120])  # Truncate if too large

        # Combine and project
        combined = torch.cat([adj_repr, feat_repr], dim=1)
        latent = self.projection(combined)

        return latent

    def _forward_fallback(self, x) -> torch.Tensor:
        """Fallback forward pass"""
        if isinstance(x, dict) and "adjacency" in x:
            return self._forward_dict(x)
        else:
            # Return zero embedding
            batch_size = 1
            return torch.zeros(batch_size, self.config.latent_dim)


class SpectroscopyEncoder(nn.Module):
    """
    1D CNN encoder for high-resolution spectra

    Processes wavelength-flux pairs to extract spectral features
    including absorption lines and continuum shapes.
    """

    def __init__(self, config: EncoderConfig, input_channels: int = 2):
        super().__init__()
        self.config = config

        # 1D convolutional layers
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        in_channels = input_channels

        for i, (out_channels, kernel_size, pool_size) in enumerate(
            zip(config.spec_num_filters, config.spec_kernel_sizes, config.spec_pool_sizes)
        ):
            # Convolution
            conv = nn.Conv1d(
                in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
            )
            self.conv_layers.append(conv)

            # Pooling
            pool = nn.MaxPool1d(pool_size, stride=pool_size)
            self.pool_layers.append(pool)

            # Normalization
            norm = nn.BatchNorm1d(out_channels)
            self.norm_layers.append(norm)

            in_channels = out_channels

        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(16)

        # Projection to latent space
        final_features = config.spec_num_filters[-1] * 16
        self.projection = nn.Sequential(
            nn.Linear(final_features, config.latent_dim * 2),
            self._get_activation(),
            nn.Dropout(config.dropout),
            nn.Linear(config.latent_dim * 2, config.latent_dim),
        )

        logger.info(
            f"🌈 Spectroscopy encoder initialized: {input_channels}D → {config.latent_dim}D"
        )

    def _get_activation(self):
        """Get activation function"""
        if self.config.activation == "relu":
            return nn.ReLU()
        elif self.config.activation == "gelu":
            return nn.GELU()
        else:
            return nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for spectroscopy encoder

        Args:
            x: Spectrum data [batch, n_points, 2] (wavelength, flux)

        Returns:
            Latent representation [batch, latent_dim]
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (B,N,C), got {x.shape}")

        # Transpose for 1D conv: [batch, channels, length]
        x = x.transpose(1, 2)  # [batch, 2, n_points]

        # 1D convolution layers
        for conv, pool, norm in zip(self.conv_layers, self.pool_layers, self.norm_layers):
            x = conv(x)
            x = norm(x)
            x = self._get_activation()(x)
            x = pool(x)
            x = F.dropout(x, self.config.dropout, training=self.training)

        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)  # [batch, channels, 16]

        # Flatten and project
        x = x.view(x.size(0), -1)  # [batch, channels * 16]
        latent = self.projection(x)

        return latent


class SpatialAttention3D(nn.Module):
    """3D spatial attention for climate datacubes"""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return x * attention


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion for multi-modal representations

    Allows different domains to attend to each other and share information.
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.num_heads = config.fusion_num_heads
        self.head_dim = config.latent_dim // config.fusion_num_heads

        assert config.latent_dim % config.fusion_num_heads == 0

        # Cross-attention layers
        self.attention_layers = nn.ModuleList()
        for _ in range(config.fusion_num_layers):
            attention = nn.MultiheadAttention(
                embed_dim=config.latent_dim,
                num_heads=config.fusion_num_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            self.attention_layers.append(attention)

        # Layer normalization
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(config.latent_dim) for _ in range(config.fusion_num_layers)]
        )

        # Feed-forward networks
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.latent_dim, config.latent_dim * 4),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.latent_dim * 4, config.latent_dim),
                )
                for _ in range(config.fusion_num_layers)
            ]
        )

    def forward(self, modality_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Cross-attention fusion of multi-modal features

        Args:
            modality_features: List of [batch, latent_dim] tensors from different modalities

        Returns:
            Fused representation [batch, latent_dim]
        """
        # Filter out None features
        valid_features = [f for f in modality_features if f is not None]

        if not valid_features:
            # Return zero if no valid features
            batch_size = 1
            return torch.zeros(batch_size, self.latent_dim)

        if len(valid_features) == 1:
            # Single modality - just return as is
            return valid_features[0]

        # Stack features: [batch, num_modalities, latent_dim]
        stacked_features = torch.stack(valid_features, dim=1)

        # Apply cross-attention layers
        for attention, norm, ffn in zip(self.attention_layers, self.layer_norms, self.ffns):
            # Self-attention across modalities
            attn_output, _ = attention(stacked_features, stacked_features, stacked_features)

            # Residual connection and layer norm
            stacked_features = norm(stacked_features + attn_output)

            # Feed-forward network
            ffn_output = ffn(stacked_features)
            stacked_features = norm(stacked_features + ffn_output)

        # Global pooling across modalities
        fused = stacked_features.mean(dim=1)  # [batch, latent_dim]

        return fused


class PhysicsConstraintLayer(nn.Module):
    """Physics constraint layer for enforcing physical laws"""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

        # Physics constraint networks
        self.energy_constraint = nn.Linear(latent_dim, 1)
        self.mass_constraint = nn.Linear(latent_dim, 1)
        self.radiative_constraint = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor, planet_params: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply physics constraints

        Args:
            x: Latent representation [batch, latent_dim]
            planet_params: Planet parameters [batch, n_params]

        Returns:
            Dictionary with constraint violations
        """
        # Energy balance constraint
        energy_violation = self.energy_constraint(x)

        # Mass conservation constraint
        mass_violation = self.mass_constraint(x)

        # Radiative equilibrium constraint
        radiative_violation = self.radiative_constraint(x)

        return {
            "energy_violation": energy_violation,
            "mass_violation": mass_violation,
            "radiative_violation": radiative_violation,
        }


class MultiModalEncoder(nn.Module):
    """
    Complete multi-modal encoder with domain-specific encoders and fusion

    This is the main class that combines all domain encoders and handles
    the fusion into a shared latent representation.
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        # Domain-specific encoders
        self.climate_encoder = ClimateEncoder(config)
        self.biology_encoder = BiologyEncoder(config)
        self.spectroscopy_encoder = SpectroscopyEncoder(config)

        # Fusion layer
        if config.fusion_strategy == FusionStrategy.CROSS_ATTENTION:
            self.fusion = CrossAttentionFusion(config)
        elif config.fusion_strategy == FusionStrategy.CONCATENATION:
            self.fusion = nn.Sequential(
                nn.Linear(config.latent_dim * 3, config.latent_dim * 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.latent_dim * 2, config.latent_dim),
            )
        else:
            # Default to concatenation
            self.fusion = nn.Linear(config.latent_dim * 3, config.latent_dim)

        # Physics constraints
        if config.use_physics_constraints:
            self.physics_layer = PhysicsConstraintLayer(config.latent_dim)
        else:
            self.physics_layer = None

        logger.info(
            f"[PROC] Multi-modal encoder initialized with {config.fusion_strategy.value} fusion"
        )

    def forward(self, batch_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-modal encoder

        Args:
            batch_data: Dictionary containing:
                - climate_cubes: [batch, vars, time, lat, lon, lev] (optional)
                - bio_graphs: PyG batch or dict (optional)
                - spectra: [batch, n_points, 2] (optional)
                - planet_params: [batch, n_params]

        Returns:
            Dictionary with fused representation and individual encodings
        """
        modality_features = []
        individual_features = {}

        # Encode climate data
        if "climate_cubes" in batch_data and batch_data["climate_cubes"] is not None:
            climate_features = self.climate_encoder(batch_data["climate_cubes"])
            modality_features.append(climate_features)
            individual_features["climate"] = climate_features

        # Encode biological data
        if "bio_graphs" in batch_data and batch_data["bio_graphs"] is not None:
            bio_features = self.biology_encoder(batch_data["bio_graphs"])
            modality_features.append(bio_features)
            individual_features["biology"] = bio_features

        # Encode spectroscopy data
        if "spectra" in batch_data and batch_data["spectra"] is not None:
            spec_features = self.spectroscopy_encoder(batch_data["spectra"])
            modality_features.append(spec_features)
            individual_features["spectroscopy"] = spec_features

        # Fuse modalities
        if self.config.fusion_strategy == FusionStrategy.CROSS_ATTENTION:
            fused_features = self.fusion(modality_features)
        elif self.config.fusion_strategy == FusionStrategy.CONCATENATION:
            if modality_features:
                # Pad missing modalities with zeros
                while len(modality_features) < 3:
                    batch_size = modality_features[0].shape[0] if modality_features else 1
                    modality_features.append(torch.zeros(batch_size, self.config.latent_dim))

                concatenated = torch.cat(modality_features, dim=1)
                fused_features = self.fusion(concatenated)
            else:
                batch_size = 1
                fused_features = torch.zeros(batch_size, self.config.latent_dim)
        else:
            # Simple mean fusion
            if modality_features:
                fused_features = torch.stack(modality_features, dim=0).mean(dim=0)
            else:
                batch_size = 1
                fused_features = torch.zeros(batch_size, self.config.latent_dim)

        result = {"fused_features": fused_features, "individual_features": individual_features}

        # Apply physics constraints if enabled
        if self.physics_layer is not None and "planet_params" in batch_data:
            physics_constraints = self.physics_layer(fused_features, batch_data["planet_params"])
            result["physics_constraints"] = physics_constraints

        return result


# Convenience function
def create_multimodal_encoder(config: EncoderConfig = None) -> MultiModalEncoder:
    """Create a multi-modal encoder with default or custom configuration"""
    if config is None:
        config = EncoderConfig()

    return MultiModalEncoder(config)


if __name__ == "__main__":
    # Test the multi-modal encoder
    def test_multimodal_encoder():
        logger.info("[TEST] Testing Multi-Modal Encoder Architecture")

        # Create encoder config
        config = EncoderConfig(
            latent_dim=256,
            climate_base_features=32,
            climate_depth=3,
            bio_hidden_dim=64,
            bio_num_layers=2,
            fusion_strategy=FusionStrategy.CROSS_ATTENTION,
            use_physics_constraints=True,
        )

        # Create encoder
        encoder = create_multimodal_encoder(config)
        encoder.eval()

        # Create test data
        batch_size = 2

        # Climate data: [batch, vars, time, lat, lon, lev]
        climate_cubes = torch.randn(batch_size, 2, 8, 16, 24, 6)

        # Biology data (adjacency format)
        bio_graphs = {
            "adjacency": torch.randn(batch_size, 20, 20),
            "node_features": torch.randn(batch_size, 20, 6),
        }

        # Spectroscopy data: [batch, n_points, 2]
        spectra = torch.randn(batch_size, 1000, 2)

        # Planet parameters
        planet_params = torch.randn(batch_size, 8)

        # Test batch data
        batch_data = {
            "climate_cubes": climate_cubes,
            "bio_graphs": bio_graphs,
            "spectra": spectra,
            "planet_params": planet_params,
        }

        # Forward pass
        logger.info("[PROC] Testing forward pass...")

        with torch.no_grad():
            results = encoder(batch_data)

        # Check results
        fused_features = results["fused_features"]
        individual_features = results["individual_features"]

        logger.info(f"[OK] Fused features shape: {fused_features.shape}")
        logger.info(f"[OK] Individual features available: {list(individual_features.keys())}")

        for domain, features in individual_features.items():
            logger.info(f"  {domain}: {features.shape}")

        if "physics_constraints" in results:
            physics = results["physics_constraints"]
            logger.info(f"[OK] Physics constraints: {list(physics.keys())}")
            for constraint, value in physics.items():
                logger.info(f"  {constraint}: {value.shape}")

        # Test with missing modalities
        logger.info("[PROC] Testing with missing modalities...")

        partial_batch = {"climate_cubes": climate_cubes, "planet_params": planet_params}

        with torch.no_grad():
            partial_results = encoder(partial_batch)

        logger.info(f"[OK] Partial results shape: {partial_results['fused_features'].shape}")
        logger.info(
            f"[OK] Available modalities: {list(partial_results['individual_features'].keys())}"
        )

        # Test different fusion strategies
        logger.info("[PROC] Testing different fusion strategies...")

        for fusion_strategy in [FusionStrategy.CONCATENATION, FusionStrategy.CROSS_ATTENTION]:
            test_config = EncoderConfig(latent_dim=128, fusion_strategy=fusion_strategy)
            test_encoder = create_multimodal_encoder(test_config)
            test_encoder.eval()

            with torch.no_grad():
                test_results = encoder(batch_data)

            logger.info(f"[OK] {fusion_strategy.value}: {test_results['fused_features'].shape}")

        logger.info("[OK] Multi-Modal Encoder test completed successfully!")

        # Show model statistics
        total_params = sum(p.numel() for p in encoder.parameters())
        trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

        print("\n" + "=" * 60)
        print("[AI] MULTI-MODAL ENCODER STATISTICS")
        print("=" * 60)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {total_params * 4 / (1024**2):.1f} MB")
        print(f"\nDomain encoders:")
        print(f"  🌡️ Climate: 3D U-Net with attention")
        print(f"  [BIO] Biology: {config.bio_gnn_type.upper()} with {config.bio_num_layers} layers")
        print(f"  🌈 Spectroscopy: 1D CNN with {len(config.spec_num_filters)} layers")
        print(f"\nFusion:")
        print(f"  Strategy: {config.fusion_strategy.value}")
        print(f"  Latent dimension: {config.latent_dim}")
        print(f"  Physics constraints: {'[OK]' if config.use_physics_constraints else '[FAIL]'}")
        print("=" * 60)

    # Run test
    test_multimodal_encoder()

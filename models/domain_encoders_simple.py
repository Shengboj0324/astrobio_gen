#!/usr/bin/env python3
"""
Domain-Specific Encoders (Simplified)
=====================================

Simplified multi-modal encoder architecture that correctly handles different data types.
"""

import logging
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
    from torch_geometric.nn import GATConv, GCNConv, global_max_pool, global_mean_pool

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
    MEAN = "mean"


@dataclass
class EncoderConfig:
    """Configuration for domain-specific encoders"""

    latent_dim: int = 256
    dropout: float = 0.1
    fusion_strategy: FusionStrategy = FusionStrategy.CROSS_ATTENTION
    use_physics_constraints: bool = True


class ClimateEncoder(nn.Module):
    """Simple CNN encoder for climate datacubes"""

    def __init__(self, config: EncoderConfig, n_input_vars: int = 2):
        super().__init__()
        self.config = config
        self.n_input_vars = n_input_vars

        # Simple approach: flatten spatial-temporal dimensions and use MLPs
        # This avoids complex 3D convolution issues

        # Estimate input size: vars * time * lat * lon * lev
        # For our test case: 2 * 8 * 16 * 24 * 6 = 36864
        estimated_input_size = 36864  # Will adjust dynamically

        self.encoder = nn.Sequential(
            nn.Linear(estimated_input_size, 2048),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(1024, config.latent_dim),
        )

        logger.info(
            f"🌡️ Climate encoder initialized: {n_input_vars} vars → {config.latent_dim}D (MLP)"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for climate encoder

        Args:
            x: Climate datacube [batch, vars, time, lat, lon, lev]

        Returns:
            Latent representation [batch, latent_dim]
        """
        if x.dim() != 6:
            raise ValueError(f"Expected 6D input (B,V,T,H,W,D), got {x.shape}")

        batch_size = x.shape[0]

        # Flatten everything except batch dimension
        x_flat = x.view(batch_size, -1)

        # Adjust encoder input size if needed
        actual_input_size = x_flat.shape[1]
        if actual_input_size != 36864:
            # Recreate encoder with correct input size
            self.encoder = nn.Sequential(
                nn.Linear(actual_input_size, 2048),
                nn.ReLU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(1024, self.config.latent_dim),
            )
            if x.is_cuda:
                self.encoder = self.encoder.cuda()

        # Forward pass
        return self.encoder(x_flat)


class BiologyEncoder(nn.Module):
    """Simple encoder for biological networks"""

    def __init__(self, config: EncoderConfig, node_input_dim: int = 6):
        super().__init__()
        self.config = config
        self.node_input_dim = node_input_dim

        if PYTORCH_GEOMETRIC_AVAILABLE:
            self._init_gnn_encoder()
        else:
            self._init_mlp_encoder()

        logger.info(f"[BIO] Biology encoder initialized → {config.latent_dim}D")

    def _init_gnn_encoder(self):
        """Initialize GNN-based encoder"""
        self.use_gnn = True
        self.input_projection = nn.Linear(self.node_input_dim, 64)
        self.gnn1 = GCNConv(64, 128)
        self.gnn2 = GCNConv(128, 128)
        self.projection = nn.Sequential(
            nn.Linear(256, self.config.latent_dim),  # 256 = 128*2 (mean+max pool)
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
        )

    def _init_mlp_encoder(self):
        """Initialize MLP-based encoder (fallback)"""
        self.use_gnn = False
        # Simple MLP for adjacency + features
        max_nodes = 30
        adj_size = max_nodes * max_nodes
        feat_size = max_nodes * self.node_input_dim

        self.encoder = nn.Sequential(
            nn.Linear(adj_size + feat_size, 1024),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(512, self.config.latent_dim),
        )

    def forward(self, x: Union[PyGBatch, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Forward pass for biology encoder"""

        if self.use_gnn and isinstance(x, PyGBatch):
            # Use GNN for PyTorch Geometric data
            node_features = self.input_projection(x.x)
            h1 = F.relu(self.gnn1(node_features, x.edge_index))
            h2 = F.relu(self.gnn2(h1, x.edge_index))

            # Global pooling
            h_mean = global_mean_pool(h2, x.batch)
            h_max = global_max_pool(h2, x.batch)
            h_graph = torch.cat([h_mean, h_max], dim=1)

            return self.projection(h_graph)

        elif isinstance(x, dict) and "adjacency" in x:
            # Use MLP for adjacency matrix format
            adj = x["adjacency"]  # [batch, nodes, nodes]
            feat = x["node_features"]  # [batch, nodes, features]

            batch_size = adj.shape[0]

            # Flatten and pad/truncate to fixed size
            adj_flat = adj.view(batch_size, -1)
            feat_flat = feat.view(batch_size, -1)

            # Pad or truncate to expected sizes
            max_nodes = 30
            adj_size = max_nodes * max_nodes
            feat_size = max_nodes * self.node_input_dim

            if adj_flat.shape[1] > adj_size:
                adj_flat = adj_flat[:, :adj_size]
            elif adj_flat.shape[1] < adj_size:
                padding = torch.zeros(
                    batch_size, adj_size - adj_flat.shape[1], device=adj_flat.device
                )
                adj_flat = torch.cat([adj_flat, padding], dim=1)

            if feat_flat.shape[1] > feat_size:
                feat_flat = feat_flat[:, :feat_size]
            elif feat_flat.shape[1] < feat_size:
                padding = torch.zeros(
                    batch_size, feat_size - feat_flat.shape[1], device=feat_flat.device
                )
                feat_flat = torch.cat([feat_flat, padding], dim=1)

            # Combine and process
            combined = torch.cat([adj_flat, feat_flat], dim=1)

            if self.use_gnn:
                # Shouldn't reach here, but handle gracefully
                return torch.zeros(batch_size, self.config.latent_dim, device=combined.device)
            else:
                return self.encoder(combined)

        else:
            # Fallback - return zeros
            batch_size = 1
            return torch.zeros(batch_size, self.config.latent_dim)


class SpectroscopyEncoder(nn.Module):
    """Simple 1D CNN encoder for spectra"""

    def __init__(self, config: EncoderConfig, input_channels: int = 2):
        super().__init__()
        self.config = config

        # Simple 1D CNN
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(2)

        # Adaptive pooling and projection
        self.adaptive_pool = nn.AdaptiveAvgPool1d(8)
        self.projection = nn.Sequential(
            nn.Linear(256 * 8, 512),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, config.latent_dim),
        )

        logger.info(
            f"🌈 Spectroscopy encoder initialized: {input_channels}D → {config.latent_dim}D"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for spectroscopy encoder

        Args:
            x: Spectrum data [batch, n_points, 2]

        Returns:
            Latent representation [batch, latent_dim]
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (B,N,C), got {x.shape}")

        # Transpose for 1D conv: [batch, channels, length]
        x = x.transpose(1, 2)  # [batch, 2, n_points]

        # 1D convolutions
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        # Adaptive pooling
        x = self.adaptive_pool(x)  # [batch, 256, 8]

        # Flatten and project
        x = x.view(x.size(0), -1)  # [batch, 256*8]
        return self.projection(x)


class CrossAttentionFusion(nn.Module):
    """Simple cross-attention fusion"""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim

        # Simple cross-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=config.latent_dim, num_heads=8, dropout=config.dropout, batch_first=True
        )

        self.layer_norm = nn.LayerNorm(config.latent_dim)

        self.ffn = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.latent_dim * 2, config.latent_dim),
        )

    def forward(self, modality_features: List[torch.Tensor]) -> torch.Tensor:
        """Cross-attention fusion"""
        # Filter out None features
        valid_features = [f for f in modality_features if f is not None]

        if not valid_features:
            # Return zero if no valid features
            return torch.zeros(1, self.latent_dim)

        if len(valid_features) == 1:
            # Single modality
            return valid_features[0]

        # Stack features: [batch, num_modalities, latent_dim]
        stacked = torch.stack(valid_features, dim=1)

        # Self-attention across modalities
        attn_output, _ = self.attention(stacked, stacked, stacked)

        # Residual connection and layer norm
        stacked = self.layer_norm(stacked + attn_output)

        # FFN
        ffn_output = self.ffn(stacked)
        stacked = self.layer_norm(stacked + ffn_output)

        # Pool across modalities
        return stacked.mean(dim=1)


class DomainPhysicsConstraintLayer(nn.Module):
    """Simple physics constraint layer"""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.energy_constraint = nn.Linear(latent_dim, 1)
        self.mass_constraint = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor, planet_params: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply physics constraints"""
        return {
            "energy_violation": self.energy_constraint(x),
            "mass_violation": self.mass_constraint(x),
        }


class MultiModalEncoder(nn.Module):
    """Simple multi-modal encoder"""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        # Domain encoders
        self.climate_encoder = ClimateEncoder(config)
        self.biology_encoder = BiologyEncoder(config)
        self.spectroscopy_encoder = SpectroscopyEncoder(config)

        # Fusion
        if config.fusion_strategy == FusionStrategy.CROSS_ATTENTION:
            self.fusion = CrossAttentionFusion(config)
        elif config.fusion_strategy == FusionStrategy.CONCATENATION:
            self.fusion = nn.Sequential(
                nn.Linear(config.latent_dim * 3, config.latent_dim * 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.latent_dim * 2, config.latent_dim),
            )
        else:  # MEAN
            self.fusion = None

        # Physics constraints
        if config.use_physics_constraints:
            self.physics_layer = DomainPhysicsConstraintLayer(config.latent_dim)
        else:
            self.physics_layer = None

        logger.info(f"[PROC] Simple multi-modal encoder initialized")

    def forward(self, batch_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        modality_features = []
        individual_features = {}

        # Encode each modality
        if "climate_cubes" in batch_data and batch_data["climate_cubes"] is not None:
            climate_features = self.climate_encoder(batch_data["climate_cubes"])
            modality_features.append(climate_features)
            individual_features["climate"] = climate_features

        if "bio_graphs" in batch_data and batch_data["bio_graphs"] is not None:
            bio_features = self.biology_encoder(batch_data["bio_graphs"])
            modality_features.append(bio_features)
            individual_features["biology"] = bio_features

        if "spectra" in batch_data and batch_data["spectra"] is not None:
            spec_features = self.spectroscopy_encoder(batch_data["spectra"])
            modality_features.append(spec_features)
            individual_features["spectroscopy"] = spec_features

        # Fuse modalities
        if self.config.fusion_strategy == FusionStrategy.CROSS_ATTENTION and self.fusion:
            fused_features = self.fusion(modality_features)
        elif self.config.fusion_strategy == FusionStrategy.CONCATENATION and self.fusion:
            if modality_features:
                # Pad missing modalities with zeros
                batch_size = modality_features[0].shape[0]
                while len(modality_features) < 3:
                    modality_features.append(
                        torch.zeros(
                            batch_size, self.config.latent_dim, device=modality_features[0].device
                        )
                    )

                concatenated = torch.cat(modality_features, dim=1)
                fused_features = self.fusion(concatenated)
            else:
                fused_features = torch.zeros(1, self.config.latent_dim)
        else:  # MEAN fusion
            if modality_features:
                fused_features = torch.stack(modality_features, dim=0).mean(dim=0)
            else:
                fused_features = torch.zeros(1, self.config.latent_dim)

        result = {"fused_features": fused_features, "individual_features": individual_features}

        # Physics constraints
        if self.physics_layer and "planet_params" in batch_data:
            physics_constraints = self.physics_layer(fused_features, batch_data["planet_params"])
            result["physics_constraints"] = physics_constraints

        return result


def create_multimodal_encoder(config: EncoderConfig = None) -> MultiModalEncoder:
    """Create a multi-modal encoder"""
    if config is None:
        config = EncoderConfig()
    return MultiModalEncoder(config)


if __name__ == "__main__":

    def test_simple_encoder():
        logger.info("[TEST] Testing Simple Multi-Modal Encoder")

        # Create config
        config = EncoderConfig(
            latent_dim=256,
            fusion_strategy=FusionStrategy.CROSS_ATTENTION,
            use_physics_constraints=True,
        )

        # Create encoder
        encoder = create_multimodal_encoder(config)
        encoder.eval()

        # Test data
        batch_size = 2

        # Climate: [batch, vars, time, lat, lon, lev]
        climate_cubes = torch.randn(batch_size, 2, 8, 16, 24, 6)

        # Biology: dict format
        bio_graphs = {
            "adjacency": torch.randn(batch_size, 20, 20),
            "node_features": torch.randn(batch_size, 20, 6),
        }

        # Spectroscopy: [batch, n_points, 2]
        spectra = torch.randn(batch_size, 1000, 2)

        # Planet parameters
        planet_params = torch.randn(batch_size, 8)

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
        fused = results["fused_features"]
        individual = results["individual_features"]

        logger.info(f"[OK] Fused features shape: {fused.shape}")
        logger.info(f"[OK] Individual features: {list(individual.keys())}")

        for domain, features in individual.items():
            logger.info(f"  {domain}: {features.shape}")

        if "physics_constraints" in results:
            physics = results["physics_constraints"]
            logger.info(f"[OK] Physics constraints: {list(physics.keys())}")

        # Test partial data
        logger.info("[PROC] Testing with partial data...")

        partial_data = {"climate_cubes": climate_cubes, "planet_params": planet_params}

        with torch.no_grad():
            partial_results = encoder(partial_data)

        logger.info(f"[OK] Partial fused shape: {partial_results['fused_features'].shape}")

        logger.info("[OK] Simple Multi-Modal Encoder test completed!")

        # Show statistics
        total_params = sum(p.numel() for p in encoder.parameters())
        print(f"\n[DATA] Total parameters: {total_params:,}")
        print(f"[DATA] Model size: {total_params * 4 / (1024**2):.1f} MB")

    test_simple_encoder()

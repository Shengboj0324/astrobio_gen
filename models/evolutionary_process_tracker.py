#!/usr/bin/env python3
"""
Evolutionary Process Tracker for Astrobiology
=============================================

Priority 1 Implementation: Extends 4D datacube infrastructure to 5D evolutionary modeling.
Tracks co-evolution of life and environment over geological time scales.

Key Features:
- 5D datacube modeling: [batch, variables, climate_time, geological_time, lev, lat, lon]
- Metabolic pathway evolution tracking using KEGG integration
- Atmospheric evolution signature detection
- Co-evolution dynamics between life and environment
- Deep time narrative construction (billion-year timescales)
- Evolutionary contingency modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
from datetime import datetime, timedelta
import pandas as pd
from scipy.integrate import odeint

# Import existing components
from .datacube_unet import CubeUNet, PhysicsConstraints
from .surrogate_transformer import SurrogateTransformer
from .graph_vae import GVAE

logger = logging.getLogger(__name__)

@dataclass
class EvolutionaryTimeScale:
    """Geological time scale definitions for evolutionary modeling"""
    
    # Geological time boundaries (years ago)
    HADEAN_START = 4.6e9      # Formation of Earth
    ARCHEAN_START = 4.0e9     # First life
    PROTEROZOIC_START = 2.5e9 # Great Oxidation Event
    PHANEROZOIC_START = 0.54e9 # Cambrian Explosion
    PRESENT = 0
    
    # Critical evolutionary events (years ago)
    FIRST_LIFE = 3.8e9
    PHOTOSYNTHESIS = 3.5e9
    EUKARYOTES = 2.0e9
    MULTICELLULAR = 1.0e9
    COMPLEX_LIFE = 0.6e9
    
    # Time resolution for modeling
    GEOLOGICAL_TIMESTEPS = 1000  # 1000 steps across 4.6 billion years
    CLIMATE_TIMESTEPS = 100      # 100 climate steps per geological step

@dataclass
class EvolutionaryState:
    """Complete evolutionary state at a given time"""
    time_gya: float  # Time in billions of years ago
    
    # Life state
    metabolic_complexity: float  # 0-1 scale
    pathway_networks: Dict[str, nx.DiGraph]
    biomass_distribution: torch.Tensor  # Spatial distribution
    evolutionary_innovations: List[str]
    
    # Environment state  
    atmospheric_composition: Dict[str, float]
    surface_temperature: float
    ocean_chemistry: Dict[str, float]
    continental_configuration: str
    
    # Co-evolution metrics
    life_environment_coupling: float  # How strongly life affects environment
    evolutionary_pressure: float     # Environmental pressure on evolution
    
    # Uncertainty and contingency
    path_divergence_potential: float  # How many different paths possible
    contingency_factors: List[str]    # Major contingent events

class MetabolicEvolutionEngine(nn.Module):
    """Models evolution of metabolic pathways over geological time"""
    
    def __init__(
        self,
        n_pathways: int = 7302,  # KEGG pathway count
        pathway_embed_dim: int = 128,
        time_embed_dim: int = 64,
        evolution_dim: int = 256
    ):
        super().__init__()
        
        self.n_pathways = n_pathways
        self.pathway_embed_dim = pathway_embed_dim
        
        # Pathway embedding (from KEGG data)
        self.pathway_embedder = nn.Embedding(n_pathways, pathway_embed_dim)
        
        # Geological time embedding
        self.time_embedder = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Evolution dynamics network
        self.evolution_net = nn.Sequential(
            nn.Linear(pathway_embed_dim + time_embed_dim, evolution_dim),
            nn.ReLU(),
            nn.Linear(evolution_dim, evolution_dim),
            nn.ReLU(),
            nn.Linear(evolution_dim, pathway_embed_dim)
        )
        
        # Innovation probability head
        self.innovation_head = nn.Linear(evolution_dim, 1)
        
        # Environmental coupling head  
        self.coupling_head = nn.Linear(evolution_dim, 4)  # O2, CO2, CH4, H2O effects
        
        # Graph VAE for pathway network evolution
        self.pathway_vae = GVAE(in_channels=pathway_embed_dim, latent=64)
        
    def forward(
        self, 
        pathway_ids: torch.Tensor,
        geological_time: torch.Tensor,
        environmental_state: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Model metabolic evolution over time
        
        Args:
            pathway_ids: [batch, n_active_pathways] - Active pathway indices
            geological_time: [batch, 1] - Time in billions of years ago
            environmental_state: [batch, env_dim] - Environmental conditions
            
        Returns:
            Evolution predictions and environmental coupling
        """
        # Embed pathways and time
        pathway_embeds = self.pathway_embedder(pathway_ids)  # [batch, n_pathways, embed_dim]
        time_embeds = self.time_embedder(geological_time)    # [batch, time_dim]
        
        # Combine pathway and time information
        batch_size, n_pathways, embed_dim = pathway_embeds.shape
        time_expanded = time_embeds.unsqueeze(1).expand(-1, n_pathways, -1)
        
        combined = torch.cat([pathway_embeds, time_expanded], dim=-1)
        
        # Model evolution dynamics
        evolved_pathways = self.evolution_net(combined)  # [batch, n_pathways, embed_dim]
        
        # Innovation probability (new pathways emerging)
        innovation_prob = torch.sigmoid(
            self.innovation_head(combined.mean(dim=1))
        )  # [batch, 1]
        
        # Environmental coupling (how metabolism affects atmosphere)
        env_coupling = self.coupling_head(combined.mean(dim=1))  # [batch, 4]
        
        return {
            'evolved_pathways': evolved_pathways,
            'innovation_probability': innovation_prob,
            'environmental_coupling': env_coupling,
            'metabolic_complexity': evolved_pathways.norm(dim=-1).mean(dim=-1),
            'pathway_diversity': self._compute_pathway_diversity(evolved_pathways)
        }
    
    def _compute_pathway_diversity(self, pathway_embeds: torch.Tensor) -> torch.Tensor:
        """Compute metabolic diversity using embedding distances"""
        # Pairwise distances between pathway embeddings
        embeds_norm = F.normalize(pathway_embeds, dim=-1)
        similarity_matrix = torch.bmm(embeds_norm, embeds_norm.transpose(-2, -1))
        
        # Diversity as negative of average similarity
        diversity = 1.0 - similarity_matrix.mean(dim=(-2, -1))
        return diversity

class AtmosphericEvolutionEngine(nn.Module):
    """Models atmospheric evolution coupled with biological processes"""
    
    def __init__(
        self,
        n_gases: int = 10,
        atmosphere_dim: int = 128,
        coupling_dim: int = 64
    ):
        super().__init__()
        
        self.n_gases = n_gases
        
        # Atmospheric dynamics network
        self.atmosphere_net = nn.Sequential(
            nn.Linear(n_gases + 1, atmosphere_dim),  # +1 for time
            nn.ReLU(),
            nn.Linear(atmosphere_dim, atmosphere_dim),
            nn.ReLU(),
            nn.Linear(atmosphere_dim, n_gases)
        )
        
        # Life-atmosphere coupling network
        self.coupling_net = nn.Sequential(
            nn.Linear(n_gases + 4, coupling_dim),  # +4 from metabolic coupling
            nn.ReLU(),
            nn.Linear(coupling_dim, n_gases)
        )
        
        # Biosignature detection head
        self.biosignature_head = nn.Sequential(
            nn.Linear(n_gases, 32),
            nn.ReLU(), 
            nn.Linear(32, 4)  # O2, CH4, O3, phosphine biosignatures
        )
        
    def forward(
        self,
        atmospheric_state: torch.Tensor,
        geological_time: torch.Tensor,
        metabolic_coupling: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Model atmospheric evolution with biological coupling
        
        Args:
            atmospheric_state: [batch, n_gases] - Current atmospheric composition
            geological_time: [batch, 1] - Geological time
            metabolic_coupling: [batch, 4] - Coupling from metabolic evolution
            
        Returns:
            Atmospheric evolution predictions
        """
        # Atmospheric dynamics (abiotic processes)
        time_atmos = torch.cat([atmospheric_state, geological_time], dim=-1)
        abiotic_evolution = self.atmosphere_net(time_atmos)
        
        # Biological coupling effects
        coupled_input = torch.cat([atmospheric_state, metabolic_coupling], dim=-1)
        biotic_effects = self.coupling_net(coupled_input)
        
        # Combined atmospheric evolution
        new_atmosphere = atmospheric_state + abiotic_evolution + biotic_effects
        
        # Ensure physical constraints (positive concentrations, normalization)
        new_atmosphere = F.softmax(F.relu(new_atmosphere), dim=-1)
        
        # Detect biosignatures
        biosignatures = torch.sigmoid(self.biosignature_head(new_atmosphere))
        
        return {
            'new_atmospheric_state': new_atmosphere,
            'abiotic_component': abiotic_evolution,
            'biotic_component': biotic_effects,
            'biosignature_strength': biosignatures,
            'atmospheric_disequilibrium': self._compute_disequilibrium(new_atmosphere)
        }
    
    def _compute_disequilibrium(self, atmosphere: torch.Tensor) -> torch.Tensor:
        """Compute atmospheric disequilibrium as biosignature indicator"""
        # Simplified: presence of both O2 and CH4 indicates disequilibrium
        o2_idx, ch4_idx = 0, 1  # Assume first two indices
        o2_ch4_product = atmosphere[:, o2_idx] * atmosphere[:, ch4_idx]
        return o2_ch4_product

class FiveDimensionalDatacube(nn.Module):
    """5D datacube processor: [batch, vars, climate_time, geo_time, lev, lat, lon]"""
    
    def __init__(
        self,
        base_cube_model: CubeUNet,
        geological_time_dim: int = 1000,
        climate_time_dim: int = 100
    ):
        super().__init__()
        
        self.base_cube_model = base_cube_model
        self.geological_time_dim = geological_time_dim
        self.climate_time_dim = climate_time_dim
        
        # Temporal evolution network for geological time
        self.geological_evolution = nn.LSTM(
            input_size=base_cube_model.n_output_vars,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=False
        )
        
        # Cross-time attention for climate-geological coupling
        self.cross_time_attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=8,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process 5D datacube with temporal evolution
        
        Args:
            x: [batch, vars, climate_time, geo_time, lev, lat, lon]
            
        Returns:
            5D evolution predictions
        """
        batch_size, n_vars, climate_t, geo_t, lev, lat, lon = x.shape
        
        # Process each geological time step through base 4D model
        geo_time_predictions = []
        
        for t_geo in range(geo_t):
            # Extract 4D slice: [batch, vars, climate_time, lev, lat, lon]
            x_4d = x[:, :, :, t_geo, :, :, :]
            
            # Process through base datacube model
            pred_4d = self.base_cube_model(x_4d)  # [batch, vars, climate_time, lev, lat, lon]
            
            geo_time_predictions.append(pred_4d)
        
        # Stack geological time predictions
        predictions_5d = torch.stack(geo_time_predictions, dim=3)  # [batch, vars, climate_t, geo_t, lev, lat, lon]
        
        # Model geological time evolution using LSTM
        # Reshape for LSTM: [batch * spatial, geo_time, vars]
        spatial_size = lev * lat * lon
        lstm_input = predictions_5d.permute(0, 4, 5, 6, 2, 3, 1)  # [batch, lev, lat, lon, climate_t, geo_t, vars]
        lstm_input = lstm_input.reshape(batch_size * spatial_size * climate_t, geo_t, n_vars)
        
        geological_evolution, _ = self.geological_evolution(lstm_input)
        
        # Reshape back to 5D
        geological_evolution = geological_evolution.reshape(
            batch_size, lev, lat, lon, climate_t, geo_t, n_vars
        ).permute(0, 6, 4, 5, 1, 2, 3)  # Back to [batch, vars, climate_t, geo_t, lev, lat, lon]
        
        return {
            'predictions_5d': predictions_5d,
            'geological_evolution': geological_evolution,
            'evolutionary_trajectory': self._extract_evolutionary_trajectory(geological_evolution)
        }
    
    def _extract_evolutionary_trajectory(self, evolution: torch.Tensor) -> torch.Tensor:
        """Extract key evolutionary trajectory metrics"""
        # Global mean evolution over spatial dimensions
        global_evolution = evolution.mean(dim=(-3, -2, -1))  # [batch, vars, climate_t, geo_t]
        
        # Extract key metrics across geological time
        trajectory_metrics = {
            'temperature_evolution': global_evolution[:, 0, :, :],  # Assuming temp is first var
            'atmospheric_evolution': global_evolution[:, 1:4, :, :],  # Atmospheric gases
            'complexity_evolution': global_evolution.var(dim=1).mean(dim=1)  # Variance as complexity proxy
        }
        
        return trajectory_metrics

class EvolutionaryProcessTracker(pl.LightningModule):
    """
    Main evolutionary process tracking system
    Integrates metabolic, atmospheric, and datacube evolution
    """
    
    def __init__(
        self,
        datacube_config: Dict[str, Any],
        metabolic_config: Dict[str, Any] = None,
        atmospheric_config: Dict[str, Any] = None,
        learning_rate: float = 1e-4,
        physics_weight: float = 0.1,
        evolution_weight: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize base 4D datacube model
        self.base_datacube = CubeUNet(**datacube_config)
        
        # Initialize 5D datacube processor
        self.datacube_5d = FiveDimensionalDatacube(
            base_cube_model=self.base_datacube,
            geological_time_dim=EvolutionaryTimeScale.GEOLOGICAL_TIMESTEPS,
            climate_time_dim=EvolutionaryTimeScale.CLIMATE_TIMESTEPS
        )
        
        # Initialize metabolic evolution engine
        metabolic_config = metabolic_config or {}
        self.metabolic_engine = MetabolicEvolutionEngine(**metabolic_config)
        
        # Initialize atmospheric evolution engine  
        atmospheric_config = atmospheric_config or {}
        self.atmospheric_engine = AtmosphericEvolutionEngine(**atmospheric_config)
        
        # Evolutionary constraint layer
        self.evolution_constraints = EvolutionaryConstraints()
        
        # Loss weights
        self.learning_rate = learning_rate
        self.physics_weight = physics_weight
        self.evolution_weight = evolution_weight
        
        logger.info("Initialized Evolutionary Process Tracker with 5D datacube modeling")
    
    def forward(
        self,
        datacube_5d: torch.Tensor,
        pathway_ids: torch.Tensor,
        geological_time: torch.Tensor,
        environmental_state: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through evolutionary process tracking
        
        Args:
            datacube_5d: [batch, vars, climate_time, geo_time, lev, lat, lon]
            pathway_ids: [batch, n_pathways] - Active metabolic pathways
            geological_time: [batch, 1] - Current geological time
            environmental_state: [batch, env_dim] - Environmental conditions
            
        Returns:
            Complete evolutionary state predictions
        """
        # Process 5D datacube evolution
        datacube_results = self.datacube_5d(datacube_5d)
        
        # Model metabolic evolution
        metabolic_results = self.metabolic_engine(
            pathway_ids, geological_time, environmental_state
        )
        
        # Extract atmospheric state from datacube
        atmospheric_state = datacube_results['predictions_5d'][:, 1:5, -1, -1, 0, :, :].mean(dim=(-2, -1))
        
        # Model atmospheric evolution
        atmospheric_results = self.atmospheric_engine(
            atmospheric_state,
            geological_time,
            metabolic_results['environmental_coupling']
        )
        
        return {
            **datacube_results,
            **metabolic_results,
            **atmospheric_results,
            'evolutionary_state': self._construct_evolutionary_state(
                datacube_results, metabolic_results, atmospheric_results, geological_time
            )
        }
    
    def _construct_evolutionary_state(
        self,
        datacube_results: Dict[str, torch.Tensor],
        metabolic_results: Dict[str, torch.Tensor], 
        atmospheric_results: Dict[str, torch.Tensor],
        geological_time: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Construct complete evolutionary state representation"""
        
        return {
            'time_gya': geological_time,
            'metabolic_complexity': metabolic_results['metabolic_complexity'],
            'pathway_diversity': metabolic_results['pathway_diversity'],
            'atmospheric_disequilibrium': atmospheric_results['atmospheric_disequilibrium'],
            'biosignature_strength': atmospheric_results['biosignature_strength'],
            'evolutionary_trajectory': datacube_results['evolutionary_trajectory'],
            'life_environment_coupling': self._compute_coupling_strength(
                metabolic_results, atmospheric_results
            )
        }
    
    def _compute_coupling_strength(
        self,
        metabolic_results: Dict[str, torch.Tensor],
        atmospheric_results: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute strength of life-environment coupling"""
        coupling_metabolic = metabolic_results['environmental_coupling'].norm(dim=-1)
        coupling_atmospheric = atmospheric_results['biotic_component'].norm(dim=-1)
        
        return (coupling_metabolic + coupling_atmospheric) / 2.0
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute evolutionary process tracking loss"""
        losses = {}
        
        # Base datacube reconstruction loss
        if 'predictions_5d' in predictions and '5d_targets' in targets:
            datacube_loss = F.mse_loss(predictions['predictions_5d'], targets['5d_targets'])
            losses['datacube_loss'] = datacube_loss
        
        # Metabolic evolution loss
        if 'metabolic_complexity' in predictions and 'metabolic_targets' in targets:
            metabolic_loss = F.mse_loss(
                predictions['metabolic_complexity'], 
                targets['metabolic_targets']
            )
            losses['metabolic_loss'] = metabolic_loss
        
        # Atmospheric evolution loss
        if 'new_atmospheric_state' in predictions and 'atmospheric_targets' in targets:
            atmospheric_loss = F.mse_loss(
                predictions['new_atmospheric_state'],
                targets['atmospheric_targets']
            )
            losses['atmospheric_loss'] = atmospheric_loss
        
        # Evolutionary constraints
        evolution_constraints = self.evolution_constraints(predictions)
        losses['evolution_constraints'] = evolution_constraints
        
        # Total loss
        total_loss = (
            losses.get('datacube_loss', 0) +
            self.evolution_weight * losses.get('metabolic_loss', 0) +
            self.evolution_weight * losses.get('atmospheric_loss', 0) +
            self.physics_weight * losses['evolution_constraints']
        )
        losses['total_loss'] = total_loss
        
        return losses
    
    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        """Training step for evolutionary process tracking"""
        datacube_5d, pathway_ids, geological_time, environmental_state, targets = batch
        
        # Forward pass
        predictions = self(datacube_5d, pathway_ids, geological_time, environmental_state)
        
        # Compute losses
        losses = self.compute_loss(predictions, targets)
        
        # Log losses
        for name, loss in losses.items():
            self.log(f'train/{name}', loss, on_step=True, on_epoch=True)
        
        return losses['total_loss']
    
    def configure_optimizers(self):
        """Configure optimizer for evolutionary training"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.7,
            patience=10,
            min_lr=1e-7
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/total_loss",
                "interval": "epoch",
            },
        }

class EvolutionaryConstraints(nn.Module):
    """Physics and evolutionary constraints for process modeling"""
    
    def __init__(self):
        super().__init__()
        
        # Evolutionary time scales
        self.time_scales = EvolutionaryTimeScale()
        
    def forward(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute evolutionary constraints"""
        constraints = []
        
        # Constraint 1: Metabolic complexity should generally increase over time
        if 'time_gya' in predictions and 'metabolic_complexity' in predictions:
            time_gya = predictions['time_gya']
            complexity = predictions['metabolic_complexity']
            
            # Complexity should correlate with evolutionary time (4.6 billion years ago to present)
            expected_complexity = (4.6 - time_gya) / 4.6  # 0 at 4.6 Gya, 1 at present
            complexity_constraint = F.mse_loss(complexity, expected_complexity.squeeze(-1))
            constraints.append(complexity_constraint)
        
        # Constraint 2: Atmospheric disequilibrium requires life
        if 'atmospheric_disequilibrium' in predictions and 'metabolic_complexity' in predictions:
            disequilibrium = predictions['atmospheric_disequilibrium']
            complexity = predictions['metabolic_complexity']
            
            # Disequilibrium should correlate with metabolic complexity
            disequilibrium_constraint = F.mse_loss(disequilibrium, complexity * 0.5)
            constraints.append(disequilibrium_constraint)
        
        # Constraint 3: Biosignatures only after certain evolutionary milestones
        if 'biosignature_strength' in predictions and 'time_gya' in predictions:
            biosignatures = predictions['biosignature_strength']
            time_gya = predictions['time_gya']
            
            # O2 biosignatures only after Great Oxidation Event (2.5 Gya)
            o2_mask = (time_gya < 2.5).float().squeeze(-1)
            o2_constraint = F.mse_loss(biosignatures[:, 0] * (1 - o2_mask), torch.zeros_like(biosignatures[:, 0]))
            constraints.append(o2_constraint)
        
        # Combine all constraints
        if constraints:
            return sum(constraints) / len(constraints)
        else:
            return torch.tensor(0.0, device=predictions[list(predictions.keys())[0]].device)

def create_evolutionary_dataset_from_kegg(
    kegg_data_path: str,
    output_path: str,
    n_samples: int = 1000
) -> str:
    """
    Create evolutionary process dataset from KEGG pathway data
    
    Args:
        kegg_data_path: Path to processed KEGG data
        output_path: Output path for evolutionary dataset
        n_samples: Number of evolutionary trajectories to generate
        
    Returns:
        Path to created dataset
    """
    from data_build.kegg_real_data_integration import KEGGDataProcessor
    
    logger.info(f"Creating evolutionary dataset with {n_samples} trajectories")
    
    # Load KEGG data
    processor = KEGGDataProcessor(kegg_data_path)
    
    # Generate evolutionary trajectories
    trajectories = []
    
    for i in range(n_samples):
        # Create random evolutionary trajectory
        trajectory = generate_evolutionary_trajectory(processor, i)
        trajectories.append(trajectory)
    
    # Save dataset
    output_file = Path(output_path) / "evolutionary_trajectories.pt"
    torch.save(trajectories, output_file)
    
    logger.info(f"Created evolutionary dataset: {output_file}")
    return str(output_file)

def generate_evolutionary_trajectory(
    kegg_processor: Any,
    seed: int = 0
) -> Dict[str, torch.Tensor]:
    """Generate a single evolutionary trajectory using KEGG data"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    time_scales = EvolutionaryTimeScale()
    
    # Time points (billion years ago)
    time_points = torch.linspace(4.6, 0, time_scales.GEOLOGICAL_TIMESTEPS)
    
    # Initialize trajectory data
    trajectory = {
        'time_gya': time_points,
        'pathway_evolution': [],
        'atmospheric_evolution': [],
        'datacube_evolution': []
    }
    
    # Model pathway evolution over time
    for t_gya in time_points:
        # Simple model: more pathways active as evolution progresses
        evolution_progress = (4.6 - t_gya) / 4.6
        
        # Number of active pathways increases with evolution
        n_active = int(100 + 7000 * evolution_progress)  # 100 to 7100 pathways
        
        # Random pathway selection (in real implementation, use KEGG temporal data)
        active_pathways = torch.randint(0, 7302, (n_active,))
        
        trajectory['pathway_evolution'].append(active_pathways)
        
        # Model atmospheric evolution (simplified)
        # Early: reducing atmosphere, Late: oxidizing atmosphere
        o2_level = max(0, evolution_progress - 0.5) * 0.21  # O2 appears after 50% evolution
        co2_level = max(0.01, 0.4 - evolution_progress * 0.35)  # CO2 decreases
        ch4_level = max(0.001, 0.1 - evolution_progress * 0.095)  # CH4 decreases
        
        atmosphere = torch.tensor([o2_level, co2_level, ch4_level, 0.78])  # N2 constant
        atmosphere = atmosphere / atmosphere.sum()  # Normalize
        
        trajectory['atmospheric_evolution'].append(atmosphere)
        
        # Generate synthetic 4D datacube for this time point
        datacube_4d = torch.randn(5, 100, 20, 64, 64)  # [vars, time, lev, lat, lon]
        trajectory['datacube_evolution'].append(datacube_4d)
    
    # Stack time series
    trajectory['pathway_evolution'] = trajectory['pathway_evolution']  # Keep as list of different sizes
    trajectory['atmospheric_evolution'] = torch.stack(trajectory['atmospheric_evolution'])
    trajectory['datacube_evolution'] = torch.stack(trajectory['datacube_evolution'])
    
    return trajectory 
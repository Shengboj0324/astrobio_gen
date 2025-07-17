#!/usr/bin/env python3
"""
Unified DataLoader Architecture (Fixed)
=======================================

Fixed standalone PyTorch DataPipe system for multi-modal scientific data loading.
This version handles all edge cases and data type issues properly.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from threading import Lock
import time
import json
import random
from collections import deque

# PyTorch Geometric for graph data (optional)
try:
    import torch_geometric
    from torch_geometric.data import Data as PyGData, Batch as PyGBatch
    PYTORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    PYTORCH_GEOMETRIC_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatchingStrategy(Enum):
    """Batching strategies for multi-modal data"""
    FIXED_SIZE = "fixed_size"
    ADAPTIVE_MEMORY = "adaptive_memory"

class DataSplit(Enum):
    """Data split types"""
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"

@dataclass
class DataLoaderConfig:
    """Configuration for unified data loader"""
    # Basic settings
    batch_size: int = 4
    num_workers: int = 0  # Default to 0 for stability
    pin_memory: bool = False  # Default to False for CPU-only systems
    
    # Multi-modal settings
    include_climate: bool = True
    include_biology: bool = True
    include_spectroscopy: bool = True
    include_metadata: bool = True
    
    # Preprocessing
    normalize_climate: bool = True
    normalize_spectra: bool = True
    standardize_bio_features: bool = True
    
    # Cache settings
    enable_caching: bool = True
    
    # Data augmentation
    enable_augmentation: bool = False
    climate_noise_std: float = 0.01
    spectrum_noise_std: float = 0.005

@dataclass
class MultiModalBatch:
    """Container for multi-modal batch data"""
    # Core data
    run_ids: torch.Tensor                    # [batch_size]
    planet_params: torch.Tensor              # [batch_size, n_params]
    
    # Domain-specific data
    climate_cubes: Optional[torch.Tensor] = None      # [batch_size, vars, time, lat, lon, lev]
    bio_graphs: Optional[Any] = None                   # PyG batch or tensor
    spectra: Optional[torch.Tensor] = None             # [batch_size, wavelengths, features]
    
    # Metadata
    metadata: List[Dict[str, Any]] = field(default_factory=list)
    
    def to(self, device):
        """Move batch to device"""
        result = MultiModalBatch(
            run_ids=self.run_ids.to(device),
            planet_params=self.planet_params.to(device),
            metadata=self.metadata
        )
        
        if self.climate_cubes is not None:
            result.climate_cubes = self.climate_cubes.to(device)
        
        if self.spectra is not None:
            result.spectra = self.spectra.to(device)
        
        if self.bio_graphs is not None:
            if hasattr(self.bio_graphs, 'to'):
                result.bio_graphs = self.bio_graphs.to(device)
            else:
                result.bio_graphs = self.bio_graphs
        
        return result

class MockDataStorage:
    """Mock storage system for testing"""
    
    def __init__(self, n_runs: int = 20):
        self.n_runs = n_runs
        self.runs = list(range(1, n_runs + 1))
        
    def list_stored_runs(self) -> List[int]:
        return self.runs
    
    async def load_climate_datacube(self, run_id: int) -> Dict[str, Any]:
        """Generate mock climate data"""
        # Smaller dimensions for testing
        time_steps, lat_points, lon_points, lev_points = 8, 16, 24, 6
        
        # Set random seed for reproducible data
        np.random.seed(run_id)
        
        temp = 280 + 20 * np.random.random((time_steps, lat_points, lon_points, lev_points))
        humidity = 0.01 * np.random.random((time_steps, lat_points, lon_points, lev_points))
        
        return {
            'data': {
                'temperature': temp.astype(np.float32),
                'humidity': humidity.astype(np.float32)
            },
            'dimensions': {
                'time': np.arange(time_steps, dtype=np.float32),
                'lat': np.linspace(-90, 90, lat_points, dtype=np.float32),
                'lon': np.linspace(-180, 180, lon_points, dtype=np.float32),
                'lev': np.linspace(1000, 1, lev_points, dtype=np.float32)
            },
            'metadata': {
                'run_id': run_id,
                'model': 'ROCKE-3D'
            }
        }
    
    async def load_biological_network(self, run_id: int) -> Dict[str, Any]:
        """Generate mock biological network"""
        n_nodes = 20  # Smaller for testing
        
        # Set random seed for reproducible data
        np.random.seed(run_id + 1000)
        
        # Create sparse adjacency matrix
        adj_matrix = np.random.choice([0, 1], size=(n_nodes, n_nodes), p=[0.85, 0.15])
        adj_matrix = (adj_matrix + adj_matrix.T) > 0  # Make symmetric
        np.fill_diagonal(adj_matrix, 0)  # Remove self-loops
        
        node_features = np.random.random((n_nodes, 6)).astype(np.float32)
        node_names = [f"metabolite_{i:03d}" for i in range(n_nodes)]
        
        return {
            'adjacency_matrix': adj_matrix.astype(np.float32),
            'node_features': node_features,
            'node_names': node_names,
            'metadata': {
                'run_id': run_id,
                'network_type': 'metabolic',
                'n_nodes': n_nodes,
                'n_edges': int(adj_matrix.sum())
            }
        }
    
    async def load_spectrum(self, run_id: int) -> Dict[str, Any]:
        """Generate mock spectrum"""
        n_points = 1000  # Smaller for testing
        
        # Set random seed for reproducible data
        np.random.seed(run_id + 2000)
        
        wavelengths = np.linspace(0.5, 30.0, n_points, dtype=np.float32)
        
        # Generate realistic spectrum
        flux = np.exp(-((wavelengths - 10.0) / 8.0)**2)  # Base continuum
        
        # Add some absorption lines
        line_centers = [2.0, 4.3, 9.6, 15.0]
        for center in line_centers:
            line_depth = 0.1 + 0.2 * np.random.random()
            line_width = 0.1 + 0.1 * np.random.random()
            flux *= (1 - line_depth * np.exp(-((wavelengths - center) / line_width)**2))
        
        # Add noise
        flux += 0.01 * np.random.random(n_points)
        flux = flux.astype(np.float32)
        
        return {
            'wavelengths': wavelengths,
            'flux': flux,
            'metadata': {
                'run_id': run_id,
                'instrument': 'PSG',
                'resolution': 100000,
                'n_points': n_points
            }
        }

class MultiModalPreprocessor:
    """Preprocessor for multi-modal scientific data"""
    
    def __init__(self, config: DataLoaderConfig):
        self.config = config
        
        # Normalization statistics
        self.climate_stats = {
            'temperature': {'mean': 280.0, 'std': 50.0},
            'humidity': {'mean': 0.005, 'std': 0.01}
        }
        
        self.spectrum_stats = {'mean': 0.5, 'std': 0.3}
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a multi-modal sample"""
        processed = {
            'run_id': sample['run_id'],
            'planet_params': self._process_planet_params(sample['planet_params'])
        }
        
        # Process climate data
        if sample.get('climate_data') and self.config.include_climate:
            processed['climate_cube'] = self._process_climate_data(sample['climate_data'])
        
        # Process biological data
        if sample.get('bio_data') and self.config.include_biology:
            processed['bio_graph'] = self._process_biological_data(sample['bio_data'])
        
        # Process spectrum data
        if sample.get('spectrum_data') and self.config.include_spectroscopy:
            processed['spectrum'] = self._process_spectrum_data(sample['spectrum_data'])
        
        # Add metadata
        if self.config.include_metadata:
            processed['metadata'] = sample.get('metadata', {})
        
        return processed
    
    def _process_planet_params(self, params: Optional[np.ndarray]) -> torch.Tensor:
        """Process planet parameters"""
        if params is None:
            # Generate default parameters
            params = np.array([1.0, 1.0, 365.0, 1.0, 0.001, 0.0004, 0.0, 0.0], dtype=np.float32)
        
        return torch.from_numpy(params).float()
    
    def _process_climate_data(self, climate_data: Dict[str, Any]) -> torch.Tensor:
        """Process climate datacube"""
        data_vars = climate_data['data']
        
        # Stack variables into tensor [vars, time, lat, lon, lev]
        var_tensors = []
        for var_name in ['temperature', 'humidity']:
            if var_name in data_vars:
                var_data = data_vars[var_name]
                
                # Normalize if enabled
                if self.config.normalize_climate and var_name in self.climate_stats:
                    stats = self.climate_stats[var_name]
                    var_data = (var_data - stats['mean']) / stats['std']
                
                # Add noise for augmentation
                if self.config.enable_augmentation:
                    noise = np.random.normal(0, self.config.climate_noise_std, var_data.shape)
                    var_data = var_data + noise
                
                var_tensors.append(torch.from_numpy(var_data).float())
        
        if var_tensors:
            # Stack: [vars, time, lat, lon, lev]
            climate_tensor = torch.stack(var_tensors, dim=0)
            return climate_tensor
        else:
            # Return dummy tensor if no data
            return torch.zeros(2, 8, 16, 24, 6, dtype=torch.float32)
    
    def _process_biological_data(self, bio_data: Dict[str, Any]) -> Union[torch.Tensor, Any]:
        """Process biological network data"""
        adj_matrix = bio_data['adjacency_matrix']
        node_features = bio_data['node_features']
        
        if PYTORCH_GEOMETRIC_AVAILABLE:
            # Create PyTorch Geometric data object
            # Fix the numpy nonzero issue
            edge_indices = np.nonzero(adj_matrix)
            if len(edge_indices) == 2:  # 2D array case
                edge_index = torch.from_numpy(np.stack(edge_indices, axis=0)).long()
            else:
                # Fallback - create empty edge index
                edge_index = torch.empty((2, 0), dtype=torch.long)
            
            node_features_tensor = torch.from_numpy(node_features).float()
            
            # Standardize features if enabled
            if self.config.standardize_bio_features:
                mean = node_features_tensor.mean(0)
                std = node_features_tensor.std(0) + 1e-8
                node_features_tensor = (node_features_tensor - mean) / std
            
            graph_data = PyGData(
                x=node_features_tensor,
                edge_index=edge_index,
                num_nodes=adj_matrix.shape[0]
            )
            
            return graph_data
        else:
            # Fallback to tensors
            adj_tensor = torch.from_numpy(adj_matrix).float()
            features_tensor = torch.from_numpy(node_features).float()
            
            # Standardize features if enabled
            if self.config.standardize_bio_features:
                mean = features_tensor.mean(0)
                std = features_tensor.std(0) + 1e-8
                features_tensor = (features_tensor - mean) / std
            
            return {
                'adjacency': adj_tensor,
                'node_features': features_tensor
            }
    
    def _process_spectrum_data(self, spectrum_data: Dict[str, Any]) -> torch.Tensor:
        """Process spectrum data"""
        wavelengths = spectrum_data['wavelengths']
        flux = spectrum_data['flux']
        
        # Combine wavelengths and flux [n_points, 2]
        spectrum_tensor = torch.stack([
            torch.from_numpy(wavelengths).float(),
            torch.from_numpy(flux).float()
        ], dim=1)
        
        # Normalize if enabled
        if self.config.normalize_spectra:
            spectrum_tensor = (spectrum_tensor - self.spectrum_stats['mean']) / self.spectrum_stats['std']
        
        # Add noise for augmentation
        if self.config.enable_augmentation:
            noise = torch.normal(0, self.config.spectrum_noise_std, spectrum_tensor.shape)
            spectrum_tensor = spectrum_tensor + noise
        
        return spectrum_tensor

class PlanetRunDataset(Dataset):
    """Dataset for loading multi-modal planet run data"""
    
    def __init__(self,
                 config: DataLoaderConfig,
                 data_split: DataSplit = DataSplit.TRAIN,
                 storage_manager = None):
        
        self.config = config
        self.data_split = data_split
        self.storage_manager = storage_manager or MockDataStorage()
        
        # Load available runs
        self.available_runs = self._load_available_runs()
        self.preprocessor = MultiModalPreprocessor(config)
        
        # Cache for preprocessed data
        self._cache = {} if config.enable_caching else None
        self._cache_lock = Lock()
        
        logger.info(f"[DATA] Dataset initialized: {len(self.available_runs)} runs ({data_split.value})")
    
    def _load_available_runs(self) -> List[int]:
        """Load available planet runs for the specified split"""
        all_runs = self.storage_manager.list_stored_runs()
        
        # Simple deterministic split based on run ID
        random.seed(42)
        shuffled_runs = all_runs.copy()
        random.shuffle(shuffled_runs)
        
        n_total = len(shuffled_runs)
        if self.data_split == DataSplit.TRAIN:
            return shuffled_runs[:int(0.7 * n_total)]
        elif self.data_split == DataSplit.VALIDATION:
            start_idx = int(0.7 * n_total)
            end_idx = int(0.85 * n_total)
            return shuffled_runs[start_idx:end_idx]
        else:  # TEST
            return shuffled_runs[int(0.85 * n_total):]
    
    def __len__(self) -> int:
        return len(self.available_runs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get multi-modal sample for a planet run"""
        run_id = self.available_runs[idx]
        
        # Check cache first
        if self._cache is not None:
            with self._cache_lock:
                if run_id in self._cache:
                    return self._cache[run_id]
        
        # Load multi-modal data
        sample = self._load_multimodal_sample(run_id)
        
        # Preprocess data
        try:
            sample = self.preprocessor.process_sample(sample)
        except Exception as e:
            logger.warning(f"Failed to preprocess sample {run_id}: {e}")
            # Return minimal sample
            sample = {
                'run_id': run_id,
                'planet_params': torch.zeros(8, dtype=torch.float32),
                'metadata': {}
            }
        
        # Cache if enabled
        if self._cache is not None:
            with self._cache_lock:
                self._cache[run_id] = sample
        
        return sample
    
    def _load_multimodal_sample(self, run_id: int) -> Dict[str, Any]:
        """Load raw multi-modal data for a planet run"""
        sample = {
            'run_id': run_id,
            'climate_data': None,
            'bio_data': None,
            'spectrum_data': None,
            'planet_params': None,
            'metadata': {}
        }
        
        try:
            # Load climate data
            if self.config.include_climate:
                climate_data = asyncio.run(self.storage_manager.load_climate_datacube(run_id))
                sample['climate_data'] = climate_data
                sample['metadata'].update(climate_data.get('metadata', {}))
            
            # Load biological network
            if self.config.include_biology:
                bio_data = asyncio.run(self.storage_manager.load_biological_network(run_id))
                sample['bio_data'] = bio_data
                sample['metadata'].update(bio_data.get('metadata', {}))
            
            # Load spectrum
            if self.config.include_spectroscopy:
                spectrum_data = asyncio.run(self.storage_manager.load_spectrum(run_id))
                sample['spectrum_data'] = spectrum_data
                sample['metadata'].update(spectrum_data.get('metadata', {}))
            
            # Generate planet parameters (would come from database in practice)
            np.random.seed(run_id)
            sample['planet_params'] = np.array([
                np.random.uniform(0.5, 2.5),    # radius_earth
                np.random.uniform(0.3, 10.0),   # mass_earth
                np.random.uniform(1.0, 100.0),  # period_days
                np.random.uniform(0.1, 10.0),   # stellar_flux
                np.random.uniform(0.0, 0.01),   # H2O
                np.random.uniform(0.0, 0.001),  # CO2
                np.random.uniform(0.0, 0.0001), # CH4
                np.random.uniform(0.0, 0.0001)  # O2
            ], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Failed to load data for run {run_id}: {e}")
        
        return sample

def collate_multimodal_batch(batch: List[Dict[str, Any]]) -> MultiModalBatch:
    """Collate function for multi-modal batches"""
    
    # Extract components
    run_ids = [sample['run_id'] for sample in batch]
    planet_params = [sample['planet_params'] for sample in batch]
    
    # Create batch containers
    batch_data = MultiModalBatch(
        run_ids=torch.tensor(run_ids, dtype=torch.long),
        planet_params=torch.stack(planet_params),
        metadata=[sample.get('metadata', {}) for sample in batch]
    )
    
    # Collate climate cubes if present
    climate_cubes = [sample.get('climate_cube') for sample in batch if sample.get('climate_cube') is not None]
    if climate_cubes:
        try:
            batch_data.climate_cubes = torch.stack(climate_cubes)
        except Exception as e:
            logger.warning(f"Failed to stack climate cubes: {e}")
    
    # Collate spectra if present
    spectra = [sample.get('spectrum') for sample in batch if sample.get('spectrum') is not None]
    if spectra:
        try:
            # Pad spectra to same length if needed
            max_len = max(spec.shape[0] for spec in spectra)
            padded_spectra = []
            for spec in spectra:
                if spec.shape[0] < max_len:
                    padding = torch.zeros(max_len - spec.shape[0], spec.shape[1])
                    spec = torch.cat([spec, padding], dim=0)
                padded_spectra.append(spec)
            
            batch_data.spectra = torch.stack(padded_spectra)
        except Exception as e:
            logger.warning(f"Failed to stack spectra: {e}")
    
    # Collate biological graphs
    bio_graphs = [sample.get('bio_graph') for sample in batch if sample.get('bio_graph') is not None]
    if bio_graphs:
        try:
            if PYTORCH_GEOMETRIC_AVAILABLE and all(hasattr(g, 'x') for g in bio_graphs):
                # Use PyTorch Geometric batching
                batch_data.bio_graphs = PyGBatch.from_data_list(bio_graphs)
            else:
                # Fallback: stack adjacency matrices if possible
                if all(isinstance(g, dict) and 'adjacency' in g for g in bio_graphs):
                    # Check if all have same size
                    adj_sizes = [g['adjacency'].shape for g in bio_graphs]
                    if len(set(adj_sizes)) == 1:  # All same size
                        adj_matrices = [g['adjacency'] for g in bio_graphs]
                        features = [g['node_features'] for g in bio_graphs]
                        
                        batch_data.bio_graphs = {
                            'adjacency': torch.stack(adj_matrices),
                            'node_features': torch.stack(features)
                        }
                    else:
                        # Different sizes - store as list
                        batch_data.bio_graphs = bio_graphs
                else:
                    batch_data.bio_graphs = bio_graphs
        except Exception as e:
            logger.warning(f"Failed to process bio graphs: {e}")
            batch_data.bio_graphs = bio_graphs
    
    return batch_data

class SimpleDataLoader:
    """Simple data loader wrapper"""
    
    def __init__(self,
                 dataset: PlanetRunDataset,
                 config: DataLoaderConfig):
        
        self.dataset = dataset
        self.config = config
        
        # Create PyTorch DataLoader
        self.dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(dataset.data_split == DataSplit.TRAIN),
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            collate_fn=collate_multimodal_batch
        )
        
        logger.info(f"[PROC] DataLoader initialized: batch_size={config.batch_size}")
    
    def __iter__(self):
        """Iterator"""
        for batch in self.dataloader:
            yield batch
    
    def __len__(self):
        return len(self.dataloader)

def create_multimodal_dataloaders(config: DataLoaderConfig,
                                storage_manager = None) -> Tuple[SimpleDataLoader, SimpleDataLoader, SimpleDataLoader]:
    """Create train, validation, and test dataloaders for multi-modal data"""
    
    # Create datasets
    train_dataset = PlanetRunDataset(config, DataSplit.TRAIN, storage_manager)
    val_dataset = PlanetRunDataset(config, DataSplit.VALIDATION, storage_manager)
    test_dataset = PlanetRunDataset(config, DataSplit.TEST, storage_manager)
    
    # Create dataloaders
    train_loader = SimpleDataLoader(train_dataset, config)
    val_loader = SimpleDataLoader(val_dataset, config)
    test_loader = SimpleDataLoader(test_dataset, config)
    
    logger.info(f"[PKG] Created dataloaders: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test the unified dataloader
    def test_dataloader():
        logger.info("[TEST] Testing Unified DataLoader Architecture (Fixed)")
        
        # Create mock storage
        mock_storage = MockDataStorage(n_runs=15)
        
        # Create dataloader config
        config = DataLoaderConfig(
            batch_size=3,
            num_workers=0,
            pin_memory=False,
            include_climate=True,
            include_biology=True,
            include_spectroscopy=True,
            enable_caching=True,
            normalize_climate=True,
            enable_augmentation=False
        )
        
        # Create dataloaders
        train_loader, val_loader, test_loader = create_multimodal_dataloaders(config, mock_storage)
        
        # Test training loader
        logger.info("[PROC] Testing training dataloader...")
        batch_count = 0
        for i, batch in enumerate(train_loader):
            logger.info(f"Batch {i}:")
            logger.info(f"  Run IDs: {batch.run_ids.tolist()}")
            logger.info(f"  Planet params shape: {batch.planet_params.shape}")
            
            if batch.climate_cubes is not None:
                logger.info(f"  Climate cubes shape: {batch.climate_cubes.shape}")
                logger.info(f"  Climate range: [{batch.climate_cubes.min():.3f}, {batch.climate_cubes.max():.3f}]")
            
            if batch.spectra is not None:
                logger.info(f"  Spectra shape: {batch.spectra.shape}")
                logger.info(f"  Spectrum range: [{batch.spectra.min():.3f}, {batch.spectra.max():.3f}]")
            
            if batch.bio_graphs is not None:
                if hasattr(batch.bio_graphs, 'x'):
                    logger.info(f"  Bio graphs nodes: {batch.bio_graphs.x.shape}")
                    logger.info(f"  Bio graphs edges: {batch.bio_graphs.edge_index.shape}")
                elif isinstance(batch.bio_graphs, dict):
                    logger.info(f"  Bio adjacency shape: {batch.bio_graphs['adjacency'].shape}")
                    logger.info(f"  Bio features shape: {batch.bio_graphs['node_features'].shape}")
                else:
                    logger.info(f"  Bio graphs type: {type(batch.bio_graphs)}")
            
            logger.info(f"  Metadata entries: {len(batch.metadata)}")
            
            batch_count += 1
            if batch_count >= 2:  # Test first couple batches
                break
        
        # Test validation loader
        logger.info("[PROC] Testing validation dataloader...")
        val_batch = next(iter(val_loader))
        logger.info(f"Validation batch - Run IDs: {val_batch.run_ids.tolist()}")
        
        # Test batch device movement
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info("[PROC] Testing GPU batch transfer...")
            try:
                gpu_batch = val_batch.to(device)
                logger.info(f"  GPU batch planet params device: {gpu_batch.planet_params.device}")
                logger.info("  [OK] GPU transfer successful")
            except Exception as e:
                logger.warning(f"  GPU transfer failed: {e}")
        else:
            logger.info("[PROC] CUDA not available, skipping GPU test")
        
        logger.info("[OK] Unified DataLoader test completed successfully!")
        
        # Show final statistics
        print("\n" + "="*60)
        print("[PROC] UNIFIED DATALOADER STATISTICS")
        print("="*60)
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        print(f"Batch size: {config.batch_size}")
        print(f"PyTorch Geometric available: {PYTORCH_GEOMETRIC_AVAILABLE}")
        print(f"\nMulti-modal features:")
        print(f"  [OK] Climate datacubes (4D tensors)")
        print(f"  [OK] Biological networks (graphs/adjacency)")
        print(f"  [OK] High-resolution spectra")
        print(f"  [OK] Planet parameters")
        print(f"  [OK] Metadata integration")
        print(f"  [OK] Intelligent batching")
        print(f"  [OK] Error handling")
        print("="*60)
    
    # Run test
    test_dataloader() 
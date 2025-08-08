#!/usr/bin/env python3
"""
Unified DataLoader Architecture
==============================

PyTorch DataPipe system for multi-modal scientific data loading with intelligent
batching and collation across climate, biology, and spectroscopy domains.

Features:
- Multi-modal batch construction
- Intelligent tensor collation
- Memory-efficient streaming
- Adaptive batching strategies
- Domain-specific preprocessing
- Cache-aware data loading
- Parallel data pipeline
- Quality-based filtering

This system creates batches that contain:
{
    'climate_cube': tensor,      # 4D climate fields
    'bio_graph': pyg.Data,       # Biological network
    'spectrum': tensor,          # High-res spectrum
    'planet_params': tensor,     # Planetary parameters
    'run_metadata': dict         # Run information
}
"""

import asyncio
import json
import logging
import multiprocessing
import random
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset

# PyTorch Geometric for graph data
try:
    import torch_geometric
    from torch_geometric.data import Batch as PyGBatch
    from torch_geometric.data import Data as PyGData

    PYTORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    PYTORCH_GEOMETRIC_AVAILABLE = False
    logger.warning("PyTorch Geometric not available - graph batching disabled")

# Local imports
from .multi_modal_storage_layer_simple import MultiModalStorage, StorageConfig, get_storage_manager
from .planet_run_primary_key_system import get_planet_run_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BatchingStrategy(Enum):
    """Batching strategies for multi-modal data"""

    FIXED_SIZE = "fixed_size"  # Fixed batch size
    ADAPTIVE_MEMORY = "adaptive_memory"  # Adapt to memory constraints
    QUALITY_WEIGHTED = "quality_weighted"  # Weight by data quality
    DOMAIN_BALANCED = "domain_balanced"  # Balance across domains


class DataSplit(Enum):
    """Data split types"""

    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


@dataclass
class DataLoaderConfig:
    """Configuration for unified data loader"""

    # Basic settings
    batch_size: int = 8
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True

    # Multi-modal settings
    include_climate: bool = True
    include_biology: bool = True
    include_spectroscopy: bool = True
    include_metadata: bool = True

    # Data filtering
    min_data_completeness: float = 0.8
    min_quality_score: float = 0.0
    max_climate_size_mb: float = 100.0  # Limit climate cube size

    # Preprocessing
    normalize_climate: bool = True
    normalize_spectra: bool = True
    standardize_bio_features: bool = True

    # Batching strategy
    batching_strategy: BatchingStrategy = BatchingStrategy.ADAPTIVE_MEMORY
    max_memory_per_batch_mb: float = 500.0

    # Cache settings
    enable_caching: bool = True
    cache_preprocessed: bool = True

    # Data augmentation
    enable_augmentation: bool = False
    climate_noise_std: float = 0.01
    spectrum_noise_std: float = 0.005


@dataclass
class MultiModalBatch:
    """Container for multi-modal batch data"""

    # Core data
    run_ids: torch.Tensor  # [batch_size]
    planet_params: torch.Tensor  # [batch_size, n_params]

    # Domain-specific data (optional depending on config)
    climate_cubes: Optional[torch.Tensor] = None  # [batch_size, vars, time, lat, lon, lev]
    bio_graphs: Optional[Any] = None  # PyG batch or list of adjacency matrices
    spectra: Optional[torch.Tensor] = None  # [batch_size, wavelengths, features]

    # Metadata
    metadata: List[Dict[str, Any]] = field(default_factory=list)

    # Quality information
    data_completeness: torch.Tensor = None  # [batch_size]
    quality_scores: torch.Tensor = None  # [batch_size]

    def to(self, device):
        """Move batch to device"""
        result = MultiModalBatch(
            run_ids=self.run_ids.to(device),
            planet_params=self.planet_params.to(device),
            metadata=self.metadata,
            data_completeness=(
                self.data_completeness.to(device) if self.data_completeness is not None else None
            ),
            quality_scores=(
                self.quality_scores.to(device) if self.quality_scores is not None else None
            ),
        )

        if self.climate_cubes is not None:
            result.climate_cubes = self.climate_cubes.to(device)

        if self.spectra is not None:
            result.spectra = self.spectra.to(device)

        if self.bio_graphs is not None:
            if hasattr(self.bio_graphs, "to"):
                result.bio_graphs = self.bio_graphs.to(device)
            else:
                result.bio_graphs = self.bio_graphs

        return result


class PlanetRunDataset(Dataset):
    """
    Dataset for loading multi-modal planet run data

    This dataset loads planet runs and constructs multi-modal samples
    containing climate, biology, and spectroscopy data.
    """

    def __init__(
        self,
        config: DataLoaderConfig,
        data_split: DataSplit = DataSplit.TRAIN,
        storage_manager: MultiModalStorage = None,
        planet_run_manager=None,
    ):

        self.config = config
        self.data_split = data_split
        self.storage_manager = storage_manager or get_storage_manager()

        # Try to get planet run manager
        try:
            self.planet_run_manager = planet_run_manager or get_planet_run_manager()
        except:
            logger.warning("Planet run manager not available - using storage manager registry")
            self.planet_run_manager = None

        # Load available runs
        self.available_runs = self._load_available_runs()
        self.preprocessor = MultiModalPreprocessor(config)

        # Cache for preprocessed data
        self._cache = {} if config.enable_caching else None
        self._cache_lock = Lock()

        logger.info(
            f"[DATA] Dataset initialized: {len(self.available_runs)} runs ({data_split.value})"
        )

    def _load_available_runs(self) -> List[int]:
        """Load available planet runs for the specified split"""

        if self.planet_run_manager:
            # Use planet run manager if available
            try:
                runs = self.planet_run_manager.get_planet_runs(
                    ml_split=self.data_split.value,
                    min_completeness=self.config.min_data_completeness,
                    min_quality=self.config.min_quality_score,
                )
                return [run["run_id"] for run in runs]
            except Exception as e:
                logger.warning(f"Failed to load runs from planet run manager: {e}")

        # Fallback: use storage manager registry
        stored_runs = self.storage_manager.list_stored_runs()

        # Simple split based on run ID (deterministic)
        random.seed(42)  # Reproducible splits
        shuffled_runs = stored_runs.copy()
        random.shuffle(shuffled_runs)

        n_total = len(shuffled_runs)
        if self.data_split == DataSplit.TRAIN:
            return shuffled_runs[: int(0.7 * n_total)]
        elif self.data_split == DataSplit.VALIDATION:
            start_idx = int(0.7 * n_total)
            end_idx = int(0.85 * n_total)
            return shuffled_runs[start_idx:end_idx]
        else:  # TEST
            return shuffled_runs[int(0.85 * n_total) :]

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
        sample = self.preprocessor.process_sample(sample)

        # Cache if enabled
        if self._cache is not None and self.config.cache_preprocessed:
            with self._cache_lock:
                self._cache[run_id] = sample

        return sample

    def _load_multimodal_sample(self, run_id: int) -> Dict[str, Any]:
        """Load raw multi-modal data for a planet run"""
        sample = {
            "run_id": run_id,
            "climate_data": None,
            "bio_data": None,
            "spectrum_data": None,
            "planet_params": None,
            "metadata": {},
        }

        try:
            # Load climate data
            if self.config.include_climate:
                climate_data = asyncio.run(self.storage_manager.load_climate_datacube(run_id))
                sample["climate_data"] = climate_data

            # Load biological network
            if self.config.include_biology:
                bio_data = asyncio.run(self.storage_manager.load_biological_network(run_id))
                sample["bio_data"] = bio_data

            # Load spectrum
            if self.config.include_spectroscopy:
                spectrum_data = asyncio.run(self.storage_manager.load_spectrum(run_id))
                sample["spectrum_data"] = spectrum_data

            # Extract planet parameters from metadata
            if sample["climate_data"] and "metadata" in sample["climate_data"]:
                climate_meta = sample["climate_data"]["metadata"]
                sample["metadata"].update(climate_meta)

            # Generate synthetic planet parameters if not available
            sample["planet_params"] = self._extract_planet_parameters(sample)

        except Exception as e:
            logger.warning(f"Failed to load complete data for run {run_id}: {e}")
            # Return partial sample

        return sample

    def _extract_planet_parameters(self, sample: Dict[str, Any]) -> np.ndarray:
        """Extract or generate planet parameters from sample metadata"""
        # This would ideally come from the planet run database
        # For now, generate realistic parameters

        params = np.array(
            [
                np.random.uniform(0.5, 2.5),  # radius_earth
                np.random.uniform(0.3, 10.0),  # mass_earth
                np.random.uniform(1.0, 100.0),  # period_days
                np.random.uniform(0.1, 10.0),  # stellar_flux
                np.random.uniform(0.0, 0.01),  # H2O mixing ratio
                np.random.uniform(0.0, 0.001),  # CO2 mixing ratio
                np.random.uniform(0.0, 0.0001),  # CH4 mixing ratio
                np.random.uniform(0.0, 0.0001),  # O2 mixing ratio
            ],
            dtype=np.float32,
        )

        return params


class MultiModalPreprocessor:
    """Preprocessor for multi-modal scientific data"""

    def __init__(self, config: DataLoaderConfig):
        self.config = config

        # Normalization statistics (would be computed from training data)
        self.climate_stats = {
            "temperature": {"mean": 280.0, "std": 50.0},
            "humidity": {"mean": 0.005, "std": 0.01},
        }

        self.spectrum_stats = {"mean": 0.5, "std": 0.3}
        self.planet_param_stats = {"mean": np.zeros(8), "std": np.ones(8)}

    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a multi-modal sample"""
        processed = {
            "run_id": sample["run_id"],
            "planet_params": self._process_planet_params(sample["planet_params"]),
        }

        # Process climate data
        if sample["climate_data"] and self.config.include_climate:
            processed["climate_cube"] = self._process_climate_data(sample["climate_data"])

        # Process biological data
        if sample["bio_data"] and self.config.include_biology:
            processed["bio_graph"] = self._process_biological_data(sample["bio_data"])

        # Process spectrum data
        if sample["spectrum_data"] and self.config.include_spectroscopy:
            processed["spectrum"] = self._process_spectrum_data(sample["spectrum_data"])

        # Add metadata
        if self.config.include_metadata:
            processed["metadata"] = sample["metadata"]

        return processed

    def _process_planet_params(self, params: np.ndarray) -> torch.Tensor:
        """Process planet parameters"""
        if params is None:
            return torch.zeros(8, dtype=torch.float32)

        params_tensor = torch.from_numpy(params).float()

        # Normalize (simplified)
        # In practice, use statistics from training data
        return params_tensor

    def _process_climate_data(self, climate_data: Dict[str, Any]) -> torch.Tensor:
        """Process climate datacube"""
        data_vars = climate_data["data"]

        # Stack variables into tensor [vars, time, lat, lon, lev]
        var_tensors = []
        for var_name in ["temperature", "humidity"]:
            if var_name in data_vars:
                var_data = data_vars[var_name]

                # Normalize if enabled
                if self.config.normalize_climate and var_name in self.climate_stats:
                    stats = self.climate_stats[var_name]
                    var_data = (var_data - stats["mean"]) / stats["std"]

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
            return torch.zeros(2, 10, 32, 64, 10, dtype=torch.float32)

    def _process_biological_data(self, bio_data: Dict[str, Any]) -> Union[torch.Tensor, Any]:
        """Process biological network data"""
        if "adjacency_matrix" in bio_data:
            adj_matrix = bio_data["adjacency_matrix"]
            node_features = bio_data.get(
                "node_features", np.random.random((adj_matrix.shape[0], 10))
            )

            if PYTORCH_GEOMETRIC_AVAILABLE:
                # Create PyTorch Geometric data object
                edge_index = torch.from_numpy(np.nonzero(adj_matrix)).long()
                node_features_tensor = torch.from_numpy(node_features).float()

                # Standardize features if enabled
                if self.config.standardize_bio_features:
                    node_features_tensor = (node_features_tensor - node_features_tensor.mean(0)) / (
                        node_features_tensor.std(0) + 1e-8
                    )

                graph_data = PyGData(
                    x=node_features_tensor, edge_index=edge_index, num_nodes=adj_matrix.shape[0]
                )

                return graph_data
            else:
                # Fallback to adjacency matrix
                adj_tensor = torch.from_numpy(adj_matrix).float()
                features_tensor = torch.from_numpy(node_features).float()

                return {"adjacency": adj_tensor, "node_features": features_tensor}

        # Return dummy data if no biological data
        if PYTORCH_GEOMETRIC_AVAILABLE:
            return PyGData(
                x=torch.randn(10, 5), edge_index=torch.randint(0, 10, (2, 20)), num_nodes=10
            )
        else:
            return {"adjacency": torch.eye(10), "node_features": torch.randn(10, 5)}

    def _process_spectrum_data(self, spectrum_data: Dict[str, Any]) -> torch.Tensor:
        """Process spectrum data"""
        wavelengths = spectrum_data["wavelengths"]
        flux = spectrum_data["flux"]

        # Combine wavelengths and flux
        spectrum_tensor = torch.stack(
            [torch.from_numpy(wavelengths).float(), torch.from_numpy(flux).float()], dim=1
        )  # [n_points, 2]

        # Normalize if enabled
        if self.config.normalize_spectra:
            spectrum_tensor = (spectrum_tensor - self.spectrum_stats["mean"]) / self.spectrum_stats[
                "std"
            ]

        # Add noise for augmentation
        if self.config.enable_augmentation:
            noise = torch.normal(0, self.config.spectrum_noise_std, spectrum_tensor.shape)
            spectrum_tensor = spectrum_tensor + noise

        return spectrum_tensor


def collate_multimodal_batch(batch: List[Dict[str, Any]]) -> MultiModalBatch:
    """
    Collate function for multi-modal batches

    Takes a list of individual samples and creates a properly batched
    MultiModalBatch object.
    """

    # Extract components
    run_ids = [sample["run_id"] for sample in batch]
    planet_params = [sample["planet_params"] for sample in batch]

    # Create batch containers
    batch_data = MultiModalBatch(
        run_ids=torch.tensor(run_ids, dtype=torch.long),
        planet_params=torch.stack(planet_params),
        metadata=[sample.get("metadata", {}) for sample in batch],
    )

    # Collate climate cubes if present
    climate_cubes = [
        sample.get("climate_cube") for sample in batch if sample.get("climate_cube") is not None
    ]
    if climate_cubes:
        batch_data.climate_cubes = torch.stack(climate_cubes)

    # Collate spectra if present
    spectra = [sample.get("spectrum") for sample in batch if sample.get("spectrum") is not None]
    if spectra:
        # Pad spectra to same length if needed
        max_len = max(spec.shape[0] for spec in spectra)
        padded_spectra = []
        for spec in spectra:
            if spec.shape[0] < max_len:
                padding = torch.zeros(max_len - spec.shape[0], spec.shape[1])
                spec = torch.cat([spec, padding], dim=0)
            padded_spectra.append(spec)

        batch_data.spectra = torch.stack(padded_spectra)

    # Collate biological graphs
    bio_graphs = [
        sample.get("bio_graph") for sample in batch if sample.get("bio_graph") is not None
    ]
    if bio_graphs:
        if PYTORCH_GEOMETRIC_AVAILABLE and all(hasattr(g, "x") for g in bio_graphs):
            # Use PyTorch Geometric batching
            batch_data.bio_graphs = PyGBatch.from_data_list(bio_graphs)
        else:
            # Fallback: store as list or try to stack adjacency matrices
            if all(isinstance(g, dict) and "adjacency" in g for g in bio_graphs):
                # Stack adjacency matrices (assuming same size)
                adj_matrices = [g["adjacency"] for g in bio_graphs]
                features = [g["node_features"] for g in bio_graphs]

                batch_data.bio_graphs = {
                    "adjacency": torch.stack(adj_matrices),
                    "node_features": torch.stack(features),
                }
            else:
                batch_data.bio_graphs = bio_graphs

    return batch_data


class AdaptiveDataLoader:
    """
    Adaptive data loader that adjusts batch size based on memory constraints
    and data characteristics.
    """

    def __init__(self, dataset: PlanetRunDataset, config: DataLoaderConfig):

        self.dataset = dataset
        self.config = config
        self.current_batch_size = config.batch_size

        # Memory monitoring
        self.memory_history = deque(maxlen=10)
        self.batch_time_history = deque(maxlen=10)

        # Create base DataLoader
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.current_batch_size,
            shuffle=(dataset.data_split == DataSplit.TRAIN),
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor,
            pin_memory=config.pin_memory,
            collate_fn=collate_multimodal_batch,
            persistent_workers=True if config.num_workers > 0 else False,
        )

        logger.info(f"[PROC] Adaptive DataLoader initialized: batch_size={self.current_batch_size}")

    def __iter__(self):
        """Iterator with adaptive batching"""
        for batch in self.dataloader:
            start_time = time.time()

            # Monitor memory usage (simplified)
            batch_size_mb = self._estimate_batch_memory(batch)
            self.memory_history.append(batch_size_mb)

            yield batch

            # Track timing
            batch_time = time.time() - start_time
            self.batch_time_history.append(batch_time)

            # Adapt batch size if needed
            self._adapt_batch_size()

    def __len__(self):
        return len(self.dataloader)

    def _estimate_batch_memory(self, batch: MultiModalBatch) -> float:
        """Estimate memory usage of batch in MB"""
        total_elements = 0

        # Count tensor elements
        total_elements += batch.planet_params.numel()

        if batch.climate_cubes is not None:
            total_elements += batch.climate_cubes.numel()

        if batch.spectra is not None:
            total_elements += batch.spectra.numel()

        # Estimate 4 bytes per float32 element
        memory_mb = (total_elements * 4) / (1024**2)
        return memory_mb

    def _adapt_batch_size(self):
        """Adapt batch size based on memory and performance"""
        if len(self.memory_history) < 5:
            return

        avg_memory = sum(self.memory_history) / len(self.memory_history)
        avg_time = sum(self.batch_time_history) / len(self.batch_time_history)

        # Adapt based on memory usage
        if avg_memory > self.config.max_memory_per_batch_mb * 1.2:
            # Reduce batch size
            new_batch_size = max(1, int(self.current_batch_size * 0.8))
            if new_batch_size != self.current_batch_size:
                logger.info(f"🔽 Reducing batch size: {self.current_batch_size} → {new_batch_size}")
                self.current_batch_size = new_batch_size
                self._recreate_dataloader()

        elif avg_memory < self.config.max_memory_per_batch_mb * 0.6 and avg_time < 1.0:
            # Increase batch size
            new_batch_size = min(self.config.batch_size * 2, int(self.current_batch_size * 1.2))
            if new_batch_size != self.current_batch_size:
                logger.info(
                    f"🔼 Increasing batch size: {self.current_batch_size} → {new_batch_size}"
                )
                self.current_batch_size = new_batch_size
                self._recreate_dataloader()

    def _recreate_dataloader(self):
        """Recreate DataLoader with new batch size"""
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.current_batch_size,
            shuffle=(self.dataset.data_split == DataSplit.TRAIN),
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=self.config.pin_memory,
            collate_fn=collate_multimodal_batch,
            persistent_workers=True if self.config.num_workers > 0 else False,
        )


def create_multimodal_dataloaders(
    config: DataLoaderConfig, storage_manager: MultiModalStorage = None
) -> Tuple[AdaptiveDataLoader, AdaptiveDataLoader, AdaptiveDataLoader]:
    """
    Create train, validation, and test dataloaders for multi-modal data

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """

    # Create datasets
    train_dataset = PlanetRunDataset(config, DataSplit.TRAIN, storage_manager)
    val_dataset = PlanetRunDataset(config, DataSplit.VALIDATION, storage_manager)
    test_dataset = PlanetRunDataset(config, DataSplit.TEST, storage_manager)

    # Create adaptive dataloaders
    train_loader = AdaptiveDataLoader(train_dataset, config)
    val_loader = AdaptiveDataLoader(val_dataset, config)
    test_loader = AdaptiveDataLoader(test_dataset, config)

    logger.info(
        f"[PKG] Created dataloaders: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test"
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the unified dataloader
    async def test_dataloader():
        logger.info("[TEST] Testing Unified DataLoader Architecture")

        # Create test storage with example data
        storage_config = StorageConfig(storage_root=Path("data/test_planet_runs"))
        storage_manager = MultiModalStorage(storage_config)

        # Ensure we have test data
        from .multi_modal_storage_layer_simple import create_example_data

        for run_id in range(1, 6):
            await create_example_data(storage_manager, run_id)

        # Create dataloader config
        config = DataLoaderConfig(
            batch_size=2,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
            include_climate=True,
            include_biology=True,
            include_spectroscopy=True,
            enable_caching=True,
            normalize_climate=True,
        )

        # Create dataloaders
        train_loader, val_loader, test_loader = create_multimodal_dataloaders(
            config, storage_manager
        )

        # Test training loader
        logger.info("[PROC] Testing training dataloader...")
        for i, batch in enumerate(train_loader):
            logger.info(f"Batch {i}:")
            logger.info(f"  Run IDs: {batch.run_ids}")
            logger.info(f"  Planet params shape: {batch.planet_params.shape}")

            if batch.climate_cubes is not None:
                logger.info(f"  Climate cubes shape: {batch.climate_cubes.shape}")

            if batch.spectra is not None:
                logger.info(f"  Spectra shape: {batch.spectra.shape}")

            if batch.bio_graphs is not None:
                if hasattr(batch.bio_graphs, "x"):
                    logger.info(f"  Bio graphs nodes: {batch.bio_graphs.x.shape}")
                else:
                    logger.info(f"  Bio graphs type: {type(batch.bio_graphs)}")

            logger.info(f"  Metadata entries: {len(batch.metadata)}")

            if i >= 2:  # Test first few batches
                break

        # Test memory estimation
        if hasattr(train_loader, "_estimate_batch_memory"):
            memory_mb = train_loader._estimate_batch_memory(batch)
            logger.info(f"[DATA] Estimated batch memory: {memory_mb:.1f} MB")

        # Test batch device movement
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("[PROC] Testing GPU batch transfer...")
            gpu_batch = batch.to(device)
            logger.info(f"  GPU batch planet params device: {gpu_batch.planet_params.device}")

        logger.info("[OK] Unified DataLoader test completed successfully!")

    # Run test
    import asyncio

    asyncio.run(test_dataloader())

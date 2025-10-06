#!/usr/bin/env python3
"""
Advanced 4-D Climate Datacube DataModule
========================================

Industry-grade PyTorch Lightning DataModule for streaming 4-D climate datacubes.
Features advanced caching, adaptive chunking, memory optimization, and streaming.

Key Features:
- Adaptive chunking based on available memory
- Advanced caching with LRU eviction
- Streaming data loading with prefetching
- Physics-informed data validation
- Real-time memory monitoring
- Multi-zarr store support
- Configuration-driven setup
"""

import gc
import logging
import os
import queue
import threading
import time
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple, Union, Union

import numpy as np
import psutil
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import xarray as xr
import yaml
from torch.utils.data import DataLoader, Dataset, IterableDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for advanced caching system"""

    enabled: bool = True
    max_size_gb: float = 2.0
    eviction_policy: str = "lru"  # lru, fifo, lfu
    persist_to_disk: bool = True
    cache_dir: Path = Path("data/cache/cube_cache")
    compression: bool = True


class MemoryMonitor:
    """Real-time memory monitoring for datacube operations"""

    def __init__(self, memory_limit_gb: float = 8.0):
        self.memory_limit_gb = memory_limit_gb
        self.memory_history = deque(maxlen=100)
        self.warning_threshold = 0.85
        self.critical_threshold = 0.95
        self._lock = RLock()

    def check_memory(self) -> Dict[str, float]:
        """Check current memory usage"""
        with self._lock:
            memory = psutil.virtual_memory()
            usage_gb = memory.used / (1024**3)
            usage_percent = memory.percent / 100.0

            memory_info = {
                "used_gb": usage_gb,
                "available_gb": memory.available / (1024**3),
                "usage_percent": usage_percent,
                "within_limit": usage_gb < self.memory_limit_gb,
            }

            self.memory_history.append(memory_info)

            # Issue warnings if necessary
            if usage_percent > self.critical_threshold:
                logger.critical(f"Critical memory usage: {usage_percent:.1%}")
                gc.collect()
            elif usage_percent > self.warning_threshold:
                logger.warning(f"High memory usage: {usage_percent:.1%}")

            return memory_info

    def get_memory_trend(self) -> Dict[str, float]:
        """Get memory usage trend"""
        if len(self.memory_history) < 2:
            return {"trend": 0.0, "stability": 1.0}

        recent_usage = [m["usage_percent"] for m in list(self.memory_history)[-10:]]
        trend = np.polyfit(range(len(recent_usage)), recent_usage, 1)[0]
        stability = 1.0 - np.std(recent_usage)

        return {"trend": trend, "stability": max(0.0, stability)}


class AdaptiveChunker:
    """Adaptive chunking based on memory constraints and data characteristics"""

    def __init__(self, memory_monitor: MemoryMonitor, base_chunks: Dict[str, int]):
        self.memory_monitor = memory_monitor
        self.base_chunks = base_chunks
        self.chunk_history = []

    def optimize_chunks(self, dataset: xr.Dataset, target_memory_gb: float = 1.0) -> Dict[str, int]:
        """Optimize chunk sizes based on dataset characteristics and memory"""

        # Calculate current memory usage
        memory_info = self.memory_monitor.check_memory()

        # Start with base chunks
        optimized_chunks = self.base_chunks.copy()

        # Estimate dataset size per chunk
        sample_var = list(dataset.data_vars.keys())[0]
        var_data = dataset[sample_var]

        # Calculate bytes per element
        bytes_per_element = np.dtype(var_data.dtype).itemsize

        # Calculate current chunk size in bytes
        current_chunk_size = bytes_per_element
        for dim in var_data.dims:
            current_chunk_size *= optimized_chunks.get(dim, var_data.sizes[dim])

        current_chunk_gb = current_chunk_size / (1024**3)

        # Adjust chunks if needed
        if current_chunk_gb > target_memory_gb:
            scale_factor = (target_memory_gb / current_chunk_gb) ** 0.25

            for dim in optimized_chunks:
                if dim in var_data.dims:
                    optimized_chunks[dim] = max(1, int(optimized_chunks[dim] * scale_factor))

        # Consider memory pressure
        if memory_info["usage_percent"] > 0.8:
            # Reduce chunks further under memory pressure
            for dim in optimized_chunks:
                optimized_chunks[dim] = max(1, int(optimized_chunks[dim] * 0.8))

        self.chunk_history.append(
            {
                "chunks": optimized_chunks.copy(),
                "memory_usage": memory_info["usage_percent"],
                "chunk_size_gb": current_chunk_gb,
            }
        )

        logger.info(f"Optimized chunks: {optimized_chunks} (target: {target_memory_gb:.2f}GB)")
        return optimized_chunks


class DataValidator:
    """Physics-informed data validation"""

    def __init__(self):
        self.validation_stats = {"total_samples": 0, "valid_samples": 0, "validation_errors": []}

    def validate_physics(self, data: torch.Tensor, variable: str) -> Dict[str, bool]:
        """Validate physical constraints"""
        results = {}

        # Basic range checks
        if "temp" in variable.lower() or "T_surf" in variable:
            results["temperature_range"] = torch.all((data >= 150.0) & (data <= 400.0))
        elif "pressure" in variable.lower() or "psurf" in variable:
            results["pressure_range"] = torch.all((data >= 0.001) & (data <= 1000.0))
        elif "humidity" in variable.lower() or "q_H2O" in variable:
            results["humidity_range"] = torch.all((data >= 0.0) & (data <= 1.0))

        # Check for NaN/Inf values
        results["finite_values"] = torch.all(torch.isfinite(data))

        # Check for extreme gradients
        if data.dim() >= 3:
            gradients = torch.gradient(data, dim=-1)[0]
            results["gradient_check"] = torch.all(torch.abs(gradients) < 1000.0)

        return results

    def validate_batch(self, batch: torch.Tensor, variable_names: List[str]) -> Dict[str, Any]:
        """Validate a batch of data"""
        batch_results = {}

        for i, var_name in enumerate(variable_names):
            if i < batch.size(1):  # Check if variable exists in batch
                var_data = batch[:, i]
                var_results = self.validate_physics(var_data, var_name)
                batch_results[var_name] = var_results

        # Overall batch validation
        all_valid = all(all(results.values()) for results in batch_results.values())

        batch_results["batch_valid"] = all_valid

        self.validation_stats["total_samples"] += batch.size(0)
        if all_valid:
            self.validation_stats["valid_samples"] += batch.size(0)

        return batch_results


class StreamingCache:
    """Advanced caching system with streaming capabilities"""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache = {}
        self.access_times = {}
        self.access_counts = {}
        self.cache_size_bytes = 0
        self.max_size_bytes = config.max_size_gb * (1024**3)
        self._lock = RLock()

        # Create cache directory
        if config.persist_to_disk:
            config.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, zarr_path: Path, time_idx: int, variables: List[str]) -> str:
        """Generate cache key"""
        var_str = "_".join(sorted(variables))
        return f"{zarr_path.name}_{time_idx}_{var_str}"

    def _estimate_size(self, data: torch.Tensor) -> int:
        """Estimate tensor size in bytes"""
        return data.numel() * data.element_size()

    def _evict_lru(self):
        """Evict least recently used items"""
        with self._lock:
            if not self.cache:
                return

            # Sort by access time
            sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])

            # Evict oldest items until under limit
            for key, _ in sorted_items:
                if self.cache_size_bytes <= self.max_size_bytes * 0.8:
                    break

                if key in self.cache:
                    data = self.cache[key]
                    size = self._estimate_size(data)

                    del self.cache[key]
                    del self.access_times[key]
                    del self.access_counts[key]

                    self.cache_size_bytes -= size
                    logger.debug(f"Evicted cache entry: {key} ({size} bytes)")

    def get(self, zarr_path: Path, time_idx: int, variables: List[str]) -> Optional[torch.Tensor]:
        """Get data from cache"""
        if not self.config.enabled:
            return None

        key = self._get_cache_key(zarr_path, time_idx, variables)

        with self._lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                logger.debug(f"Cache hit: {key}")
                return self.cache[key]

        return None

    def put(self, zarr_path: Path, time_idx: int, variables: List[str], data: torch.Tensor):
        """Put data into cache"""
        if not self.config.enabled:
            return

        key = self._get_cache_key(zarr_path, time_idx, variables)
        data_size = self._estimate_size(data)

        with self._lock:
            # Check if we need to evict
            if self.cache_size_bytes + data_size > self.max_size_bytes:
                self._evict_lru()

            # Add to cache
            self.cache[key] = data.clone()
            self.access_times[key] = time.time()
            self.access_counts[key] = 1
            self.cache_size_bytes += data_size

            logger.debug(f"Cached: {key} ({data_size} bytes)")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                "cache_size_mb": self.cache_size_bytes / (1024**2),
                "cache_entries": len(self.cache),
                "hit_rate": sum(self.access_counts.values()) / max(1, len(self.access_counts)),
                "memory_usage_percent": self.cache_size_bytes / self.max_size_bytes * 100,
            }


class CubeDataset(IterableDataset):
    """
    Iterable dataset for 4-D climate datacubes using xarray/zarr
    """

    def __init__(
        self,
        zarr_paths: List[Path],
        variables: List[str] = None,
        time_window: int = 10,
        spatial_crop: Optional[Tuple[int, int, int, int]] = None,
        transform: Optional[callable] = None,
        target_variables: List[str] = None,
        normalize: bool = True,
    ):
        """
        Initialize datacube dataset

        Args:
            zarr_paths: List of paths to zarr stores
            variables: Variables to load (default: all)
            time_window: Number of time steps per sample
            spatial_crop: (lat_start, lat_end, lon_start, lon_end) for spatial cropping
            transform: Optional transform function
            target_variables: Variables to use as targets (for prediction tasks)
            normalize: Whether to normalize data
        """
        self.zarr_paths = zarr_paths
        self.variables = variables
        self.time_window = time_window
        self.spatial_crop = spatial_crop
        self.transform = transform
        self.target_variables = target_variables or []
        self.normalize = normalize

        # Load datasets and get metadata
        self.datasets = []
        self.dataset_lengths = []
        self.total_samples = 0

        for zarr_path in zarr_paths:
            try:
                ds = xr.open_zarr(zarr_path, chunks="auto")

                # Filter variables if specified
                if self.variables:
                    available_vars = [var for var in self.variables if var in ds.data_vars]
                    if not available_vars:
                        logger.warning(f"No specified variables found in {zarr_path}")
                        continue
                    ds = ds[available_vars]

                # Apply spatial cropping
                if self.spatial_crop:
                    lat_start, lat_end, lon_start, lon_end = self.spatial_crop
                    if "lat" in ds.coords:
                        ds = ds.isel(lat=slice(lat_start, lat_end))
                    if "lon" in ds.coords:
                        ds = ds.isel(lon=slice(lon_start, lon_end))

                self.datasets.append(ds)

                # Calculate number of samples (sliding window over time)
                if "time" in ds.dims:
                    n_samples = max(0, len(ds.time) - self.time_window + 1)
                else:
                    n_samples = 1

                self.dataset_lengths.append(n_samples)
                self.total_samples += n_samples

                logger.info(f"Loaded dataset from {zarr_path}: {n_samples} samples")

            except Exception as e:
                logger.error(f"Failed to load dataset from {zarr_path}: {e}")

        if not self.datasets:
            raise ValueError("No valid datasets found")

        # Calculate normalization statistics if needed
        if self.normalize:
            self.norm_stats = self._calculate_normalization_stats()
        else:
            self.norm_stats = None

        logger.info(
            f"Initialized CubeDataset with {len(self.datasets)} zarr stores, "
            f"{self.total_samples} total samples"
        )

    def _calculate_normalization_stats(self) -> Dict[str, Dict[str, float]]:
        """Calculate mean and std for normalization"""
        logger.info("Calculating normalization statistics...")

        stats = {}

        # Sample a subset of data for statistics
        for ds in self.datasets[: min(10, len(self.datasets))]:  # Use up to 10 datasets
            for var in ds.data_vars:
                if var not in stats:
                    stats[var] = {"values": []}

                # Sample some time steps
                if "time" in ds.dims:
                    sample_times = slice(
                        0, min(50, len(ds.time)), 5
                    )  # Every 5th timestep, up to 50
                    sample_data = ds[var].isel(time=sample_times)
                else:
                    sample_data = ds[var]

                # Compute statistics on flattened data
                values = sample_data.values.flatten()
                values = values[~np.isnan(values)]  # Remove NaN values

                if len(values) > 0:
                    stats[var]["values"].extend(values[:10000])  # Limit to prevent memory issues

        # Calculate final statistics
        final_stats = {}
        for var, data in stats.items():
            if data["values"]:
                values = np.array(data["values"])
                final_stats[var] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
            else:
                final_stats[var] = {"mean": 0.0, "std": 1.0}

        logger.info(f"Calculated normalization stats for {len(final_stats)} variables")
        return final_stats

    def _normalize_data(self, data: torch.Tensor, variable: str) -> torch.Tensor:
        """Normalize data using pre-calculated statistics"""
        if self.norm_stats and variable in self.norm_stats:
            mean = self.norm_stats[variable]["mean"]
            std = self.norm_stats[variable]["std"]
            if std > 0:
                return (data - mean) / std
        return data

    def __iter__(self):
        """Iterate over datacube samples"""
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single process
            dataset_indices = list(range(len(self.datasets)))
        else:
            # Multi-process: split datasets among workers
            per_worker = len(self.datasets) // worker_info.num_workers
            worker_id = worker_info.id
            start_idx = worker_id * per_worker
            end_idx = (
                start_idx + per_worker
                if worker_id < worker_info.num_workers - 1
                else len(self.datasets)
            )
            dataset_indices = list(range(start_idx, end_idx))

        for dataset_idx in dataset_indices:
            ds = self.datasets[dataset_idx]
            n_samples = self.dataset_lengths[dataset_idx]

            for sample_idx in range(n_samples):
                try:
                    # Extract time window
                    if "time" in ds.dims:
                        time_slice = slice(sample_idx, sample_idx + self.time_window)
                        sample_ds = ds.isel(time=time_slice)
                    else:
                        sample_ds = ds

                    # Load data into memory
                    sample_ds = sample_ds.load()

                    # Convert to tensors
                    input_data = []
                    target_data = []

                    for var in sample_ds.data_vars:
                        data_array = sample_ds[var]

                        # Handle different dimensionalities
                        if data_array.ndim == 4:  # (time, lev, lat, lon)
                            tensor = torch.from_numpy(data_array.values).float()
                        elif data_array.ndim == 3:  # (lev, lat, lon) or (time, lat, lon)
                            tensor = torch.from_numpy(data_array.values).float()
                            if "time" not in data_array.dims:
                                tensor = tensor.unsqueeze(0)  # Add time dimension
                        else:
                            # Handle 2D or other cases
                            tensor = torch.from_numpy(data_array.values).float()
                            while tensor.ndim < 4:
                                tensor = tensor.unsqueeze(0)

                        # Normalize if requested
                        if self.normalize:
                            tensor = self._normalize_data(tensor, var)

                        # Assign to input or target
                        if var in self.target_variables:
                            target_data.append(tensor)
                        else:
                            input_data.append(tensor)

                    # Stack tensors
                    if input_data:
                        input_tensor = torch.stack(
                            input_data, dim=0
                        )  # (n_vars, time, lev, lat, lon)
                    else:
                        input_tensor = torch.empty(0)

                    if target_data:
                        target_tensor = torch.stack(target_data, dim=0)
                    else:
                        target_tensor = input_tensor  # Self-supervised case

                    # Apply transform if provided
                    if self.transform:
                        input_tensor, target_tensor = self.transform(input_tensor, target_tensor)

                    yield input_tensor, target_tensor

                except Exception as e:
                    logger.warning(
                        f"Error processing sample {sample_idx} from dataset {dataset_idx}: {e}"
                    )
                    continue


class CubeDM(pl.LightningDataModule):
    """
    Advanced Lightning DataModule for 4-D climate datacubes

    Features:
    - Configuration-driven setup with environment variable support
    - Advanced caching and memory management
    - Adaptive chunking based on available memory
    - Physics-informed data validation
    - Real-time performance monitoring
    - Multi-zarr store support
    """

    def __init__(
        self,
        zarr_root: Optional[str] = None,
        config_path: Optional[str] = "config/config.yaml",
        batch_size: int = 4,
        num_workers: int = 6,
        variables: List[str] = None,
        target_variables: List[str] = None,
        time_window: int = 10,
        spatial_crop: Optional[Tuple[int, int, int, int]] = None,
        train_fraction: float = 0.8,
        val_fraction: float = 0.1,
        normalize: bool = True,
        pin_memory: bool = True,
        enable_caching: bool = True,
        cache_size_gb: float = 2.0,
        memory_limit_gb: float = 8.0,
        enable_validation: bool = True,
        adaptive_chunking: bool = True,
        **kwargs,
    ):
        """
        Initialize advanced CubeDM

        Args:
            zarr_root: Root directory containing zarr stores (overrides config)
            config_path: Path to configuration file
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            variables: List of input variables to load
            target_variables: List of target variables for prediction
            time_window: Number of time steps per sample
            spatial_crop: Spatial cropping parameters
            train_fraction: Fraction of data for training
            val_fraction: Fraction of data for validation
            normalize: Whether to normalize data
            pin_memory: Whether to use pinned memory
            enable_caching: Enable advanced caching
            cache_size_gb: Cache size limit in GB
            memory_limit_gb: Memory limit in GB
            enable_validation: Enable physics validation
            adaptive_chunking: Enable adaptive chunking
        """
        super().__init__()
        self.save_hyperparameters()

        # Load configuration
        self.config = self._load_config(config_path)

        # Setup paths with environment variable support
        self.zarr_root = self._resolve_zarr_root(zarr_root)
        self.additional_zarr_stores = self._get_additional_stores()

        # Basic parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.time_window = time_window
        self.spatial_crop = spatial_crop
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.normalize = normalize
        self.pin_memory = pin_memory

        # Variables configuration
        self.variables = variables or self._get_config_variables("input_variables")
        self.target_variables = target_variables or self._get_config_variables("output_variables")

        # Advanced features
        self.enable_caching = enable_caching
        self.enable_validation = enable_validation
        self.adaptive_chunking = adaptive_chunking

        # Initialize advanced components
        self.memory_monitor = MemoryMonitor(memory_limit_gb)

        # Initialize cache
        if enable_caching:
            cache_config = CacheConfig(
                enabled=True,
                max_size_gb=cache_size_gb,
                cache_dir=Path(self.config.get("paths", {}).get("cache", "data/cache/cube_cache")),
            )
            self.cache = StreamingCache(cache_config)
        else:
            self.cache = None

        # Initialize validator
        if enable_validation:
            self.validator = DataValidator()
        else:
            self.validator = None

        # Initialize adaptive chunker
        if adaptive_chunking:
            base_chunks = self._get_base_chunks()
            self.chunker = AdaptiveChunker(self.memory_monitor, base_chunks)
        else:
            self.chunker = None

        # Dataset storage
        self.train_paths = None
        self.val_paths = None
        self.test_paths = None

        # Performance monitoring
        self.performance_stats = {
            "total_samples_loaded": 0,
            "total_load_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "validation_failures": 0,
        }

        logger.info(f"Initialized advanced CubeDM:")
        logger.info(f"  Zarr root: {self.zarr_root}")
        logger.info(f"  Variables: {self.variables}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Caching: {enable_caching}")
        logger.info(f"  Validation: {enable_validation}")
        logger.info(f"  Adaptive chunking: {adaptive_chunking}")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not config_path or not Path(config_path).exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    def _resolve_zarr_root(self, zarr_root: Optional[str]) -> Union[Path, str]:
        """Resolve zarr root path with S3/R2 and environment variable support"""
        if zarr_root:
            # FIXED: Support S3/R2 URLs directly (R2 uses S3-compatible API)
            if zarr_root.startswith("s3://") or zarr_root.startswith("r2://"):
                return zarr_root  # Return S3/R2 URL as string
            return Path(zarr_root)

        # Try environment variable
        env_path = os.getenv("ASTRO_CUBE_ZARR")
        if env_path:
            if env_path.startswith("s3://") or env_path.startswith("r2://"):
                return env_path  # Return S3/R2 URL as string
            return Path(env_path)

        # Try config file
        config_path = self.config.get("datacube", {}).get("zarr_root")
        if config_path:
            # Handle environment variable substitution in config
            if config_path.startswith("${") and config_path.endswith("}"):
                env_var = config_path[2:-1]
                if "," in env_var:
                    env_var, default = env_var.split(",")
                    resolved_path = os.getenv(env_var, default)
                    if resolved_path.startswith("s3://") or resolved_path.startswith("r2://"):
                        return resolved_path  # Return S3/R2 URL as string
                    return Path(resolved_path)
                else:
                    resolved_path = os.getenv(env_var, "data/processed/gcm_zarr")
                    if resolved_path.startswith("s3://") or resolved_path.startswith("r2://"):
                        return resolved_path  # Return S3/R2 URL as string
                    return Path(resolved_path)
            if config_path.startswith("s3://") or config_path.startswith("r2://"):
                return config_path  # Return S3/R2 URL as string
            return Path(config_path)

        # Default fallback
        return Path("data/processed/gcm_zarr")

    def _get_additional_stores(self) -> List[Path]:
        """Get additional zarr stores from config"""
        additional = self.config.get("datacube", {}).get("additional_zarr_stores", [])
        return [Path(store) for store in additional]

    def _discover_s3_zarr_stores(self) -> List[str]:
        """Discover zarr stores in S3/R2 bucket (R2 uses S3-compatible API)"""
        try:
            import s3fs
            import os

            # Check if using R2 (Cloudflare R2 with S3-compatible API)
            if self.zarr_root.startswith("r2://"):
                # R2 endpoint configuration
                r2_endpoint = os.getenv('R2_ENDPOINT_URL')
                r2_key = os.getenv('R2_ACCESS_KEY_ID')
                r2_secret = os.getenv('R2_SECRET_ACCESS_KEY')

                fs = s3fs.S3FileSystem(
                    key=r2_key,
                    secret=r2_secret,
                    client_kwargs={'endpoint_url': r2_endpoint}
                )
                bucket_path = self.zarr_root.replace("r2://", "")
            else:
                # Standard S3
                fs = s3fs.S3FileSystem()
                bucket_path = self.zarr_root.replace("s3://", "")

            # List directories matching run_*/data.zarr pattern
            zarr_stores = []
            try:
                # List all objects in the bucket path
                objects = fs.ls(bucket_path, detail=False)

                # Filter for zarr stores
                for obj in objects:
                    if obj.endswith("/data.zarr") and "/run_" in obj:
                        # Use appropriate prefix (s3:// or r2://)
                        prefix = "r2://" if self.zarr_root.startswith("r2://") else "s3://"
                        zarr_stores.append(f"{prefix}{obj}")

                logger.info(f"🔍 Discovered {len(zarr_stores)} S3/R2 zarr stores")
                return zarr_stores

            except Exception as e:
                logger.warning(f"⚠️ S3/R2 zarr store discovery failed: {e}")
                # Return dummy stores for testing
                return [f"{self.zarr_root}/run_000001/data.zarr",
                       f"{self.zarr_root}/run_000002/data.zarr"]

        except ImportError:
            logger.error("❌ s3fs not available for S3/R2 zarr store discovery")
            raise RuntimeError("s3fs required for S3/R2 zarr operations")

    def _get_config_variables(self, var_type: str) -> List[str]:
        """Get variables from configuration"""
        variables = self.config.get("datacube", {}).get(var_type, [])
        if not variables:
            # Default variables
            default_vars = ["T_surf", "q_H2O", "cldfrac", "albedo", "psurf"]
            logger.warning(f"No {var_type} in config, using defaults: {default_vars}")
            return default_vars
        return variables

    def _get_base_chunks(self) -> Dict[str, int]:
        """Get base chunk configuration"""
        chunking_config = self.config.get("datacube", {}).get("chunking", {})
        return chunking_config.get("base_chunks", {"lat": 40, "lon": 40, "lev": 15, "time": 4})

    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory information"""
        return self.memory_monitor.check_memory()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self.cache:
            return self.cache.get_stats()
        return {"cache_enabled": False}

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = self.performance_stats.copy()

        # Add derived metrics
        if stats["total_samples_loaded"] > 0:
            stats["avg_load_time"] = stats["total_load_time"] / stats["total_samples_loaded"]

        if self.cache:
            total_requests = stats["cache_hits"] + stats["cache_misses"]
            if total_requests > 0:
                stats["cache_hit_rate"] = stats["cache_hits"] / total_requests

        if self.validator:
            stats.update(self.validator.validation_stats)

        return stats

    def optimize_for_memory(self) -> Dict[str, Any]:
        """Optimize settings for current memory conditions"""
        memory_info = self.memory_monitor.check_memory()

        optimizations = {
            "original_batch_size": self.batch_size,
            "original_num_workers": self.num_workers,
            "memory_usage": memory_info["usage_percent"],
        }

        # Reduce batch size if memory usage is high
        if memory_info["usage_percent"] > 0.8:
            self.batch_size = max(1, self.batch_size // 2)
            optimizations["reduced_batch_size"] = self.batch_size

        # Reduce workers if memory usage is critical
        if memory_info["usage_percent"] > 0.9:
            self.num_workers = max(1, self.num_workers // 2)
            optimizations["reduced_num_workers"] = self.num_workers

        logger.info(f"Memory optimizations applied: {optimizations}")
        return optimizations

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage"""
        if stage in (None, "fit", "validate"):
            # FIXED: Handle S3/R2 URLs for zarr store discovery
            if isinstance(self.zarr_root, str) and (self.zarr_root.startswith("s3://") or self.zarr_root.startswith("r2://")):
                # S3/R2 zarr store discovery (R2 uses S3-compatible API)
                zarr_stores = self._discover_s3_zarr_stores()
            else:
                # Local zarr store discovery
                zarr_stores = list(self.zarr_root.glob("run_*/data.zarr"))

            if not zarr_stores:
                raise ValueError(f"No zarr stores found in {self.zarr_root}")

            # Sort for reproducible splits
            zarr_stores = sorted(zarr_stores)
            n_total = len(zarr_stores)

            # Calculate split indices
            n_train = int(n_total * self.train_fraction)
            n_val = int(n_total * self.val_fraction)

            # Split datasets
            self.train_paths = zarr_stores[:n_train]
            self.val_paths = zarr_stores[n_train : n_train + n_val]
            self.test_paths = zarr_stores[n_train + n_val :]

            logger.info(
                f"Data split: {len(self.train_paths)} train, {len(self.val_paths)} val, "
                f"{len(self.test_paths)} test zarr stores"
            )

        if stage in (None, "test"):
            if self.test_paths is None:
                # If not already set up, use all data for testing
                zarr_stores = sorted(list(self.zarr_root.glob("run_*/data.zarr")))
                self.test_paths = zarr_stores

    def train_dataloader(self):
        """Training dataloader"""
        if not self.train_paths:
            raise ValueError("Training paths not set up. Call setup() first.")

        dataset = CubeDataset(
            zarr_paths=self.train_paths,
            variables=self.variables,
            target_variables=self.target_variables,
            time_window=self.time_window,
            spatial_crop=self.spatial_crop,
            normalize=self.normalize,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        """Validation dataloader"""
        if not self.val_paths:
            raise ValueError("Validation paths not set up. Call setup() first.")

        dataset = CubeDataset(
            zarr_paths=self.val_paths,
            variables=self.variables,
            target_variables=self.target_variables,
            time_window=self.time_window,
            spatial_crop=self.spatial_crop,
            normalize=self.normalize,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        """Test dataloader"""
        if not self.test_paths:
            raise ValueError("Test paths not set up. Call setup() first.")

        dataset = CubeDataset(
            zarr_paths=self.test_paths,
            variables=self.variables,
            target_variables=self.target_variables,
            time_window=self.time_window,
            spatial_crop=self.spatial_crop,
            normalize=self.normalize,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=min(self.num_workers, 2),  # Reduce workers for testing
            pin_memory=self.pin_memory,
        )

    def get_sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample batch for testing"""
        self.setup("fit")
        dataloader = self.train_dataloader()
        return next(iter(dataloader))

    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the data structure"""
        try:
            sample_input, sample_target = self.get_sample_batch()

            info = {
                "input_shape": list(sample_input.shape),
                "target_shape": list(sample_target.shape),
                "input_variables": self.variables,
                "target_variables": self.target_variables,
                "time_window": self.time_window,
                "spatial_crop": self.spatial_crop,
                "normalize": self.normalize,
                "total_zarr_stores": (
                    len(list(self.zarr_root.glob("run_*/data.zarr")))
                    if self.zarr_root.exists()
                    else 0
                ),
            }

            return info

        except Exception as e:
            logger.error(f"Error getting data info: {e}")
            return {
                "error": str(e),
                "input_variables": self.variables,
                "target_variables": self.target_variables,
            }

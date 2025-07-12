#!/usr/bin/env python3
"""
4-D Climate Datacube DataModule
==============================

PyTorch Lightning DataModule for streaming 4-D climate datacubes.
Handles zarr-formatted GCM simulation data with chunked loading.

Author: AI Assistant
Date: 2025
"""

import torch
import pytorch_lightning as pl
import xarray as xr
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import logging
from torch.utils.data import Dataset, DataLoader, IterableDataset
import warnings

# Configure logging
logger = logging.getLogger(__name__)

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
        normalize: bool = True
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
                    if 'lat' in ds.coords:
                        ds = ds.isel(lat=slice(lat_start, lat_end))
                    if 'lon' in ds.coords:
                        ds = ds.isel(lon=slice(lon_start, lon_end))
                
                self.datasets.append(ds)
                
                # Calculate number of samples (sliding window over time)
                if 'time' in ds.dims:
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
        
        logger.info(f"Initialized CubeDataset with {len(self.datasets)} zarr stores, "
                   f"{self.total_samples} total samples")
    
    def _calculate_normalization_stats(self) -> Dict[str, Dict[str, float]]:
        """Calculate mean and std for normalization"""
        logger.info("Calculating normalization statistics...")
        
        stats = {}
        
        # Sample a subset of data for statistics
        for ds in self.datasets[:min(10, len(self.datasets))]:  # Use up to 10 datasets
            for var in ds.data_vars:
                if var not in stats:
                    stats[var] = {'values': []}
                
                # Sample some time steps
                if 'time' in ds.dims:
                    sample_times = slice(0, min(50, len(ds.time)), 5)  # Every 5th timestep, up to 50
                    sample_data = ds[var].isel(time=sample_times)
                else:
                    sample_data = ds[var]
                
                # Compute statistics on flattened data
                values = sample_data.values.flatten()
                values = values[~np.isnan(values)]  # Remove NaN values
                
                if len(values) > 0:
                    stats[var]['values'].extend(values[:10000])  # Limit to prevent memory issues
        
        # Calculate final statistics
        final_stats = {}
        for var, data in stats.items():
            if data['values']:
                values = np.array(data['values'])
                final_stats[var] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values))
                }
            else:
                final_stats[var] = {'mean': 0.0, 'std': 1.0}
        
        logger.info(f"Calculated normalization stats for {len(final_stats)} variables")
        return final_stats
    
    def _normalize_data(self, data: torch.Tensor, variable: str) -> torch.Tensor:
        """Normalize data using pre-calculated statistics"""
        if self.norm_stats and variable in self.norm_stats:
            mean = self.norm_stats[variable]['mean']
            std = self.norm_stats[variable]['std']
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
            end_idx = start_idx + per_worker if worker_id < worker_info.num_workers - 1 else len(self.datasets)
            dataset_indices = list(range(start_idx, end_idx))
        
        for dataset_idx in dataset_indices:
            ds = self.datasets[dataset_idx]
            n_samples = self.dataset_lengths[dataset_idx]
            
            for sample_idx in range(n_samples):
                try:
                    # Extract time window
                    if 'time' in ds.dims:
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
                            if 'time' not in data_array.dims:
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
                        input_tensor = torch.stack(input_data, dim=0)  # (n_vars, time, lev, lat, lon)
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
                    logger.warning(f"Error processing sample {sample_idx} from dataset {dataset_idx}: {e}")
                    continue

class CubeDM(pl.LightningDataModule):
    """
    Lightning DataModule for 4-D climate datacubes
    """
    
    def __init__(
        self,
        zarr_root: str = "data/processed/gcm_zarr",
        batch_size: int = 4,
        num_workers: int = 4,
        variables: List[str] = None,
        target_variables: List[str] = None,
        time_window: int = 10,
        spatial_crop: Optional[Tuple[int, int, int, int]] = None,
        train_fraction: float = 0.8,
        val_fraction: float = 0.1,
        normalize: bool = True,
        pin_memory: bool = True,
        **kwargs
    ):
        """
        Initialize CubeDM
        
        Args:
            zarr_root: Root directory containing zarr stores
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
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.zarr_root = Path(zarr_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.variables = variables or ['temp', 'humidity', 'pressure', 'wind_u', 'wind_v']
        self.target_variables = target_variables or []
        self.time_window = time_window
        self.spatial_crop = spatial_crop
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.normalize = normalize
        self.pin_memory = pin_memory
        
        # Dataset storage
        self.train_paths = None
        self.val_paths = None
        self.test_paths = None
        
        logger.info(f"Initialized CubeDM with zarr_root={zarr_root}, batch_size={batch_size}")
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage"""
        if stage in (None, "fit", "validate"):
            # Find all zarr stores
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
            self.val_paths = zarr_stores[n_train:n_train + n_val]
            self.test_paths = zarr_stores[n_train + n_val:]
            
            logger.info(f"Data split: {len(self.train_paths)} train, {len(self.val_paths)} val, "
                       f"{len(self.test_paths)} test zarr stores")
        
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
            normalize=self.normalize
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
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
            normalize=self.normalize
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
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
            normalize=self.normalize
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=min(self.num_workers, 2),  # Reduce workers for testing
            pin_memory=self.pin_memory
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
                'input_shape': list(sample_input.shape),
                'target_shape': list(sample_target.shape),
                'input_variables': self.variables,
                'target_variables': self.target_variables,
                'time_window': self.time_window,
                'spatial_crop': self.spatial_crop,
                'normalize': self.normalize,
                'total_zarr_stores': len(list(self.zarr_root.glob("run_*/data.zarr"))) if self.zarr_root.exists() else 0
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting data info: {e}")
            return {
                'error': str(e),
                'input_variables': self.variables,
                'target_variables': self.target_variables
            } 
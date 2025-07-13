#!/usr/bin/env python3
"""
Advanced NetCDF to Zarr Conversion Utility
==========================================

High-performance conversion utility for climate model outputs.
Supports parallel processing, memory optimization, and quality validation.

Features:
- Parallel processing with dask
- Memory-efficient streaming
- Automatic chunking optimization
- Quality validation and metadata preservation
- Progress tracking and error recovery
- Physics-informed variable validation

Usage:
    python data_build/nc_to_zarr.py \
        --src_dir my_cubes/nc_raw \
        --dst_dir my_cubes/rocke3d_all_runs.zarr \
        --vars T_surf q_H2O cldfrac albedo psurf \
        --chunk "lat:40,lon:40,lev:15,time:4" \
        --parallel 8 \
        --quality_check
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import json

import numpy as np
import xarray as xr
import dask
import dask.array as da
from dask.distributed import Client, LocalCluster
import zarr
from tqdm import tqdm
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nc_to_zarr.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ConversionConfig:
    """Configuration for NetCDF to Zarr conversion"""
    
    src_dir: Path
    dst_dir: Path
    variables: List[str]
    chunk_spec: Dict[str, int]
    
    # Processing options
    parallel_workers: int = 8
    memory_limit: str = "4GB"
    quality_check: bool = True
    compress: bool = True
    compression_level: int = 3
    
    # Validation options
    validate_physics: bool = True
    validate_metadata: bool = True
    
    # Performance tuning
    buffer_size: int = 1000
    max_concurrent_files: int = 10
    
    # Quality thresholds
    max_missing_fraction: float = 0.05
    temperature_range: Tuple[float, float] = (150.0, 400.0)  # K
    pressure_range: Tuple[float, float] = (0.001, 1000.0)   # bar
    
    def __post_init__(self):
        self.src_dir = Path(self.src_dir)
        self.dst_dir = Path(self.dst_dir)
        
        if not self.src_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.src_dir}")
            
        self.dst_dir.mkdir(parents=True, exist_ok=True)

class PhysicsValidator:
    """Validates physical consistency of climate data"""
    
    def __init__(self, config: ConversionConfig):
        self.config = config
        self.validation_results = []
        
    def validate_temperature(self, data: xr.DataArray) -> Dict[str, bool]:
        """Validate temperature data"""
        results = {}
        
        # Check for reasonable temperature range
        valid_range = (data >= self.config.temperature_range[0]) & \
                     (data <= self.config.temperature_range[1])
        results['temperature_range'] = valid_range.all().compute()
        
        # Check for physical gradients
        if 'lev' in data.dims:
            temp_gradient = data.diff('lev')
            # Temperature should generally decrease with altitude
            results['temperature_gradient'] = (temp_gradient <= 0).mean() > 0.7
        
        return results
    
    def validate_pressure(self, data: xr.DataArray) -> Dict[str, bool]:
        """Validate pressure data"""
        results = {}
        
        # Check for reasonable pressure range
        valid_range = (data >= self.config.pressure_range[0]) & \
                     (data <= self.config.pressure_range[1])
        results['pressure_range'] = valid_range.all().compute()
        
        # Check for hydrostatic consistency
        if 'lev' in data.dims:
            pressure_gradient = data.diff('lev')
            # Pressure should increase with depth
            results['hydrostatic_consistency'] = (pressure_gradient >= 0).mean() > 0.9
        
        return results
    
    def validate_humidity(self, data: xr.DataArray) -> Dict[str, bool]:
        """Validate humidity data"""
        results = {}
        
        # Check for reasonable humidity range (mixing ratio)
        results['humidity_range'] = ((data >= 0) & (data <= 1)).all().compute()
        
        # Check for condensation consistency
        if 'lev' in data.dims:
            # Higher humidity should be near surface
            surface_humidity = data.isel(lev=-1)
            upper_humidity = data.isel(lev=0)
            results['vertical_humidity'] = (surface_humidity >= upper_humidity).mean() > 0.8
        
        return results
    
    def validate_dataset(self, ds: xr.Dataset) -> Dict[str, Dict[str, bool]]:
        """Validate entire dataset"""
        results = {}
        
        for var in ds.data_vars:
            if 'T_surf' in var or 'temp' in var.lower():
                results[var] = self.validate_temperature(ds[var])
            elif 'psurf' in var or 'pressure' in var.lower():
                results[var] = self.validate_pressure(ds[var])
            elif 'q_H2O' in var or 'humid' in var.lower():
                results[var] = self.validate_humidity(ds[var])
        
        return results

class MemoryOptimizer:
    """Optimizes memory usage during conversion"""
    
    def __init__(self, config: ConversionConfig):
        self.config = config
        self.memory_stats = []
        
    def get_optimal_chunks(self, dataset: xr.Dataset) -> Dict[str, int]:
        """Calculate optimal chunk sizes based on memory constraints"""
        
        # Start with base chunks
        chunks = self.config.chunk_spec.copy()
        
        # Get available memory
        available_memory = psutil.virtual_memory().available
        memory_limit_bytes = self._parse_memory_limit(self.config.memory_limit)
        
        # Calculate dataset size in memory
        dataset_size = sum(
            var.nbytes for var in dataset.data_vars.values()
            if hasattr(var, 'nbytes')
        )
        
        # Adjust chunks if dataset is too large
        if dataset_size > memory_limit_bytes:
            scale_factor = (memory_limit_bytes / dataset_size) ** 0.5
            
            for dim in chunks:
                if dim in dataset.dims:
                    chunks[dim] = max(1, int(chunks[dim] * scale_factor))
        
        logger.info(f"Optimal chunks: {chunks}")
        return chunks
    
    def _parse_memory_limit(self, memory_str: str) -> int:
        """Parse memory limit string to bytes"""
        units = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
        
        for unit in units:
            if memory_str.endswith(unit):
                return int(memory_str[:-len(unit)]) * units[unit]
        
        return int(memory_str)  # Assume bytes
    
    def monitor_memory(self):
        """Monitor memory usage during conversion"""
        memory_info = psutil.virtual_memory()
        self.memory_stats.append({
            'timestamp': time.time(),
            'used_percent': memory_info.percent,
            'available_gb': memory_info.available / (1024**3)
        })
        
        if memory_info.percent > 90:
            logger.warning(f"High memory usage: {memory_info.percent}%")

class ZarrConverter:
    """Main converter class"""
    
    def __init__(self, config: ConversionConfig):
        self.config = config
        self.physics_validator = PhysicsValidator(config)
        self.memory_optimizer = MemoryOptimizer(config)
        self.conversion_stats = {
            'files_processed': 0,
            'files_failed': 0,
            'total_size_gb': 0,
            'processing_time': 0,
            'validation_results': {}
        }
        
    def setup_dask_client(self) -> Client:
        """Setup optimized Dask client"""
        
        cluster = LocalCluster(
            n_workers=self.config.parallel_workers,
            threads_per_worker=2,
            memory_limit=self.config.memory_limit,
            dashboard_address=':8787'
        )
        
        client = Client(cluster)
        logger.info(f"Dask dashboard: {client.dashboard_link}")
        
        return client
    
    def discover_netcdf_files(self) -> List[Path]:
        """Discover all NetCDF files in source directory"""
        
        patterns = ['*.nc', '*.nc4', '*.netcdf']
        files = []
        
        for pattern in patterns:
            files.extend(self.config.src_dir.glob(pattern))
            files.extend(self.config.src_dir.rglob(pattern))
        
        logger.info(f"Found {len(files)} NetCDF files")
        return sorted(files)
    
    def preprocess_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Preprocess dataset before conversion"""
        
        # Filter variables
        if self.config.variables:
            available_vars = [var for var in self.config.variables if var in ds.data_vars]
            ds = ds[available_vars]
        
        # Handle missing values
        ds = ds.fillna(0)  # Or use more sophisticated interpolation
        
        # Standardize coordinate names
        coord_mapping = {
            'latitude': 'lat',
            'longitude': 'lon',
            'level': 'lev',
            'pressure': 'lev'
        }
        
        for old_name, new_name in coord_mapping.items():
            if old_name in ds.coords and new_name not in ds.coords:
                ds = ds.rename({old_name: new_name})
        
        # Add metadata
        ds.attrs.update({
            'conversion_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'conversion_tool': 'nc_to_zarr.py',
            'source_format': 'NetCDF',
            'target_format': 'Zarr'
        })
        
        return ds
    
    def convert_file(self, nc_file: Path) -> bool:
        """Convert a single NetCDF file"""
        
        try:
            logger.info(f"Processing {nc_file}")
            start_time = time.time()
            
            # Open dataset
            with xr.open_dataset(nc_file, chunks='auto') as ds:
                
                # Preprocess
                ds = self.preprocess_dataset(ds)
                
                # Validate if required
                if self.config.quality_check:
                    validation_results = self.physics_validator.validate_dataset(ds)
                    self.conversion_stats['validation_results'][str(nc_file)] = validation_results
                
                # Optimize chunking
                optimal_chunks = self.memory_optimizer.get_optimal_chunks(ds)
                ds = ds.chunk(optimal_chunks)
                
                # Monitor memory
                self.memory_optimizer.monitor_memory()
                
                # Convert to zarr
                zarr_path = self.config.dst_dir / f"{nc_file.stem}.zarr"
                
                # Setup zarr store
                store = zarr.DirectoryStore(str(zarr_path))
                
                # Write to zarr with compression
                encoding = {}
                if self.config.compress:
                    for var in ds.data_vars:
                        encoding[var] = {
                            'compressor': zarr.Blosc(cname='lz4', clevel=self.config.compression_level),
                            'chunks': tuple(optimal_chunks.get(dim, ds[var].sizes[dim]) 
                                          for dim in ds[var].dims)
                        }
                
                ds.to_zarr(store, mode='w', encoding=encoding)
                
                processing_time = time.time() - start_time
                file_size_gb = nc_file.stat().st_size / (1024**3)
                
                logger.info(f"✓ Converted {nc_file} in {processing_time:.2f}s ({file_size_gb:.2f} GB)")
                
                self.conversion_stats['files_processed'] += 1
                self.conversion_stats['total_size_gb'] += file_size_gb
                self.conversion_stats['processing_time'] += processing_time
                
                return True
                
        except Exception as e:
            logger.error(f"✗ Failed to convert {nc_file}: {e}")
            self.conversion_stats['files_failed'] += 1
            return False
    
    def merge_zarr_stores(self) -> None:
        """Merge individual zarr stores into single consolidated store"""
        
        logger.info("Merging zarr stores...")
        
        # Find all zarr stores
        zarr_stores = list(self.config.dst_dir.glob("*.zarr"))
        
        if not zarr_stores:
            logger.warning("No zarr stores found to merge")
            return
        
        # Open all stores
        datasets = []
        for store in zarr_stores:
            try:
                ds = xr.open_zarr(store)
                datasets.append(ds)
            except Exception as e:
                logger.warning(f"Failed to open {store}: {e}")
        
        if not datasets:
            logger.error("No valid zarr stores found")
            return
        
        # Concatenate along time dimension
        try:
            merged_ds = xr.concat(datasets, dim='time')
            
            # Save merged dataset
            merged_path = self.config.dst_dir / "merged_dataset.zarr"
            merged_ds.to_zarr(merged_path, mode='w')
            
            logger.info(f"Merged dataset saved to {merged_path}")
            
        except Exception as e:
            logger.error(f"Failed to merge datasets: {e}")
    
    def generate_report(self) -> Dict:
        """Generate conversion report"""
        
        report = {
            'conversion_summary': self.conversion_stats,
            'memory_stats': self.memory_optimizer.memory_stats,
            'configuration': {
                'source_dir': str(self.config.src_dir),
                'destination_dir': str(self.config.dst_dir),
                'variables': self.config.variables,
                'chunk_spec': self.config.chunk_spec,
                'parallel_workers': self.config.parallel_workers
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save report
        report_path = self.config.dst_dir / "conversion_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Conversion report saved to {report_path}")
        return report
    
    def convert_all(self) -> None:
        """Convert all NetCDF files to zarr"""
        
        logger.info("Starting NetCDF to Zarr conversion...")
        
        # Setup Dask client
        client = self.setup_dask_client()
        
        try:
            # Discover files
            nc_files = self.discover_netcdf_files()
            
            if not nc_files:
                logger.warning("No NetCDF files found")
                return
            
            # Convert files
            with ProcessPoolExecutor(max_workers=self.config.parallel_workers) as executor:
                futures = {executor.submit(self.convert_file, nc_file): nc_file 
                          for nc_file in nc_files}
                
                for future in tqdm(as_completed(futures), total=len(futures), 
                                 desc="Converting files"):
                    nc_file = futures[future]
                    try:
                        success = future.result()
                    except Exception as e:
                        logger.error(f"Error processing {nc_file}: {e}")
            
            # Merge zarr stores if multiple files
            if len(nc_files) > 1:
                self.merge_zarr_stores()
            
            # Generate report
            report = self.generate_report()
            
            # Print summary
            logger.info("Conversion completed!")
            logger.info(f"Files processed: {self.conversion_stats['files_processed']}")
            logger.info(f"Files failed: {self.conversion_stats['files_failed']}")
            logger.info(f"Total size: {self.conversion_stats['total_size_gb']:.2f} GB")
            logger.info(f"Processing time: {self.conversion_stats['processing_time']:.2f}s")
            
        finally:
            client.close()

def parse_chunk_spec(chunk_str: str) -> Dict[str, int]:
    """Parse chunk specification string"""
    
    chunks = {}
    for chunk_def in chunk_str.split(','):
        dim, size = chunk_def.split(':')
        chunks[dim.strip()] = int(size.strip())
    
    return chunks

def main():
    """Main conversion function"""
    
    parser = argparse.ArgumentParser(description="Convert NetCDF files to Zarr format")
    
    parser.add_argument("--src_dir", required=True, help="Source directory with NetCDF files")
    parser.add_argument("--dst_dir", required=True, help="Destination directory for Zarr stores")
    parser.add_argument("--vars", nargs='+', help="Variables to convert")
    parser.add_argument("--chunk", required=True, help="Chunk specification (e.g., 'lat:40,lon:40,lev:15,time:4')")
    parser.add_argument("--parallel", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--memory_limit", default="4GB", help="Memory limit per worker")
    parser.add_argument("--quality_check", action="store_true", help="Enable quality validation")
    parser.add_argument("--compress", action="store_true", default=True, help="Enable compression")
    parser.add_argument("--compression_level", type=int, default=3, help="Compression level (1-9)")
    
    args = parser.parse_args()
    
    try:
        # Parse chunk specification
        chunk_spec = parse_chunk_spec(args.chunk)
        
        # Create configuration
        config = ConversionConfig(
            src_dir=args.src_dir,
            dst_dir=args.dst_dir,
            variables=args.vars,
            chunk_spec=chunk_spec,
            parallel_workers=args.parallel,
            memory_limit=args.memory_limit,
            quality_check=args.quality_check,
            compress=args.compress,
            compression_level=args.compression_level
        )
        
        # Create converter and run
        converter = ZarrConverter(config)
        converter.convert_all()
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
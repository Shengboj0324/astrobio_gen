#!/usr/bin/env python3
"""
Multi-Modal Storage Layer (Simplified)
======================================

Simplified storage architecture for multi-disciplinary scientific data with
intelligent format selection and caching strategies.

Storage Strategy:
- Climate 4D datacubes: NPZ/HDF5 (reliable, cross-platform)
- Biological networks: NPZ (numpy arrays, fast loading)
- High-resolution spectra: NPZ (simple, reliable)
- Metadata/parameters: JSON (human-readable)

This simplified version focuses on reliability and compatibility.
"""

import os
import json
import numpy as np
import logging
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from threading import Lock
import time
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor
import tempfile
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataDomain(Enum):
    """Scientific data domains"""
    BIOLOGY = "biology"
    ASTRONOMY = "astronomy" 
    CLIMATE = "climate"
    SPECTROSCOPY = "spectroscopy"
    PHYSICS = "physics"
    OBSERVATIONS = "observations"

class DataFormat(Enum):
    """Supported data storage formats"""
    NPZ = "npz"
    JSON = "json"
    HDF5 = "hdf5"

class StorageTier(Enum):
    """Storage performance tiers"""
    HOT = "hot"           # SSD, frequent access
    WARM = "warm"         # HDD, occasional access
    COLD = "cold"         # Cloud, archival

class CompressionLevel(Enum):
    """Compression level options"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class StorageConfig:
    """Configuration for multi-modal storage"""
    # Base storage paths
    storage_root: Path = Path("data/planet_runs")
    cache_root: Path = Path("data/cache")
    temp_root: Path = Path("data/temp")
    
    # Performance settings
    max_memory_cache_gb: float = 8.0
    parallel_workers: int = 4
    
    # Compression settings
    default_compression: CompressionLevel = CompressionLevel.MEDIUM
    
    # Data validation
    verify_checksums: bool = True

@dataclass
class DataFileInfo:
    """Information about stored data file"""
    run_id: int
    domain: DataDomain
    data_type: str
    file_path: Path
    file_format: DataFormat
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class MultiModalStorage:
    """
    Simplified storage manager for multi-modal scientific data
    
    Handles storage and retrieval across domains using reliable formats:
    - Climate: NPZ arrays for 4D datacubes
    - Biology: NPZ arrays for graphs and networks
    - Spectroscopy: NPZ for high-resolution data
    - Metadata: JSON for parameters and configuration
    """
    
    def __init__(self, config: StorageConfig = None):
        self.config = config or StorageConfig()
        
        # Initialize storage directories
        self._initialize_storage()
        
        # Cache management
        self._memory_cache = {}
        self._cache_lock = Lock()
        self._access_stats = {}
        
        # File registry
        self._file_registry = {}
        self._registry_lock = Lock()
        
        # Thread pool for parallel I/O
        self._executor = ThreadPoolExecutor(max_workers=self.config.parallel_workers)
        
        logger.info(f"🗄️ Multi-Modal Storage initialized: {self.config.storage_root}")
    
    def _initialize_storage(self):
        """Initialize storage directory structure"""
        directories = [
            self.config.storage_root,
            self.config.cache_root,
            self.config.temp_root
        ]
        
        # Create domain-specific subdirectories
        for domain in DataDomain:
            directories.append(self.config.storage_root / domain.value)
            directories.append(self.config.cache_root / domain.value)
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _get_run_directory(self, run_id: int) -> Path:
        """Get directory for planet run"""
        return self.config.storage_root / f"run_{run_id:06d}"
    
    def _register_file(self, file_info: DataFileInfo):
        """Register file in internal registry"""
        with self._registry_lock:
            key = (file_info.run_id, file_info.domain, file_info.data_type)
            self._file_registry[key] = file_info
    
    def _get_file_info(self, run_id: int, domain: DataDomain, data_type: str) -> Optional[DataFileInfo]:
        """Get file information from registry"""
        with self._registry_lock:
            key = (run_id, domain, data_type)
            return self._file_registry.get(key)
    
    # === CLIMATE DATA ===
    
    async def store_climate_datacube(self,
                                   run_id: int,
                                   data_dict: Dict[str, np.ndarray],
                                   dimensions: Dict[str, np.ndarray],
                                   variables: List[str] = None) -> Path:
        """
        Store climate datacube in NPZ format
        
        Args:
            run_id: Planet run identifier
            data_dict: Dictionary of variable_name -> data_array
            dimensions: Dictionary of dimension_name -> coordinate_array
            variables: List of variables to store (None = all)
            
        Returns:
            Path to stored NPZ file
        """
        logger.info(f"[PKG] Storing climate datacube for run {run_id}")
        
        run_dir = self._get_run_directory(run_id)
        climate_dir = run_dir / "climate"
        climate_dir.mkdir(parents=True, exist_ok=True)
        
        npz_path = climate_dir / "datacube.npz"
        
        # Prepare data for storage
        save_dict = {}
        
        # Add coordinate arrays with prefix
        for dim_name, coord_array in dimensions.items():
            save_dict[f"coord_{dim_name}"] = coord_array.astype(np.float32)
        
        # Add data variables
        for var_name, var_data in data_dict.items():
            if variables is None or var_name in variables:
                save_dict[f"data_{var_name}"] = var_data.astype(np.float32)
        
        # Store metadata
        metadata = {
            'run_id': run_id,
            'domain': DataDomain.CLIMATE.value,
            'data_type': '4d_datacube',
            'variables': list(data_dict.keys()),
            'dimensions': list(dimensions.keys()),
            'shapes': {k: list(v.shape) for k, v in data_dict.items()},
            'created_at': datetime.now(timezone.utc).isoformat(),
            'storage_format': 'npz_compressed'
        }
        save_dict['_metadata'] = json.dumps(metadata)
        
        # Save with compression
        if self.config.default_compression != CompressionLevel.NONE:
            np.savez_compressed(npz_path, **save_dict)
        else:
            np.savez(npz_path, **save_dict)
        
        # Register file
        file_info = DataFileInfo(
            run_id=run_id,
            domain=DataDomain.CLIMATE,
            data_type="4d_datacube",
            file_path=npz_path,
            file_format=DataFormat.NPZ,
            created_at=datetime.now(timezone.utc),
            metadata=metadata
        )
        self._register_file(file_info)
        
        logger.info(f"[OK] Stored climate datacube: {npz_path}")
        return npz_path
    
    async def load_climate_datacube(self, 
                                  run_id: int,
                                  variables: List[str] = None) -> Dict[str, Any]:
        """
        Load climate datacube with optional variable filtering
        
        Args:
            run_id: Planet run identifier
            variables: Variables to load (None = all)
            
        Returns:
            Dictionary with 'data', 'dimensions', and 'metadata'
        """
        # Check cache first
        cache_key = f"climate_{run_id}_{hash(tuple(variables) if variables else None)}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        
        # Get file path
        file_info = self._get_file_info(run_id, DataDomain.CLIMATE, "4d_datacube")
        if file_info is None:
            # Try to find file directly
            run_dir = self._get_run_directory(run_id)
            npz_path = run_dir / "climate" / "datacube.npz"
        else:
            npz_path = file_info.file_path
        
        if not npz_path.exists():
            raise FileNotFoundError(f"Climate datacube not found for run {run_id}")
        
        # Load NPZ data
        data = dict(np.load(npz_path, allow_pickle=True))
        
        # Extract metadata
        metadata = {}
        if '_metadata' in data:
            try:
                metadata = json.loads(str(data['_metadata']))
                del data['_metadata']
            except:
                pass
        
        # Separate coordinates and data variables
        dimensions = {}
        data_dict = {}
        
        for key, value in data.items():
            if key.startswith('coord_'):
                coord_name = key[6:]  # Remove 'coord_' prefix
                dimensions[coord_name] = value
            elif key.startswith('data_'):
                var_name = key[5:]  # Remove 'data_' prefix
                if variables is None or var_name in variables:
                    data_dict[var_name] = value
        
        result = {
            'data': data_dict,
            'dimensions': dimensions,
            'metadata': metadata
        }
        
        # Cache result
        self._add_to_cache(cache_key, result)
        
        return result
    
    # === BIOLOGICAL DATA ===
    
    async def store_biological_network(self,
                                     run_id: int,
                                     network_data: Dict[str, Any],
                                     network_type: str = "metabolic") -> Path:
        """
        Store biological network data in NPZ format
        
        Args:
            run_id: Planet run identifier
            network_data: Dictionary with network components
            network_type: Type of biological network
            
        Returns:
            Path to stored NPZ file
        """
        logger.info(f"[BIO] Storing {network_type} network for run {run_id}")
        
        run_dir = self._get_run_directory(run_id)
        bio_dir = run_dir / "biosphere"
        bio_dir.mkdir(parents=True, exist_ok=True)
        
        npz_path = bio_dir / f"{network_type}_network.npz"
        
        # Prepare data for storage
        save_dict = {}
        
        # Convert any special data types to numpy arrays
        for key, value in network_data.items():
            if key == 'graph' and hasattr(value, 'nodes'):
                # Skip NetworkX graphs - will be reconstructed from adjacency matrix
                continue
            elif isinstance(value, (list, tuple)):
                save_dict[key] = np.array(value)
            else:
                save_dict[key] = value
        
        # Store metadata
        metadata = {
            'run_id': run_id,
            'domain': DataDomain.BIOLOGY.value,
            'data_type': f"{network_type}_network",
            'network_type': network_type,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'storage_format': 'npz_compressed'
        }
        save_dict['_metadata'] = json.dumps(metadata)
        
        # Save with compression
        np.savez_compressed(npz_path, **save_dict)
        
        # Register file
        file_info = DataFileInfo(
            run_id=run_id,
            domain=DataDomain.BIOLOGY,
            data_type=f"{network_type}_network",
            file_path=npz_path,
            file_format=DataFormat.NPZ,
            created_at=datetime.now(timezone.utc),
            metadata=metadata
        )
        self._register_file(file_info)
        
        logger.info(f"[OK] Stored biological network: {npz_path}")
        return npz_path
    
    async def load_biological_network(self, 
                                    run_id: int,
                                    network_type: str = "metabolic") -> Dict[str, Any]:
        """
        Load biological network data
        
        Args:
            run_id: Planet run identifier
            network_type: Type of network to load
            
        Returns:
            Dictionary with network data
        """
        # Check cache
        cache_key = f"bio_{run_id}_{network_type}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        
        # Get file path
        file_info = self._get_file_info(run_id, DataDomain.BIOLOGY, f"{network_type}_network")
        if file_info is None:
            # Try to find file directly
            run_dir = self._get_run_directory(run_id)
            npz_path = run_dir / "biosphere" / f"{network_type}_network.npz"
        else:
            npz_path = file_info.file_path
        
        if not npz_path.exists():
            raise FileNotFoundError(f"Biological network not found for run {run_id}")
        
        # Load NPZ data
        data = dict(np.load(npz_path, allow_pickle=True))
        
        # Extract and remove metadata
        metadata = {}
        if '_metadata' in data:
            try:
                metadata = json.loads(str(data['_metadata']))
                del data['_metadata']
            except:
                pass
        
        data['metadata'] = metadata
        
        # Cache result
        self._add_to_cache(cache_key, data)
        
        return data
    
    # === SPECTROSCOPY DATA ===
    
    async def store_spectrum(self,
                           run_id: int,
                           wavelengths: np.ndarray,
                           flux: np.ndarray,
                           resolution: float,
                           instrument: str = "PSG",
                           metadata: Dict[str, Any] = None) -> Path:
        """
        Store high-resolution spectrum in NPZ format
        
        Args:
            run_id: Planet run identifier
            wavelengths: Wavelength array (microns)
            flux: Flux array (same length as wavelengths)
            resolution: Spectral resolution
            instrument: Instrument name
            metadata: Additional metadata
            
        Returns:
            Path to stored NPZ file
        """
        logger.info(f"🌈 Storing spectrum for run {run_id} (R={resolution})")
        
        run_dir = self._get_run_directory(run_id)
        spec_dir = run_dir / "spectroscopy"
        spec_dir.mkdir(parents=True, exist_ok=True)
        
        npz_path = spec_dir / f"spectrum_{instrument.lower()}.npz"
        
        # Prepare data for storage
        save_data = {
            'wavelengths': wavelengths.astype(np.float32),
            'flux': flux.astype(np.float32),
            'resolution': float(resolution),
            'instrument': instrument,
            'num_points': len(wavelengths),
            'wavelength_min': float(wavelengths.min()),
            'wavelength_max': float(wavelengths.max())
        }
        
        # Add additional metadata
        if metadata:
            for key, value in metadata.items():
                if not key.startswith('_'):  # Avoid conflicts with internal metadata
                    save_data[key] = value
        
        # Store metadata
        spec_metadata = {
            'run_id': run_id,
            'domain': DataDomain.SPECTROSCOPY.value,
            'data_type': f"spectrum_{instrument.lower()}",
            'instrument': instrument,
            'resolution': float(resolution),
            'num_points': len(wavelengths),
            'wavelength_range': [float(wavelengths.min()), float(wavelengths.max())],
            'created_at': datetime.now(timezone.utc).isoformat(),
            'storage_format': 'npz_compressed'
        }
        save_data['_metadata'] = json.dumps(spec_metadata)
        
        # Save with compression
        np.savez_compressed(npz_path, **save_data)
        
        # Register file
        file_info = DataFileInfo(
            run_id=run_id,
            domain=DataDomain.SPECTROSCOPY,
            data_type=f"spectrum_{instrument.lower()}",
            file_path=npz_path,
            file_format=DataFormat.NPZ,
            created_at=datetime.now(timezone.utc),
            metadata=spec_metadata
        )
        self._register_file(file_info)
        
        logger.info(f"[OK] Stored spectrum: {npz_path}")
        return npz_path
    
    async def load_spectrum(self,
                          run_id: int,
                          instrument: str = "PSG",
                          wavelength_range: Tuple[float, float] = None) -> Dict[str, Any]:
        """
        Load high-resolution spectrum with optional wavelength filtering
        
        Args:
            run_id: Planet run identifier
            instrument: Instrument name
            wavelength_range: (min_wl, max_wl) in microns for filtering
            
        Returns:
            Dictionary with spectrum data and metadata
        """
        # Check cache
        cache_key = f"spectrum_{run_id}_{instrument}_{hash(wavelength_range)}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        
        # Get file path
        file_info = self._get_file_info(run_id, DataDomain.SPECTROSCOPY, f"spectrum_{instrument.lower()}")
        if file_info is None:
            # Try to find file directly
            run_dir = self._get_run_directory(run_id)
            npz_path = run_dir / "spectroscopy" / f"spectrum_{instrument.lower()}.npz"
        else:
            npz_path = file_info.file_path
        
        if not npz_path.exists():
            raise FileNotFoundError(f"Spectrum not found for run {run_id}, instrument {instrument}")
        
        # Load NPZ data
        data = dict(np.load(npz_path, allow_pickle=True))
        
        # Extract wavelengths and flux
        wavelengths = data.pop('wavelengths', None)
        flux = data.pop('flux', None)
        
        if wavelengths is None or flux is None:
            raise ValueError(f"Invalid spectrum data for run {run_id}")
        
        # Apply wavelength filtering if requested
        if wavelength_range:
            min_wl, max_wl = wavelength_range
            mask = (wavelengths >= min_wl) & (wavelengths <= max_wl)
            wavelengths = wavelengths[mask]
            flux = flux[mask]
        
        # Extract metadata
        metadata = {}
        if '_metadata' in data:
            try:
                metadata = json.loads(str(data['_metadata']))
                del data['_metadata']
            except:
                pass
        
        # Add remaining data to metadata
        metadata.update(data)
        
        result = {
            'wavelengths': wavelengths,
            'flux': flux,
            'metadata': metadata
        }
        
        # Cache result
        self._add_to_cache(cache_key, result)
        
        return result
    
    # === CACHE MANAGEMENT ===
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get item from memory cache"""
        with self._cache_lock:
            if key in self._memory_cache:
                item, timestamp = self._memory_cache[key]
                
                # Update access stats
                if key not in self._access_stats:
                    self._access_stats[key] = {'count': 0, 'last_access': timestamp}
                
                self._access_stats[key]['count'] += 1
                self._access_stats[key]['last_access'] = time.time()
                
                return item
        
        return None
    
    def _add_to_cache(self, key: str, item: Any):
        """Add item to memory cache with size management"""
        with self._cache_lock:
            # Simple cache size management
            max_cache_items = 100
            
            if len(self._memory_cache) >= max_cache_items:
                # Remove least recently used items
                if self._access_stats:
                    sorted_items = sorted(
                        self._access_stats.items(),
                        key=lambda x: x[1]['last_access']
                    )
                    
                    # Remove oldest 20% of items
                    num_to_remove = max(1, len(sorted_items) // 5)
                    for old_key, _ in sorted_items[:num_to_remove]:
                        if old_key in self._memory_cache:
                            del self._memory_cache[old_key]
                        if old_key in self._access_stats:
                            del self._access_stats[old_key]
            
            self._memory_cache[key] = (item, time.time())
    
    def clear_cache(self):
        """Clear memory cache"""
        with self._cache_lock:
            self._memory_cache.clear()
            self._access_stats.clear()
        logger.info("🧹 Memory cache cleared")
    
    # === STATISTICS AND UTILITIES ===
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics"""
        with self._registry_lock:
            registry_size = len(self._file_registry)
            
            # Count files by domain
            domain_counts = {}
            for domain in DataDomain:
                count = sum(1 for (_, d, _) in self._file_registry.keys() if d == domain)
                domain_counts[domain.value] = count
        
        with self._cache_lock:
            cache_size = len(self._memory_cache)
            total_cache_hits = sum(s['count'] for s in self._access_stats.values())
        
        return {
            'registry_size': registry_size,
            'cache_size': cache_size,
            'total_cache_hits': total_cache_hits,
            'storage_root': str(self.config.storage_root),
            'domain_file_counts': domain_counts,
            'supported_formats': ['NPZ', 'JSON']
        }
    
    def list_stored_runs(self) -> List[int]:
        """List all stored planet run IDs"""
        with self._registry_lock:
            run_ids = set(run_id for (run_id, _, _) in self._file_registry.keys())
            return sorted(run_ids)
    
    def get_run_info(self, run_id: int) -> Dict[str, Any]:
        """Get information about stored data for a planet run"""
        with self._registry_lock:
            run_files = [(d, dt) for (rid, d, dt) in self._file_registry.keys() if rid == run_id]
        
        return {
            'run_id': run_id,
            'stored_domains': [d.value for d, _ in run_files],
            'data_types': [dt for _, dt in run_files],
            'file_count': len(run_files)
        }

# Convenience functions
def get_storage_manager(config: StorageConfig = None) -> MultiModalStorage:
    """Get storage manager instance"""
    return MultiModalStorage(config)

async def create_example_data(storage: MultiModalStorage, run_id: int):
    """Create example data for testing storage system"""
    logger.info(f"[TEST] Creating example data for run {run_id}")
    
    # Create example climate datacube
    time_steps = 10
    lat_points = 32
    lon_points = 64
    lev_points = 10
    
    # Create coordinate arrays
    time = np.arange(time_steps, dtype=np.float32)
    lat = np.linspace(-90, 90, lat_points, dtype=np.float32)
    lon = np.linspace(-180, 180, lon_points, dtype=np.float32)
    lev = np.linspace(1000, 1, lev_points, dtype=np.float32)
    
    # Create data arrays
    shape = (time_steps, lat_points, lon_points, lev_points)
    temp = 280 + 20 * np.random.random(shape).astype(np.float32)
    humidity = 0.01 * np.random.random(shape).astype(np.float32)
    
    climate_data = {
        'temperature': temp,
        'humidity': humidity
    }
    
    dimensions = {
        'time': time,
        'lat': lat,
        'lon': lon,
        'lev': lev
    }
    
    await storage.store_climate_datacube(run_id, climate_data, dimensions)
    
    # Create example biological network
    n_nodes = 50
    network_data = {
        'adjacency_matrix': np.random.choice([0, 1], size=(n_nodes, n_nodes), p=[0.9, 0.1]).astype(np.float32),
        'node_features': np.random.random((n_nodes, 10)).astype(np.float32),
        'node_names': [f"metabolite_{i:03d}" for i in range(n_nodes)],
        'node_types': np.random.choice(['substrate', 'product', 'enzyme'], size=n_nodes)
    }
    
    await storage.store_biological_network(run_id, network_data, "metabolic")
    
    # Create example spectrum
    n_points = 5000
    wavelengths = np.linspace(0.5, 30.0, n_points, dtype=np.float32)
    flux = (np.exp(-((wavelengths - 10.0) / 5.0)**2) + 
            0.1 * np.random.random(n_points)).astype(np.float32)
    
    spectrum_metadata = {
        'planet_type': 'rocky',
        'atmosphere_model': 'clear_sky',
        'observation_geometry': 'transit'
    }
    
    await storage.store_spectrum(run_id, wavelengths, flux, 
                                resolution=100000, instrument="PSG",
                                metadata=spectrum_metadata)
    
    logger.info(f"[OK] Created example data for run {run_id}")

if __name__ == "__main__":
    import asyncio
    
    # Test the storage system
    async def test_storage():
        logger.info("[TEST] Testing Multi-Modal Storage System")
        
        # Initialize storage
        config = StorageConfig(
            storage_root=Path("data/test_planet_runs"),
            max_memory_cache_gb=1.0
        )
        storage = MultiModalStorage(config)
        
        # Create test data for multiple runs
        test_run_ids = [1, 2, 3]
        
        for run_id in test_run_ids:
            await create_example_data(storage, run_id)
        
        # Test loading data
        logger.info("[PROC] Testing data loading...")
        
        for run_id in test_run_ids:
            try:
                # Load climate data
                climate_data = await storage.load_climate_datacube(run_id)
                temp_shape = climate_data['data']['temperature'].shape
                logger.info(f"  Run {run_id} - Climate temperature shape: {temp_shape}")
                
                # Load biological network
                bio_data = await storage.load_biological_network(run_id)
                adj_shape = bio_data['adjacency_matrix'].shape
                logger.info(f"  Run {run_id} - Network adjacency shape: {adj_shape}")
                
                # Load spectrum
                spectrum = await storage.load_spectrum(run_id)
                n_wavelengths = len(spectrum['wavelengths'])
                logger.info(f"  Run {run_id} - Spectrum points: {n_wavelengths}")
                
                # Test wavelength filtering
                filtered_spectrum = await storage.load_spectrum(
                    run_id, wavelength_range=(5.0, 15.0)
                )
                n_filtered = len(filtered_spectrum['wavelengths'])
                logger.info(f"  Run {run_id} - Filtered spectrum points: {n_filtered}")
                
            except Exception as e:
                logger.error(f"Error loading data for run {run_id}: {e}")
        
        # Test cache performance
        logger.info("[FAST] Testing cache performance...")
        start_time = time.time()
        
        # Load same data multiple times (should hit cache)
        for _ in range(3):
            for run_id in test_run_ids:
                await storage.load_climate_datacube(run_id)
                await storage.load_biological_network(run_id)
                await storage.load_spectrum(run_id)
        
        cache_time = time.time() - start_time
        logger.info(f"Cache performance test completed in {cache_time:.2f} seconds")
        
        # Show comprehensive statistics
        stats = storage.get_storage_statistics()
        print("\n" + "="*60)
        print("🗄️ MULTI-MODAL STORAGE STATISTICS")
        print("="*60)
        print(f"Registered files: {stats['registry_size']}")
        print(f"Cache size: {stats['cache_size']} items")
        print(f"Total cache hits: {stats['total_cache_hits']}")
        print(f"Storage root: {stats['storage_root']}")
        print(f"Supported formats: {', '.join(stats['supported_formats'])}")
        
        print("\nFiles by domain:")
        for domain, count in stats['domain_file_counts'].items():
            print(f"  {domain.title()}: {count}")
        
        print("\nStored planet runs:")
        stored_runs = storage.list_stored_runs()
        print(f"  Run IDs: {stored_runs}")
        
        for run_id in stored_runs:
            run_info = storage.get_run_info(run_id)
            print(f"  Run {run_id}: {run_info['file_count']} files, "
                  f"domains: {', '.join(run_info['stored_domains'])}")
        
        print("="*60)
        
        # Clean up cache
        storage.clear_cache()
        logger.info("[OK] Multi-Modal Storage test completed successfully!")
    
    # Run test
    asyncio.run(test_storage()) 
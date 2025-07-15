#!/usr/bin/env python3
"""
Multi-Modal Storage Layer (Fixed)
=================================

Optimized storage architecture for multi-disciplinary scientific data with
intelligent format selection, caching strategies, and access patterns.

Storage Strategy:
- Climate 4D datacubes: ZARR (chunked, cloud-native, parallel access)
- Biological networks: NPZ (numpy arrays, fast loading, small files)
- High-resolution spectra: HDF5 (columnar, random access, compression)
- Telescope observations: FITS (preserve WCS, astronomical standards)
- Metadata/parameters: JSON (human-readable, flexible schema)
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
from contextlib import contextmanager
import tempfile
import shutil

# Core dependencies
try:
    import h5py
    HDF5_SUPPORT = True
except ImportError:
    HDF5_SUPPORT = False
    logger.warning("h5py not available - HDF5 support disabled")

try:
    import zarr
    import xarray as xr
    ZARR_SUPPORT = True
except ImportError:
    ZARR_SUPPORT = False
    logger.warning("zarr/xarray not available - Zarr support disabled")

try:
    from astropy.io import fits
    FITS_SUPPORT = True
except ImportError:
    FITS_SUPPORT = False
    logger.warning("astropy not available - FITS support disabled")

try:
    import networkx as nx
    NETWORKX_SUPPORT = True
except ImportError:
    NETWORKX_SUPPORT = False
    logger.warning("networkx not available - Graph support disabled")

# Cloud storage support
try:
    import fsspec
    CLOUD_SUPPORT = True
except ImportError:
    CLOUD_SUPPORT = False

# Local imports
try:
    from .planet_run_primary_key_system import (
        PlanetRunManager, DataDomain, DataFormat, 
        get_planet_run_manager
    )
except ImportError:
    # Fallback definitions for testing
    from enum import Enum
    
    class DataDomain(Enum):
        BIOLOGY = "biology"
        ASTRONOMY = "astronomy"
        CLIMATE = "climate"
        SPECTROSCOPY = "spectroscopy"
        PHYSICS = "physics"
        OBSERVATIONS = "observations"
    
    class DataFormat(Enum):
        ZARR = "zarr"
        NPZ = "npz"
        HDF5 = "hdf5"
        FITS = "fits"
        JSON = "json"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StorageTier(Enum):
    """Storage performance tiers"""
    HOT = "hot"           # SSD, frequent access
    WARM = "warm"         # HDD, occasional access  
    COLD = "cold"         # Cloud, archival
    FROZEN = "frozen"     # Tape, long-term archive

class CompressionLevel(Enum):
    """Compression level options"""
    NONE = "none"
    LOW = "low"           # Fast compression
    MEDIUM = "medium"     # Balanced
    HIGH = "high"         # Maximum compression

@dataclass
class StorageConfig:
    """Configuration for multi-modal storage"""
    # Base storage paths
    storage_root: Path = Path("data/planet_runs")
    cache_root: Path = Path("data/cache")
    temp_root: Path = Path("data/temp")
    
    # Performance settings
    max_memory_cache_gb: float = 8.0
    max_disk_cache_gb: float = 50.0
    parallel_workers: int = 4
    
    # Compression settings
    default_compression: CompressionLevel = CompressionLevel.MEDIUM
    zarr_compressor: str = "blosc"
    hdf5_compression: str = "gzip"
    
    # Chunk sizes for optimal performance
    zarr_chunks: Dict[str, int] = field(default_factory=lambda: {
        "time": 10, "lat": 32, "lon": 32, "lev": 10
    })
    
    # Data validation
    verify_checksums: bool = True
    auto_repair: bool = True

class MultiModalStorage:
    """
    Central storage manager for multi-modal scientific data
    
    Handles optimal storage and retrieval across domains:
    - Climate: Zarr stores with optimized chunking
    - Biology: NPZ arrays for graphs and networks
    - Spectroscopy: HDF5 for high-resolution data
    - Observations: FITS with WCS preservation
    - Metadata: JSON with schema validation
    """
    
    def __init__(self, config: StorageConfig = None):
        self.config = config or StorageConfig()
        
        # Try to get planet run manager
        try:
            self.planet_run_manager = get_planet_run_manager()
        except:
            logger.warning("Planet run manager not available - operating in standalone mode")
            self.planet_run_manager = None
        
        # Initialize storage directories
        self._initialize_storage()
        
        # Cache management
        self._memory_cache = {}
        self._cache_lock = Lock()
        self._access_stats = {}
        
        # Thread pool for parallel I/O
        self._executor = ThreadPoolExecutor(max_workers=self.config.parallel_workers)
        
        logger.info(f"🗄️ Multi-Modal Storage initialized: {self.config.storage_root}")
    
    def _initialize_storage(self):
        """Initialize storage directory structure"""
        directories = [
            self.config.storage_root,
            self.config.cache_root,
            self.config.temp_root,
            self.config.cache_root / "zarr",
            self.config.cache_root / "npz", 
            self.config.cache_root / "hdf5",
            self.config.cache_root / "fits",
            self.config.cache_root / "metadata"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    # === CLIMATE DATA (ZARR) ===
    
    async def store_climate_datacube(self,
                                   run_id: int,
                                   data_dict: Dict[str, np.ndarray],
                                   dimensions: Dict[str, np.ndarray],
                                   variables: List[str] = None) -> Path:
        """
        Store climate datacube in optimized format
        
        Args:
            run_id: Planet run identifier
            data_dict: Dictionary of variable_name -> data_array
            dimensions: Dictionary of dimension_name -> coordinate_array
            variables: List of variables to store (None = all)
            
        Returns:
            Path to stored data
        """
        logger.info(f"📦 Storing climate datacube for run {run_id}")
        
        run_dir = self.config.storage_root / f"run_{run_id:06d}"
        
        if ZARR_SUPPORT:
            # Use Zarr for optimal performance
            zarr_path = run_dir / "climate" / "datacube.zarr"
            zarr_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create zarr group
            store = zarr.DirectoryStore(str(zarr_path))
            root = zarr.group(store=store, overwrite=True)
            
            # Store coordinates
            for dim_name, coord_array in dimensions.items():
                root.create_dataset(
                    f"coords/{dim_name}",
                    data=coord_array,
                    chunks=True,
                    compressor=zarr.Blosc(cname='lz4')
                )
            
            # Store variables
            for var_name, var_data in data_dict.items():
                if variables is None or var_name in variables:
                    root.create_dataset(
                        f"data/{var_name}",
                        data=var_data.astype(np.float32),
                        chunks=True,
                        compressor=zarr.Blosc(cname='lz4')
                    )
            
            storage_path = zarr_path
            
        else:
            # Fallback to NPZ
            npz_path = run_dir / "climate" / "datacube.npz"
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Combine data and dimensions
            save_dict = {**data_dict, **{f"coord_{k}": v for k, v in dimensions.items()}}
            
            if variables:
                save_dict = {k: v for k, v in save_dict.items() 
                           if k.startswith("coord_") or k in variables}
            
            np.savez_compressed(npz_path, **save_dict)
            storage_path = npz_path
        
        # Register with planet run manager if available
        if self.planet_run_manager:
            try:
                self.planet_run_manager.register_data_file(
                    run_id=run_id,
                    domain=DataDomain.CLIMATE,
                    data_type="4d_datacube",
                    file_path=storage_path,
                    file_format=DataFormat.ZARR if ZARR_SUPPORT else DataFormat.NPZ,
                    dimensions={
                        'variables': list(data_dict.keys()),
                        'dimensions': list(dimensions.keys())
                    },
                    variables=list(data_dict.keys())
                )
            except Exception as e:
                logger.warning(f"Failed to register with planet run manager: {e}")
        
        logger.info(f"✅ Stored climate datacube: {storage_path}")
        return storage_path
    
    async def load_climate_datacube(self, run_id: int, variables: List[str] = None) -> Dict[str, Any]:
        """Load climate datacube with optional variable filtering"""
        # Check cache first
        cache_key = f"climate_{run_id}_{hash(tuple(variables) if variables else None)}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        
        run_dir = self.config.storage_root / f"run_{run_id:06d}"
        
        # Try different storage formats
        zarr_path = run_dir / "climate" / "datacube.zarr"
        npz_path = run_dir / "climate" / "datacube.npz"
        
        result = None
        
        if ZARR_SUPPORT and zarr_path.exists():
            # Load from Zarr
            store = zarr.DirectoryStore(str(zarr_path))
            root = zarr.group(store=store, mode='r')
            
            # Load coordinates
            dimensions = {}
            if 'coords' in root:
                for coord_name in root['coords']:
                    dimensions[coord_name] = root[f'coords/{coord_name}'][:]
            
            # Load data variables
            data_dict = {}
            if 'data' in root:
                for var_name in root['data']:
                    if variables is None or var_name in variables:
                        data_dict[var_name] = root[f'data/{var_name}'][:]
            
            result = {'data': data_dict, 'dimensions': dimensions}
            
        elif npz_path.exists():
            # Load from NPZ
            data = dict(np.load(npz_path))
            
            # Separate data and coordinates
            data_dict = {}
            dimensions = {}
            
            for key, value in data.items():
                if key.startswith('coord_'):
                    coord_name = key[6:]  # Remove 'coord_' prefix
                    dimensions[coord_name] = value
                else:
                    if variables is None or key in variables:
                        data_dict[key] = value
            
            result = {'data': data_dict, 'dimensions': dimensions}
        
        if result is None:
            raise FileNotFoundError(f"Climate data not found for run {run_id}")
        
        # Cache result
        self._add_to_cache(cache_key, result)
        return result
    
    # === BIOLOGICAL DATA (NPZ) ===
    
    async def store_biological_network(self,
                                     run_id: int,
                                     network_data: Dict[str, Any],
                                     network_type: str = "metabolic") -> Path:
        """Store biological network data in NPZ format"""
        logger.info(f"🧬 Storing {network_type} network for run {run_id}")
        
        run_dir = self.config.storage_root / f"run_{run_id:06d}"
        npz_path = run_dir / "biosphere" / f"{network_type}_network.npz"
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert NetworkX graph to arrays if needed
        if NETWORKX_SUPPORT and 'graph' in network_data and hasattr(network_data['graph'], 'nodes'):
            graph = network_data['graph']
            adjacency = nx.adjacency_matrix(graph).toarray()
            node_features = np.array([graph.nodes[n].get('features', []) 
                                    for n in graph.nodes()])
            
            network_data = {
                'adjacency_matrix': adjacency,
                'node_features': node_features,
                'node_names': list(graph.nodes()),
                **{k: v for k, v in network_data.items() if k != 'graph'}
            }
        
        # Save with compression
        np.savez_compressed(npz_path, **network_data)
        
        # Register with planet run manager if available
        if self.planet_run_manager:
            try:
                self.planet_run_manager.register_data_file(
                    run_id=run_id,
                    domain=DataDomain.BIOLOGY,
                    data_type=f"{network_type}_network",
                    file_path=npz_path,
                    file_format=DataFormat.NPZ,
                    dimensions={
                        'adjacency_shape': network_data.get('adjacency_matrix', np.array([])).shape,
                        'num_nodes': len(network_data.get('node_names', [])),
                    },
                    variables=list(network_data.keys())
                )
            except Exception as e:
                logger.warning(f"Failed to register biological network: {e}")
        
        logger.info(f"✅ Stored biological network: {npz_path}")
        return npz_path
    
    async def load_biological_network(self, 
                                    run_id: int,
                                    network_type: str = "metabolic") -> Dict[str, Any]:
        """Load biological network data"""
        # Check cache
        cache_key = f"bio_{run_id}_{network_type}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        
        run_dir = self.config.storage_root / f"run_{run_id:06d}"
        npz_path = run_dir / "biosphere" / f"{network_type}_network.npz"
        
        if not npz_path.exists():
            raise FileNotFoundError(f"Biological network not found for run {run_id}")
        
        # Load NPZ data
        data = dict(np.load(npz_path, allow_pickle=True))
        
        # Cache result
        self._add_to_cache(cache_key, data)
        
        return data
    
    # === SPECTROSCOPY DATA (HDF5) ===
    
    async def store_spectrum(self,
                           run_id: int,
                           wavelengths: np.ndarray,
                           flux: np.ndarray,
                           resolution: float,
                           instrument: str = "PSG",
                           metadata: Dict[str, Any] = None) -> Path:
        """Store high-resolution spectrum in HDF5 or NPZ format"""
        logger.info(f"🌈 Storing spectrum for run {run_id} (R={resolution})")
        
        run_dir = self.config.storage_root / f"run_{run_id:06d}"
        
        if HDF5_SUPPORT:
            # Use HDF5 for optimal spectrum storage
            hdf5_path = run_dir / "spectroscopy" / f"spectrum_{instrument.lower()}.h5"
            hdf5_path.parent.mkdir(parents=True, exist_ok=True)
            
            with h5py.File(hdf5_path, 'w') as f:
                # Main spectral data
                f.create_dataset(
                    'wavelengths', 
                    data=wavelengths.astype(np.float32),
                    compression=self.config.hdf5_compression,
                    shuffle=True,
                    chunks=True
                )
                
                f.create_dataset(
                    'flux',
                    data=flux.astype(np.float32), 
                    compression=self.config.hdf5_compression,
                    shuffle=True,
                    chunks=True
                )
                
                # Metadata
                f.attrs['resolution'] = resolution
                f.attrs['instrument'] = instrument
                f.attrs['num_points'] = len(wavelengths)
                f.attrs['wavelength_min'] = float(wavelengths.min())
                f.attrs['wavelength_max'] = float(wavelengths.max())
                f.attrs['created_at'] = datetime.now(timezone.utc).isoformat()
                
                if metadata:
                    for key, value in metadata.items():
                        if isinstance(value, (str, int, float, bool)):
                            f.attrs[key] = value
            
            storage_path = hdf5_path
            
        else:
            # Fallback to NPZ
            npz_path = run_dir / "spectroscopy" / f"spectrum_{instrument.lower()}.npz"
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            
            save_data = {
                'wavelengths': wavelengths.astype(np.float32),
                'flux': flux.astype(np.float32),
                'resolution': resolution,
                'instrument': instrument,
                'num_points': len(wavelengths),
                'wavelength_min': float(wavelengths.min()),
                'wavelength_max': float(wavelengths.max())
            }
            
            if metadata:
                save_data.update(metadata)
            
            np.savez_compressed(npz_path, **save_data)
            storage_path = npz_path
        
        # Register with planet run manager if available
        if self.planet_run_manager:
            try:
                self.planet_run_manager.register_data_file(
                    run_id=run_id,
                    domain=DataDomain.SPECTROSCOPY,
                    data_type=f"spectrum_{instrument.lower()}",
                    file_path=storage_path,
                    file_format=DataFormat.HDF5 if HDF5_SUPPORT else DataFormat.NPZ,
                    dimensions={
                        'num_points': len(wavelengths),
                        'wavelength_range': [float(wavelengths.min()), float(wavelengths.max())],
                        'resolution': resolution
                    },
                    variables=['wavelengths', 'flux']
                )
            except Exception as e:
                logger.warning(f"Failed to register spectrum: {e}")
        
        logger.info(f"✅ Stored spectrum: {storage_path}")
        return storage_path
    
    async def load_spectrum(self,
                          run_id: int,
                          instrument: str = "PSG") -> Dict[str, Any]:
        """Load high-resolution spectrum"""
        # Check cache
        cache_key = f"spectrum_{run_id}_{instrument}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        
        run_dir = self.config.storage_root / f"run_{run_id:06d}"
        
        # Try different formats
        hdf5_path = run_dir / "spectroscopy" / f"spectrum_{instrument.lower()}.h5"
        npz_path = run_dir / "spectroscopy" / f"spectrum_{instrument.lower()}.npz"
        
        result = None
        
        if HDF5_SUPPORT and hdf5_path.exists():
            # Load from HDF5
            with h5py.File(hdf5_path, 'r') as f:
                result = {
                    'wavelengths': f['wavelengths'][:],
                    'flux': f['flux'][:],
                    'metadata': dict(f.attrs)
                }
                
        elif npz_path.exists():
            # Load from NPZ
            data = dict(np.load(npz_path, allow_pickle=True))
            
            wavelengths = data.pop('wavelengths', None)
            flux = data.pop('flux', None)
            
            if wavelengths is not None and flux is not None:
                result = {
                    'wavelengths': wavelengths,
                    'flux': flux,
                    'metadata': data
                }
        
        if result is None:
            raise FileNotFoundError(f"Spectrum not found for run {run_id}, instrument {instrument}")
        
        # Cache result
        self._add_to_cache(cache_key, result)
        
        return result
    
    # === UTILITY METHODS ===
    
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
            if len(self._memory_cache) > 100:  # Max 100 items
                # Remove oldest items
                sorted_items = sorted(
                    self._access_stats.items(),
                    key=lambda x: x[1]['last_access']
                )
                
                for old_key, _ in sorted_items[:20]:  # Remove oldest 20
                    if old_key in self._memory_cache:
                        del self._memory_cache[old_key]
                    if old_key in self._access_stats:
                        del self._access_stats[old_key]
            
            self._memory_cache[key] = (item, time.time())
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics"""
        stats = {
            'cache_size': len(self._memory_cache),
            'total_cache_hits': sum(s['count'] for s in self._access_stats.values()),
            'storage_root': str(self.config.storage_root),
            'supported_formats': {
                'zarr': ZARR_SUPPORT,
                'hdf5': HDF5_SUPPORT,
                'fits': FITS_SUPPORT,
                'networkx': NETWORKX_SUPPORT,
                'cloud': CLOUD_SUPPORT
            }
        }
        
        return stats

# Convenience functions
def get_storage_manager(config: StorageConfig = None) -> MultiModalStorage:
    """Get storage manager"""
    return MultiModalStorage(config)

async def create_example_data(storage: MultiModalStorage, run_id: int):
    """Create example data for testing storage system"""
    logger.info(f"🧪 Creating example data for run {run_id}")
    
    # Create example climate datacube
    time = np.arange(10)
    lat = np.linspace(-90, 90, 32)
    lon = np.linspace(-180, 180, 64)
    lev = np.linspace(1000, 1, 10)
    
    # Create meshgrid for proper broadcasting
    T, LAT, LON, LEV = np.meshgrid(time, lat, lon, lev, indexing='ij')
    
    temp = 280 + 20 * np.random.random(T.shape)
    humidity = 0.01 * np.random.random(T.shape)
    
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
    network_data = {
        'adjacency_matrix': np.random.choice([0, 1], size=(50, 50), p=[0.9, 0.1]).astype(np.float32),
        'node_features': np.random.random((50, 10)).astype(np.float32),
        'node_names': [f"metabolite_{i:03d}" for i in range(50)]
    }
    
    await storage.store_biological_network(run_id, network_data, "metabolic")
    
    # Create example spectrum
    wavelengths = np.linspace(0.5, 30.0, 5000)
    flux = np.exp(-((wavelengths - 10.0) / 5.0)**2) + 0.1 * np.random.random(len(wavelengths))
    
    await storage.store_spectrum(run_id, wavelengths, flux, resolution=100000, instrument="PSG")
    
    logger.info(f"✅ Created example data for run {run_id}")

if __name__ == "__main__":
    import asyncio
    
    # Test the storage system
    async def test_storage():
        # Initialize storage
        config = StorageConfig(
            storage_root=Path("data/test_planet_runs"),
            max_memory_cache_gb=2.0
        )
        storage = MultiModalStorage(config)
        
        # Create test data for a few runs
        test_run_ids = [1, 2, 3]
        
        for run_id in test_run_ids:
            await create_example_data(storage, run_id)
        
        # Test loading data
        logger.info("🔄 Testing data loading...")
        
        for run_id in test_run_ids:
            try:
                # Load climate data
                climate_data = await storage.load_climate_datacube(run_id)
                logger.info(f"  Climate data variables: {list(climate_data['data'].keys())}")
                
                # Load biological network
                bio_data = await storage.load_biological_network(run_id)
                logger.info(f"  Network size: {bio_data['adjacency_matrix'].shape}")
                
                # Load spectrum
                spectrum = await storage.load_spectrum(run_id)
                logger.info(f"  Spectrum points: {len(spectrum['wavelengths'])}")
                
            except Exception as e:
                logger.error(f"Error loading data for run {run_id}: {e}")
        
        # Show statistics
        stats = storage.get_storage_statistics()
        print("\n" + "="*60)
        print("🗄️ MULTI-MODAL STORAGE STATISTICS")
        print("="*60)
        print(f"Cache size: {stats['cache_size']} items")
        print(f"Cache hits: {stats['total_cache_hits']}")
        print(f"Storage root: {stats['storage_root']}")
        print("\nSupported formats:")
        for fmt, supported in stats['supported_formats'].items():
            print(f"  {fmt.upper()}: {'✅' if supported else '❌'}")
        print("="*60)
    
    # Run test
    asyncio.run(test_storage()) 
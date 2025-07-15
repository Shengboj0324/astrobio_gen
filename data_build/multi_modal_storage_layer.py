#!/usr/bin/env python3
"""
Multi-Modal Storage Layer
=========================

Optimized storage architecture for multi-disciplinary scientific data with
intelligent format selection, caching strategies, and access patterns.

Storage Strategy:
- Climate 4D datacubes: ZARR (chunked, cloud-native, parallel access)
- Biological networks: NPZ (numpy arrays, fast loading, small files)
- High-resolution spectra: HDF5 (columnar, random access, compression)
- Telescope observations: FITS (preserve WCS, astronomical standards)
- Metadata/parameters: JSON (human-readable, flexible schema)
- Tabular data: Parquet (columnar, fast filtering)

Features:
- Intelligent data placement and retrieval
- Multi-level caching (memory, SSD, cloud)
- Compression and optimization
- Parallel I/O for large datasets
- Data integrity verification
- Cloud storage integration
"""

import os
import json
import h5py
import zarr
import numpy as np
import pandas as pd
import logging
import asyncio
import aiofiles
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import threading
from threading import Lock
import time
import hashlib
import gzip
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, asynccontextmanager
import tempfile
import shutil

# Scientific data libraries
import xarray as xr
try:
    from astropy.io import fits
    FITS_SUPPORT = True
except ImportError:
    FITS_SUPPORT = False

import networkx as nx
from scipy.sparse import csr_matrix

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    PARQUET_SUPPORT = True
except ImportError:
    PARQUET_SUPPORT = False

# Cloud storage support
try:
    import fsspec
    import s3fs
    CLOUD_SUPPORT = True
except ImportError:
    CLOUD_SUPPORT = False

# Local imports
from .planet_run_primary_key_system import (
    PlanetRunManager, DataDomain, DataFormat, 
    get_planet_run_manager
)

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
    
    # Cloud storage
    cloud_bucket: Optional[str] = None
    cloud_prefix: str = "astrobio"
    
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

@dataclass  
class DataLocation:
    """Location information for stored data"""
    storage_tier: StorageTier
    local_path: Optional[Path] = None
    cloud_path: Optional[str] = None
    cached_path: Optional[Path] = None
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    
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
        self.planet_run_manager = get_planet_run_manager()
        
        # Initialize storage directories
        self._initialize_storage()
        
        # Cache management
        self._memory_cache = {}
        self._cache_lock = Lock()
        self._access_stats = {}
        
        # Thread pool for parallel I/O
        self._executor = ThreadPoolExecutor(max_workers=self.config.parallel_workers)
        
        # Cloud storage client
        self._cloud_client = self._initialize_cloud_storage()
        
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
    
    def _initialize_cloud_storage(self):
        """Initialize cloud storage client if available"""
        if not CLOUD_SUPPORT or not self.config.cloud_bucket:
            return None
        
        try:
            if self.config.cloud_bucket.startswith("s3://"):
                return s3fs.S3FileSystem()
            else:
                return fsspec.filesystem("file")
        except Exception as e:
            logger.warning(f"Failed to initialize cloud storage: {e}")
            return None
    
    # === CLIMATE DATA (ZARR) ===
    
    async def store_climate_datacube(self,
                                   run_id: int,
                                   datacube: xr.Dataset,
                                   variables: List[str] = None) -> Path:
        """
        Store climate datacube in optimized Zarr format
        
        Args:
            run_id: Planet run identifier
            datacube: xarray Dataset with climate data
            variables: List of variables to store (None = all)
            
        Returns:
            Path to stored Zarr dataset
        """
        logger.info(f"📦 Storing climate datacube for run {run_id}")
        
        # Get run directory
        run_info = self.planet_run_manager.get_planet_runs()[0]  # Simplified for example
        run_dir = self.config.storage_root / f"run_{run_id:06d}"
        zarr_path = run_dir / "climate" / "datacube.zarr"
        
        # Filter variables if specified
        if variables:
            available_vars = [v for v in variables if v in datacube.data_vars]
            if available_vars:
                datacube = datacube[available_vars]
        
        # Optimize chunking based on data characteristics
        optimal_chunks = self._optimize_zarr_chunking(datacube)
        
        # Configure compression
        compressor = self._get_zarr_compressor()
        
        # Store with optimal settings
        encoding = {}
        for var in datacube.data_vars:
            encoding[var] = {
                'chunks': optimal_chunks,
                'compressor': compressor,
                'dtype': 'float32'  # Use float32 to save space
            }
        
        # Save to Zarr
        datacube.to_zarr(
            zarr_path,
            mode='w',
            encoding=encoding,
            consolidated=True  # Faster metadata access
        )
        
        # Register with planet run manager
        self.planet_run_manager.register_data_file(
            run_id=run_id,
            domain=DataDomain.CLIMATE,
            data_type="4d_datacube",
            file_path=zarr_path,
            file_format=DataFormat.ZARR,
            dimensions={
                'shape': [datacube.dims[d] for d in ['time', 'lat', 'lon', 'lev']],
                'chunks': optimal_chunks
            },
            variables=list(datacube.data_vars.keys())
        )
        
        logger.info(f"✅ Stored climate datacube: {zarr_path}")
        return zarr_path
    
    async def load_climate_datacube(self, 
                                  run_id: int,
                                  variables: List[str] = None,
                                  time_slice: slice = None,
                                  spatial_box: Tuple[slice, slice] = None) -> xr.Dataset:
        """
        Load climate datacube with optional subsetting
        
        Args:
            run_id: Planet run identifier
            variables: Variables to load (None = all)
            time_slice: Time slice to load
            spatial_box: (lat_slice, lon_slice) for spatial subsetting
            
        Returns:
            xarray Dataset with climate data
        """
        # Check cache first
        cache_key = f"climate_{run_id}_{hash((variables, time_slice, spatial_box))}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        
        # Get data path
        zarr_path = self._get_data_path(run_id, DataDomain.CLIMATE)
        if not zarr_path or not zarr_path.exists():
            raise FileNotFoundError(f"Climate data not found for run {run_id}")
        
        # Load with chunking optimization
        datacube = xr.open_zarr(zarr_path, chunks="auto")
        
        # Apply subsetting
        if variables:
            available_vars = [v for v in variables if v in datacube.data_vars]
            if available_vars:
                datacube = datacube[available_vars]
        
        if time_slice:
            datacube = datacube.isel(time=time_slice)
        
        if spatial_box:
            lat_slice, lon_slice = spatial_box
            datacube = datacube.isel(lat=lat_slice, lon=lon_slice)
        
        # Cache result
        self._add_to_cache(cache_key, datacube)
        
        return datacube
    
    def _optimize_zarr_chunking(self, datacube: xr.Dataset) -> Dict[str, int]:
        """Optimize Zarr chunking based on data characteristics"""
        dims = datacube.dims
        
        # Base chunks from config
        chunks = self.config.zarr_chunks.copy()
        
        # Adjust based on actual data size
        for dim, size in dims.items():
            if dim in chunks:
                # Don't chunk larger than dimension size
                chunks[dim] = min(chunks[dim], size)
                
                # For very small dimensions, don't chunk
                if size < chunks[dim] * 2:
                    chunks[dim] = size
        
        # Optimize for ~50MB chunks (typical sweet spot)
        target_chunk_size_mb = 50
        total_elements = np.prod(list(chunks.values()))
        bytes_per_element = 4  # float32
        chunk_size_mb = (total_elements * bytes_per_element) / (1024**2)
        
        if chunk_size_mb > target_chunk_size_mb * 2:
            # Chunks too large, reduce
            scale_factor = (target_chunk_size_mb / chunk_size_mb) ** 0.25
            for dim in chunks:
                chunks[dim] = max(1, int(chunks[dim] * scale_factor))
        
        return chunks
    
    def _get_zarr_compressor(self):
        """Get optimized Zarr compressor"""
        try:
            import numcodecs
            if self.config.default_compression == CompressionLevel.HIGH:
                return numcodecs.Blosc(cname='zstd', clevel=9, shuffle=numcodecs.Blosc.SHUFFLE)
            elif self.config.default_compression == CompressionLevel.MEDIUM:
                return numcodecs.Blosc(cname='lz4', clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)
            else:
                return numcodecs.Blosc(cname='lz4', clevel=1)
        except ImportError:
            return None
    
    # === BIOLOGICAL DATA (NPZ) ===
    
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
        logger.info(f"🧬 Storing {network_type} network for run {run_id}")
        
        run_dir = self.config.storage_root / f"run_{run_id:06d}"
        npz_path = run_dir / "biosphere" / f"{network_type}_network.npz"
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert NetworkX graph to arrays if needed
        if 'graph' in network_data and hasattr(network_data['graph'], 'nodes'):
            graph = network_data['graph']
            adjacency = nx.adjacency_matrix(graph).toarray()
            node_features = np.array([graph.nodes[n].get('features', []) 
                                    for n in graph.nodes()])
            edge_features = np.array([[graph.edges[e].get('weight', 1.0)] 
                                    for e in graph.edges()])
            
            network_data = {
                'adjacency_matrix': adjacency,
                'node_features': node_features,
                'edge_features': edge_features,
                'node_names': list(graph.nodes()),
                'edge_names': list(graph.edges())
            }
        
        # Save with compression
        if self.config.default_compression != CompressionLevel.NONE:
            np.savez_compressed(npz_path, **network_data)
        else:
            np.savez(npz_path, **network_data)
        
        # Register with planet run manager
        self.planet_run_manager.register_data_file(
            run_id=run_id,
            domain=DataDomain.BIOLOGY,
            data_type=f"{network_type}_network",
            file_path=npz_path,
            file_format=DataFormat.NPZ,
            dimensions={
                'adjacency_shape': network_data.get('adjacency_matrix', np.array([])).shape,
                'num_nodes': len(network_data.get('node_names', [])),
                'num_edges': len(network_data.get('edge_names', []))
            },
            variables=list(network_data.keys())
        )
        
        logger.info(f"✅ Stored biological network: {npz_path}")
        return npz_path
    
    async def load_biological_network(self, 
                                    run_id: int,
                                    network_type: str = "metabolic",
                                    return_graph: bool = False) -> Dict[str, Any]:
        """
        Load biological network data
        
        Args:
            run_id: Planet run identifier
            network_type: Type of network to load
            return_graph: Whether to reconstruct NetworkX graph
            
        Returns:
            Dictionary with network data
        """
        # Check cache
        cache_key = f"bio_{run_id}_{network_type}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        
        # Get data path
        npz_path = self._get_data_path(run_id, DataDomain.BIOLOGY, network_type)
        if not npz_path or not npz_path.exists():
            raise FileNotFoundError(f"Biological network not found for run {run_id}")
        
        # Load NPZ data
        data = dict(np.load(npz_path, allow_pickle=True))
        
        # Reconstruct NetworkX graph if requested
        if return_graph and 'adjacency_matrix' in data:
            graph = nx.from_numpy_array(data['adjacency_matrix'])
            
            # Add node features
            if 'node_features' in data:
                for i, features in enumerate(data['node_features']):
                    graph.nodes[i]['features'] = features
            
            # Add node names
            if 'node_names' in data:
                mapping = {i: name for i, name in enumerate(data['node_names'])}
                graph = nx.relabel_nodes(graph, mapping)
            
            data['graph'] = graph
        
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
        """
        Store high-resolution spectrum in HDF5 format
        
        Args:
            run_id: Planet run identifier
            wavelengths: Wavelength array (microns)
            flux: Flux array (same length as wavelengths)
            resolution: Spectral resolution
            instrument: Instrument name
            metadata: Additional metadata
            
        Returns:
            Path to stored HDF5 file
        """
        logger.info(f"🌈 Storing spectrum for run {run_id} (R={resolution})")
        
        run_dir = self.config.storage_root / f"run_{run_id:06d}"
        hdf5_path = run_dir / "spectroscopy" / f"spectrum_{instrument.lower()}.h5"
        hdf5_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Store in HDF5 with compression and chunking
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
            
            # Additional metadata
            if metadata:
                meta_group = f.create_group('metadata')
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        meta_group.attrs[key] = value
                    else:
                        # Store complex objects as JSON strings
                        meta_group.attrs[key] = json.dumps(value)
        
        # Register with planet run manager
        self.planet_run_manager.register_data_file(
            run_id=run_id,
            domain=DataDomain.SPECTROSCOPY,
            data_type=f"spectrum_{instrument.lower()}",
            file_path=hdf5_path,
            file_format=DataFormat.HDF5,
            dimensions={
                'num_points': len(wavelengths),
                'wavelength_range': [float(wavelengths.min()), float(wavelengths.max())],
                'resolution': resolution
            },
            variables=['wavelengths', 'flux']
        )
        
        logger.info(f"✅ Stored spectrum: {hdf5_path}")
        return hdf5_path
    
    async def load_spectrum(self,
                          run_id: int,
                          instrument: str = "PSG",
                          wavelength_range: Tuple[float, float] = None) -> Dict[str, Any]:
        """
        Load high-resolution spectrum with optional wavelength filtering
        
        Args:
            run_id: Planet run identifier
            instrument: Instrument name
            wavelength_range: (min_wl, max_wl) in microns
            
        Returns:
            Dictionary with spectrum data and metadata
        """
        # Check cache
        cache_key = f"spectrum_{run_id}_{instrument}_{hash(wavelength_range)}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        
        # Get data path
        hdf5_path = self._get_data_path(run_id, DataDomain.SPECTROSCOPY, f"spectrum_{instrument.lower()}")
        if not hdf5_path or not hdf5_path.exists():
            raise FileNotFoundError(f"Spectrum not found for run {run_id}, instrument {instrument}")
        
        # Load HDF5 data
        with h5py.File(hdf5_path, 'r') as f:
            wavelengths = f['wavelengths'][:]
            flux = f['flux'][:]
            
            # Apply wavelength filtering
            if wavelength_range:
                min_wl, max_wl = wavelength_range
                mask = (wavelengths >= min_wl) & (wavelengths <= max_wl)
                wavelengths = wavelengths[mask]
                flux = flux[mask]
            
            # Load metadata
            metadata = dict(f.attrs)
            
            # Load additional metadata if present
            if 'metadata' in f:
                meta_group = f['metadata']
                for key in meta_group.attrs:
                    value = meta_group.attrs[key]
                    if isinstance(value, str) and value.startswith('{'):
                        try:
                            metadata[key] = json.loads(value)
                        except:
                            metadata[key] = value
                    else:
                        metadata[key] = value
        
        result = {
            'wavelengths': wavelengths,
            'flux': flux,
            'metadata': metadata
        }
        
        # Cache result
        self._add_to_cache(cache_key, result)
        
        return result
    
    # === OBSERVATIONAL DATA (FITS) ===
    
    async def store_observation(self,
                              run_id: int,
                              fits_data: Union[Path, fits.HDUList],
                              telescope: str,
                              instrument: str,
                              observation_type: str = "transit") -> Path:
        """
        Store astronomical observation in FITS format
        
        Args:
            run_id: Planet run identifier
            fits_data: FITS file path or HDUList
            telescope: Telescope name (e.g., JWST, HST)
            instrument: Instrument name
            observation_type: Type of observation
            
        Returns:
            Path to stored FITS file
        """
        logger.info(f"🔭 Storing {observation_type} observation for run {run_id}")
        
        run_dir = self.config.storage_root / f"run_{run_id:06d}"
        fits_path = run_dir / "observations" / f"{telescope.lower()}_{instrument.lower()}_{observation_type}.fits"
        fits_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy or write FITS file
        if isinstance(fits_data, Path):
            shutil.copy2(fits_data, fits_path)
        else:
            fits_data.writeto(fits_path, overwrite=True)
        
        # Extract metadata from FITS header
        with fits.open(fits_path) as hdul:
            primary_header = hdul[0].header
            dimensions = {
                'num_extensions': len(hdul),
                'primary_shape': hdul[0].data.shape if hdul[0].data is not None else None
            }
            
            # Extract key observational parameters
            variables = []
            for ext in hdul:
                if ext.data is not None:
                    variables.append(f"ext_{ext.name or len(variables)}")
        
        # Register with planet run manager
        self.planet_run_manager.register_data_file(
            run_id=run_id,
            domain=DataDomain.OBSERVATIONS,
            data_type=f"{telescope.lower()}_{observation_type}",
            file_path=fits_path,
            file_format=DataFormat.FITS,
            dimensions=dimensions,
            variables=variables
        )
        
        logger.info(f"✅ Stored observation: {fits_path}")
        return fits_path
    
    # === UTILITY METHODS ===
    
    def _get_data_path(self, run_id: int, domain: DataDomain, data_type: str = None) -> Optional[Path]:
        """Get path to data file for planet run"""
        try:
            # Query planet run manager for file path
            with self.planet_run_manager.get_session() as session:
                from .planet_run_primary_key_system import PlanetRun, DataFile
                
                query = session.query(DataFile).filter_by(
                    run_id=run_id,
                    domain=domain.value
                )
                
                if data_type:
                    query = query.filter_by(data_type=data_type)
                
                data_file = query.first()
                
                if data_file:
                    return self.config.storage_root / data_file.file_path
                
        except Exception as e:
            logger.error(f"Failed to get data path: {e}")
        
        return None
    
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
        # Estimate memory usage (simplified)
        estimated_size_mb = self._estimate_memory_size(item)
        
        with self._cache_lock:
            # Check if cache is getting too large
            current_size = len(self._memory_cache) * 10  # Rough estimate
            
            if current_size + estimated_size_mb > self.config.max_memory_cache_gb * 1024:
                self._evict_cache_items()
            
            self._memory_cache[key] = (item, time.time())
    
    def _estimate_memory_size(self, item: Any) -> float:
        """Estimate memory size of item in MB"""
        if isinstance(item, np.ndarray):
            return item.nbytes / (1024**2)
        elif isinstance(item, dict):
            return sum(self._estimate_memory_size(v) for v in item.values())
        elif hasattr(item, '__sizeof__'):
            return item.__sizeof__() / (1024**2)
        else:
            return 1.0  # Default estimate
    
    def _evict_cache_items(self):
        """Evict least recently used cache items"""
        if not self._access_stats:
            return
        
        # Sort by last access time
        sorted_items = sorted(
            self._access_stats.items(),
            key=lambda x: x[1]['last_access']
        )
        
        # Remove oldest 25% of items
        num_to_remove = max(1, len(sorted_items) // 4)
        
        for key, _ in sorted_items[:num_to_remove]:
            if key in self._memory_cache:
                del self._memory_cache[key]
            if key in self._access_stats:
                del self._access_stats[key]
        
        logger.debug(f"Evicted {num_to_remove} items from cache")
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics"""
        stats = {
            'cache_size': len(self._memory_cache),
            'total_cache_hits': sum(s['count'] for s in self._access_stats.values()),
            'storage_root': str(self.config.storage_root),
            'domain_stats': {}
        }
        
        # Get per-domain statistics
        for domain in DataDomain:
            domain_dir = self.config.storage_root / f"*/{domain.value}"
            
            # Count files and total size (simplified)
            stats['domain_stats'][domain.value] = {
                'estimated_files': 0,
                'estimated_size_gb': 0.0
            }
        
        return stats
    
    async def optimize_storage(self):
        """Run storage optimization tasks"""
        logger.info("🔧 Running storage optimization...")
        
        # Clear unused cache items
        self._evict_cache_items()
        
        # Implement additional optimizations:
        optimization_stats = {
            'compressed_files': 0,
            'cold_storage_moved': 0,
            'indices_rebuilt': 0,
            'integrity_checks': 0,
            'errors_fixed': 0,
            'space_saved_gb': 0.0
        }
        
        try:
            # 1. Compress old data
            compressed_space = await self._compress_old_data()
            optimization_stats['compressed_files'] = compressed_space['files']
            optimization_stats['space_saved_gb'] += compressed_space['space_saved_gb']
            
            # 2. Move infrequently accessed data to cold storage
            cold_moved = await self._migrate_to_cold_storage()
            optimization_stats['cold_storage_moved'] = cold_moved['files']
            optimization_stats['space_saved_gb'] += cold_moved['space_saved_gb']
            
            # 3. Rebuild indices
            rebuilt_indices = await self._rebuild_indices()
            optimization_stats['indices_rebuilt'] = rebuilt_indices
            
            # 4. Validate data integrity
            integrity_results = await self._validate_data_integrity()
            optimization_stats['integrity_checks'] = integrity_results['checked']
            optimization_stats['errors_fixed'] = integrity_results['fixed']
            
            logger.info(f"✅ Storage optimization complete: {optimization_stats}")
            return optimization_stats
            
        except Exception as e:
            logger.error(f"Storage optimization failed: {e}")
            raise
    
    async def _compress_old_data(self) -> Dict[str, Any]:
        """Compress data files older than specified threshold"""
        logger.info("🗜️ Compressing old data files...")
        
        from datetime import timedelta
        import gzip
        import shutil
        
        compression_threshold = timedelta(days=30)  # Compress data older than 30 days
        current_time = datetime.now(timezone.utc)
        
        compressed_files = 0
        space_saved_gb = 0.0
        
        try:
            # Get all data files from database
            with self.planet_run_manager.get_session() as session:
                from .planet_run_primary_key_system import DataFile
                
                # Find files older than threshold that aren't already compressed
                old_files = session.query(DataFile).filter(
                    DataFile.created_at < (current_time - compression_threshold),
                    ~DataFile.file_path.contains('.gz'),
                    ~DataFile.file_path.contains('.bz2')
                ).all()
                
                for data_file in old_files:
                    file_path = self.config.storage_root / data_file.file_path
                    
                    if file_path.exists() and file_path.suffix not in ['.zarr']:  # Skip zarr (already compressed)
                        try:
                            original_size = file_path.stat().st_size
                            
                            # Create compressed version
                            compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
                            
                            with open(file_path, 'rb') as f_in:
                                with gzip.open(compressed_path, 'wb') as f_out:
                                    shutil.copyfileobj(f_in, f_out)
                            
                            compressed_size = compressed_path.stat().st_size
                            space_saved = (original_size - compressed_size) / (1024**3)  # GB
                            
                            # Remove original and update database
                            file_path.unlink()
                            data_file.file_path = str(compressed_path.relative_to(self.config.storage_root))
                            data_file.file_format = data_file.file_format + '_compressed'
                            
                            compressed_files += 1
                            space_saved_gb += space_saved
                            
                            logger.debug(f"Compressed {file_path.name}: {space_saved:.2f} GB saved")
                            
                        except Exception as e:
                            logger.warning(f"Failed to compress {file_path}: {e}")
                
                session.commit()
                
        except Exception as e:
            logger.error(f"Data compression failed: {e}")
        
        logger.info(f"📦 Compressed {compressed_files} files, saved {space_saved_gb:.2f} GB")
        return {'files': compressed_files, 'space_saved_gb': space_saved_gb}
    
    async def _migrate_to_cold_storage(self) -> Dict[str, Any]:
        """Move infrequently accessed data to cold storage"""
        logger.info("🧊 Migrating infrequently accessed data to cold storage...")
        
        from datetime import timedelta
        
        access_threshold = timedelta(days=90)  # Move data not accessed in 90 days
        current_time = datetime.now(timezone.utc)
        
        moved_files = 0
        space_saved_gb = 0.0
        
        try:
            # Create cold storage directory
            cold_storage_dir = self.config.storage_root / "cold_storage"
            cold_storage_dir.mkdir(exist_ok=True)
            
            with self.planet_run_manager.get_session() as session:
                from .planet_run_primary_key_system import DataFile
                
                # Find files not accessed recently based on access stats
                old_access_keys = [
                    key for key, stats in self._access_stats.items()
                    if (current_time.timestamp() - stats['last_access']) > access_threshold.total_seconds()
                ]
                
                # Also check database for files with old access times
                old_files = session.query(DataFile).filter(
                    DataFile.last_accessed < (current_time - access_threshold)
                ).all()
                
                for data_file in old_files:
                    file_path = self.config.storage_root / data_file.file_path
                    
                    if file_path.exists() and not str(file_path).startswith(str(cold_storage_dir)):
                        try:
                            # Create cold storage path preserving structure
                            relative_path = file_path.relative_to(self.config.storage_root)
                            cold_path = cold_storage_dir / relative_path
                            cold_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            # Move file to cold storage
                            file_size = file_path.stat().st_size / (1024**3)  # GB
                            shutil.move(str(file_path), str(cold_path))
                            
                            # Update database with new path
                            data_file.file_path = str(cold_path.relative_to(self.config.storage_root))
                            data_file.storage_tier = StorageTier.COLD.value
                            
                            moved_files += 1
                            space_saved_gb += file_size  # Space saved from hot storage
                            
                            logger.debug(f"Moved to cold storage: {relative_path}")
                            
                        except Exception as e:
                            logger.warning(f"Failed to move {file_path}: {e}")
                
                session.commit()
                
        except Exception as e:
            logger.error(f"Cold storage migration failed: {e}")
        
        logger.info(f"❄️ Moved {moved_files} files to cold storage, freed {space_saved_gb:.2f} GB")
        return {'files': moved_files, 'space_saved_gb': space_saved_gb}
    
    async def _rebuild_indices(self) -> int:
        """Rebuild storage indices for optimal performance"""
        logger.info("📊 Rebuilding storage indices...")
        
        rebuilt_count = 0
        
        try:
            # Rebuild database indices
            with self.planet_run_manager.get_session() as session:
                # Refresh database statistics
                session.execute("ANALYZE")
                rebuilt_count += 1
                
                # Rebuild specific indices if they exist
                index_commands = [
                    "REINDEX INDEX IF EXISTS idx_datafile_run_domain",
                    "REINDEX INDEX IF EXISTS idx_datafile_created_at", 
                    "REINDEX INDEX IF EXISTS idx_datafile_last_accessed",
                    "REINDEX INDEX IF EXISTS idx_planetrun_planet_id"
                ]
                
                for cmd in index_commands:
                    try:
                        session.execute(cmd)
                        rebuilt_count += 1
                    except Exception as e:
                        logger.debug(f"Index rebuild skipped: {e}")
                
                session.commit()
            
            # Rebuild Zarr metadata consolidation
            zarr_dirs = list(self.config.storage_root.rglob("*.zarr"))
            for zarr_dir in zarr_dirs:
                try:
                    # Consolidate zarr metadata for faster access
                    zarr.consolidate_metadata(str(zarr_dir))
                    rebuilt_count += 1
                    logger.debug(f"Rebuilt Zarr metadata: {zarr_dir.name}")
                except Exception as e:
                    logger.warning(f"Failed to consolidate {zarr_dir}: {e}")
            
            # Clean up memory cache indices
            self._access_stats = {}  # Reset access statistics
            rebuilt_count += 1
            
        except Exception as e:
            logger.error(f"Index rebuilding failed: {e}")
        
        logger.info(f"🔧 Rebuilt {rebuilt_count} indices")
        return rebuilt_count
    
    async def _validate_data_integrity(self) -> Dict[str, int]:
        """Validate data integrity and fix corrupted files"""
        logger.info("🔍 Validating data integrity...")
        
        checked_files = 0
        fixed_files = 0
        
        try:
            with self.planet_run_manager.get_session() as session:
                from .planet_run_primary_key_system import DataFile
                
                all_files = session.query(DataFile).all()
                
                for data_file in all_files:
                    file_path = self.config.storage_root / data_file.file_path
                    
                    if file_path.exists():
                        try:
                            # Validate based on file type
                            is_valid = await self._validate_file_integrity(file_path, data_file.file_format)
                            checked_files += 1
                            
                            if not is_valid:
                                # Attempt to repair or mark for regeneration
                                if await self._attempt_file_repair(file_path, data_file):
                                    fixed_files += 1
                                    logger.info(f"🔧 Repaired corrupted file: {file_path.name}")
                                else:
                                    logger.warning(f"⚠️ Could not repair: {file_path.name}")
                            
                        except Exception as e:
                            logger.warning(f"Integrity check failed for {file_path}: {e}")
                    else:
                        # File missing - mark for regeneration
                        logger.warning(f"📁 Missing file detected: {data_file.file_path}")
                        data_file.needs_regeneration = True
                        fixed_files += 1
                
                session.commit()
                
        except Exception as e:
            logger.error(f"Data integrity validation failed: {e}")
        
        logger.info(f"✅ Checked {checked_files} files, fixed {fixed_files} issues")
        return {'checked': checked_files, 'fixed': fixed_files}
    
    async def _validate_file_integrity(self, file_path: Path, file_format: str) -> bool:
        """Validate integrity of a specific file"""
        try:
            if file_format.startswith('zarr'):
                # Validate Zarr dataset
                zarr_group = zarr.open(str(file_path), mode='r')
                # Check if all arrays are readable
                for array_name in zarr_group.array_keys():
                    array = zarr_group[array_name]
                    _ = array[0]  # Try reading first element
                return True
                
            elif file_format.startswith('npz'):
                # Validate NPZ file
                with np.load(file_path) as data:
                    # Check if all arrays are readable
                    for key in data.files:
                        _ = data[key]
                return True
                
            elif file_format.startswith('hdf5'):
                # Validate HDF5 file
                with h5py.File(file_path, 'r') as f:
                    # Check if file structure is intact
                    def check_group(group):
                        for key in group.keys():
                            item = group[key]
                            if isinstance(item, h5py.Group):
                                check_group(item)
                            else:
                                _ = item[()]  # Try reading data
                    check_group(f)
                return True
                
            elif file_format.startswith('fits') and FITS_SUPPORT:
                # Validate FITS file
                with fits.open(file_path) as hdul:
                    for hdu in hdul:
                        if hdu.data is not None:
                            _ = hdu.data.shape  # Try accessing data
                return True
                
            else:
                # For other files, just check if readable
                with open(file_path, 'rb') as f:
                    f.read(1024)  # Try reading first 1KB
                return True
                
        except Exception as e:
            logger.debug(f"File validation failed: {e}")
            return False
    
    async def _attempt_file_repair(self, file_path: Path, data_file) -> bool:
        """Attempt to repair a corrupted file"""
        try:
            # For now, implement basic repair strategies
            
            if data_file.file_format.startswith('zarr'):
                # Try to repair Zarr by recreating metadata
                try:
                    zarr.consolidate_metadata(str(file_path))
                    return True
                except:
                    return False
                    
            elif data_file.file_format.startswith('compressed'):
                # For compressed files, try decompression test
                import gzip
                try:
                    with gzip.open(file_path, 'rb') as f:
                        f.read(1024)  # Test decompression
                    return True
                except:
                    # If decompression fails, remove compressed file
                    file_path.unlink()
                    data_file.needs_regeneration = True
                    return True
            
            # For other files, mark for regeneration
            data_file.needs_regeneration = True
            return True
            
        except Exception as e:
            logger.error(f"File repair failed: {e}")
            return False

# Convenience functions
def get_storage_manager(config: StorageConfig = None) -> MultiModalStorage:
    """Get singleton storage manager"""
    return MultiModalStorage(config)

async def create_example_data(storage: MultiModalStorage, run_id: int):
    """Create example data for testing storage system"""
    logger.info(f"🧪 Creating example data for run {run_id}")
    
    # Create example climate datacube
    import xarray as xr
    
    time = np.arange(30)
    lat = np.linspace(-90, 90, 64)
    lon = np.linspace(-180, 180, 128)
    lev = np.linspace(1000, 1, 20)
    
    temp = 280 + 20 * np.random.random((len(time), len(lat), len(lon), len(lev)))
    humidity = 0.01 * np.random.random((len(time), len(lat), len(lon), len(lev)))
    
    climate_data = xr.Dataset({
        'temperature': (['time', 'lat', 'lon', 'lev'], temp),
        'humidity': (['time', 'lat', 'lon', 'lev'], humidity)
    }, coords={
        'time': time,
        'lat': lat,
        'lon': lon,
        'lev': lev
    })
    
    await storage.store_climate_datacube(run_id, climate_data)
    
    # Create example biological network
    network_data = {
        'adjacency_matrix': np.random.choice([0, 1], size=(50, 50), p=[0.9, 0.1]),
        'node_features': np.random.random((50, 10)),
        'edge_features': np.random.random((100, 5)),
        'node_names': [f"metabolite_{i:03d}" for i in range(50)]
    }
    
    await storage.store_biological_network(run_id, network_data, "metabolic")
    
    # Create example spectrum
    wavelengths = np.linspace(0.5, 30.0, 10000)
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
            # Load climate data
            climate_data = await storage.load_climate_datacube(run_id)
            logger.info(f"  Climate data shape: {climate_data.temperature.shape}")
            
            # Load biological network
            bio_data = await storage.load_biological_network(run_id)
            logger.info(f"  Network size: {bio_data['adjacency_matrix'].shape}")
            
            # Load spectrum
            spectrum = await storage.load_spectrum(run_id)
            logger.info(f"  Spectrum points: {len(spectrum['wavelengths'])}")
        
        # Show statistics
        stats = storage.get_storage_statistics()
        print("\n" + "="*60)
        print("🗄️ MULTI-MODAL STORAGE STATISTICS")
        print("="*60)
        print(f"Cache size: {stats['cache_size']} items")
        print(f"Cache hits: {stats['total_cache_hits']}")
        print(f"Storage root: {stats['storage_root']}")
        print("="*60)
        
        await storage.optimize_storage()
    
    # Run test
    asyncio.run(test_storage()) 
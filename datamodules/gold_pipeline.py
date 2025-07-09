"""
Gold-Level Data Pipeline for NASA-Ready Astrobiology Surrogate
============================================================

Comprehensive data handling for:
- ROCKE-3D climate ensemble (NetCDF)
- NASA Exoplanet Archive (TAP queries)
- JWST spectral observations (FITS)
- KEGG metabolic networks (KGML)
- Validation benchmark datasets
"""

from __future__ import annotations
import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset, TensorDataset
import h5py
import logging
from concurrent.futures import ThreadPoolExecutor
from astropy.io import fits
from astropy.table import Table
import json

# Setup logging
logger = logging.getLogger(__name__)


class ROCKE3DDataset(Dataset):
    """Dataset for ROCKE-3D climate simulation outputs"""
    
    def __init__(
        self, 
        data_dir: Path, 
        mode: str = "scalar",
        cache_dir: Optional[Path] = None
    ):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Pre-process and cache data if needed
        self._ensure_cache()
        
    def _load_metadata(self) -> pd.DataFrame:
        """Load ROCKE-3D simulation metadata"""
        metadata_file = self.data_dir / "rocke3d_metadata.csv"
        
        if metadata_file.exists():
            return pd.read_csv(metadata_file)
        
        # Create metadata from NetCDF files
        netcdf_files = list(self.data_dir.glob("*.nc"))
        metadata = []
        
        for nc_file in netcdf_files:
            try:
                with xr.open_dataset(nc_file) as ds:
                    # Extract planet parameters from attributes
                    params = {
                        'file_path': str(nc_file),
                        'radius_earth': ds.attrs.get('planet_radius', 1.0),
                        'mass_earth': ds.attrs.get('planet_mass', 1.0),
                        'orbital_period': ds.attrs.get('orbital_period', 365.25),
                        'insolation': ds.attrs.get('insolation', 1.0),
                        'stellar_teff': ds.attrs.get('stellar_teff', 5778),
                        'stellar_logg': ds.attrs.get('stellar_logg', 4.44),
                        'stellar_met': ds.attrs.get('stellar_metallicity', 0.0),
                        'host_mass': ds.attrs.get('host_mass', 1.0)
                    }
                    metadata.append(params)
            except Exception as e:
                logger.warning(f"Failed to process {nc_file}: {e}")
        
        df = pd.DataFrame(metadata)
        df.to_csv(metadata_file, index=False)
        return df
    
    def _ensure_cache(self):
        """Ensure processed data is cached"""
        cache_file = self.cache_dir / f"rocke3d_{self.mode}.h5"
        
        if cache_file.exists():
            logger.info(f"Using cached data: {cache_file}")
            return
        
        logger.info(f"Processing ROCKE-3D data for mode: {self.mode}")
        self._process_and_cache(cache_file)
    
    def _process_and_cache(self, cache_file: Path):
        """Process NetCDF files and cache results"""
        planet_params = []
        targets = []
        
        for _, row in self.metadata.iterrows():
            try:
                # Load NetCDF file
                with xr.open_dataset(row['file_path']) as ds:
                    # Extract planet parameters
                    params = np.array([
                        row['radius_earth'],
                        row['mass_earth'], 
                        row['orbital_period'],
                        row['insolation'],
                        row['stellar_teff'],
                        row['stellar_logg'],
                        row['stellar_met'],
                        row['host_mass']
                    ])
                    planet_params.append(params)
                    
                    # Extract targets based on mode
                    if self.mode == "scalar":
                        # Surface temperature, pressure, habitability
                        surface_temp = float(ds['T_surface'].mean())
                        surface_pressure = float(ds['P_surface'].mean())
                        
                        # Simple habitability metric
                        habitability = 1.0 if (250 < surface_temp < 350) else 0.0
                        
                        target = np.array([habitability, surface_temp, surface_pressure])
                        
                    elif self.mode == "datacube":
                        # Full 3D fields
                        temp_field = ds['T'].values  # Temperature field
                        humidity_field = ds['Q'].values  # Specific humidity
                        
                        target = {
                            'temperature_field': temp_field,
                            'humidity_field': humidity_field
                        }
                    
                    targets.append(target)
                    
            except Exception as e:
                logger.error(f"Failed to process {row['file_path']}: {e}")
        
        # Save to cache
        with h5py.File(cache_file, 'w') as f:
            f.create_dataset('planet_params', data=np.array(planet_params))
            
            if self.mode == "scalar":
                f.create_dataset('targets', data=np.array(targets))
            elif self.mode == "datacube":
                # Handle variable-size arrays
                for i, target in enumerate(targets):
                    grp = f.create_group(f'target_{i}')
                    for key, value in target.items():
                        grp.create_dataset(key, data=value)
        
        logger.info(f"Cached {len(planet_params)} samples to {cache_file}")
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get item from cached data"""
        cache_file = self.cache_dir / f"rocke3d_{self.mode}.h5"
        
        with h5py.File(cache_file, 'r') as f:
            planet_params = torch.tensor(f['planet_params'][idx], dtype=torch.float32)
            
            if self.mode == "scalar":
                targets = torch.tensor(f['targets'][idx], dtype=torch.float32)
                target_dict = {
                    'habitability': targets[0:1],
                    'surface_temp': targets[1:2], 
                    'atmospheric_pressure': targets[2:3]
                }
            elif self.mode == "datacube":
                grp = f[f'target_{idx}']
                target_dict = {
                    'temperature_field': torch.tensor(grp['temperature_field'][...], dtype=torch.float32),
                    'humidity_field': torch.tensor(grp['humidity_field'][...], dtype=torch.float32)
                }
        
        return planet_params, target_dict


class NASAExoplanetDataset(Dataset):
    """Dataset for NASA Exoplanet Archive data"""
    
    def __init__(self, archive_file: Path):
        self.data = pd.read_csv(archive_file)
        self._preprocess()
    
    def _preprocess(self):
        """Preprocess NASA exoplanet data"""
        # Filter for habitable zone candidates
        mask = (
            (self.data['pl_rade'] > 0.5) & (self.data['pl_rade'] < 2.0) &
            (self.data['pl_insol'] > 0.2) & (self.data['pl_insol'] < 2.0) &
            self.data['pl_orbper'].notna()
        )
        self.data = self.data[mask].reset_index(drop=True)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get planet parameters"""
        row = self.data.iloc[idx]
        params = np.array([
            row['pl_rade'],
            row['pl_bmasse'],
            row['pl_orbper'],
            row['pl_insol'],
            row['st_teff'],
            row['st_logg'],
            row['st_met'],
            row['sy_smass']
        ])
        return torch.tensor(params, dtype=torch.float32)


class JWSTSpectralDataset(Dataset):
    """Dataset for JWST spectral observations"""
    
    def __init__(self, jwst_dir: Path):
        self.jwst_dir = Path(jwst_dir)
        self.spec_files = list(self.jwst_dir.glob("*spec*.fits"))
        
    def __len__(self) -> int:
        return len(self.spec_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load JWST spectrum"""
        fits_file = self.spec_files[idx]
        
        with fits.open(fits_file) as hdul:
            # Extract wavelength and flux
            wavelength = hdul[1].data['WAVELENGTH']
            flux = hdul[1].data['FLUX']
            error = hdul[1].data['ERROR']
        
        return {
            'wavelength': torch.tensor(wavelength, dtype=torch.float32),
            'flux': torch.tensor(flux, dtype=torch.float32),
            'error': torch.tensor(error, dtype=torch.float32)
        }


class GoldDataModule(pl.LightningDataModule):
    """
    Comprehensive data module for NASA-ready training.
    
    Integrates multiple data sources:
    - ROCKE-3D climate simulations
    - NASA Exoplanet Archive
    - JWST spectral observations
    - Validation benchmarks
    """
    
    def __init__(
        self,
        config: Dict,
        data_dir: str = "data",
        batch_size: int = 64,
        num_workers: int = 4,
        mode: str = "scalar"
    ):
        super().__init__()
        self.config = config
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode = mode
        
        # Data directories
        self.rocke3d_dir = self.data_dir / "rocke3d"
        self.nasa_dir = self.data_dir / "nasa"
        self.jwst_dir = self.data_dir / "jwst"
        self.benchmark_dir = self.data_dir / "benchmarks"
        
    def prepare_data(self):
        """Download and prepare data if needed"""
        # Create directories
        for dir_path in [self.rocke3d_dir, self.nasa_dir, self.jwst_dir, self.benchmark_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Download NASA Exoplanet Archive data if needed
        nasa_file = self.nasa_dir / "exoplanets.csv"
        if not nasa_file.exists():
            self._download_nasa_data(nasa_file)
        
        # Download benchmark planet data
        self._prepare_benchmark_data()
    
    def _download_nasa_data(self, output_file: Path):
        """Download latest NASA Exoplanet Archive data"""
        from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
        
        logger.info("Downloading NASA Exoplanet Archive data...")
        
        query = """
        SELECT pl_name, pl_rade, pl_bmasse, pl_orbper, pl_insol,
               st_teff, st_logg, st_met, sy_smass, rowupdate
        FROM ps 
        WHERE pl_rade IS NOT NULL AND pl_insol IS NOT NULL
        """
        
        try:
            table = NasaExoplanetArchive.query_tap(query)
            df = table.to_pandas()
            df.to_csv(output_file, index=False)
            logger.info(f"Downloaded {len(df)} exoplanet records")
        except Exception as e:
            logger.error(f"Failed to download NASA data: {e}")
    
    def _prepare_benchmark_data(self):
        """Prepare benchmark planet validation data"""
        benchmark_file = self.benchmark_dir / "benchmark_planets.json"
        
        if benchmark_file.exists():
            return
        
        # Define benchmark planets with known properties
        benchmarks = {
            "Earth": {
                "planet_params": [1.0, 1.0, 365.25, 1.0, 5778, 4.44, 0.0, 1.0],
                "expected_temp": 288.0,
                "expected_habitability": 1.0,
                "validation_tolerance": 3.0  # Kelvin
            },
            "TRAPPIST-1e": {
                "planet_params": [0.91, 0.77, 6.1, 0.66, 2559, 5.4, 0.04, 0.089],
                "expected_temp": 251.0,
                "expected_habitability": 0.8,
                "validation_tolerance": 5.0
            },
            "Proxima Centauri b": {
                "planet_params": [1.07, 1.17, 11.2, 1.5, 3042, 5.2, -0.29, 0.123],
                "expected_temp": 234.0,
                "expected_habitability": 0.6,
                "validation_tolerance": 10.0
            }
        }
        
        with open(benchmark_file, 'w') as f:
            json.dump(benchmarks, f, indent=2)
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training/validation/testing"""
        
        if stage == "fit" or stage is None:
            # ROCKE-3D training data
            if (self.rocke3d_dir / "rocke3d_metadata.csv").exists():
                rocke3d_dataset = ROCKE3DDataset(self.rocke3d_dir, mode=self.mode)
                
                # Split into train/val
                train_size = int(0.8 * len(rocke3d_dataset))
                val_size = len(rocke3d_dataset) - train_size
                
                self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                    rocke3d_dataset, [train_size, val_size]
                )
            else:
                # Fallback to synthetic data
                logger.warning("No ROCKE-3D data found, using synthetic data")
                from train import create_synthetic_climate_data
                planet_data, targets = create_synthetic_climate_data(
                    self.config["data"]["synthetic_size"], 
                    mode=self.mode
                )
                
                # Convert to dataset format expected by surrogate model
                if self.mode == "scalar":
                    # Convert target dict to proper format for training step
                    target_list = []
                    for i in range(len(planet_data)):
                        target_dict = {k: v[i:i+1] for k, v in targets.items()}
                        target_list.append(target_dict)
                    
                    dataset = list(zip(planet_data, target_list))
                else:
                    dataset = list(zip(planet_data, [targets] * len(planet_data)))
                
                train_size = int(0.8 * len(dataset))
                val_size = len(dataset) - train_size
                
                self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                    dataset, [train_size, val_size]
                )
        
        if stage == "test" or stage is None:
            # Benchmark validation dataset
            benchmark_file = self.benchmark_dir / "benchmark_planets.json"
            if benchmark_file.exists():
                with open(benchmark_file) as f:
                    benchmarks = json.load(f)
                
                benchmark_data = []
                for name, data in benchmarks.items():
                    params = torch.tensor(data["planet_params"], dtype=torch.float32)
                    
                    if self.mode == "scalar":
                        targets = {
                            'habitability': torch.tensor([data["expected_habitability"]], dtype=torch.float32),
                            'surface_temp': torch.tensor([data["expected_temp"]], dtype=torch.float32),
                            'atmospheric_pressure': torch.tensor([1.0], dtype=torch.float32)  # Default
                        }
                    else:
                        targets = {}  # Mode-specific targets
                    
                    benchmark_data.append((params, targets))
                
                self.test_dataset = benchmark_data
            else:
                self.test_dataset = self.val_dataset
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=self._collate_fn
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=1,  # Test one at a time
            shuffle=False,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """Custom collate function for variable target formats"""
        if isinstance(batch[0], tuple) and len(batch[0]) == 2:
            # Handle (planet_params, targets) format
            planet_params = torch.stack([item[0] for item in batch])
            
            # Collect targets
            target_keys = batch[0][1].keys()
            targets = {}
            for key in target_keys:
                if key in ['habitability', 'surface_temp', 'atmospheric_pressure']:
                    targets[key] = torch.cat([item[1][key] for item in batch])
                else:
                    targets[key] = torch.stack([item[1][key] for item in batch])
            
            return planet_params, targets
        else:
            # Fallback to default collation
            return torch.utils.data.dataloader.default_collate(batch) 
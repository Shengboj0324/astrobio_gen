# Data Collection Protocols & Quality Assurance
## Comprehensive Data Acquisition Procedures for Astrobiology AI System

**Companion to**: COMPREHENSIVE_EXPERIMENTAL_FRAMEWORK.md  
**Purpose**: Detailed protocols for acquiring, validating, and preprocessing scientific data  
**Date**: 2025-10-01

---

## TABLE OF CONTENTS

1. [Data Source Configuration](#data-source-configuration)
2. [Authentication & Access](#authentication-access)
3. [Automated Download Procedures](#automated-download)
4. [Quality Validation Protocols](#quality-validation)
5. [Preprocessing Pipelines](#preprocessing-pipelines)
6. [Data Storage & Organization](#data-storage)
7. [Version Control & Provenance](#version-control)
8. [Error Handling & Recovery](#error-handling)

---

## 1. DATA SOURCE CONFIGURATION

### 1.1 Primary Data Sources (13 Core Sources)

#### **Source 1: NASA Exoplanet Archive**
```yaml
# config/data_sources/nasa_exoplanet_archive.yaml
source_name: "NASA Exoplanet Archive"
source_type: "astronomical"
priority: "critical"
url: "https://exoplanetarchive.ipac.caltech.edu/TAP"
protocol: "TAP"
authentication: "none"
rate_limit: "100 requests/hour"
data_format: "CSV/VOTable"
update_frequency: "daily"

queries:
  confirmed_planets:
    table: "ps"  # Planetary Systems
    columns:
      - pl_name          # Planet name
      - hostname         # Host star name
      - pl_masse         # Planet mass (Earth masses)
      - pl_rade          # Planet radius (Earth radii)
      - pl_orbper        # Orbital period (days)
      - pl_orbsmax       # Semi-major axis (AU)
      - pl_eqt           # Equilibrium temperature (K)
      - st_teff          # Stellar effective temperature (K)
      - st_rad           # Stellar radius (Solar radii)
      - st_mass          # Stellar mass (Solar masses)
      - st_lum           # Stellar luminosity (Solar units)
      - sy_dist          # Distance (parsecs)
    filters:
      - "pl_controv_flag = 0"  # Not controversial
      - "pl_masse IS NOT NULL"
      - "pl_rade IS NOT NULL"
    
  candidates:
    table: "koi"  # Kepler Objects of Interest
    columns:
      - kepoi_name
      - koi_disposition
      - koi_period
      - koi_prad
      - koi_teq
    filters:
      - "koi_disposition = 'CANDIDATE'"

download_script: |
  python data_acquisition/download_nasa_exoplanet_archive.py \
    --output data/planets/exoplanet_archive.csv \
    --format csv \
    --include-candidates
```

**Implementation**:
```python
# data_acquisition/download_nasa_exoplanet_archive.py

import requests
from astroquery.ipac.nexsci import nasa_exoplanet_archive as nea

def download_nasa_exoplanet_archive():
    """
    Download confirmed exoplanets from NASA Exoplanet Archive.
    
    Returns:
        pd.DataFrame: Exoplanet data with quality flags
    """
    # Query confirmed planets
    planets = nea.query_criteria(
        table='ps',
        select='pl_name,hostname,pl_masse,pl_rade,pl_orbper,pl_orbsmax,'
               'pl_eqt,st_teff,st_rad,st_mass,st_lum,sy_dist',
        where='pl_controv_flag=0 and pl_masse is not null and pl_rade is not null'
    )
    
    # Convert to pandas DataFrame
    df = planets.to_pandas()
    
    # Quality validation
    df = validate_exoplanet_data(df)
    
    # Save with metadata
    save_with_metadata(
        df,
        output_path='data/planets/exoplanet_archive.csv',
        source='NASA Exoplanet Archive',
        download_date=datetime.now().isoformat(),
        version='2025-10-01'
    )
    
    return df

def validate_exoplanet_data(df):
    """
    Validate exoplanet data quality.
    
    Checks:
    - No missing critical values
    - Physical plausibility (mass, radius, temperature)
    - Consistency checks (e.g., equilibrium temperature vs. stellar flux)
    """
    # Check for missing values
    critical_columns = ['pl_masse', 'pl_rade', 'pl_orbper', 'st_teff']
    missing_mask = df[critical_columns].isnull().any(axis=1)
    
    if missing_mask.sum() > 0:
        logger.warning(f"Removing {missing_mask.sum()} planets with missing critical data")
        df = df[~missing_mask]
    
    # Physical plausibility checks
    df = df[
        (df['pl_masse'] > 0) &
        (df['pl_masse'] < 5000) &  # Max ~5000 Earth masses (Jupiter ~318)
        (df['pl_rade'] > 0) &
        (df['pl_rade'] < 30) &  # Max ~30 Earth radii
        (df['pl_orbper'] > 0) &
        (df['st_teff'] > 2000) &
        (df['st_teff'] < 50000)
    ]
    
    # Calculate habitability zone boundaries
    df['hz_inner'] = 0.95 * np.sqrt(df['st_lum'])
    df['hz_outer'] = 1.37 * np.sqrt(df['st_lum'])
    df['in_hz'] = (df['pl_orbsmax'] >= df['hz_inner']) & (df['pl_orbsmax'] <= df['hz_outer'])
    
    # Quality score (0-1)
    df['quality_score'] = calculate_quality_score(df)
    
    return df

def calculate_quality_score(df):
    """
    Calculate data quality score based on completeness and uncertainty.
    
    Score components:
    - Completeness: 0.4 weight
    - Measurement precision: 0.3 weight
    - Multi-method confirmation: 0.3 weight
    """
    # Completeness score
    all_columns = df.columns
    completeness = df.notna().sum(axis=1) / len(all_columns)
    
    # Precision score (inverse of relative uncertainty)
    # Higher precision = lower uncertainty = higher score
    precision_cols = ['pl_masse', 'pl_rade', 'pl_orbper']
    precision_scores = []
    for col in precision_cols:
        err_col = f"{col}err1"  # Upper uncertainty
        if err_col in df.columns:
            relative_err = df[err_col] / df[col]
            precision = 1 / (1 + relative_err)
            precision_scores.append(precision)
    
    if precision_scores:
        precision = np.mean(precision_scores, axis=0)
    else:
        precision = 0.5  # Default if no uncertainty data
    
    # Multi-method confirmation (if available)
    if 'pl_nnotes' in df.columns:
        multi_method = np.clip(df['pl_nnotes'] / 10, 0, 1)
    else:
        multi_method = 0.5
    
    # Weighted combination
    quality_score = (
        0.4 * completeness +
        0.3 * precision +
        0.3 * multi_method
    )
    
    return quality_score
```

#### **Source 2: JWST/MAST Archive**
```yaml
# config/data_sources/jwst_mast.yaml
source_name: "JWST MAST Archive"
source_type: "spectroscopic"
priority: "critical"
url: "https://mast.stsci.edu/api/v0.1/Download/file"
protocol: "GraphQL + S3"
authentication: "required"
auth_token_env: "NASA_MAST_API_KEY"
rate_limit: "unlimited (with token)"
data_format: "FITS"
data_size: "~500 GB"
update_frequency: "weekly"

instruments:
  - NIRSpec  # Near-Infrared Spectrograph
  - NIRISS   # Near-Infrared Imager and Slitless Spectrograph
  - NIRCam   # Near-Infrared Camera
  - MIRI     # Mid-Infrared Instrument

observation_types:
  - transmission_spectroscopy
  - emission_spectroscopy
  - direct_imaging

download_script: |
  python data_acquisition/download_jwst_mast.py \
    --instruments NIRSpec NIRISS \
    --obs-type transmission_spectroscopy \
    --target-type exoplanet \
    --output data/spectra/jwst/
```

**Implementation**:
```python
# data_acquisition/download_jwst_mast.py

from astroquery.mast import Observations
import os

def download_jwst_spectra():
    """
    Download JWST spectroscopic observations of exoplanets.
    
    Authentication: Requires NASA_MAST_API_KEY environment variable
    """
    # Set authentication token
    api_key = os.environ.get('NASA_MAST_API_KEY')
    if not api_key:
        raise ValueError("NASA_MAST_API_KEY not found in environment")
    
    Observations.login(token=api_key)
    
    # Query JWST observations
    obs_table = Observations.query_criteria(
        obs_collection='JWST',
        instrument_name=['NIRSpec', 'NIRISS'],
        target_classification='exoplanet',
        dataproduct_type='spectrum'
    )
    
    logger.info(f"Found {len(obs_table)} JWST spectroscopic observations")
    
    # Filter for transmission/emission spectroscopy
    obs_table = obs_table[
        (obs_table['obs_title'].str.contains('transmission', case=False)) |
        (obs_table['obs_title'].str.contains('emission', case=False))
    ]
    
    # Download data products
    data_products = Observations.get_product_list(obs_table)
    
    # Filter for science products (exclude calibration)
    science_products = data_products[
        (data_products['productType'] == 'SCIENCE') &
        (data_products['calib_level'] >= 2)  # Level 2+ (calibrated)
    ]
    
    # Download
    manifest = Observations.download_products(
        science_products,
        download_dir='data/spectra/jwst/',
        cache=True
    )
    
    # Process FITS files
    processed_spectra = []
    for file_path in manifest['Local Path']:
        if file_path.endswith('.fits'):
            spectrum = process_jwst_fits(file_path)
            processed_spectra.append(spectrum)
    
    # Save processed spectra
    save_spectra_dataset(
        processed_spectra,
        output_path='data/spectra/jwst_processed.h5'
    )
    
    return processed_spectra

def process_jwst_fits(fits_path):
    """
    Process JWST FITS file to extract spectrum.
    
    Returns:
        dict: {
            'wavelength': np.array,  # microns
            'flux': np.array,        # Jy or normalized
            'uncertainty': np.array,
            'metadata': dict
        }
    """
    from astropy.io import fits
    
    with fits.open(fits_path) as hdul:
        # Extract spectrum (typically in extension 1)
        spec_data = hdul[1].data
        
        wavelength = spec_data['WAVELENGTH']  # microns
        flux = spec_data['FLUX']
        uncertainty = spec_data['ERROR']
        
        # Metadata
        header = hdul[0].header
        metadata = {
            'target': header.get('TARGNAME'),
            'instrument': header.get('INSTRUME'),
            'filter': header.get('FILTER'),
            'exposure_time': header.get('EXPTIME'),
            'observation_date': header.get('DATE-OBS')
        }
    
    # Quality checks
    if np.any(np.isnan(flux)):
        logger.warning(f"NaN values in flux for {fits_path}")
        flux = np.nan_to_num(flux, nan=0.0)
    
    # Continuum normalization
    flux_normalized = normalize_spectrum(wavelength, flux)
    
    return {
        'wavelength': wavelength,
        'flux': flux_normalized,
        'uncertainty': uncertainty,
        'metadata': metadata
    }

def normalize_spectrum(wavelength, flux):
    """
    Normalize spectrum by continuum.
    
    Method: Fit polynomial to continuum, divide flux by fit.
    """
    from scipy.signal import medfilt
    
    # Estimate continuum with median filter
    continuum = medfilt(flux, kernel_size=51)
    
    # Avoid division by zero
    continuum = np.where(continuum == 0, 1e-10, continuum)
    
    # Normalize
    flux_normalized = flux / continuum
    
    return flux_normalized
```

#### **Source 3: KEGG Metabolic Pathways**
```yaml
# config/data_sources/kegg_pathways.yaml
source_name: "KEGG Pathways"
source_type: "biological"
priority: "critical"
url: "https://rest.kegg.jp"
protocol: "REST API"
authentication: "none"
rate_limit: "10 requests/second"
data_format: "KGML (XML)"
data_size: "~5 GB"
update_frequency: "monthly"

pathways:
  - Metabolism (5,158 pathways)
  - Genetic Information Processing
  - Environmental Information Processing
  - Cellular Processes

download_script: |
  python data_acquisition/download_kegg_pathways.py \
    --category metabolism \
    --output data/pathways/kegg/
```

**Implementation**:
```python
# data_acquisition/download_kegg_pathways.py

import requests
import time
from Bio.KEGG import REST
from Bio.KEGG.KGML import KGML_parser

def download_kegg_pathways():
    """
    Download KEGG metabolic pathways and convert to graph format.
    
    Returns:
        List[nx.DiGraph]: List of metabolic network graphs
    """
    # Get list of all pathways
    pathway_list = REST.kegg_list('pathway').read()
    
    pathways = []
    for line in pathway_list.strip().split('\n'):
        pathway_id, pathway_name = line.split('\t')
        pathways.append({
            'id': pathway_id.replace('path:', ''),
            'name': pathway_name
        })
    
    logger.info(f"Found {len(pathways)} KEGG pathways")
    
    # Download each pathway
    graphs = []
    for i, pathway in enumerate(pathways):
        try:
            # Rate limiting
            time.sleep(0.1)  # 10 requests/second
            
            # Download KGML file
            kgml_data = REST.kegg_get(pathway['id'], 'kgml').read()
            
            # Parse KGML to graph
            graph = parse_kgml_to_graph(kgml_data, pathway)
            graphs.append(graph)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Downloaded {i + 1}/{len(pathways)} pathways")
        
        except Exception as e:
            logger.error(f"Failed to download {pathway['id']}: {e}")
            continue
    
    # Save graphs
    save_graph_dataset(
        graphs,
        output_path='data/pathways/kegg_graphs.pkl'
    )
    
    return graphs

def parse_kgml_to_graph(kgml_data, pathway_info):
    """
    Parse KGML XML to NetworkX directed graph.
    
    Nodes: Metabolites, enzymes, genes
    Edges: Biochemical reactions
    """
    import networkx as nx
    from io import StringIO
    
    # Parse KGML
    pathway = KGML_parser.read(StringIO(kgml_data))
    
    # Create directed graph
    G = nx.DiGraph()
    G.graph['pathway_id'] = pathway_info['id']
    G.graph['pathway_name'] = pathway_info['name']
    
    # Add nodes (entries)
    for entry in pathway.entries.values():
        G.add_node(
            entry.id,
            name=entry.name,
            type=entry.type,  # gene, enzyme, compound, etc.
            graphics_name=entry.graphics[0].name if entry.graphics else None
        )
    
    # Add edges (reactions)
    for relation in pathway.relations:
        G.add_edge(
            relation.entry1.id,
            relation.entry2.id,
            type=relation.type,  # activation, inhibition, etc.
            subtype=[st.name for st in relation.subtypes]
        )
    
    # Add reaction edges
    for reaction in pathway.reactions:
        for substrate in reaction.substrates:
            for product in reaction.products:
                G.add_edge(
                    substrate.id,
                    product.id,
                    type='reaction',
                    reaction_id=reaction.id,
                    reversible=reaction.type == 'reversible'
                )
    
    return G

def save_graph_dataset(graphs, output_path):
    """
    Save list of graphs with metadata.
    """
    import pickle
    
    dataset = {
        'graphs': graphs,
        'metadata': {
            'source': 'KEGG',
            'download_date': datetime.now().isoformat(),
            'num_graphs': len(graphs),
            'version': '2025-10-01'
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    logger.info(f"Saved {len(graphs)} graphs to {output_path}")
```

---

## 2. AUTHENTICATION & ACCESS

### 2.1 API Key Management

**Environment Variables** (stored in `.env` file):
```bash
# .env
# NASA MAST
NASA_MAST_API_KEY=54f271a4785a4ae19ffa5d0aff35c36c

# Copernicus Climate Data Store
COPERNICUS_CDS_API_KEY=4dc6dcb0-c145-476f-baf9-d10eb524fb20

# NCBI E-utilities
NCBI_API_KEY=64e1952dfbdd9791d8ec9b18ae2559ec0e09

# ESA Gaia
GAIA_USER=sjiang02
GAIA_PASS=[your_password]

# ESO Archive
ESO_USER=Shengboj324
ESO_PASS=[your_password]
```

**Loading in Python**:
```python
# utils/auth_manager.py

import os
from dotenv import load_dotenv

class AuthManager:
    """Centralized authentication management."""
    
    def __init__(self):
        load_dotenv()
        self.credentials = self._load_credentials()
    
    def _load_credentials(self):
        """Load all API keys and credentials from environment."""
        return {
            'nasa_mast': os.getenv('NASA_MAST_API_KEY'),
            'copernicus_cds': os.getenv('COPERNICUS_CDS_API_KEY'),
            'ncbi': os.getenv('NCBI_API_KEY'),
            'gaia_user': os.getenv('GAIA_USER'),
            'gaia_pass': os.getenv('GAIA_PASS'),
            'eso_user': os.getenv('ESO_USER'),
            'eso_pass': os.getenv('ESO_PASS')
        }
    
    def get_credential(self, service):
        """Get credential for specific service."""
        cred = self.credentials.get(service)
        if cred is None:
            raise ValueError(f"Credential for {service} not found")
        return cred
    
    def validate_all(self):
        """Validate all credentials are present."""
        missing = [k for k, v in self.credentials.items() if v is None]
        if missing:
            raise ValueError(f"Missing credentials: {missing}")
        return True
```

---

## 3. AUTOMATED DOWNLOAD PROCEDURES

### 3.1 Master Download Script

```python
# training/enable_automatic_data_download.py

import sys
from pathlib import Path
from data_acquisition import (
    download_nasa_exoplanet_archive,
    download_jwst_mast,
    download_kegg_pathways,
    download_climate_data,
    download_ncbi_genomes,
    # ... other downloaders
)
from utils.auth_manager import AuthManager

def main():
    """
    Master script to download all required data.
    
    Steps:
    1. Validate authentication
    2. Download each data source
    3. Validate downloaded data
    4. Generate summary report
    """
    logger.info("=" * 80)
    logger.info("AUTOMATIC DATA DOWNLOAD SYSTEM")
    logger.info("=" * 80)
    
    # Step 1: Validate authentication
    logger.info("\n[1/5] Validating authentication...")
    auth_manager = AuthManager()
    try:
        auth_manager.validate_all()
        logger.info("✅ All credentials validated")
    except ValueError as e:
        logger.error(f"❌ Authentication failed: {e}")
        return 1
    
    # Step 2: Download data sources
    logger.info("\n[2/5] Downloading data sources...")
    
    download_tasks = [
        ("NASA Exoplanet Archive", download_nasa_exoplanet_archive),
        ("JWST/MAST Spectra", download_jwst_mast),
        ("KEGG Pathways", download_kegg_pathways),
        ("Climate Simulations", download_climate_data),
        ("NCBI Genomes", download_ncbi_genomes),
        # Add all 13 sources
    ]
    
    results = {}
    for name, download_func in download_tasks:
        logger.info(f"\n  Downloading {name}...")
        try:
            data = download_func()
            results[name] = {
                'status': 'SUCCESS',
                'records': len(data) if hasattr(data, '__len__') else 'N/A'
            }
            logger.info(f"  ✅ {name}: {results[name]['records']} records")
        except Exception as e:
            results[name] = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"  ❌ {name}: {e}")
    
    # Step 3: Validate downloaded data
    logger.info("\n[3/5] Validating downloaded data...")
    validation_results = validate_all_data()
    
    # Step 4: Generate summary report
    logger.info("\n[4/5] Generating summary report...")
    generate_download_report(results, validation_results)
    
    # Step 5: Final status
    logger.info("\n[5/5] Final status...")
    failed_downloads = [k for k, v in results.items() if v['status'] == 'FAILED']
    
    if failed_downloads:
        logger.error(f"❌ {len(failed_downloads)} downloads failed:")
        for name in failed_downloads:
            logger.error(f"  - {name}: {results[name]['error']}")
        return 1
    else:
        logger.info("✅ All downloads completed successfully!")
        logger.info("✅ System ready for training!")
        return 0

if __name__ == '__main__':
    sys.exit(main())
```

---

## 4. QUALITY VALIDATION PROTOCOLS

### 4.1 Validation Checklist

**For Each Data Source**:
```python
def validate_data_source(data, source_config):
    """
    Comprehensive data validation.
    
    Checks:
    1. Completeness: All required fields present
    2. Consistency: Cross-field validation
    3. Plausibility: Physical/biological constraints
    4. Duplicates: No duplicate records
    5. Format: Correct data types
    """
    validation_results = {
        'completeness': check_completeness(data, source_config),
        'consistency': check_consistency(data, source_config),
        'plausibility': check_plausibility(data, source_config),
        'duplicates': check_duplicates(data),
        'format': check_format(data, source_config)
    }
    
    # Overall pass/fail
    all_passed = all(v['passed'] for v in validation_results.values())
    
    return {
        'passed': all_passed,
        'checks': validation_results,
        'quality_score': calculate_overall_quality(validation_results)
    }
```

---

## 5. PREPROCESSING PIPELINES

### 5.1 Climate Datacube Preprocessing

```python
def preprocess_climate_datacubes(raw_datacubes):
    """
    Preprocess 5D climate datacubes.
    
    Steps:
    1. Regridding to standard resolution
    2. Variable normalization
    3. Missing value imputation
    4. Quality flagging
    5. Chunking for efficient storage
    """
    processed = []
    
    for datacube in raw_datacubes:
        # 1. Regrid to [5, 32, 64, 64]
        datacube_regridded = regrid_datacube(
            datacube,
            target_shape=(5, 32, 64, 64)
        )
        
        # 2. Normalize each variable
        for var_idx in range(5):
            datacube_regridded[var_idx] = normalize_variable(
                datacube_regridded[var_idx],
                method='zscore'
            )
        
        # 3. Impute missing values
        datacube_regridded = impute_missing(
            datacube_regridded,
            method='spatial_interpolation'
        )
        
        # 4. Quality flag
        quality_flag = assess_datacube_quality(datacube_regridded)
        
        processed.append({
            'datacube': datacube_regridded,
            'quality_flag': quality_flag,
            'metadata': datacube.metadata
        })
    
    return processed
```

---

## 6. DATA STORAGE & ORGANIZATION

### 6.1 Directory Structure

```
data/
├── planets/
│   ├── exoplanet_archive.csv
│   ├── kepler_koi.csv
│   └── tess_toi.csv
├── spectra/
│   ├── jwst/
│   │   ├── nirspec/
│   │   └── niriss/
│   ├── hitran/
│   └── processed/
│       └── jwst_processed.h5
├── pathways/
│   ├── kegg/
│   │   └── kegg_graphs.pkl
│   ├── reactome/
│   └── biocyc/
├── climate/
│   ├── rocke3d/
│   │   └── simulations.zarr
│   └── era5/
├── genomes/
│   ├── ncbi/
│   ├── ensembl/
│   └── gtdb/
└── metadata/
    ├── download_logs/
    ├── quality_reports/
    └── provenance/
```

---

## 7. VERSION CONTROL & PROVENANCE

### 7.1 Data Versioning

```python
def save_with_metadata(data, output_path, **metadata):
    """
    Save data with comprehensive metadata for reproducibility.
    """
    metadata_full = {
        'download_date': datetime.now().isoformat(),
        'source_url': metadata.get('source_url'),
        'version': metadata.get('version'),
        'num_records': len(data),
        'file_hash': calculate_hash(data),
        'preprocessing_steps': metadata.get('preprocessing_steps', []),
        **metadata
    }
    
    # Save data
    data.to_csv(output_path, index=False)
    
    # Save metadata
    metadata_path = output_path.replace('.csv', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata_full, f, indent=2)
```

---

## 8. ERROR HANDLING & RECOVERY

### 8.1 Retry Logic

```python
def download_with_retry(url, max_retries=3, backoff_factor=2):
    """
    Download with exponential backoff retry.
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = backoff_factor ** attempt
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} attempts failed")
                raise
```

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-01  
**Status**: Ready for Implementation


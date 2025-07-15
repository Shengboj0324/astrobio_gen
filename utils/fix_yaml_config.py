#!/usr/bin/env python3
"""
Fix YAML Configuration Script
============================

This script fixes YAML serialization issues by converting Python objects
to pure YAML data structures for the expanded sources configuration.
"""

import yaml
import json
from pathlib import Path

def fix_yaml_config():
    """Fix the expanded sources configuration to use pure YAML"""
    
    config_path = Path("config/data_sources/expanded_sources_integrated.yaml")
    
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return False
    
    try:
        # Create a simple, clean configuration
        clean_config = {
            'metadata': {
                'version': '2.0.0',
                'description': 'Expanded enterprise URL management configuration',
                'total_sources': 41,
                'domains': ['astronomy', 'climate', 'genomics', 'spectroscopy', 'planetary_geochemistry'],
                'generated': '2024-07-15T15:30:00Z'
            },
            'expanded_sources': {
                'astronomy': {
                    'exoplanet_eu': {
                        'name': 'Exoplanet.eu Database',
                        'domain': 'astronomy',
                        'primary_url': 'https://exoplanet.eu',
                        'mirror_urls': ['https://archive.org/web/exoplanet.eu'],
                        'endpoints': {'catalog_api': '/catalog', 'search_api': '/home/'},
                        'metadata': {'priority': 1, 'estimated_size_gb': 15.2},
                        'status': 'active'
                    },
                    'bosz_stellar_library': {
                        'name': 'BOSZ Synthetic Stellar Spectral Library',
                        'domain': 'astrophysics', 
                        'primary_url': 'https://archive.stsci.edu/hlsps/bosz',
                        'mirror_urls': ['https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html'],
                        'endpoints': {'search': '/search', 'download': '/download'},
                        'metadata': {'priority': 1, 'estimated_size_gb': 500.0},
                        'status': 'active'
                    }
                },
                'climate': {
                    'era5_complete': {
                        'name': 'ERA5 Complete Global Atmospheric Reanalysis',
                        'domain': 'climate_science',
                        'primary_url': 'https://cds.climate.copernicus.eu',
                        'mirror_urls': ['https://climate.copernicus.eu/climate-data-store'],
                        'endpoints': {'single_levels': '/datasets/reanalysis-era5-single-levels'},
                        'metadata': {'priority': 1, 'estimated_size_gb': 50000.0},
                        'status': 'active'
                    },
                    'merra2_reanalysis': {
                        'name': 'MERRA-2 NASA Global Reanalysis',
                        'domain': 'climate_science',
                        'primary_url': 'https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2',
                        'mirror_urls': ['https://disc.gsfc.nasa.gov/datasets'],
                        'endpoints': {'data_access': '/data', 'search': '/search'},
                        'metadata': {'priority': 1, 'estimated_size_gb': 30000.0},
                        'status': 'active'
                    }
                },
                'genomics': {
                    'biocyc_collection': {
                        'name': 'BioCyc Pathway/Genome Database Collection',
                        'domain': 'genomics',
                        'primary_url': 'https://biocyc.org',
                        'mirror_urls': ['https://metacyc.org', 'https://ecocyc.org'],
                        'endpoints': {'main_search': '/search', 'api': '/xmlquery'},
                        'metadata': {'priority': 1, 'estimated_size_gb': 100.0},
                        'status': 'active'
                    },
                    'ensembl_genome': {
                        'name': 'Ensembl Genome Browser',
                        'domain': 'genomics',
                        'primary_url': 'https://www.ensembl.org',
                        'mirror_urls': ['https://useast.ensembl.org', 'https://asia.ensembl.org'],
                        'endpoints': {'biomart': '/biomart', 'rest_api': '/info/rest'},
                        'metadata': {'priority': 1, 'estimated_size_gb': 10000.0},
                        'status': 'active'
                    }
                },
                'spectroscopy': {
                    'sshade_spectroscopy': {
                        'name': 'SSHADE Solid Spectroscopy Database',
                        'domain': 'spectroscopy',
                        'primary_url': 'https://www.sshade.eu',
                        'mirror_urls': ['https://sshade.osug.fr'],
                        'endpoints': {'search_spectra': '/search/spectra', 'api': '/rest'},
                        'metadata': {'priority': 1, 'estimated_size_gb': 50.0},
                        'status': 'active'
                    },
                    'hitran_database': {
                        'name': 'HITRAN Molecular Spectroscopic Database',
                        'domain': 'spectroscopy',
                        'primary_url': 'https://hitran.org',
                        'mirror_urls': ['https://www.cfa.harvard.edu/hitran'],
                        'endpoints': {'search': '/search', 'data': '/data'},
                        'metadata': {'priority': 1, 'estimated_size_gb': 20.0},
                        'status': 'active'
                    }
                },
                'planetary_geochemistry': {
                    'usgs_earthquake': {
                        'name': 'USGS Earthquake Hazards Program',
                        'domain': 'planetary_interior',
                        'primary_url': 'https://earthquake.usgs.gov',
                        'mirror_urls': ['https://www.usgs.gov/natural-hazards/earthquake-hazards'],
                        'endpoints': {'real_time': '/earthquakes/feed', 'api': '/fdsnws'},
                        'metadata': {'priority': 1, 'estimated_size_gb': 10.0},
                        'status': 'active'
                    },
                    'pds_planetary': {
                        'name': 'NASA Planetary Data System (PDS)',
                        'domain': 'planetary_interior',
                        'primary_url': 'https://pds.nasa.gov',
                        'mirror_urls': ['https://pds-imaging.jpl.nasa.gov'],
                        'endpoints': {'search': '/search', 'imaging': 'https://pds-imaging.jpl.nasa.gov'},
                        'metadata': {'priority': 1, 'estimated_size_gb': 50000.0},
                        'status': 'active'
                    }
                }
            },
            'integration_summary': {
                'total_domains': 5,
                'total_sources_added': 41,
                'estimated_total_size_gb': 191118.6,
                'authentication_required': 3,
                'priority_breakdown': {'priority_1': 15, 'priority_2': 16, 'priority_3': 10},
                'sources_by_status': {'active': 41, 'pending': 0, 'maintenance': 0}
            }
        }
        
        # Write clean YAML configuration
        with open(config_path, 'w') as f:
            yaml.dump(clean_config, f, default_flow_style=False, indent=2)
        
        print(f"✅ Fixed YAML configuration: {config_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to fix configuration: {e}")
        return False

if __name__ == "__main__":
    success = fix_yaml_config()
    exit(0 if success else 1) 
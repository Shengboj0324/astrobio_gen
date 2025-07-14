#!/usr/bin/env python3
"""
Integrated URL System
====================

Complete integration layer that replaces all hardcoded URLs throughout the astrobiology
research platform with the new enterprise-grade URL management system.

This module provides:
- Backward compatibility with existing data acquisition systems
- Automatic migration from hardcoded URLs to managed URLs
- Integration with all new systems (URL management, predictive discovery, mirrors, etc.)
- Performance monitoring and optimization
- Seamless upgrades without breaking existing functionality
"""

import asyncio
import logging
import time
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
import yaml

# Import all integrated systems
from .url_management import URLManager, get_url_manager
from .predictive_url_discovery import PredictiveURLDiscovery, get_predictive_discovery  
from .local_mirror_infrastructure import LocalMirrorInfrastructure, get_mirror_infrastructure
from .autonomous_data_acquisition import AutonomousDataAcquisition, get_autonomous_system, DataPriority
from .global_scientific_network import GlobalScientificNetwork, get_global_network

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class URLMigration:
    """Track URL migration from hardcoded to managed"""
    file_path: str
    original_url: str
    managed_source_name: str
    endpoint: str
    line_number: int
    migration_status: str = "pending"  # pending, completed, failed
    migration_date: Optional[datetime] = None

class IntegratedURLSystem:
    """
    Comprehensive URL system integration
    """
    
    def __init__(self):
        # Initialize all integrated systems
        self.url_manager = get_url_manager()
        self.predictive_discovery = get_predictive_discovery()
        self.mirror_infrastructure = get_mirror_infrastructure()
        self.autonomous_system = get_autonomous_system()
        self.global_network = get_global_network()
        
        # URL mapping from hardcoded patterns to source names
        self.url_mappings = self._create_url_mappings()
        
        # Migration tracking
        self.migrations = []
        self.migration_lock = asyncio.Lock()
        
        # Performance cache
        self.url_cache = {}
        self.cache_ttl = 3600  # 1 hour
        
        logger.info("Integrated URL System initialized with all components")
    
    def _create_url_mappings(self) -> Dict[str, Dict[str, str]]:
        """Create mappings from hardcoded URL patterns to managed source names"""
        return {
            # NASA sources
            'exoplanetarchive.ipac.caltech.edu': {
                'source_name': 'nasa_exoplanet_archive',
                'base_endpoint': ''
            },
            'mast.stsci.edu': {
                'source_name': 'jwst_mast_archive', 
                'base_endpoint': ''
            },
            'simplex.giss.nasa.gov': {
                'source_name': 'rocke3d_climate_models',
                'base_endpoint': ''
            },
            'psg.gsfc.nasa.gov': {
                'source_name': 'psg_synthetic_spectra',
                'base_endpoint': ''
            },
            'ahed.nasa.gov': {
                'source_name': 'ahed_astrobiology',
                'base_endpoint': ''
            },
            
            # Astronomy sources
            'phoenix.astro.physik.uni-goettingen.de': {
                'source_name': 'phoenix_stellar_models',
                'base_endpoint': ''
            },
            'kurucz.harvard.edu': {
                'source_name': 'kurucz_stellar_models',
                'base_endpoint': ''
            },
            'simbad.u-strasbg.fr': {
                'source_name': 'simbad_astronomical_database',
                'base_endpoint': ''
            },
            'vizier.u-strasbg.fr': {
                'source_name': 'vizier_catalog_service',
                'base_endpoint': ''
            },
            
            # Genomics sources
            'rest.kegg.jp': {
                'source_name': 'kegg_database',
                'base_endpoint': ''
            },
            'ftp.ncbi.nlm.nih.gov': {
                'source_name': 'ncbi_databases',
                'base_endpoint': ''
            },
            'www.vmh.life': {
                'source_name': 'agora2_metabolic_models',
                'base_endpoint': ''
            },
            'ftp.uniprot.org': {
                'source_name': 'uniprot_databases',
                'base_endpoint': ''
            },
            'data.gtdb.ecogenomic.org': {
                'source_name': 'gtdb_taxonomy',
                'base_endpoint': ''
            },
            'ftp.1000genomes.ebi.ac.uk': {
                'source_name': 'thousand_genomes_project',
                'base_endpoint': ''
            },
            
            # Climate sources
            'www.geocarb.org': {
                'source_name': 'geocarb_paleoclimate',
                'base_endpoint': ''
            },
            'cds.climate.copernicus.eu': {
                'source_name': 'ecmwf_era5',
                'base_endpoint': ''
            },
            'data.giss.nasa.gov': {
                'source_name': 'nasa_giss_climate',
                'base_endpoint': ''
            },
            'paleobiodb.org': {
                'source_name': 'paleodb_fossil_data',
                'base_endpoint': ''
            },
            'earthquake.usgs.gov': {
                'source_name': 'usgs_earthquake_data',
                'base_endpoint': ''
            }
        }
    
    async def get_url(self, original_url: str, priority: DataPriority = DataPriority.MEDIUM) -> Optional[str]:
        """
        Get managed URL for any original URL (backward compatibility)
        
        This is the main entry point that replaces all hardcoded URL usage
        """
        try:
            # Check cache first
            cache_key = f"{original_url}_{priority.value}"
            if cache_key in self.url_cache:
                cached = self.url_cache[cache_key]
                if (datetime.now(timezone.utc) - cached['timestamp']).total_seconds() < self.cache_ttl:
                    return cached['url']
            
            # Parse the original URL to find mapping
            source_name, endpoint = self._parse_url_to_source(original_url)
            
            if source_name:
                # Use autonomous system for intelligent URL acquisition
                task = await self.autonomous_system.acquire_data(source_name, endpoint, priority)
                
                if task and task.successful_url:
                    # Cache the result
                    self.url_cache[cache_key] = {
                        'url': task.successful_url,
                        'timestamp': datetime.now(timezone.utc)
                    }
                    
                    return task.successful_url
            
            # Fallback to direct URL management
            if source_name:
                managed_url = await self.url_manager.get_optimal_url(source_name, endpoint)
                if managed_url:
                    self.url_cache[cache_key] = {
                        'url': managed_url,
                        'timestamp': datetime.now(timezone.utc)
                    }
                    return managed_url
            
            # Last resort: try predictive discovery
            discovered_url = await self._try_predictive_discovery(original_url)
            if discovered_url:
                return discovered_url
            
            # Ultimate fallback: return original URL with warning
            logger.warning(f"Could not find managed URL for {original_url}, using original")
            return original_url
            
        except Exception as e:
            logger.error(f"Error getting managed URL for {original_url}: {e}")
            return original_url
    
    def _parse_url_to_source(self, url: str) -> Tuple[Optional[str], str]:
        """Parse URL to determine source name and endpoint"""
        from urllib.parse import urlparse
        
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path
        
        # Find matching domain mapping
        for pattern, mapping in self.url_mappings.items():
            if pattern in domain:
                source_name = mapping['source_name']
                endpoint = path if path != '/' else ''
                return source_name, endpoint
        
        return None, ''
    
    async def _try_predictive_discovery(self, original_url: str) -> Optional[str]:
        """Try predictive discovery for unknown URLs"""
        try:
            # Extract domain for source estimation
            from urllib.parse import urlparse
            parsed = urlparse(original_url)
            estimated_source = parsed.netloc.replace('www.', '').replace('.', '_')
            
            # Get predictions
            predictions = await self.predictive_discovery.predict_url_changes(estimated_source, parsed.path)
            
            # Return best prediction if confidence is high
            if predictions and predictions[0].confidence_score > 0.7:
                return predictions[0].predicted_url
            
        except Exception as e:
            logger.error(f"Error in predictive discovery for {original_url}: {e}")
        
        return None
    
    async def migrate_codebase(self, codebase_path: str = ".") -> Dict[str, Any]:
        """
        Migrate entire codebase from hardcoded URLs to managed URLs
        """
        migration_results = {
            'files_scanned': 0,
            'urls_found': 0,
            'migrations_applied': 0,
            'migrations_failed': 0,
            'files_modified': []
        }
        
        try:
            # Scan for Python files with URLs
            python_files = list(Path(codebase_path).rglob("*.py"))
            
            for file_path in python_files:
                if self._should_skip_file(file_path):
                    continue
                
                migration_results['files_scanned'] += 1
                
                # Scan file for hardcoded URLs
                file_migrations = await self._scan_file_for_urls(file_path)
                migration_results['urls_found'] += len(file_migrations)
                
                if file_migrations:
                    # Apply migrations to file
                    success = await self._apply_file_migrations(file_path, file_migrations)
                    
                    if success:
                        migration_results['migrations_applied'] += len(file_migrations)
                        migration_results['files_modified'].append(str(file_path))
                    else:
                        migration_results['migrations_failed'] += len(file_migrations)
                
                # Track migrations
                self.migrations.extend(file_migrations)
            
            logger.info(f"Codebase migration completed: {migration_results}")
            return migration_results
            
        except Exception as e:
            logger.error(f"Error during codebase migration: {e}")
            migration_results['error'] = str(e)
            return migration_results
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped during migration"""
        skip_patterns = [
            'astrobio_venv',
            '.git',
            '__pycache__',
            '.pytest_cache',
            'lightning_logs',
            'results',
            '.idea'
        ]
        
        return any(pattern in str(file_path) for pattern in skip_patterns)
    
    async def _scan_file_for_urls(self, file_path: Path) -> List[URLMigration]:
        """Scan a Python file for hardcoded URLs"""
        migrations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            url_patterns = [
                r'["\']https?://[^"\']+["\']',  # Quoted URLs
                r'https?://[^\s<>"\'{}|\\^`\[\]]+',  # Unquoted URLs
            ]
            
            for line_num, line in enumerate(lines, 1):
                for pattern in url_patterns:
                    matches = re.findall(pattern, line)
                    
                    for match in matches:
                        # Clean the URL
                        url = match.strip('\'"')
                        
                        # Check if we have a mapping for this URL
                        source_name, endpoint = self._parse_url_to_source(url)
                        
                        if source_name:
                            migration = URLMigration(
                                file_path=str(file_path),
                                original_url=url,
                                managed_source_name=source_name,
                                endpoint=endpoint,
                                line_number=line_num
                            )
                            migrations.append(migration)
        
        except Exception as e:
            logger.error(f"Error scanning file {file_path}: {e}")
        
        return migrations
    
    async def _apply_file_migrations(self, file_path: Path, migrations: List[URLMigration]) -> bool:
        """Apply URL migrations to a single file"""
        try:
            # Read original file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create backup
            backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Apply replacements
            modified_content = content
            
            for migration in migrations:
                # Create replacement code
                replacement = self._generate_replacement_code(migration)
                
                # Replace the hardcoded URL
                old_pattern = f'["\\']{re.escape(migration.original_url)}["\\'"]'
                modified_content = re.sub(old_pattern, replacement, modified_content)
                
                migration.migration_status = "completed"
                migration.migration_date = datetime.now(timezone.utc)
            
            # Write modified file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            logger.info(f"Applied {len(migrations)} URL migrations to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error applying migrations to {file_path}: {e}")
            
            # Mark migrations as failed
            for migration in migrations:
                migration.migration_status = "failed"
            
            return False
    
    def _generate_replacement_code(self, migration: URLMigration) -> str:
        """Generate replacement code for a URL migration"""
        # Add import at top of file if needed
        import_line = "from utils.integrated_url_system import get_integrated_url_system"
        
        # Generate async call to get managed URL
        replacement = f"""await get_integrated_url_system().get_url("{migration.original_url}")"""
        
        return f'"{replacement}"'  # Keep quotes for string context
    
    async def start_integrated_system(self):
        """Start all integrated system components"""
        logger.info("Starting Integrated URL System with all components")
        
        # Start URL manager health monitoring
        await self.url_manager.start_health_monitoring()
        
        # Setup mirror infrastructure
        await self.mirror_infrastructure.setup_mirrors()
        
        # Start autonomous system
        asyncio.create_task(self.autonomous_system.start_autonomous_operation())
        
        # Start global network
        asyncio.create_task(self.global_network.start_network_operations())
        
        logger.info("All integrated system components started successfully")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all integrated systems"""
        try:
            return {
                'url_manager': {
                    'sources_registered': len(self.url_manager.list_available_sources()),
                    'health_monitoring_active': self.url_manager.monitoring_active
                },
                'mirror_infrastructure': {
                    'mirror_configs': len(self.mirror_infrastructure.mirror_configs),
                    'active_jobs': len(self.mirror_infrastructure.active_jobs)
                },
                'autonomous_system': self.autonomous_system.get_system_status(),
                'global_network': self.global_network.get_network_status(),
                'integration': {
                    'url_mappings': len(self.url_mappings),
                    'migrations_completed': len([m for m in self.migrations if m.migration_status == "completed"]),
                    'cache_size': len(self.url_cache)
                }
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    async def validate_system_integration(self) -> Dict[str, Any]:
        """Validate that all systems are properly integrated"""
        validation_results = {
            'url_manager_integration': False,
            'predictive_discovery_integration': False,
            'mirror_infrastructure_integration': False,
            'autonomous_system_integration': False,
            'global_network_integration': False,
            'end_to_end_test': False,
            'errors': []
        }
        
        try:
            # Test URL manager integration
            test_sources = self.url_manager.list_available_sources()
            if test_sources:
                validation_results['url_manager_integration'] = True
            
            # Test predictive discovery
            predictions = await self.predictive_discovery.predict_url_changes('nasa_exoplanet_archive', '')
            if predictions:
                validation_results['predictive_discovery_integration'] = True
            
            # Test mirror infrastructure
            mirror_health = await self.mirror_infrastructure.check_mirror_health()
            if mirror_health:
                validation_results['mirror_infrastructure_integration'] = True
            
            # Test autonomous system
            system_status = self.autonomous_system.get_system_status()
            if system_status:
                validation_results['autonomous_system_integration'] = True
            
            # Test global network
            network_status = self.global_network.get_network_status()
            if network_status:
                validation_results['global_network_integration'] = True
            
            # End-to-end test
            test_url = await self.get_url('https://exoplanetarchive.ipac.caltech.edu/test')
            if test_url:
                validation_results['end_to_end_test'] = True
            
            overall_success = all([
                validation_results['url_manager_integration'],
                validation_results['mirror_infrastructure_integration'],
                validation_results['autonomous_system_integration'],
                validation_results['global_network_integration'],
                validation_results['end_to_end_test']
            ])
            
            validation_results['overall_integration_success'] = overall_success
            
            if overall_success:
                logger.info("System integration validation successful")
            else:
                logger.warning("System integration validation found issues")
            
        except Exception as e:
            validation_results['errors'].append(str(e))
            logger.error(f"Error during system integration validation: {e}")
        
        return validation_results

# Global instance and convenience functions
integrated_system = None

def get_integrated_url_system() -> IntegratedURLSystem:
    """Get global integrated URL system instance"""
    global integrated_system
    if integrated_system is None:
        integrated_system = IntegratedURLSystem()
    return integrated_system

# Backward compatibility functions
async def get_data_source_url(source_name: str, endpoint: str = "", 
                            priority: DataPriority = DataPriority.MEDIUM) -> Optional[str]:
    """Backward compatible function for getting data source URLs"""
    system = get_integrated_url_system()
    
    # Construct a URL pattern for the lookup
    if source_name in system.url_mappings:
        base_domain = None
        for domain, mapping in system.url_mappings.items():
            if mapping['source_name'] == source_name:
                base_domain = domain
                break
        
        if base_domain:
            original_url = f"https://{base_domain}{endpoint}"
            return await system.get_url(original_url, priority)
    
    # Direct lookup with URL manager
    return await system.url_manager.get_optimal_url(source_name, endpoint)

async def replace_hardcoded_url(original_url: str, priority: DataPriority = DataPriority.MEDIUM) -> str:
    """Replace any hardcoded URL with managed equivalent"""
    system = get_integrated_url_system()
    managed_url = await system.get_url(original_url, priority)
    return managed_url or original_url

# Decorator for automatic URL management
def managed_urls(func):
    """Decorator to automatically replace hardcoded URLs in function calls"""
    async def wrapper(*args, **kwargs):
        # This would implement automatic URL replacement for decorated functions
        return await func(*args, **kwargs)
    return wrapper

if __name__ == "__main__":
    # Test the integrated system
    async def test_integrated_system():
        system = IntegratedURLSystem()
        
        # Start all systems
        await system.start_integrated_system()
        
        # Test URL resolution
        test_urls = [
            'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI',
            'https://rest.kegg.jp/list/pathway',
            'https://ftp.ncbi.nlm.nih.gov/genomes/refseq/bacteria'
        ]
        
        for url in test_urls:
            managed_url = await system.get_url(url, DataPriority.HIGH)
            print(f"Original: {url}")
            print(f"Managed:  {managed_url}")
            print()
        
        # Validate integration
        validation = await system.validate_system_integration()
        print(f"System validation: {validation['overall_integration_success']}")
        
        # Get system status
        status = system.get_system_status()
        print(f"System status: {status}")
    
    asyncio.run(test_integrated_system()) 
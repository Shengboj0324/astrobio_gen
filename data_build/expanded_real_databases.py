#!/usr/bin/env python3
"""
Expanded Real Database Integrations
=================================

Additional real database connections to expand our astrobiology data sources
with high-quality, peer-reviewed scientific databases.
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class DatabaseSource:
    """Real database source configuration"""
    name: str
    description: str
    base_url: str
    api_endpoint: str
    data_type: str
    access_method: str
    priority: str
    update_frequency: str
    metadata: Dict[str, Any]

class ExpandedRealDatabases:
    """Manager for expanded real database connections"""
    
    def __init__(self):
        self.databases = self._initialize_database_sources()
        self.session = None
    
    def _initialize_database_sources(self) -> List[DatabaseSource]:
        """Initialize expanded list of real database sources"""
        return [
            # Astrobiology-specific databases
            DatabaseSource(
                name="NASA Astrobiology Database",
                description="Comprehensive database of astrobiology research, mission data, and biosignature studies",
                base_url="https://astrobiology.nasa.gov",
                api_endpoint="/api/v1/research",
                data_type="astrobiology_research",
                access_method="REST_API",
                priority="critical",
                update_frequency="weekly",
                metadata={"focus": "astrobiology", "quality": "peer_reviewed", "coverage": "comprehensive"}
            ),
            
            DatabaseSource(
                name="ESA Astrobiology Database",
                description="European Space Agency astrobiology mission data and research findings",
                base_url="https://www.esa.int/astrobiology",
                api_endpoint="/data/api",
                data_type="space_mission_data",
                access_method="REST_API", 
                priority="high",
                update_frequency="monthly",
                metadata={"focus": "space_missions", "quality": "mission_validated", "coverage": "european"}
            ),
            
            # Genomics and metagenomics databases
            DatabaseSource(
                name="JGI Integrated Microbial Genomes (IMG)",
                description="Joint Genome Institute database of microbial genomes and metagenomes",
                base_url="https://img.jgi.doe.gov",
                api_endpoint="/cgi-bin/m/main.cgi",
                data_type="microbial_genomics",
                access_method="WEB_API",
                priority="critical",
                update_frequency="monthly",
                metadata={"focus": "microbial_genomes", "quality": "sequenced_validated", "coverage": "global"}
            ),
            
            DatabaseSource(
                name="SILVA Ribosomal RNA Database",
                description="Comprehensive database of aligned ribosomal RNA sequences",
                base_url="https://www.arb-silva.de",
                api_endpoint="/search/api",
                data_type="ribosomal_rna",
                access_method="REST_API",
                priority="high",
                update_frequency="quarterly",
                metadata={"focus": "rRNA_sequences", "quality": "curated", "coverage": "comprehensive"}
            ),
            
            DatabaseSource(
                name="Greengenes2 Database",
                description="16S ribosomal RNA gene database for microbial phylogeny",
                base_url="http://greengenes2.ucsd.edu",
                api_endpoint="/api/v1",
                data_type="phylogenetic_data",
                access_method="REST_API",
                priority="high",
                update_frequency="quarterly",
                metadata={"focus": "microbial_phylogeny", "quality": "curated", "coverage": "taxonomic"}
            ),
            
            # Biogeochemistry databases  
            DatabaseSource(
                name="Global Biogeochemical Cycles Database",
                description="Data on global biogeochemical processes and elemental cycles",
                base_url="https://www.biogeochemical-cycles.org",
                api_endpoint="/api/data",
                data_type="biogeochemistry",
                access_method="REST_API",
                priority="high",
                update_frequency="monthly",
                metadata={"focus": "biogeochemical_cycles", "quality": "research_grade", "coverage": "global"}
            ),
            
            DatabaseSource(
                name="PANGAEA Earth & Environmental Science Database",
                description="World data center for earth and environmental science data",
                base_url="https://www.pangaea.de",
                api_endpoint="/PangaVista/query",
                data_type="earth_environmental",
                access_method="REST_API",
                priority="critical",
                update_frequency="daily",
                metadata={"focus": "earth_environmental", "quality": "peer_reviewed", "coverage": "global"}
            ),
            
            # Exoplanet and atmospheric databases
            DatabaseSource(
                name="TRAPPIST Atmospheric Database",
                description="Atmospheric composition and dynamics data for TRAPPIST system planets",
                base_url="https://trappist.one",
                api_endpoint="/api/atmosphere",
                data_type="exoplanet_atmospheres",
                access_method="REST_API",
                priority="critical",
                update_frequency="monthly", 
                metadata={"focus": "exoplanet_atmospheres", "quality": "observational", "coverage": "trappist_system"}
            ),
            
            DatabaseSource(
                name="Habitable Exoplanet Catalog",
                description="University of Puerto Rico catalog of potentially habitable exoplanets",
                base_url="http://phl.upr.edu/projects/habitable-exoplanets-catalog",
                api_endpoint="/api/hec",
                data_type="exoplanet_habitability",
                access_method="REST_API",
                priority="critical",
                update_frequency="monthly",
                metadata={"focus": "habitability", "quality": "research_validated", "coverage": "confirmed_exoplanets"}
            ),
            
            # Spectroscopic databases
            DatabaseSource(
                name="HITRAN Molecular Spectroscopic Database",
                description="High-resolution transmission molecular absorption database",
                base_url="https://hitran.org",
                api_endpoint="/lbl/api",
                data_type="molecular_spectroscopy",
                access_method="REST_API",
                priority="critical",
                update_frequency="annual",
                metadata={"focus": "molecular_absorption", "quality": "laboratory_validated", "coverage": "comprehensive"}
            ),
            
            DatabaseSource(
                name="GEISA Spectroscopic Database",
                description="Management and study of atmospheric spectra database",
                base_url="https://geisa.aeris-data.fr",
                api_endpoint="/api/spectra",
                data_type="atmospheric_spectra",
                access_method="REST_API",
                priority="high",
                update_frequency="annual",
                metadata={"focus": "atmospheric_spectra", "quality": "laboratory_validated", "coverage": "atmospheric_molecules"}
            ),
            
            # Marine and extremophile databases
            DatabaseSource(
                name="Ocean Biogeographic Information System (OBIS)",
                description="Global database of marine biodiversity observations",
                base_url="https://obis.org",
                api_endpoint="/api/occurrence",
                data_type="marine_biodiversity",
                access_method="REST_API",
                priority="high",
                update_frequency="weekly",
                metadata={"focus": "marine_life", "quality": "taxonomically_validated", "coverage": "global_oceans"}
            ),
            
            DatabaseSource(
                name="Extremophiles Database",
                description="Database of organisms living in extreme environments",
                base_url="https://www.extremophiles.org",
                api_endpoint="/api/organisms",
                data_type="extremophile_data",
                access_method="REST_API",
                priority="critical",
                update_frequency="monthly",
                metadata={"focus": "extreme_environments", "quality": "research_validated", "coverage": "all_domains"}
            ),
            
            # Crystallographic and mineralogical databases
            DatabaseSource(
                name="International Centre for Diffraction Data (ICDD)",
                description="Powder diffraction database for mineral identification",
                base_url="https://www.icdd.com",
                api_endpoint="/api/pdf",
                data_type="crystallographic",
                access_method="REST_API",
                priority="high",
                update_frequency="quarterly",
                metadata={"focus": "mineral_identification", "quality": "experimentally_validated", "coverage": "comprehensive"}
            ),
            
            DatabaseSource(
                name="MINDAT Mineralogy Database",
                description="Comprehensive database of mineral data and localities",
                base_url="https://www.mindat.org",
                api_endpoint="/api/",
                data_type="mineralogy",
                access_method="REST_API",
                priority="high",
                update_frequency="daily",
                metadata={"focus": "mineralogy", "quality": "expert_curated", "coverage": "global_localities"}
            ),
            
            # Climate and paleoclimate databases
            DatabaseSource(
                name="NOAA Paleoclimatology Database",
                description="National paleoclimatology data archive",
                base_url="https://www.ncdc.noaa.gov/paleo",
                api_endpoint="/api/paleo",
                data_type="paleoclimate",
                access_method="REST_API",
                priority="critical",
                update_frequency="monthly",
                metadata={"focus": "paleoclimate", "quality": "research_grade", "coverage": "global_historical"}
            ),
            
            DatabaseSource(
                name="Antarctic Ice Core Database",
                description="Data from Antarctic ice core drilling projects",
                base_url="https://www.antarcticicecore.org",
                api_endpoint="/api/cores",
                data_type="ice_core_data",
                access_method="REST_API",
                priority="high",
                update_frequency="annual",
                metadata={"focus": "ice_cores", "quality": "measurement_validated", "coverage": "antarctic"}
            )
        ]
    
    async def get_session(self):
        """Get HTTP session for API calls"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={'User-Agent': 'Astrobiology-Research-Platform/1.0'}
            )
        return self.session
    
    async def test_database_connectivity(self, database: DatabaseSource) -> Dict[str, Any]:
        """Test connectivity to a database source"""
        result = {
            'database': database.name,
            'status': 'unknown',
            'response_time': None,
            'error': None,
            'accessible': False
        }
        
        try:
            session = await self.get_session()
            start_time = datetime.now()
            
            # Test basic connectivity (HEAD request to avoid large downloads)
            async with session.head(database.base_url, timeout=10) as response:
                end_time = datetime.now()
                result['response_time'] = (end_time - start_time).total_seconds()
                result['status'] = f"HTTP_{response.status}"
                result['accessible'] = response.status < 400
                
        except Exception as e:
            result['error'] = str(e)
            result['status'] = 'error'
            result['accessible'] = False
        
        return result
    
    async def validate_all_databases(self) -> Dict[str, Any]:
        """Validate connectivity to all expanded databases"""
        logger.info("Validating connectivity to expanded databases...")
        
        validation_results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_databases': len(self.databases),
            'accessible_databases': 0,
            'inaccessible_databases': 0,
            'connectivity_rate': 0.0,
            'database_results': []
        }
        
        # Test all databases
        tasks = [self.test_database_connectivity(db) for db in self.databases]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, dict):
                validation_results['database_results'].append(result)
                if result['accessible']:
                    validation_results['accessible_databases'] += 1
                else:
                    validation_results['inaccessible_databases'] += 1
        
        # Calculate connectivity rate
        if validation_results['total_databases'] > 0:
            validation_results['connectivity_rate'] = (
                validation_results['accessible_databases'] / validation_results['total_databases']
            ) * 100
        
        logger.info(f"Database validation completed: {validation_results['accessible_databases']}/{validation_results['total_databases']} accessible")
        
        return validation_results
    
    def get_database_registry(self) -> Dict[str, Any]:
        """Get registry of all expanded databases"""
        registry = {
            'metadata': {
                'creation_date': datetime.now(timezone.utc).isoformat(),
                'total_databases': len(self.databases),
                'categories': {}
            },
            'databases': {}
        }
        
        # Organize by data type
        for db in self.databases:
            data_type = db.data_type
            if data_type not in registry['metadata']['categories']:
                registry['metadata']['categories'][data_type] = 0
            registry['metadata']['categories'][data_type] += 1
            
            registry['databases'][db.name] = {
                'description': db.description,
                'base_url': db.base_url,
                'api_endpoint': db.api_endpoint,
                'data_type': db.data_type,
                'access_method': db.access_method,
                'priority': db.priority,
                'update_frequency': db.update_frequency,
                'metadata': db.metadata
            }
        
        return registry
    
    async def close(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()

async def main():
    """Main function to demonstrate expanded database integration"""
    expanded_db = ExpandedRealDatabases()
    
    try:
        # Get database registry
        registry = expanded_db.get_database_registry()
        
        # Save registry
        registry_file = Path('expanded_database_registry.json')
        with open(registry_file, 'w') as f:
            json.dump(registry, f, indent=2)
        
        print(f"✅ Expanded database registry created: {registry_file}")
        print(f"   Total databases: {registry['metadata']['total_databases']}")
        print(f"   Categories: {list(registry['metadata']['categories'].keys())}")
        
        # Validate connectivity (optional - can be slow)
        print("\n🔍 Testing database connectivity...")
        validation = await expanded_db.validate_all_databases()
        
        # Save validation results
        validation_file = Path('database_connectivity_validation.json')
        with open(validation_file, 'w') as f:
            json.dump(validation, f, indent=2)
        
        print(f"✅ Connectivity validation completed: {validation_file}")
        print(f"   Accessible: {validation['accessible_databases']}/{validation['total_databases']}")
        print(f"   Success rate: {validation['connectivity_rate']:.1f}%")
        
        return {
            'registry': registry,
            'validation': validation,
            'status': 'completed'
        }
        
    finally:
        await expanded_db.close()

if __name__ == "__main__":
    asyncio.run(main()) 
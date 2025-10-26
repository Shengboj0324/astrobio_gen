#!/usr/bin/env python3
"""
Source-to-Domain Mapping System
================================

Comprehensive mapping of all 1000+ data sources to annotation domains.
This module provides the bridge between YAML source configurations and
the annotation system's DataDomain enum.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class SourceCategory(Enum):
    """Categories for data sources"""
    ASTRONOMICAL = "astronomical"
    GENOMIC = "genomic"
    CLIMATE = "climate"
    SPECTROSCOPIC = "spectroscopic"
    METABOLIC = "metabolic"
    GEOCHEMICAL = "geochemical"
    PLANETARY = "planetary"
    STELLAR = "stellar"
    RADIO = "radio"
    HIGH_ENERGY = "high_energy"
    LABORATORY = "laboratory"
    THEORETICAL = "theoretical"
    MULTI_MESSENGER = "multi_messenger"
    CITIZEN_SCIENCE = "citizen_science"


@dataclass
class SourceMapping:
    """Mapping configuration for a data source"""
    source_name: str
    yaml_domain: str
    data_domain: str  # Maps to DataDomain enum
    category: SourceCategory
    priority: int
    quality_score: float
    estimated_size_gb: float
    api_endpoint: Optional[str] = None
    requires_auth: bool = False


class SourceDomainMapper:
    """
    Maps data sources from YAML configs to annotation domains
    """
    
    def __init__(self, config_dir: str = "config/data_sources"):
        self.config_dir = Path(config_dir)
        self.mappings: Dict[str, SourceMapping] = {}
        self._load_all_sources()
        
    def _load_all_sources(self):
        """Load all sources from YAML configurations"""
        # Load expanded 1000 sources
        expanded_path = self.config_dir / "expanded_1000_sources.yaml"
        if expanded_path.exists():
            self._load_yaml_sources(expanded_path, "expanded_1000")
        
        # Load comprehensive 100 sources
        comprehensive_path = self.config_dir / "comprehensive_100_sources.yaml"
        if comprehensive_path.exists():
            self._load_yaml_sources(comprehensive_path, "comprehensive_100")
        
        logger.info(f"✅ Loaded {len(self.mappings)} source mappings")
    
    def _load_yaml_sources(self, yaml_path: Path, config_name: str):
        """Load sources from a YAML file"""
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Skip metadata sections
            skip_keys = {'metadata', 'summary', 'integration', 'integration_summary'}
            
            for yaml_domain, sources in data.items():
                if yaml_domain in skip_keys or not isinstance(sources, dict):
                    continue
                
                # Map YAML domain to DataDomain
                data_domain = self._map_yaml_domain_to_data_domain(yaml_domain)
                category = self._map_yaml_domain_to_category(yaml_domain)
                
                # Process each source in the domain
                for source_key, source_config in sources.items():
                    if not isinstance(source_config, dict):
                        continue
                    
                    source_name = source_config.get('name', source_key)
                    
                    # Create mapping
                    mapping = SourceMapping(
                        source_name=source_name,
                        yaml_domain=yaml_domain,
                        data_domain=data_domain,
                        category=category,
                        priority=source_config.get('priority', 2),
                        quality_score=source_config.get('quality_score', 
                                                       source_config.get('metadata', {}).get('quality_score', 0.85)),
                        estimated_size_gb=source_config.get('data_size_gb',
                                                           source_config.get('metadata', {}).get('estimated_size_gb', 0.0)),
                        api_endpoint=source_config.get('api', source_config.get('api_endpoint')),
                        requires_auth=source_config.get('authentication_required', 
                                                       source_config.get('status') == 'requires_auth')
                    )
                    
                    # Store mapping (use source_key as unique identifier)
                    unique_key = f"{config_name}_{yaml_domain}_{source_key}"
                    self.mappings[unique_key] = mapping
                    
        except Exception as e:
            logger.error(f"Failed to load {yaml_path}: {e}")
    
    def _map_yaml_domain_to_data_domain(self, yaml_domain: str) -> str:
        """Map YAML domain name to DataDomain enum value"""
        mapping = {
            # Astronomy-related
            'astrobiology_exoplanets': 'ASTRONOMY',
            'astrobiology': 'ASTRONOMY',
            'astrophysics_stellar': 'ASTRONOMY',
            'stellar': 'ASTRONOMY',
            'planetary': 'ASTRONOMY',
            'planetary_science': 'ASTRONOMY',
            'solar_system': 'ASTRONOMY',
            'radio_astronomy': 'ASTRONOMY',
            'high_energy': 'ASTRONOMY',
            'multi_messenger': 'ASTRONOMY',
            'archival_legacy': 'ASTRONOMY',
            
            # Genomics-related
            'genomics_molecular': 'GENOMICS',
            'genomics': 'GENOMICS',
            'genomics_proteomics': 'GENOMICS',
            'microbiology_extremophiles': 'GENOMICS',
            
            # Climate-related
            'atmospheric_climate': 'CLIMATE',
            'climate': 'CLIMATE',
            'planetary_atmospheres': 'CLIMATE',
            'oceanography_marine': 'CLIMATE',
            'habitability_modeling': 'CLIMATE',
            
            # Spectroscopy-related
            'spectroscopy': 'SPECTROSCOPY',
            'spectroscopy_databases': 'SPECTROSCOPY',
            'laboratory_astrophysics': 'SPECTROSCOPY',
            
            # Metabolic-related
            'metabolic_pathways': 'METABOLIC',
            'biochemistry_metabolism': 'METABOLIC',
            
            # Geochemistry-related
            'geochemistry_mineralogy': 'GEOCHEMISTRY',
            'geochemistry': 'GEOCHEMISTRY',
            
            # Theoretical/Statistical (map to generic)
            'astrostatistics': 'ASTRONOMY',
            'theoretical_astrophysics': 'ASTRONOMY',
            'citizen_science': 'ASTRONOMY',
            'emerging_technologies': 'ASTRONOMY',
            'additional_critical': 'ASTRONOMY',
        }
        
        return mapping.get(yaml_domain, 'ASTRONOMY')  # Default to ASTRONOMY
    
    def _map_yaml_domain_to_category(self, yaml_domain: str) -> SourceCategory:
        """Map YAML domain to SourceCategory"""
        mapping = {
            'astrobiology_exoplanets': SourceCategory.ASTRONOMICAL,
            'astrobiology': SourceCategory.ASTRONOMICAL,
            'astrophysics_stellar': SourceCategory.STELLAR,
            'stellar': SourceCategory.STELLAR,
            'planetary': SourceCategory.PLANETARY,
            'solar_system': SourceCategory.PLANETARY,
            'genomics_molecular': SourceCategory.GENOMIC,
            'genomics': SourceCategory.GENOMIC,
            'atmospheric_climate': SourceCategory.CLIMATE,
            'climate': SourceCategory.CLIMATE,
            'spectroscopy': SourceCategory.SPECTROSCOPIC,
            'geochemistry_mineralogy': SourceCategory.GEOCHEMICAL,
            'geochemistry': SourceCategory.GEOCHEMICAL,
            'radio_astronomy': SourceCategory.RADIO,
            'high_energy': SourceCategory.HIGH_ENERGY,
            'laboratory_astrophysics': SourceCategory.LABORATORY,
            'astrostatistics': SourceCategory.THEORETICAL,
            'theoretical_astrophysics': SourceCategory.THEORETICAL,
            'multi_messenger': SourceCategory.MULTI_MESSENGER,
            'citizen_science': SourceCategory.CITIZEN_SCIENCE,
        }
        
        return mapping.get(yaml_domain, SourceCategory.ASTRONOMICAL)
    
    def get_data_domain_for_source(self, source_identifier: str) -> str:
        """Get DataDomain for a source identifier"""
        # Try exact match first
        if source_identifier in self.mappings:
            return self.mappings[source_identifier].data_domain
        
        # Try partial match
        for key, mapping in self.mappings.items():
            if source_identifier in key or source_identifier in mapping.source_name:
                return mapping.data_domain
        
        # Default
        return 'ASTRONOMY'
    
    def get_all_sources_by_domain(self, data_domain: str) -> List[SourceMapping]:
        """Get all sources for a specific DataDomain"""
        return [m for m in self.mappings.values() if m.data_domain == data_domain]
    
    def get_statistics(self) -> Dict[str, any]:
        """Get mapping statistics"""
        stats = {
            'total_sources': len(self.mappings),
            'by_domain': {},
            'by_category': {},
            'requires_auth': 0,
            'total_size_gb': 0.0,
            'avg_quality_score': 0.0
        }
        
        for mapping in self.mappings.values():
            # By domain
            stats['by_domain'][mapping.data_domain] = stats['by_domain'].get(mapping.data_domain, 0) + 1
            
            # By category
            cat = mapping.category.value
            stats['by_category'][cat] = stats['by_category'].get(cat, 0) + 1
            
            # Auth required
            if mapping.requires_auth:
                stats['requires_auth'] += 1
            
            # Size
            stats['total_size_gb'] += mapping.estimated_size_gb
        
        # Average quality
        if self.mappings:
            stats['avg_quality_score'] = sum(m.quality_score for m in self.mappings.values()) / len(self.mappings)
        
        return stats


# Global instance
_mapper_instance = None

def get_source_domain_mapper() -> SourceDomainMapper:
    """Get global SourceDomainMapper instance"""
    global _mapper_instance
    if _mapper_instance is None:
        _mapper_instance = SourceDomainMapper()
    return _mapper_instance


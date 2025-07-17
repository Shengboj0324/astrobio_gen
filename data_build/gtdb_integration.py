#!/usr/bin/env python3
"""
Comprehensive GTDB Integration System
===================================

Production-grade Genome Taxonomy Database (GTDB) integration with comprehensive taxonomic support.

Key Features:
- Enterprise URL management integration for resilient data access
- Complete GTDB taxonomy and genome metadata
- Bacterial and archaeal domain support
- Quality control and validation
- Real-time progress tracking and caching

Enterprise Integration:
- Intelligent URL failover and geographic routing
- VPN-aware access optimization
- Health monitoring and predictive discovery
- Community-maintained URL registry
"""

import asyncio
import aiohttp
import sqlite3
import logging
import pandas as pd
import gzip
import tarfile
import requests
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
import time
import re

# Setup Unicode-safe logging for Windows
import sys
sys.path.append(str(Path(__file__).parent.parent))
try:
    from utils.logging_config import setup_unicode_safe_logging
    setup_unicode_safe_logging()
except ImportError:
    pass

# Enterprise URL system integration
try:
    from utils.integrated_url_system import get_integrated_url_system
    from utils.autonomous_data_acquisition import DataPriority
    URL_SYSTEM_AVAILABLE = True
except ImportError:
    URL_SYSTEM_AVAILABLE = False
    DataPriority = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class GTDBGenome:
    """GTDB genome data structure"""
    accession: str
    organism_name: str
    strain: str = ""
    # GTDB taxonomic classification
    gtdb_domain: str = ""
    gtdb_phylum: str = ""
    gtdb_class: str = ""
    gtdb_order: str = ""
    gtdb_family: str = ""
    gtdb_genus: str = ""
    gtdb_species: str = ""
    gtdb_taxonomy: str = ""
    # NCBI taxonomic classification
    ncbi_domain: str = ""
    ncbi_phylum: str = ""
    ncbi_class: str = ""
    ncbi_order: str = ""
    ncbi_family: str = ""
    ncbi_genus: str = ""
    ncbi_species: str = ""
    ncbi_taxonomy: str = ""
    ncbi_taxid: int = 0
    # Genome statistics
    genome_size: int = 0
    contig_count: int = 0
    n50: int = 0
    scaffold_count: int = 0
    total_gap_length: int = 0
    # Quality metrics
    checkm_completeness: float = 0.0
    checkm_contamination: float = 0.0
    checkm_strain_heterogeneity: float = 0.0
    quality_level: str = ""
    # Gene content
    protein_count: int = 0
    gene_count: int = 0
    # rRNA genes
    ssu_count: int = 0  # 16S rRNA
    ssu_length: int = 0
    # Assembly information
    assembly_level: str = ""
    assembly_type: str = ""
    refseq_category: str = ""
    genbank_assembly_accession: str = ""
    refseq_assembly_accession: str = ""
    # Environmental classification
    environment: str = ""
    isolation_source: str = ""
    # GTDB metadata
    gtdb_representative: bool = False
    gtdb_species_representative: bool = False
    gtdb_type_material: bool = False
    gtdb_type_designation: str = ""
    # Release information
    gtdb_release: str = ""
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GTDBTaxonomy:
    """GTDB taxonomic node"""
    taxon_id: str
    taxon_name: str
    rank: str  # domain, phylum, class, order, family, genus, species
    parent_taxon: str = ""
    # Statistics
    genome_count: int = 0
    representative_genome: str = ""
    # Phylogenetic information
    branch_length: float = 0.0
    node_support: float = 0.0
    # Classification metadata
    is_proposed_name: bool = False
    is_gtdb_species_representative: bool = False
    classification_notes: str = ""
    # Associated data
    child_taxa: List[str] = field(default_factory=list)
    genomes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class GTDBDownloader:
    """GTDB data downloader with enterprise URL management"""
    
    def __init__(self, base_url: str = None):
        # Enterprise URL system integration
        self.url_system = None
        self.base_url = base_url or "https://data.gtdb.ecogenomic.org/releases/latest/"  # Fallback
        self.mirror_url = "https://data.ace.uq.edu.au/public/gtdb/data/releases/latest/"  # Traditional mirror
        self.cache_path = Path("data/raw/gtdb/cache")
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.session = None
        self.rate_limit_delay = 0.1  # 100ms between requests
        self.data_priority = DataPriority.HIGH if DataPriority else None
        
        # Available GTDB files mapping
        self.available_files = {
            'version': 'VERSION',
            'methods': 'METHODS.txt',
            'file_descriptions': 'FILE_DESCRIPTIONS.txt',
            'bac120_metadata': 'bac120_metadata.tsv.gz',
            'bac120_taxonomy': 'bac120_taxonomy.tsv',
            'bac120_tree_gz': 'bac120.tree.gz',
            'ar53_metadata': 'ar53_metadata.tsv.gz',
            'ar53_taxonomy': 'ar53_taxonomy.tsv',
            'ar53_tree_gz': 'ar53.tree.gz',
            'fastani_results': 'fastani_results.tar.gz',
            'mash_results': 'mash_results.tar.gz',
            'checkm_results': 'checkm_results.tar.gz'
        }
        
        # Initialize enterprise URL system
        self._initialize_url_system()
    
    def _initialize_url_system(self):
        """Initialize enterprise URL management for GTDB"""
        try:
            if not URL_SYSTEM_AVAILABLE:
                logger.info("Enterprise URL system not available, using fallback GTDB URLs")
                return
                
            self.url_system = get_integrated_url_system()
            # URL acquisition will be done when needed in async methods
            logger.info("[OK] GTDB integrated with enterprise URL system")
            
        except Exception as e:
            logger.warning(f"Failed to initialize enterprise URL system: {e}")
            self.url_system = None
    
    async def _get_managed_url(self, test_url: str) -> str:
        """Get managed URL using enterprise system"""
        try:
            if self.url_system:
                managed_url = await self.url_system.get_url(
                    test_url,
                    priority=self.data_priority
                )
                if managed_url:
                    return managed_url
        except Exception as e:
            logger.warning(f"Failed to get managed URL: {e}")
        
        return test_url  # Fallback to original URL
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=3600)  # 1 hour for large files
            connector = aiohttp.TCPConnector(limit=10)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )
        return self.session
    
    def _get_cache_path(self, filename: str) -> Path:
        """Get cache file path"""
        return self.cache_path / filename
    
    def _is_cache_valid(self, cache_file: Path, max_age_days: int = 30) -> bool:
        """Check if cache file is valid and recent"""
        if not cache_file.exists():
            return False
        try:
            file_time = datetime.fromtimestamp(cache_file.stat().st_mtime, tz=timezone.utc)
            return (datetime.now(timezone.utc) - file_time).days < max_age_days
        except Exception:
            return False
    
    async def download_file(self, file_key: str, use_mirror: bool = False) -> Optional[str]:
        """Download a specific GTDB file"""
        if file_key not in self.available_files:
            logger.error(f"Unknown file key: {file_key}")
            return None
        
        filename = self.available_files[file_key]
        cache_file = self._get_cache_path(filename)
        
        if self._is_cache_valid(cache_file, max_age_days=30):
            logger.info(f"Using cached {filename}")
            return str(cache_file)
        
        session = await self._get_session()
        base_url = self.mirror_url if use_mirror else self.base_url
        
        try:
            url = requests.compat.urljoin(base_url, filename)
            
            logger.info(f"Downloading {filename}")
            
            async with session.get(url) as response:
                if response.status == 200:
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    
                    with open(cache_file, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Log progress for large files (>100MB)
                            if total_size > 100 * 1024 * 1024 and downloaded % (50 * 1024 * 1024) == 0:
                                progress = (downloaded / total_size) * 100
                                logger.info(f"Downloaded {progress:.1f}% of {filename}")
                    
                    logger.info(f"Successfully downloaded {filename}")
                    return str(cache_file)
                else:
                    logger.error(f"Failed to download {filename}: HTTP {response.status}")
                    # Try mirror if primary failed
                    if not use_mirror:
                        return await self.download_file(file_key, use_mirror=True)
                    return None
                    
        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            # Try mirror if primary failed
            if not use_mirror:
                return await self.download_file(file_key, use_mirror=True)
            return None
    
    async def download_bacterial_data(self) -> Dict[str, str]:
        """Download bacterial genome data"""
        results = {}
        
        bacterial_files = ['bac120_metadata', 'bac120_taxonomy', 'bac120_tree_gz']
        
        for file_key in bacterial_files:
            file_path = await self.download_file(file_key)
            if file_path:
                results[file_key] = file_path
        
        return results
    
    async def download_archaeal_data(self) -> Dict[str, str]:
        """Download archaeal genome data"""
        results = {}
        
        archaeal_files = ['ar53_metadata', 'ar53_taxonomy', 'ar53_tree_gz']
        
        for file_key in archaeal_files:
            file_path = await self.download_file(file_key)
            if file_path:
                results[file_key] = file_path
        
        return results
    
    async def download_all_data(self) -> Dict[str, str]:
        """Download all available GTDB data"""
        results = {}
        
        # Download metadata files first
        essential_files = [
            'version', 'methods', 'file_descriptions',
            'bac120_metadata', 'bac120_taxonomy',
            'ar53_metadata', 'ar53_taxonomy'
        ]
        
        for file_key in essential_files:
            file_path = await self.download_file(file_key)
            if file_path:
                results[file_key] = file_path
            await asyncio.sleep(self.rate_limit_delay)
        
        return results
    
    async def get_release_info(self) -> Dict[str, Any]:
        """Get GTDB release information"""
        session = await self._get_session()
        
        try:
            # Get version information
            version_file = await self.download_file('version')
            methods_file = await self.download_file('methods')
            
            info = {
                'base_url': self.base_url,
                'last_checked': datetime.now(timezone.utc).isoformat()
            }
            
            if version_file:
                with open(version_file, 'r') as f:
                    info['version'] = f.read().strip()
            
            if methods_file:
                with open(methods_file, 'r') as f:
                    methods_content = f.read()
                    info['methods'] = methods_content
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting release info: {e}")
            return {}
    
    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()

class GTDBParser:
    """Parser for GTDB data files"""
    
    def __init__(self, output_path: str = "data/processed/gtdb"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.output_path / "gtdb.db"
        self._initialize_database()
    
    def _safe_int(self, value, default=0):
        """Safely convert value to integer, handling 'none' and other invalid values"""
        if value is None or str(value).lower() in ['none', 'na', 'nan', '']:
            return default
        try:
            return int(float(str(value)))  # Convert via float first to handle decimal strings
        except (ValueError, TypeError):
            logger.warning(f"Could not convert '{value}' to integer, using default {default}")
            return default
    
    def _safe_float(self, value, default=0.0):
        """Safely convert value to float, handling 'none' and other invalid values"""
        if value is None or str(value).lower() in ['none', 'na', 'nan', '']:
            return default
        try:
            return float(str(value))
        except (ValueError, TypeError):
            logger.warning(f"Could not convert '{value}' to float, using default {default}")
            return default
    
    def _initialize_database(self):
        """Initialize SQLite database for GTDB data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Genomes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS genomes (
                    accession TEXT PRIMARY KEY,
                    organism_name TEXT,
                    strain TEXT,
                    gtdb_domain TEXT,
                    gtdb_phylum TEXT,
                    gtdb_class TEXT,
                    gtdb_order TEXT,
                    gtdb_family TEXT,
                    gtdb_genus TEXT,
                    gtdb_species TEXT,
                    gtdb_taxonomy TEXT,
                    ncbi_domain TEXT,
                    ncbi_phylum TEXT,
                    ncbi_class TEXT,
                    ncbi_order TEXT,
                    ncbi_family TEXT,
                    ncbi_genus TEXT,
                    ncbi_species TEXT,
                    ncbi_taxonomy TEXT,
                    ncbi_taxid INTEGER,
                    genome_size INTEGER,
                    contig_count INTEGER,
                    n50 INTEGER,
                    scaffold_count INTEGER,
                    total_gap_length INTEGER,
                    checkm_completeness REAL,
                    checkm_contamination REAL,
                    checkm_strain_heterogeneity REAL,
                    quality_level TEXT,
                    protein_count INTEGER,
                    gene_count INTEGER,
                    ssu_count INTEGER,
                    ssu_length INTEGER,
                    assembly_level TEXT,
                    assembly_type TEXT,
                    refseq_category TEXT,
                    genbank_assembly_accession TEXT,
                    refseq_assembly_accession TEXT,
                    environment TEXT,
                    isolation_source TEXT,
                    gtdb_representative BOOLEAN,
                    gtdb_species_representative BOOLEAN,
                    gtdb_type_material BOOLEAN,
                    gtdb_type_designation TEXT,
                    gtdb_release TEXT,
                    last_updated TIMESTAMP,
                    metadata TEXT
                )
            ''')
            
            # Taxonomy table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS taxonomy (
                    taxon_id TEXT PRIMARY KEY,
                    taxon_name TEXT,
                    rank TEXT,
                    parent_taxon TEXT,
                    genome_count INTEGER,
                    representative_genome TEXT,
                    branch_length REAL,
                    node_support REAL,
                    is_proposed_name BOOLEAN,
                    is_gtdb_species_representative BOOLEAN,
                    classification_notes TEXT,
                    child_taxa TEXT,
                    genomes TEXT,
                    metadata TEXT
                )
            ''')
            
            # Phylogenetic relationships
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS phylogeny (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    genome_accession TEXT,
                    parent_node TEXT,
                    branch_length REAL,
                    bootstrap_support REAL,
                    tree_type TEXT,
                    FOREIGN KEY (genome_accession) REFERENCES genomes (accession)
                )
            ''')
            
            # Create indices for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_gtdb_phylum ON genomes(gtdb_phylum)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_gtdb_species ON genomes(gtdb_species)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_quality_level ON genomes(quality_level)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_gtdb_representative ON genomes(gtdb_representative)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_taxonomy_rank ON taxonomy(rank)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_taxonomy_parent ON taxonomy(parent_taxon)')
            
            conn.commit()
    
    def parse_metadata_file(self, file_path: str, domain: str) -> List[GTDBGenome]:
        """Parse GTDB metadata file"""
        genomes = []
        
        try:
            # Handle gzipped files
            if file_path.endswith('.gz'):
                df = pd.read_csv(file_path, sep='\t', compression='gzip', low_memory=False)
            else:
                df = pd.read_csv(file_path, sep='\t', low_memory=False)
            
            for _, row in df.iterrows():
                try:
                    # Parse GTDB taxonomy
                    gtdb_taxonomy = str(row.get('gtdb_taxonomy', ''))
                    gtdb_parts = self._parse_taxonomy_string(gtdb_taxonomy)
                    
                    # Parse NCBI taxonomy
                    ncbi_taxonomy = str(row.get('ncbi_taxonomy', ''))
                    ncbi_parts = self._parse_taxonomy_string(ncbi_taxonomy)
                    
                    genome = GTDBGenome(
                        accession=str(row.get('accession', '')),
                        organism_name=str(row.get('organism_name', '')),
                        strain=str(row.get('strain', '')),
                        # GTDB taxonomy
                        gtdb_domain=gtdb_parts.get('d', ''),
                        gtdb_phylum=gtdb_parts.get('p', ''),
                        gtdb_class=gtdb_parts.get('c', ''),
                        gtdb_order=gtdb_parts.get('o', ''),
                        gtdb_family=gtdb_parts.get('f', ''),
                        gtdb_genus=gtdb_parts.get('g', ''),
                        gtdb_species=gtdb_parts.get('s', ''),
                        gtdb_taxonomy=gtdb_taxonomy,
                        # NCBI taxonomy
                        ncbi_domain=ncbi_parts.get('d', ''),
                        ncbi_phylum=ncbi_parts.get('p', ''),
                        ncbi_class=ncbi_parts.get('c', ''),
                        ncbi_order=ncbi_parts.get('o', ''),
                        ncbi_family=ncbi_parts.get('f', ''),
                        ncbi_genus=ncbi_parts.get('g', ''),
                        ncbi_species=ncbi_parts.get('s', ''),
                        ncbi_taxonomy=ncbi_taxonomy,
                        ncbi_taxid=self._safe_int(row.get('ncbi_taxid', 0)),
                        # Genome statistics
                        genome_size=self._safe_int(row.get('genome_size', 0)),
                        contig_count=self._safe_int(row.get('contig_count', 0)),
                        n50=self._safe_int(row.get('n50_contigs', 0)),
                        scaffold_count=self._safe_int(row.get('scaffold_count', 0)),
                        total_gap_length=self._safe_int(row.get('total_gap_length', 0)),
                        # Quality metrics
                        checkm_completeness=self._safe_float(row.get('checkm_completeness', 0)),
                        checkm_contamination=self._safe_float(row.get('checkm_contamination', 0)),
                        checkm_strain_heterogeneity=self._safe_float(row.get('checkm_strain_heterogeneity', 0)),
                        # Gene content
                        protein_count=self._safe_int(row.get('protein_count', 0)),
                        gene_count=self._safe_int(row.get('gene_count', 0)),
                        ssu_count=self._safe_int(row.get('ssu_count', 0)),
                        ssu_length=self._safe_int(row.get('ssu_length', 0)),
                        # Assembly information
                        assembly_level=str(row.get('assembly_level', '')),
                        assembly_type=str(row.get('assembly_type', '')),
                        refseq_category=str(row.get('refseq_category', '')),
                        genbank_assembly_accession=str(row.get('genbank_assembly_accession', '')),
                        refseq_assembly_accession=str(row.get('refseq_assembly_accession', '')),
                        # Environmental data
                        environment=str(row.get('environment', '')),
                        isolation_source=str(row.get('isolation_source', '')),
                        # GTDB metadata
                        gtdb_representative=bool(row.get('gtdb_representative', False)),
                        gtdb_species_representative=bool(row.get('gtdb_species_representative', False)),
                        gtdb_type_material=bool(row.get('gtdb_type_material', False)),
                        gtdb_type_designation=str(row.get('gtdb_type_designation', ''))
                    )
                    
                    # Determine quality level
                    if genome.checkm_completeness >= 95 and genome.checkm_contamination <= 5:
                        genome.quality_level = "high"
                    elif genome.checkm_completeness >= 80 and genome.checkm_contamination <= 10:
                        genome.quality_level = "medium"
                    else:
                        genome.quality_level = "low"
                    
                    genomes.append(genome)
                    
                except Exception as e:
                    logger.warning(f"Error parsing genome row: {e}")
                    continue
            
            logger.info(f"Parsed {len(genomes)} {domain} genomes from {file_path}")
            return genomes
            
        except Exception as e:
            logger.error(f"Error parsing metadata file {file_path}: {e}")
            return []
    
    def parse_taxonomy_file(self, file_path: str) -> List[GTDBTaxonomy]:
        """Parse GTDB taxonomy file"""
        taxa = []
        
        try:
            df = pd.read_csv(file_path, sep='\t', header=None, names=['accession', 'taxonomy'])
            
            # Build taxonomy hierarchy
            taxonomy_dict = {}
            
            for _, row in df.iterrows():
                accession = str(row['accession'])
                taxonomy = str(row['taxonomy'])
                
                # Parse taxonomy string
                parts = self._parse_taxonomy_string(taxonomy)
                
                # Create taxa for each level
                for rank, name in parts.items():
                    taxon_id = f"{rank}__{name}"
                    
                    if taxon_id not in taxonomy_dict:
                        # Determine parent
                        parent_rank = self._get_parent_rank(rank)
                        parent_taxon = ""
                        if parent_rank and parent_rank in parts:
                            parent_taxon = f"{parent_rank}__{parts[parent_rank]}"
                        
                        taxon = GTDBTaxonomy(
                            taxon_id=taxon_id,
                            taxon_name=name,
                            rank=self._expand_rank(rank),
                            parent_taxon=parent_taxon,
                            genome_count=0,
                            genomes=[]
                        )
                        taxonomy_dict[taxon_id] = taxon
                    
                    # Add genome to taxon
                    taxonomy_dict[taxon_id].genomes.append(accession)
                    taxonomy_dict[taxon_id].genome_count += 1
            
            taxa = list(taxonomy_dict.values())
            logger.info(f"Parsed {len(taxa)} taxonomic nodes from {file_path}")
            return taxa
            
        except Exception as e:
            logger.error(f"Error parsing taxonomy file {file_path}: {e}")
            return []
    
    def _parse_taxonomy_string(self, taxonomy: str) -> Dict[str, str]:
        """Parse GTDB taxonomy string"""
        parts = {}
        
        if not taxonomy or taxonomy == 'nan':
            return parts
        
        # Split by semicolon and parse each level
        levels = taxonomy.split(';')
        
        for level in levels:
            level = level.strip()
            if '__' in level:
                rank, name = level.split('__', 1)
                if name and name != '':
                    parts[rank.lower()] = name
        
        return parts
    
    def _get_parent_rank(self, rank: str) -> Optional[str]:
        """Get parent rank for taxonomy hierarchy"""
        hierarchy = ['d', 'p', 'c', 'o', 'f', 'g', 's']
        
        try:
            idx = hierarchy.index(rank.lower())
            if idx > 0:
                return hierarchy[idx - 1]
        except ValueError:
            pass
        
        return None
    
    def _expand_rank(self, rank: str) -> str:
        """Expand rank abbreviation"""
        rank_map = {
            'd': 'domain',
            'p': 'phylum', 
            'c': 'class',
            'o': 'order',
            'f': 'family',
            'g': 'genus',
            's': 'species'
        }
        return rank_map.get(rank.lower(), rank)
    
    def store_genomes(self, genomes: List[GTDBGenome]):
        """Store genomes in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for genome in genomes:
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO genomes (
                            accession, organism_name, strain, gtdb_domain, gtdb_phylum,
                            gtdb_class, gtdb_order, gtdb_family, gtdb_genus, gtdb_species,
                            gtdb_taxonomy, ncbi_domain, ncbi_phylum, ncbi_class, ncbi_order,
                            ncbi_family, ncbi_genus, ncbi_species, ncbi_taxonomy, ncbi_taxid,
                            genome_size, contig_count, n50, scaffold_count, total_gap_length,
                            checkm_completeness, checkm_contamination, checkm_strain_heterogeneity,
                            quality_level, protein_count, gene_count, ssu_count, ssu_length,
                            assembly_level, assembly_type, refseq_category, genbank_assembly_accession,
                            refseq_assembly_accession, environment, isolation_source,
                            gtdb_representative, gtdb_species_representative, gtdb_type_material,
                            gtdb_type_designation, gtdb_release, last_updated, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        genome.accession, genome.organism_name, genome.strain,
                        genome.gtdb_domain, genome.gtdb_phylum, genome.gtdb_class,
                        genome.gtdb_order, genome.gtdb_family, genome.gtdb_genus,
                        genome.gtdb_species, genome.gtdb_taxonomy, genome.ncbi_domain,
                        genome.ncbi_phylum, genome.ncbi_class, genome.ncbi_order,
                        genome.ncbi_family, genome.ncbi_genus, genome.ncbi_species,
                        genome.ncbi_taxonomy, genome.ncbi_taxid, genome.genome_size,
                        genome.contig_count, genome.n50, genome.scaffold_count,
                        genome.total_gap_length, genome.checkm_completeness,
                        genome.checkm_contamination, genome.checkm_strain_heterogeneity,
                        genome.quality_level, genome.protein_count, genome.gene_count,
                        genome.ssu_count, genome.ssu_length, genome.assembly_level,
                        genome.assembly_type, genome.refseq_category,
                        genome.genbank_assembly_accession, genome.refseq_assembly_accession,
                        genome.environment, genome.isolation_source,
                        genome.gtdb_representative, genome.gtdb_species_representative,
                        genome.gtdb_type_material, genome.gtdb_type_designation,
                        genome.gtdb_release, genome.last_updated,
                        json.dumps(genome.metadata)
                    ))
                except Exception as e:
                    logger.error(f"Error storing genome {genome.accession}: {e}")
            
            conn.commit()
    
    def store_taxonomy(self, taxa: List[GTDBTaxonomy]):
        """Store taxonomy in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for taxon in taxa:
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO taxonomy (
                            taxon_id, taxon_name, rank, parent_taxon, genome_count,
                            representative_genome, branch_length, node_support,
                            is_proposed_name, is_gtdb_species_representative,
                            classification_notes, child_taxa, genomes, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        taxon.taxon_id, taxon.taxon_name, taxon.rank, taxon.parent_taxon,
                        taxon.genome_count, taxon.representative_genome, taxon.branch_length,
                        taxon.node_support, taxon.is_proposed_name,
                        taxon.is_gtdb_species_representative, taxon.classification_notes,
                        json.dumps(taxon.child_taxa), json.dumps(taxon.genomes),
                        json.dumps(taxon.metadata)
                    ))
                except Exception as e:
                    logger.error(f"Error storing taxon {taxon.taxon_id}: {e}")
            
            conn.commit()
    
    def export_to_csv(self) -> Dict[str, str]:
        """Export data to CSV files"""
        output_files = {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Export all genomes
                df_genomes = pd.read_sql_query('''
                    SELECT accession, organism_name, gtdb_domain, gtdb_phylum, gtdb_class,
                           gtdb_order, gtdb_family, gtdb_genus, gtdb_species, gtdb_taxonomy,
                           ncbi_taxonomy, genome_size, contig_count, n50, checkm_completeness,
                           checkm_contamination, quality_level, protein_count, gene_count,
                           assembly_level, refseq_category, gtdb_representative,
                           gtdb_species_representative, environment, isolation_source
                    FROM genomes
                ''', conn)
                
                genomes_file = self.output_path / "gtdb_genomes.csv"
                df_genomes.to_csv(genomes_file, index=False)
                output_files['genomes'] = str(genomes_file)
                
                # Export bacterial genomes
                df_bacteria = df_genomes[df_genomes['gtdb_domain'] == 'Bacteria']
                if not df_bacteria.empty:
                    bacteria_file = self.output_path / "gtdb_bacteria.csv"
                    df_bacteria.to_csv(bacteria_file, index=False)
                    output_files['bacteria'] = str(bacteria_file)
                
                # Export archaeal genomes
                df_archaea = df_genomes[df_genomes['gtdb_domain'] == 'Archaea']
                if not df_archaea.empty:
                    archaea_file = self.output_path / "gtdb_archaea.csv"
                    df_archaea.to_csv(archaea_file, index=False)
                    output_files['archaea'] = str(archaea_file)
                
                # Export high-quality genomes
                df_hq = df_genomes[df_genomes['quality_level'] == 'high']
                if not df_hq.empty:
                    hq_file = self.output_path / "gtdb_high_quality.csv"
                    df_hq.to_csv(hq_file, index=False)
                    output_files['high_quality'] = str(hq_file)
                
                # Export representative genomes
                df_reps = df_genomes[df_genomes['gtdb_representative'] == True]
                if not df_reps.empty:
                    reps_file = self.output_path / "gtdb_representatives.csv"
                    df_reps.to_csv(reps_file, index=False)
                    output_files['representatives'] = str(reps_file)
                
                # Export taxonomy
                df_taxonomy = pd.read_sql_query('''
                    SELECT taxon_id, taxon_name, rank, parent_taxon, genome_count,
                           representative_genome
                    FROM taxonomy
                ''', conn)
                
                if not df_taxonomy.empty:
                    taxonomy_file = self.output_path / "gtdb_taxonomy.csv"
                    df_taxonomy.to_csv(taxonomy_file, index=False)
                    output_files['taxonomy'] = str(taxonomy_file)
                
                logger.info(f"Exported GTDB data to {len(output_files)} CSV files")
                return output_files
                
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return {}
    
    def generate_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive statistics"""
        stats = {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Genome statistics
                cursor.execute("SELECT COUNT(*) FROM genomes")
                stats['total_genomes'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM genomes WHERE gtdb_domain = 'Bacteria'")
                stats['bacterial_genomes'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM genomes WHERE gtdb_domain = 'Archaea'")
                stats['archaeal_genomes'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM genomes WHERE quality_level = 'high'")
                stats['high_quality_genomes'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM genomes WHERE gtdb_representative = 1")
                stats['representative_genomes'] = cursor.fetchone()[0]
                
                # Taxonomic diversity
                cursor.execute("SELECT COUNT(DISTINCT gtdb_phylum) FROM genomes WHERE gtdb_phylum != ''")
                stats['phyla_count'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT gtdb_species) FROM genomes WHERE gtdb_species != ''")
                stats['species_count'] = cursor.fetchone()[0]
                
                # Quality distribution
                cursor.execute('''
                    SELECT quality_level, COUNT(*) 
                    FROM genomes 
                    GROUP BY quality_level
                ''')
                quality_dist = {level: count for level, count in cursor.fetchall()}
                stats['quality_distribution'] = quality_dist
                
                # Top phyla
                cursor.execute('''
                    SELECT gtdb_phylum, COUNT(*) 
                    FROM genomes 
                    WHERE gtdb_phylum != ''
                    GROUP BY gtdb_phylum 
                    ORDER BY COUNT(*) DESC 
                    LIMIT 20
                ''')
                top_phyla = dict(cursor.fetchall())
                stats['top_phyla'] = top_phyla
                
                # Environmental distribution
                cursor.execute('''
                    SELECT environment, COUNT(*) 
                    FROM genomes 
                    WHERE environment != ''
                    GROUP BY environment 
                    ORDER BY COUNT(*) DESC 
                    LIMIT 10
                ''')
                top_environments = dict(cursor.fetchall())
                stats['top_environments'] = top_environments
                
                return stats
                
        except Exception as e:
            logger.error(f"Error generating statistics: {e}")
            return {}

class GTDBIntegration:
    """Main integration class for GTDB data"""
    
    def __init__(self, output_path: str = "data"):
        self.output_path = Path(output_path)
        self.downloader = GTDBDownloader()
        self.parser = GTDBParser(str(self.output_path / "processed" / "gtdb"))
        
        # Progress tracking
        self.progress_file = self.output_path / "interim" / "gtdb_progress.json"
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _load_progress(self) -> Dict[str, Any]:
        """Load integration progress"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading progress: {e}")
        return {"downloads": {}, "processing": {}, "last_updated": None}
    
    def _save_progress(self, progress: Dict[str, Any]):
        """Save integration progress"""
        try:
            progress["last_updated"] = datetime.now(timezone.utc).isoformat()
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving progress: {e}")
    
    async def download_and_process_data(self, domains: List[str] = ['bacteria', 'archaea'],
                                      max_genomes_per_domain: Optional[int] = None) -> Dict[str, Any]:
        """Download and process GTDB data"""
        progress = self._load_progress()
        results = {"downloads": {}, "processing": {}, "statistics": {}}
        
        try:
            # Download data for each domain
            for domain in domains:
                logger.info(f"Processing {domain} data")
                
                if domain == 'bacteria':
                    domain_data = await self.downloader.download_bacterial_data()
                elif domain == 'archaea':
                    domain_data = await self.downloader.download_archaeal_data()
                else:
                    logger.warning(f"Unknown domain: {domain}")
                    continue
                
                if domain_data:
                    progress["downloads"][domain] = domain_data
                    results["downloads"][domain] = domain_data
                    
                    # Process metadata
                    metadata_key = f"{domain.replace('bacteria', 'bac120').replace('archaea', 'ar53')}_metadata"
                    if metadata_key in domain_data:
                        genomes = self.parser.parse_metadata_file(
                            domain_data[metadata_key], domain
                        )
                        
                        # Limit genomes if specified
                        if max_genomes_per_domain and len(genomes) > max_genomes_per_domain:
                            # Prioritize representative genomes
                            rep_genomes = [g for g in genomes if g.gtdb_representative]
                            other_genomes = [g for g in genomes if not g.gtdb_representative]
                            
                            if len(rep_genomes) >= max_genomes_per_domain:
                                genomes = rep_genomes[:max_genomes_per_domain]
                            else:
                                remaining = max_genomes_per_domain - len(rep_genomes)
                                genomes = rep_genomes + other_genomes[:remaining]
                        
                        if genomes:
                            self.parser.store_genomes(genomes)
                            results["processing"][f"{domain}_genomes"] = len(genomes)
                    
                    # Process taxonomy
                    taxonomy_key = f"{domain.replace('bacteria', 'bac120').replace('archaea', 'ar53')}_taxonomy"
                    if taxonomy_key in domain_data:
                        taxa = self.parser.parse_taxonomy_file(domain_data[taxonomy_key])
                        
                        if taxa:
                            self.parser.store_taxonomy(taxa)
                            results["processing"][f"{domain}_taxa"] = len(taxa)
                
                # Save progress after each domain
                self._save_progress(progress)
            
            # Generate statistics
            stats = self.parser.generate_statistics()
            results["statistics"] = stats
            
            return results
            
        except Exception as e:
            logger.error(f"Error in data download and processing: {e}")
            return results
        finally:
            await self.downloader.close()
    
    def export_datasets(self) -> Dict[str, str]:
        """Export all datasets to CSV format"""
        return self.parser.export_to_csv()
    
    async def run_full_integration(self, domains: List[str] = ['bacteria', 'archaea'],
                                 max_genomes_per_domain: Optional[int] = None) -> Dict[str, Any]:
        """Run complete GTDB integration"""
        logger.info("Starting comprehensive GTDB integration")
        
        start_time = time.time()
        results = {
            "status": "started",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "downloads": {},
            "processing": {},
            "exports": {},
            "statistics": {},
            "errors": []
        }
        
        try:
            # Get release information
            release_info = await self.downloader.get_release_info()
            results["release_info"] = release_info
            
            # Download and process data
            processing_results = await self.download_and_process_data(
                domains=domains,
                max_genomes_per_domain=max_genomes_per_domain
            )
            results.update(processing_results)
            
            # Export to CSV
            export_files = self.export_datasets()
            results["exports"] = export_files
            
            # Final results
            end_time = time.time()
            results.update({
                "status": "completed",
                "end_time": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": end_time - start_time,
                "total_files_exported": len(export_files)
            })
            
            logger.info(f"GTDB integration completed in {end_time - start_time:.2f} seconds")
            logger.info(f"Total genomes processed: {results.get('statistics', {}).get('total_genomes', 0)}")
            logger.info(f"Files exported: {len(export_files)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in full integration: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
            results["errors"].append(str(e))
            return results

async def main():
    """Main function for testing"""
    integration = GTDBIntegration()
    
    # Test with limited data
    results = await integration.run_full_integration(
        domains=['bacteria', 'archaea'],
        max_genomes_per_domain=5000
    )
    
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    asyncio.run(main()) 
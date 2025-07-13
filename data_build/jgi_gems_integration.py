#!/usr/bin/env python3


import os
import json
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Set
import logging
import sqlite3
import gzip
import hashlib
import time
import tarfile
from dataclasses import dataclass, field
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import ftplib
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import networkx as nx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class GEMsGenome:
    """GEMs metagenome-assembled genome data structure"""
    genome_id: str
    bin_id: str
    sample_id: str
    # Quality metrics
    completeness: float = 0.0
    contamination: float = 0.0
    quality_score: float = 0.0
    quality_level: str = ""  # high, medium, low
    # Genome statistics
    genome_size: int = 0
    contig_count: int = 0
    n50: int = 0
    gc_content: float = 0.0
    # Taxonomic classification
    domain: str = ""
    phylum: str = ""
    class_name: str = ""
    order: str = ""
    family: str = ""
    genus: str = ""
    species: str = ""
    gtdb_taxonomy: str = ""
    ncbi_taxonomy: str = ""
    # Novel species classification
    is_novel_species: bool = False
    is_candidate_species: bool = False
    # Gene content
    gene_count: int = 0
    protein_count: int = 0
    rrna_5s: int = 0
    rrna_16s: int = 0
    rrna_23s: int = 0
    trna_count: int = 0
    # Environmental context
    environment_type: str = ""
    habitat: str = ""
    sampling_site: str = ""
    geographic_location: str = ""
    # Sample metadata
    metagenome_source: str = ""
    sequencing_method: str = ""
    assembly_method: str = ""
    binning_method: str = ""
    # File paths
    genome_fasta_file: str = ""
    annotation_file: str = ""
    protein_file: str = ""
    metadata_file: str = ""
    # Additional metadata
    biosample_id: str = ""
    study_id: str = ""
    jgi_project_id: str = ""
    submission_date: str = ""
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Metagenome:
    """Metagenome sample data structure"""
    metagenome_id: str
    sample_name: str
    study_name: str
    # Environmental context
    ecosystem: str = ""
    ecosystem_category: str = ""
    ecosystem_type: str = ""
    specific_ecosystem: str = ""
    # Geographic information
    latitude: float = 0.0
    longitude: float = 0.0
    elevation: float = 0.0
    geographic_location: str = ""
    # Sample characteristics
    sampling_date: str = ""
    depth: float = 0.0
    temperature: float = 0.0
    ph: float = 0.0
    salinity: float = 0.0
    # Sequencing information
    sequencing_platform: str = ""
    sequencing_method: str = ""
    read_count: int = 0
    total_bases: int = 0
    average_read_length: float = 0.0
    # Assembly statistics
    assembly_size: int = 0
    contig_count: int = 0
    assembly_n50: int = 0
    # Binning results
    total_bins: int = 0
    high_quality_bins: int = 0
    medium_quality_bins: int = 0
    low_quality_bins: int = 0
    # File information
    raw_data_file: str = ""
    assembly_file: str = ""
    annotation_file: str = ""
    bins_directory: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class GEMsDownloader:
    """JGI GEMs data downloader"""
    
    def __init__(self, base_url: str = "https://portal.nersc.gov/GEM/"):
        self.base_url = base_url
        self.cache_path = Path("data/raw/jgi_gems/cache")
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.session = None
        self.rate_limit_delay = 0.2  # 200ms between requests
        
        # JGI portal authentication (may be required for some data)
        self.jgi_user = os.getenv('JGI_USER')
        self.jgi_password = os.getenv('JGI_PASSWORD')
        
        # Initialize requests session with retry strategy
        self.requests_session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.requests_session.mount("http://", adapter)
        self.requests_session.mount("https://", adapter)
    
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
    
    async def download_gems_catalog(self) -> Optional[str]:
        """Download the complete GEMs catalog metadata"""
        cache_file = self._get_cache_path("gems_catalog.tar.gz")
        
        if self._is_cache_valid(cache_file, max_age_days=30):
            logger.info("Using cached GEMs catalog")
            return str(cache_file)
        
        session = await self._get_session()
        
        try:
            # Try to download the complete catalog
            catalog_url = urljoin(self.base_url, "catalog.tar.gz")
            
            logger.info("Downloading GEMs catalog (this may take a while - large file)")
            
            async with session.get(catalog_url) as response:
                if response.status == 200:
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    
                    with open(cache_file, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Log progress every 500MB
                            if downloaded % (500 * 1024 * 1024) == 0:
                                if total_size > 0:
                                    progress = (downloaded / total_size) * 100
                                    logger.info(f"Downloaded {progress:.1f}% of GEMs catalog")
                    
                    logger.info("Successfully downloaded GEMs catalog")
                    return str(cache_file)
                else:
                    logger.error(f"Failed to download GEMs catalog: HTTP {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error downloading GEMs catalog: {e}")
            # Try alternative metadata files
            return await self._download_metadata_files()
    
    async def _download_metadata_files(self) -> Optional[str]:
        """Download individual metadata files if catalog is not available"""
        session = await self._get_session()
        metadata_files = {}
        
        try:
            # Common metadata files in GEMs catalog
            file_urls = {
                'genomes_metadata': 'genomes.metadata.tsv',
                'quality_report': 'quality_report.tsv',
                'taxonomy': 'taxonomy.tsv',
                'sample_metadata': 'sample_metadata.tsv',
                'environment_data': 'environment.tsv'
            }
            
            for file_key, filename in file_urls.items():
                cache_file = self._get_cache_path(filename)
                
                if self._is_cache_valid(cache_file, max_age_days=30):
                    metadata_files[file_key] = str(cache_file)
                    continue
                
                url = urljoin(self.base_url, filename)
                
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            content = await response.read()
                            
                            with open(cache_file, 'wb') as f:
                                f.write(content)
                            
                            metadata_files[file_key] = str(cache_file)
                            logger.info(f"Downloaded {filename}")
                        else:
                            logger.warning(f"Could not download {filename}: HTTP {response.status}")
                except Exception as e:
                    logger.warning(f"Error downloading {filename}: {e}")
                
                await asyncio.sleep(self.rate_limit_delay)
            
            if metadata_files:
                # Create a metadata index file
                index_file = self._get_cache_path("metadata_index.json")
                with open(index_file, 'w') as f:
                    json.dump(metadata_files, f, indent=2)
                
                return str(index_file)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error downloading metadata files: {e}")
            return None
    
    async def download_sample_genomes(self, sample_ids: List[str], max_genomes: Optional[int] = None) -> Dict[str, str]:
        """Download genome files for specific samples"""
        session = await self._get_session()
        downloaded_files = {}
        
        try:
            count = 0
            for sample_id in sample_ids:
                if max_genomes and count >= max_genomes:
                    break
                
                # Construct genome file URL
                genome_file = f"{sample_id}.fna.gz"
                cache_file = self._get_cache_path(f"genomes/{genome_file}")
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                
                if cache_file.exists():
                    downloaded_files[sample_id] = str(cache_file)
                    count += 1
                    continue
                
                url = urljoin(self.base_url, f"genomes/{genome_file}")
                
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            content = await response.read()
                            
                            with open(cache_file, 'wb') as f:
                                f.write(content)
                            
                            downloaded_files[sample_id] = str(cache_file)
                            count += 1
                            logger.debug(f"Downloaded genome {sample_id}")
                        else:
                            logger.warning(f"Could not download genome {sample_id}: HTTP {response.status}")
                except Exception as e:
                    logger.warning(f"Error downloading genome {sample_id}: {e}")
                
                await asyncio.sleep(self.rate_limit_delay)
            
            logger.info(f"Downloaded {len(downloaded_files)} genome files")
            return downloaded_files
            
        except Exception as e:
            logger.error(f"Error downloading sample genomes: {e}")
            return {}
    
    async def get_gems_statistics(self) -> Dict[str, Any]:
        """Get GEMs catalog statistics"""
        session = await self._get_session()
        
        try:
            # Try to get statistics file
            stats_url = urljoin(self.base_url, "stats.json")
            
            async with session.get(stats_url) as response:
                if response.status == 200:
                    content = await response.text()
                    return json.loads(content)
                    
        except Exception as e:
            logger.warning(f"Could not fetch GEMs statistics: {e}")
        
        # Return default statistics
        return {
            "total_genomes": 52515,
            "total_metagenomes": 10450,
            "high_quality_genomes": 9143,
            "phyla_count": 135,
            "novel_candidate_species": 12556,
            "database": "JGI GEMs",
            "last_updated": "2024-05-28"
        }
    
    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()

class GEMsParser:
    """Parser for GEMs catalog data"""
    
    def __init__(self, output_path: str = "data/processed/jgi_gems"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.output_path / "gems.db"
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database for GEMs data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Genomes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS genomes (
                    genome_id TEXT PRIMARY KEY,
                    bin_id TEXT,
                    sample_id TEXT,
                    completeness REAL,
                    contamination REAL,
                    quality_score REAL,
                    quality_level TEXT,
                    genome_size INTEGER,
                    contig_count INTEGER,
                    n50 INTEGER,
                    gc_content REAL,
                    domain TEXT,
                    phylum TEXT,
                    class_name TEXT,
                    order_name TEXT,
                    family TEXT,
                    genus TEXT,
                    species TEXT,
                    gtdb_taxonomy TEXT,
                    ncbi_taxonomy TEXT,
                    is_novel_species BOOLEAN,
                    is_candidate_species BOOLEAN,
                    gene_count INTEGER,
                    protein_count INTEGER,
                    rrna_5s INTEGER,
                    rrna_16s INTEGER,
                    rrna_23s INTEGER,
                    trna_count INTEGER,
                    environment_type TEXT,
                    habitat TEXT,
                    sampling_site TEXT,
                    geographic_location TEXT,
                    metagenome_source TEXT,
                    sequencing_method TEXT,
                    assembly_method TEXT,
                    binning_method TEXT,
                    genome_fasta_file TEXT,
                    annotation_file TEXT,
                    protein_file TEXT,
                    metadata_file TEXT,
                    biosample_id TEXT,
                    study_id TEXT,
                    jgi_project_id TEXT,
                    submission_date TEXT,
                    last_updated TIMESTAMP,
                    metadata TEXT
                )
            ''')
            
            # Metagenomes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metagenomes (
                    metagenome_id TEXT PRIMARY KEY,
                    sample_name TEXT,
                    study_name TEXT,
                    ecosystem TEXT,
                    ecosystem_category TEXT,
                    ecosystem_type TEXT,
                    specific_ecosystem TEXT,
                    latitude REAL,
                    longitude REAL,
                    elevation REAL,
                    geographic_location TEXT,
                    sampling_date TEXT,
                    depth REAL,
                    temperature REAL,
                    ph REAL,
                    salinity REAL,
                    sequencing_platform TEXT,
                    sequencing_method TEXT,
                    read_count INTEGER,
                    total_bases INTEGER,
                    average_read_length REAL,
                    assembly_size INTEGER,
                    contig_count INTEGER,
                    assembly_n50 INTEGER,
                    total_bins INTEGER,
                    high_quality_bins INTEGER,
                    medium_quality_bins INTEGER,
                    low_quality_bins INTEGER,
                    raw_data_file TEXT,
                    assembly_file TEXT,
                    annotation_file TEXT,
                    bins_directory TEXT,
                    metadata TEXT
                )
            ''')
            
            # Environment-genome associations
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS genome_environments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    genome_id TEXT,
                    environment_type TEXT,
                    habitat TEXT,
                    ecosystem TEXT,
                    geographic_location TEXT,
                    environmental_parameters TEXT,
                    FOREIGN KEY (genome_id) REFERENCES genomes (genome_id)
                )
            ''')
            
            # Create indices for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_genome_quality ON genomes(quality_level)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_genome_phylum ON genomes(phylum)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_genome_environment ON genomes(environment_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_genome_novel ON genomes(is_novel_species)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metagenome_ecosystem ON metagenomes(ecosystem)')
            
            conn.commit()
    
    def parse_catalog_file(self, catalog_file: str) -> Tuple[List[GEMsGenome], List[Metagenome]]:
        """Parse GEMs catalog file (tar.gz or metadata files)"""
        genomes = []
        metagenomes = []
        
        try:
            if catalog_file.endswith('.tar.gz'):
                # Extract and parse tar file
                with tarfile.open(catalog_file, 'r:gz') as tar:
                    # Look for metadata files
                    for member in tar.getmembers():
                        if member.name.endswith('.tsv') or member.name.endswith('.csv'):
                            f = tar.extractfile(member)
                            if f:
                                content = f.read().decode('utf-8')
                                
                                if 'genome' in member.name.lower():
                                    genomes.extend(self._parse_genomes_metadata(content))
                                elif 'sample' in member.name.lower() or 'metagenome' in member.name.lower():
                                    metagenomes.extend(self._parse_metagenomes_metadata(content))
            
            elif catalog_file.endswith('.json'):
                # Parse metadata index file
                with open(catalog_file, 'r') as f:
                    metadata_files = json.load(f)
                
                for file_type, file_path in metadata_files.items():
                    if file_type == 'genomes_metadata' and Path(file_path).exists():
                        genomes.extend(self._parse_genomes_file(file_path))
                    elif file_type == 'sample_metadata' and Path(file_path).exists():
                        metagenomes.extend(self._parse_metagenomes_file(file_path))
            
            logger.info(f"Parsed {len(genomes)} genomes and {len(metagenomes)} metagenomes")
            return genomes, metagenomes
            
        except Exception as e:
            logger.error(f"Error parsing catalog file {catalog_file}: {e}")
            return [], []
    
    def _parse_genomes_file(self, file_path: str) -> List[GEMsGenome]:
        """Parse genomes metadata file"""
        genomes = []
        
        try:
            # Try different delimiters
            for delimiter in ['\t', ',']:
                try:
                    df = pd.read_csv(file_path, delimiter=delimiter, low_memory=False)
                    if len(df.columns) > 5:  # Valid if has multiple columns
                        break
                except Exception:
                    continue
            
            for _, row in df.iterrows():
                try:
                    genome = GEMsGenome(
                        genome_id=str(row.get('genome_id', row.get('Bin_Id', ''))),
                        bin_id=str(row.get('bin_id', row.get('Bin_Id', ''))),
                        sample_id=str(row.get('sample_id', row.get('Sample_Id', ''))),
                        completeness=float(row.get('completeness', row.get('Completeness', 0))),
                        contamination=float(row.get('contamination', row.get('Contamination', 0))),
                        quality_score=float(row.get('quality_score', 0)),
                        quality_level=str(row.get('quality_level', '')),
                        genome_size=int(row.get('genome_size', row.get('Size', 0))),
                        contig_count=int(row.get('contig_count', row.get('Contigs', 0))),
                        n50=int(row.get('n50', row.get('N50', 0))),
                        gc_content=float(row.get('gc_content', row.get('GC', 0))),
                        domain=str(row.get('domain', row.get('Domain', ''))),
                        phylum=str(row.get('phylum', row.get('Phylum', ''))),
                        class_name=str(row.get('class', row.get('Class', ''))),
                        order=str(row.get('order', row.get('Order', ''))),
                        family=str(row.get('family', row.get('Family', ''))),
                        genus=str(row.get('genus', row.get('Genus', ''))),
                        species=str(row.get('species', row.get('Species', ''))),
                        gtdb_taxonomy=str(row.get('gtdb_taxonomy', '')),
                        ncbi_taxonomy=str(row.get('ncbi_taxonomy', '')),
                        is_novel_species=bool(row.get('is_novel_species', False)),
                        is_candidate_species=bool(row.get('is_candidate_species', False)),
                        gene_count=int(row.get('gene_count', 0)),
                        protein_count=int(row.get('protein_count', 0)),
                        rrna_5s=int(row.get('rrna_5s', 0)),
                        rrna_16s=int(row.get('rrna_16s', 0)),
                        rrna_23s=int(row.get('rrna_23s', 0)),
                        trna_count=int(row.get('trna_count', 0)),
                        environment_type=str(row.get('environment_type', '')),
                        habitat=str(row.get('habitat', '')),
                        sampling_site=str(row.get('sampling_site', '')),
                        geographic_location=str(row.get('geographic_location', '')),
                        metagenome_source=str(row.get('metagenome_source', '')),
                        biosample_id=str(row.get('biosample_id', '')),
                        study_id=str(row.get('study_id', '')),
                        jgi_project_id=str(row.get('jgi_project_id', ''))
                    )
                    
                    # Determine quality level if not provided
                    if not genome.quality_level:
                        if genome.completeness >= 90 and genome.contamination <= 5:
                            genome.quality_level = "high"
                        elif genome.completeness >= 50 and genome.contamination <= 10:
                            genome.quality_level = "medium"
                        else:
                            genome.quality_level = "low"
                    
                    genomes.append(genome)
                    
                except Exception as e:
                    logger.warning(f"Error parsing genome row: {e}")
                    continue
            
            return genomes
            
        except Exception as e:
            logger.error(f"Error parsing genomes file {file_path}: {e}")
            return []
    
    def _parse_metagenomes_file(self, file_path: str) -> List[Metagenome]:
        """Parse metagenomes metadata file"""
        metagenomes = []
        
        try:
            # Try different delimiters
            for delimiter in ['\t', ',']:
                try:
                    df = pd.read_csv(file_path, delimiter=delimiter, low_memory=False)
                    if len(df.columns) > 3:
                        break
                except Exception:
                    continue
            
            for _, row in df.iterrows():
                try:
                    metagenome = Metagenome(
                        metagenome_id=str(row.get('metagenome_id', row.get('Sample_Id', ''))),
                        sample_name=str(row.get('sample_name', row.get('Sample_Name', ''))),
                        study_name=str(row.get('study_name', '')),
                        ecosystem=str(row.get('ecosystem', '')),
                        ecosystem_category=str(row.get('ecosystem_category', '')),
                        ecosystem_type=str(row.get('ecosystem_type', '')),
                        specific_ecosystem=str(row.get('specific_ecosystem', '')),
                        latitude=float(row.get('latitude', row.get('lat', 0))),
                        longitude=float(row.get('longitude', row.get('lon', 0))),
                        elevation=float(row.get('elevation', 0)),
                        geographic_location=str(row.get('geographic_location', '')),
                        sampling_date=str(row.get('sampling_date', '')),
                        depth=float(row.get('depth', 0)),
                        temperature=float(row.get('temperature', 0)),
                        ph=float(row.get('ph', row.get('pH', 0))),
                        salinity=float(row.get('salinity', 0)),
                        sequencing_platform=str(row.get('sequencing_platform', '')),
                        sequencing_method=str(row.get('sequencing_method', '')),
                        read_count=int(row.get('read_count', 0)),
                        total_bases=int(row.get('total_bases', 0)),
                        average_read_length=float(row.get('average_read_length', 0)),
                        assembly_size=int(row.get('assembly_size', 0)),
                        contig_count=int(row.get('contig_count', 0)),
                        assembly_n50=int(row.get('assembly_n50', 0)),
                        total_bins=int(row.get('total_bins', 0)),
                        high_quality_bins=int(row.get('high_quality_bins', 0)),
                        medium_quality_bins=int(row.get('medium_quality_bins', 0)),
                        low_quality_bins=int(row.get('low_quality_bins', 0))
                    )
                    
                    metagenomes.append(metagenome)
                    
                except Exception as e:
                    logger.warning(f"Error parsing metagenome row: {e}")
                    continue
            
            return metagenomes
            
        except Exception as e:
            logger.error(f"Error parsing metagenomes file {file_path}: {e}")
            return []
    
    def _parse_genomes_metadata(self, content: str) -> List[GEMsGenome]:
        """Parse genomes metadata from string content"""
        # Create temporary file and use existing parser
        temp_file = self.output_path / "temp_genomes.tsv"
        try:
            with open(temp_file, 'w') as f:
                f.write(content)
            result = self._parse_genomes_file(str(temp_file))
            temp_file.unlink()
            return result
        except Exception as e:
            logger.error(f"Error parsing genomes metadata: {e}")
            if temp_file.exists():
                temp_file.unlink()
            return []
    
    def _parse_metagenomes_metadata(self, content: str) -> List[Metagenome]:
        """Parse metagenomes metadata from string content"""
        # Create temporary file and use existing parser
        temp_file = self.output_path / "temp_metagenomes.tsv"
        try:
            with open(temp_file, 'w') as f:
                f.write(content)
            result = self._parse_metagenomes_file(str(temp_file))
            temp_file.unlink()
            return result
        except Exception as e:
            logger.error(f"Error parsing metagenomes metadata: {e}")
            if temp_file.exists():
                temp_file.unlink()
            return []
    
    def store_genomes(self, genomes: List[GEMsGenome]):
        """Store genomes in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for genome in genomes:
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO genomes (
                            genome_id, bin_id, sample_id, completeness, contamination,
                            quality_score, quality_level, genome_size, contig_count, n50,
                            gc_content, domain, phylum, class_name, order_name, family,
                            genus, species, gtdb_taxonomy, ncbi_taxonomy, is_novel_species,
                            is_candidate_species, gene_count, protein_count, rrna_5s,
                            rrna_16s, rrna_23s, trna_count, environment_type, habitat,
                            sampling_site, geographic_location, metagenome_source,
                            sequencing_method, assembly_method, binning_method,
                            genome_fasta_file, annotation_file, protein_file, metadata_file,
                            biosample_id, study_id, jgi_project_id, submission_date,
                            last_updated, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        genome.genome_id, genome.bin_id, genome.sample_id,
                        genome.completeness, genome.contamination, genome.quality_score,
                        genome.quality_level, genome.genome_size, genome.contig_count,
                        genome.n50, genome.gc_content, genome.domain, genome.phylum,
                        genome.class_name, genome.order, genome.family, genome.genus,
                        genome.species, genome.gtdb_taxonomy, genome.ncbi_taxonomy,
                        genome.is_novel_species, genome.is_candidate_species,
                        genome.gene_count, genome.protein_count, genome.rrna_5s,
                        genome.rrna_16s, genome.rrna_23s, genome.trna_count,
                        genome.environment_type, genome.habitat, genome.sampling_site,
                        genome.geographic_location, genome.metagenome_source,
                        genome.sequencing_method, genome.assembly_method,
                        genome.binning_method, genome.genome_fasta_file,
                        genome.annotation_file, genome.protein_file, genome.metadata_file,
                        genome.biosample_id, genome.study_id, genome.jgi_project_id,
                        genome.submission_date, genome.last_updated,
                        json.dumps(genome.metadata)
                    ))
                except Exception as e:
                    logger.error(f"Error storing genome {genome.genome_id}: {e}")
            
            conn.commit()
    
    def store_metagenomes(self, metagenomes: List[Metagenome]):
        """Store metagenomes in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for metagenome in metagenomes:
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO metagenomes (
                            metagenome_id, sample_name, study_name, ecosystem,
                            ecosystem_category, ecosystem_type, specific_ecosystem,
                            latitude, longitude, elevation, geographic_location,
                            sampling_date, depth, temperature, ph, salinity,
                            sequencing_platform, sequencing_method, read_count,
                            total_bases, average_read_length, assembly_size,
                            contig_count, assembly_n50, total_bins, high_quality_bins,
                            medium_quality_bins, low_quality_bins, raw_data_file,
                            assembly_file, annotation_file, bins_directory, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        metagenome.metagenome_id, metagenome.sample_name,
                        metagenome.study_name, metagenome.ecosystem,
                        metagenome.ecosystem_category, metagenome.ecosystem_type,
                        metagenome.specific_ecosystem, metagenome.latitude,
                        metagenome.longitude, metagenome.elevation,
                        metagenome.geographic_location, metagenome.sampling_date,
                        metagenome.depth, metagenome.temperature, metagenome.ph,
                        metagenome.salinity, metagenome.sequencing_platform,
                        metagenome.sequencing_method, metagenome.read_count,
                        metagenome.total_bases, metagenome.average_read_length,
                        metagenome.assembly_size, metagenome.contig_count,
                        metagenome.assembly_n50, metagenome.total_bins,
                        metagenome.high_quality_bins, metagenome.medium_quality_bins,
                        metagenome.low_quality_bins, metagenome.raw_data_file,
                        metagenome.assembly_file, metagenome.annotation_file,
                        metagenome.bins_directory, json.dumps(metagenome.metadata)
                    ))
                except Exception as e:
                    logger.error(f"Error storing metagenome {metagenome.metagenome_id}: {e}")
            
            conn.commit()
    
    def export_to_csv(self) -> Dict[str, str]:
        """Export data to CSV files"""
        output_files = {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Export genomes
                df_genomes = pd.read_sql_query('''
                    SELECT genome_id, bin_id, sample_id, completeness, contamination,
                           quality_level, genome_size, contig_count, n50, gc_content,
                           domain, phylum, class_name, order_name, family, genus, species,
                           gtdb_taxonomy, is_novel_species, is_candidate_species,
                           gene_count, protein_count, environment_type, habitat,
                           geographic_location, metagenome_source
                    FROM genomes
                ''', conn)
                
                genomes_file = self.output_path / "jgi_gems_genomes.csv"
                df_genomes.to_csv(genomes_file, index=False)
                output_files['genomes'] = str(genomes_file)
                
                # Export metagenomes
                df_metagenomes = pd.read_sql_query('''
                    SELECT metagenome_id, sample_name, study_name, ecosystem,
                           ecosystem_category, ecosystem_type, specific_ecosystem,
                           latitude, longitude, geographic_location, sampling_date,
                           temperature, ph, salinity, sequencing_platform,
                           assembly_size, total_bins, high_quality_bins
                    FROM metagenomes
                ''', conn)
                
                metagenomes_file = self.output_path / "jgi_gems_metagenomes.csv"
                df_metagenomes.to_csv(metagenomes_file, index=False)
                output_files['metagenomes'] = str(metagenomes_file)
                
                # Export high-quality genomes
                df_hq_genomes = df_genomes[df_genomes['quality_level'] == 'high']
                if not df_hq_genomes.empty:
                    hq_file = self.output_path / "jgi_gems_high_quality_genomes.csv"
                    df_hq_genomes.to_csv(hq_file, index=False)
                    output_files['high_quality_genomes'] = str(hq_file)
                
                # Export novel species
                df_novel = df_genomes[df_genomes['is_novel_species'] == True]
                if not df_novel.empty:
                    novel_file = self.output_path / "jgi_gems_novel_species.csv"
                    df_novel.to_csv(novel_file, index=False)
                    output_files['novel_species'] = str(novel_file)
                
                logger.info(f"Exported JGI GEMs data to {len(output_files)} CSV files")
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
                
                cursor.execute("SELECT COUNT(*) FROM genomes WHERE quality_level = 'high'")
                stats['high_quality_genomes'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT phylum) FROM genomes WHERE phylum != ''")
                stats['phyla_count'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM genomes WHERE is_novel_species = 1")
                stats['novel_species'] = cursor.fetchone()[0]
                
                # Metagenome statistics
                cursor.execute("SELECT COUNT(*) FROM metagenomes")
                stats['total_metagenomes'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT ecosystem) FROM metagenomes WHERE ecosystem != ''")
                stats['ecosystems_count'] = cursor.fetchone()[0]
                
                # Quality distribution
                cursor.execute('''
                    SELECT quality_level, COUNT(*) 
                    FROM genomes 
                    GROUP BY quality_level
                ''')
                quality_dist = {level: count for level, count in cursor.fetchall()}
                stats['quality_distribution'] = quality_dist
                
                # Taxonomic distribution
                cursor.execute('''
                    SELECT phylum, COUNT(*) 
                    FROM genomes 
                    WHERE phylum != ''
                    GROUP BY phylum 
                    ORDER BY COUNT(*) DESC 
                    LIMIT 10
                ''')
                top_phyla = dict(cursor.fetchall())
                stats['top_phyla'] = top_phyla
                
                # Environmental distribution
                cursor.execute('''
                    SELECT environment_type, COUNT(*) 
                    FROM genomes 
                    WHERE environment_type != ''
                    GROUP BY environment_type 
                    ORDER BY COUNT(*) DESC 
                    LIMIT 10
                ''')
                top_environments = dict(cursor.fetchall())
                stats['top_environments'] = top_environments
                
                return stats
                
        except Exception as e:
            logger.error(f"Error generating statistics: {e}")
            return {}

class JGIGEMsIntegration:
    """Main integration class for JGI GEMs data"""
    
    def __init__(self, output_path: str = "data"):
        self.output_path = Path(output_path)
        self.downloader = GEMsDownloader()
        self.parser = GEMsParser(str(self.output_path / "processed" / "jgi_gems"))
        
        # Progress tracking
        self.progress_file = self.output_path / "interim" / "jgi_gems_progress.json"
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
    
    async def download_catalog_data(self, max_genomes: Optional[int] = None) -> Dict[str, Any]:
        """Download and process GEMs catalog"""
        progress = self._load_progress()
        results = {"downloads": {}, "processing": {}, "statistics": {}}
        
        try:
            # Download catalog
            catalog_file = await self.downloader.download_gems_catalog()
            
            if catalog_file:
                progress["downloads"]["catalog"] = catalog_file
                results["downloads"]["catalog"] = catalog_file
                
                # Parse catalog
                genomes, metagenomes = self.parser.parse_catalog_file(catalog_file)
                
                # Limit genomes if specified
                if max_genomes and len(genomes) > max_genomes:
                    # Prioritize high-quality genomes
                    hq_genomes = [g for g in genomes if g.quality_level == 'high']
                    other_genomes = [g for g in genomes if g.quality_level != 'high']
                    
                    if len(hq_genomes) >= max_genomes:
                        genomes = hq_genomes[:max_genomes]
                    else:
                        remaining = max_genomes - len(hq_genomes)
                        genomes = hq_genomes + other_genomes[:remaining]
                
                # Store data
                if genomes:
                    self.parser.store_genomes(genomes)
                    results["processing"]["genomes"] = len(genomes)
                
                if metagenomes:
                    self.parser.store_metagenomes(metagenomes)
                    results["processing"]["metagenomes"] = len(metagenomes)
                
                # Generate statistics
                stats = self.parser.generate_statistics()
                results["statistics"] = stats
                
                # Save progress
                progress.update(results)
                self._save_progress(progress)
                
                logger.info(f"Processed {len(genomes)} genomes and {len(metagenomes)} metagenomes")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in catalog data download: {e}")
            return results
        finally:
            await self.downloader.close()
    
    async def download_sample_genomes(self, sample_ids: List[str], max_downloads: Optional[int] = None) -> Dict[str, Any]:
        """Download specific genome files"""
        results = {"downloads": {}, "errors": []}
        
        try:
            if max_downloads:
                sample_ids = sample_ids[:max_downloads]
            
            downloaded_files = await self.downloader.download_sample_genomes(sample_ids, max_downloads)
            results["downloads"] = downloaded_files
            
            logger.info(f"Downloaded {len(downloaded_files)} genome files")
            return results
            
        except Exception as e:
            logger.error(f"Error downloading sample genomes: {e}")
            results["errors"].append(str(e))
            return results
        finally:
            await self.downloader.close()
    
    def export_datasets(self) -> Dict[str, str]:
        """Export all datasets to CSV format"""
        return self.parser.export_to_csv()
    
    async def run_full_integration(self, max_genomes: Optional[int] = None,
                                 download_genome_files: bool = False) -> Dict[str, Any]:
        """Run complete JGI GEMs integration"""
        logger.info("Starting comprehensive JGI GEMs integration")
        
        start_time = time.time()
        results = {
            "status": "started",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "catalog_data": {},
            "genome_downloads": {},
            "exports": {},
            "statistics": {},
            "errors": []
        }
        
        try:
            # Get database statistics
            gems_stats = await self.downloader.get_gems_statistics()
            results["database_info"] = gems_stats
            
            # Download catalog data
            catalog_results = await self.download_catalog_data(max_genomes=max_genomes)
            results["catalog_data"] = catalog_results
            
            # Optionally download genome files
            if download_genome_files and catalog_results.get("processing", {}).get("genomes", 0) > 0:
                # Get sample IDs from stored genomes
                with sqlite3.connect(self.parser.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT DISTINCT sample_id FROM genomes WHERE quality_level = 'high' LIMIT 100")
                    sample_ids = [row[0] for row in cursor.fetchall() if row[0]]
                
                if sample_ids:
                    genome_results = await self.download_sample_genomes(sample_ids, max_downloads=50)
                    results["genome_downloads"] = genome_results
            
            # Export to CSV
            export_files = self.export_datasets()
            results["exports"] = export_files
            
            # Generate final statistics
            stats = self.parser.generate_statistics()
            results["statistics"] = stats
            
            # Final results
            end_time = time.time()
            results.update({
                "status": "completed",
                "end_time": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": end_time - start_time,
                "total_files_exported": len(export_files)
            })
            
            logger.info(f"JGI GEMs integration completed in {end_time - start_time:.2f} seconds")
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
    integration = JGIGEMsIntegration()
    
    # Test with limited data
    results = await integration.run_full_integration(
        max_genomes=1000,
        download_genome_files=False
    )
    
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    asyncio.run(main()) 
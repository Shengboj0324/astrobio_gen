#!/usr/bin/env python3
"""
Comprehensive NCBI/AGORA2 Integration System
===========================================

Advanced system for downloading, processing, and integrating real genomic data:
- AGORA2 metabolic reconstructions (7,302 human microorganisms)
- NCBI genome assemblies and annotations
- Microbiome taxonomic data
- Metabolic network reconstructions
- Pathway-genome associations
- Quality control and validation

NASA-grade data processing with comprehensive error handling.

Author: AI Assistant
Date: 2025
"""

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
import pickle
import gzip
import hashlib
import time
from urllib.parse import urljoin
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import ftplib
import tarfile
from io import BytesIO
import zipfile
from scipy.sparse import csr_matrix
from scipy.io import mmwrite, mmread
import networkx as nx
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import cobra
from cobra.io import read_sbml_model, write_sbml_model
from cobra.core import Model, Reaction, Metabolite

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AGORA2Model:
    """AGORA2 metabolic model data structure"""
    model_id: str
    organism: str
    strain: str
    taxonomy: str
    domain: str
    phylum: str
    class_name: str
    order: str
    family: str
    genus: str
    species: str
    reactions: int = 0
    metabolites: int = 0
    genes: int = 0
    biomass_reaction: str = ""
    growth_medium: str = ""
    model_file: str = ""
    quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class NCBIGenome:
    """NCBI genome data structure"""
    assembly_accession: str
    organism_name: str
    strain: str
    taxid: int
    assembly_level: str
    genome_size: int = 0
    contig_count: int = 0
    scaffold_count: int = 0
    annotation_provider: str = ""
    annotation_date: str = ""
    ftp_path: str = ""
    checksum: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class MetabolicReaction:
    """Metabolic reaction data structure"""
    reaction_id: str
    name: str
    equation: str
    reversible: bool = True
    lower_bound: float = -1000.0
    upper_bound: float = 1000.0
    gene_reaction_rule: str = ""
    subsystem: str = ""
    ec_number: str = ""
    confidence: float = 0.0
    organism: str = ""
    metabolites: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Metabolite:
    """Metabolite data structure"""
    metabolite_id: str
    name: str
    formula: str = ""
    charge: int = 0
    compartment: str = ""
    kegg_id: str = ""
    chebi_id: str = ""
    bigg_id: str = ""
    pathways: List[str] = field(default_factory=list)
    reactions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class AGORA2Downloader:
    """Advanced AGORA2 data downloader"""
    
    def __init__(self, base_url: str = "https://www.vmh.life/files/reconstructions/AGORA2/"):
        self.base_url = base_url
        self.cache_path = Path("data/raw/agora2/cache")
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.session = None
        self.rate_limit_delay = 0.5  # 500ms between requests
        self.max_retries = 3
        
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
            timeout = aiohttp.ClientTimeout(total=600)  # 10 minutes for large files
            connector = aiohttp.TCPConnector(limit=10)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )
        return self.session
    
    def _get_cache_path(self, filename: str) -> Path:
        """Get cache file path"""
        return self.cache_path / filename
    
    def _load_from_cache(self, filename: str) -> Optional[Any]:
        """Load data from cache"""
        cache_file = self._get_cache_path(filename)
        if cache_file.exists():
            try:
                # Check if cache is recent (within 7 days)
                file_time = datetime.fromtimestamp(cache_file.stat().st_mtime, tz=timezone.utc)
                if (datetime.now(timezone.utc) - file_time).days < 7:
                    if filename.endswith('.json'):
                        with open(cache_file, 'r') as f:
                            return json.load(f)
                    elif filename.endswith('.pkl'):
                        with open(cache_file, 'rb') as f:
                            return pickle.load(f)
                    else:
                        return cache_file
            except Exception as e:
                logger.warning(f"Error loading cache {cache_file}: {e}")
        return None
    
    def _save_to_cache(self, filename: str, data: Any):
        """Save data to cache"""
        cache_file = self._get_cache_path(filename)
        try:
            if filename.endswith('.json'):
                with open(cache_file, 'w') as f:
                    json.dump(data, f, indent=2)
            elif filename.endswith('.pkl'):
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
            else:
                # For binary data
                with open(cache_file, 'wb') as f:
                    f.write(data)
        except Exception as e:
            logger.warning(f"Error saving cache {cache_file}: {e}")
    
    async def fetch_agora2_model_list(self) -> List[Dict[str, Any]]:
        """Fetch AGORA2 model list from info file"""
        cache_data = self._load_from_cache("agora2_model_list.json")
        if cache_data:
            return cache_data
        
        session = await self._get_session()
        
        try:
            # Download AGORA2 info file
            info_url = urljoin(self.base_url, "AGORA2_infoFile.xlsx")
            
            async with session.get(info_url) as response:
                if response.status == 200:
                    content = await response.read()
                    
                    # Save to temporary file
                    temp_file = self.cache_path / "temp_agora2_info.xlsx"
                    with open(temp_file, 'wb') as f:
                        f.write(content)
                    
                    # Read Excel file
                    df = pd.read_excel(temp_file)
                    temp_file.unlink()  # Clean up
                    
                    # Process model list
                    models = []
                    for _, row in df.iterrows():
                        model_data = {
                            'model_id': row.get('modelID', ''),
                            'organism': row.get('organism', ''),
                            'strain': row.get('strain', ''),
                            'taxonomy': row.get('taxonomy', ''),
                            'domain': row.get('domain', ''),
                            'phylum': row.get('phylum', ''),
                            'class': row.get('class', ''),
                            'order': row.get('order', ''),
                            'family': row.get('family', ''),
                            'genus': row.get('genus', ''),
                            'species': row.get('species', ''),
                            'reactions': row.get('reactions', 0),
                            'metabolites': row.get('metabolites', 0),
                            'genes': row.get('genes', 0),
                            'biomass_reaction': row.get('biomass', ''),
                            'growth_medium': row.get('growthMedium', ''),
                            'model_file': row.get('modelFile', '')
                        }
                        models.append(model_data)
                    
                    # Cache results
                    self._save_to_cache("agora2_model_list.json", models)
                    
                    logger.info(f"Fetched {len(models)} AGORA2 models")
                    return models
                else:
                    logger.error(f"Failed to fetch AGORA2 info file: HTTP {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching AGORA2 model list: {e}")
            return []
    
    async def download_agora2_model(self, model_id: str, model_file: str) -> Optional[str]:
        """Download individual AGORA2 model file"""
        cache_file = self._get_cache_path(f"{model_id}.xml")
        if cache_file.exists():
            return str(cache_file)
        
        session = await self._get_session()
        
        try:
            # Construct download URL
            model_url = urljoin(self.base_url, f"individual_models/{model_file}")
            
            await asyncio.sleep(self.rate_limit_delay)
            
            async with session.get(model_url) as response:
                if response.status == 200:
                    content = await response.read()
                    
                    # Save model file
                    with open(cache_file, 'wb') as f:
                        f.write(content)
                    
                    logger.debug(f"Downloaded model: {model_id}")
                    return str(cache_file)
                else:
                    logger.warning(f"Failed to download model {model_id}: HTTP {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error downloading model {model_id}: {e}")
            return None
    
    async def download_agora2_collection(self) -> Optional[str]:
        """Download complete AGORA2 collection"""
        cache_file = self._get_cache_path("AGORA2_collection.zip")
        if cache_file.exists():
            return str(cache_file)
        
        session = await self._get_session()
        
        try:
            # Download complete collection
            collection_url = urljoin(self.base_url, "AGORA2.zip")
            
            logger.info("Downloading AGORA2 complete collection (this may take a while)")
            
            async with session.get(collection_url) as response:
                if response.status == 200:
                    content = await response.read()
                    
                    # Save collection
                    with open(cache_file, 'wb') as f:
                        f.write(content)
                    
                    logger.info("Downloaded AGORA2 complete collection")
                    return str(cache_file)
                else:
                    logger.error(f"Failed to download AGORA2 collection: HTTP {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error downloading AGORA2 collection: {e}")
            return None
    
    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()

class NCBIGenomeDownloader:
    """Advanced NCBI genome data downloader"""
    
    def __init__(self, ftp_host: str = "ftp.ncbi.nlm.nih.gov"):
        self.ftp_host = ftp_host
        self.cache_path = Path("data/raw/ncbi/cache")
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.session = None
        self.assembly_summary_cache = {}
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=600)
            connector = aiohttp.TCPConnector(limit=10)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )
        return self.session
    
    def _get_cache_path(self, filename: str) -> Path:
        """Get cache file path"""
        return self.cache_path / filename
    
    async def fetch_assembly_summary(self, domain: str = "bacteria") -> List[Dict[str, Any]]:
        """Fetch NCBI assembly summary for a domain"""
        cache_file = self._get_cache_path(f"assembly_summary_{domain}.json")
        
        # Check cache
        if cache_file.exists():
            try:
                file_time = datetime.fromtimestamp(cache_file.stat().st_mtime, tz=timezone.utc)
                if (datetime.now(timezone.utc) - file_time).days < 7:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading cache {cache_file}: {e}")
        
        session = await self._get_session()
        
        try:
            # Download assembly summary
            summary_url = f"https://ftp.ncbi.nlm.nih.gov/genomes/refseq/{domain}/assembly_summary.txt"
            
            async with session.get(summary_url) as response:
                if response.status == 200:
                    content = await response.text()
                    
                    assemblies = []
                    lines = content.strip().split('\n')
                    
                    # Find header line
                    header_idx = 0
                    for i, line in enumerate(lines):
                        if line.startswith('# assembly_accession'):
                            header_idx = i
                            break
                    
                    # Parse header
                    header_line = lines[header_idx].lstrip('# ')
                    headers = header_line.split('\t')
                    
                    # Parse data lines
                    for line in lines[header_idx + 1:]:
                        if line.startswith('#'):
                            continue
                        
                        parts = line.split('\t')
                        if len(parts) >= len(headers):
                            assembly_data = {}
                            for i, header in enumerate(headers):
                                if i < len(parts):
                                    assembly_data[header.strip()] = parts[i].strip()
                            assemblies.append(assembly_data)
                    
                    # Cache results
                    with open(cache_file, 'w') as f:
                        json.dump(assemblies, f, indent=2)
                    
                    logger.info(f"Fetched {len(assemblies)} {domain} assemblies")
                    return assemblies
                else:
                    logger.error(f"Failed to fetch assembly summary: HTTP {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching assembly summary for {domain}: {e}")
            return []
    
    async def download_genome_annotation(self, ftp_path: str, assembly_accession: str) -> Optional[str]:
        """Download genome annotation file"""
        cache_file = self._get_cache_path(f"{assembly_accession}_genomic.gff.gz")
        if cache_file.exists():
            return str(cache_file)
        
        session = await self._get_session()
        
        try:
            # Construct annotation file URL
            if ftp_path.startswith('ftp://'):
                ftp_path = ftp_path.replace('ftp://', 'https://')
            
            annotation_url = f"{ftp_path}/{assembly_accession}_genomic.gff.gz"
            
            async with session.get(annotation_url) as response:
                if response.status == 200:
                    content = await response.read()
                    
                    # Save annotation file
                    with open(cache_file, 'wb') as f:
                        f.write(content)
                    
                    logger.debug(f"Downloaded annotation: {assembly_accession}")
                    return str(cache_file)
                else:
                    logger.warning(f"Failed to download annotation {assembly_accession}: HTTP {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error downloading annotation {assembly_accession}: {e}")
            return None
    
    async def download_genome_sequence(self, ftp_path: str, assembly_accession: str) -> Optional[str]:
        """Download genome sequence file"""
        cache_file = self._get_cache_path(f"{assembly_accession}_genomic.fna.gz")
        if cache_file.exists():
            return str(cache_file)
        
        session = await self._get_session()
        
        try:
            # Construct sequence file URL
            if ftp_path.startswith('ftp://'):
                ftp_path = ftp_path.replace('ftp://', 'https://')
            
            sequence_url = f"{ftp_path}/{assembly_accession}_genomic.fna.gz"
            
            async with session.get(sequence_url) as response:
                if response.status == 200:
                    content = await response.read()
                    
                    # Save sequence file
                    with open(cache_file, 'wb') as f:
                        f.write(content)
                    
                    logger.debug(f"Downloaded sequence: {assembly_accession}")
                    return str(cache_file)
                else:
                    logger.warning(f"Failed to download sequence {assembly_accession}: HTTP {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error downloading sequence {assembly_accession}: {e}")
            return None
    
    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()

class MetabolicModelProcessor:
    """Advanced metabolic model processor"""
    
    def __init__(self, output_path: str = "data/processed/agora2"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.output_path / "metabolic_models.db"
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database for metabolic models"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # AGORA2 models table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agora2_models (
                    model_id TEXT PRIMARY KEY,
                    organism TEXT,
                    strain TEXT,
                    taxonomy TEXT,
                    domain TEXT,
                    phylum TEXT,
                    class_name TEXT,
                    order_name TEXT,
                    family TEXT,
                    genus TEXT,
                    species TEXT,
                    reactions INTEGER DEFAULT 0,
                    metabolites INTEGER DEFAULT 0,
                    genes INTEGER DEFAULT 0,
                    biomass_reaction TEXT,
                    growth_medium TEXT,
                    model_file TEXT,
                    quality_score REAL DEFAULT 0.0,
                    file_path TEXT,
                    last_updated TIMESTAMP
                )
            ''')
            
            # Reactions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS reactions (
                    reaction_id TEXT,
                    model_id TEXT,
                    name TEXT,
                    equation TEXT,
                    reversible BOOLEAN DEFAULT TRUE,
                    lower_bound REAL DEFAULT -1000.0,
                    upper_bound REAL DEFAULT 1000.0,
                    gene_reaction_rule TEXT,
                    subsystem TEXT,
                    ec_number TEXT,
                    confidence REAL DEFAULT 0.0,
                    PRIMARY KEY (reaction_id, model_id),
                    FOREIGN KEY (model_id) REFERENCES agora2_models(model_id)
                )
            ''')
            
            # Metabolites table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metabolites (
                    metabolite_id TEXT,
                    model_id TEXT,
                    name TEXT,
                    formula TEXT,
                    charge INTEGER DEFAULT 0,
                    compartment TEXT,
                    kegg_id TEXT,
                    chebi_id TEXT,
                    bigg_id TEXT,
                    PRIMARY KEY (metabolite_id, model_id),
                    FOREIGN KEY (model_id) REFERENCES agora2_models(model_id)
                )
            ''')
            
            # Reaction-metabolite stoichiometry
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS reaction_metabolites (
                    reaction_id TEXT,
                    metabolite_id TEXT,
                    model_id TEXT,
                    coefficient REAL,
                    PRIMARY KEY (reaction_id, metabolite_id, model_id),
                    FOREIGN KEY (reaction_id, model_id) REFERENCES reactions(reaction_id, model_id),
                    FOREIGN KEY (metabolite_id, model_id) REFERENCES metabolites(metabolite_id, model_id)
                )
            ''')
            
            # Genes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS genes (
                    gene_id TEXT,
                    model_id TEXT,
                    name TEXT,
                    functional_annotation TEXT,
                    PRIMARY KEY (gene_id, model_id),
                    FOREIGN KEY (model_id) REFERENCES agora2_models(model_id)
                )
            ''')
            
            # NCBI genomes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ncbi_genomes (
                    assembly_accession TEXT PRIMARY KEY,
                    organism_name TEXT,
                    strain TEXT,
                    taxid INTEGER,
                    assembly_level TEXT,
                    genome_size INTEGER DEFAULT 0,
                    contig_count INTEGER DEFAULT 0,
                    scaffold_count INTEGER DEFAULT 0,
                    annotation_provider TEXT,
                    annotation_date TEXT,
                    ftp_path TEXT,
                    checksum TEXT,
                    sequence_file TEXT,
                    annotation_file TEXT,
                    last_updated TIMESTAMP
                )
            ''')
            
            # Model-genome associations
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_genome_associations (
                    model_id TEXT,
                    assembly_accession TEXT,
                    confidence REAL DEFAULT 0.0,
                    match_method TEXT,
                    PRIMARY KEY (model_id, assembly_accession),
                    FOREIGN KEY (model_id) REFERENCES agora2_models(model_id),
                    FOREIGN KEY (assembly_accession) REFERENCES ncbi_genomes(assembly_accession)
                )
            ''')
            
            conn.commit()
    
    def store_agora2_model(self, model: AGORA2Model, file_path: str):
        """Store AGORA2 model metadata in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO agora2_models 
                (model_id, organism, strain, taxonomy, domain, phylum, class_name, order_name,
                 family, genus, species, reactions, metabolites, genes, biomass_reaction,
                 growth_medium, model_file, quality_score, file_path, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model.model_id,
                model.organism,
                model.strain,
                model.taxonomy,
                model.domain,
                model.phylum,
                model.class_name,
                model.order,
                model.family,
                model.genus,
                model.species,
                model.reactions,
                model.metabolites,
                model.genes,
                model.biomass_reaction,
                model.growth_medium,
                model.model_file,
                model.quality_score,
                file_path,
                model.last_updated
            ))
            
            conn.commit()
    
    def parse_sbml_model(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Parse SBML model file and extract detailed information"""
        try:
            # Load COBRA model
            model = read_sbml_model(file_path)
            
            # Extract reactions
            reactions = []
            for reaction in model.reactions:
                reaction_data = {
                    'reaction_id': reaction.id,
                    'name': reaction.name,
                    'equation': str(reaction.reaction),
                    'reversible': reaction.reversibility,
                    'lower_bound': reaction.lower_bound,
                    'upper_bound': reaction.upper_bound,
                    'gene_reaction_rule': reaction.gene_reaction_rule,
                    'subsystem': reaction.subsystem,
                    'metabolites': {met.id: coeff for met, coeff in reaction.metabolites.items()}
                }
                reactions.append(reaction_data)
            
            # Extract metabolites
            metabolites = []
            for metabolite in model.metabolites:
                metabolite_data = {
                    'metabolite_id': metabolite.id,
                    'name': metabolite.name,
                    'formula': metabolite.formula,
                    'charge': metabolite.charge,
                    'compartment': metabolite.compartment
                }
                metabolites.append(metabolite_data)
            
            # Extract genes
            genes = []
            for gene in model.genes:
                gene_data = {
                    'gene_id': gene.id,
                    'name': gene.name,
                    'functional_annotation': gene.functional_annotation
                }
                genes.append(gene_data)
            
            return {
                'model_id': model.id,
                'name': model.name,
                'reactions': reactions,
                'metabolites': metabolites,
                'genes': genes,
                'objective': str(model.objective),
                'compartments': dict(model.compartments),
                'statistics': {
                    'reaction_count': len(reactions),
                    'metabolite_count': len(metabolites),
                    'gene_count': len(genes)
                }
            }
            
        except Exception as e:
            logger.error(f"Error parsing SBML model {file_path}: {e}")
            return None
    
    def store_model_details(self, model_id: str, model_details: Dict[str, Any]):
        """Store detailed model information in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Store reactions
            for reaction in model_details['reactions']:
                cursor.execute('''
                    INSERT OR REPLACE INTO reactions 
                    (reaction_id, model_id, name, equation, reversible, lower_bound, upper_bound,
                     gene_reaction_rule, subsystem, ec_number, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    reaction['reaction_id'],
                    model_id,
                    reaction['name'],
                    reaction['equation'],
                    reaction['reversible'],
                    reaction['lower_bound'],
                    reaction['upper_bound'],
                    reaction['gene_reaction_rule'],
                    reaction['subsystem'],
                    reaction.get('ec_number', ''),
                    reaction.get('confidence', 0.0)
                ))
                
                # Store reaction-metabolite relationships
                for metabolite_id, coefficient in reaction['metabolites'].items():
                    cursor.execute('''
                        INSERT OR REPLACE INTO reaction_metabolites 
                        (reaction_id, metabolite_id, model_id, coefficient)
                        VALUES (?, ?, ?, ?)
                    ''', (reaction['reaction_id'], metabolite_id, model_id, coefficient))
            
            # Store metabolites
            for metabolite in model_details['metabolites']:
                cursor.execute('''
                    INSERT OR REPLACE INTO metabolites 
                    (metabolite_id, model_id, name, formula, charge, compartment, kegg_id, chebi_id, bigg_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metabolite['metabolite_id'],
                    model_id,
                    metabolite['name'],
                    metabolite['formula'],
                    metabolite['charge'],
                    metabolite['compartment'],
                    metabolite.get('kegg_id', ''),
                    metabolite.get('chebi_id', ''),
                    metabolite.get('bigg_id', '')
                ))
            
            # Store genes
            for gene in model_details['genes']:
                cursor.execute('''
                    INSERT OR REPLACE INTO genes 
                    (gene_id, model_id, name, functional_annotation)
                    VALUES (?, ?, ?, ?)
                ''', (
                    gene['gene_id'],
                    model_id,
                    gene['name'],
                    gene['functional_annotation']
                ))
            
            conn.commit()
    
    def store_ncbi_genome(self, genome: NCBIGenome, sequence_file: str = "", annotation_file: str = ""):
        """Store NCBI genome metadata in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO ncbi_genomes 
                (assembly_accession, organism_name, strain, taxid, assembly_level, genome_size,
                 contig_count, scaffold_count, annotation_provider, annotation_date, ftp_path,
                 checksum, sequence_file, annotation_file, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                genome.assembly_accession,
                genome.organism_name,
                genome.strain,
                genome.taxid,
                genome.assembly_level,
                genome.genome_size,
                genome.contig_count,
                genome.scaffold_count,
                genome.annotation_provider,
                genome.annotation_date,
                genome.ftp_path,
                genome.checksum,
                sequence_file,
                annotation_file,
                genome.last_updated
            ))
            
            conn.commit()
    
    def associate_model_genome(self, model_id: str, assembly_accession: str, confidence: float, method: str):
        """Associate AGORA2 model with NCBI genome"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO model_genome_associations 
                (model_id, assembly_accession, confidence, match_method)
                VALUES (?, ?, ?, ?)
            ''', (model_id, assembly_accession, confidence, method))
            
            conn.commit()
    
    def create_metabolic_network(self, model_id: str) -> Optional[nx.DiGraph]:
        """Create metabolic network graph for a model"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get reaction-metabolite relationships
            cursor.execute('''
                SELECT r.reaction_id, rm.metabolite_id, rm.coefficient, r.reversible
                FROM reactions r
                JOIN reaction_metabolites rm ON r.reaction_id = rm.reaction_id AND r.model_id = rm.model_id
                WHERE r.model_id = ?
            ''', (model_id,))
            
            relationships = cursor.fetchall()
            
            if not relationships:
                return None
            
            # Create directed graph
            G = nx.DiGraph()
            
            # Group by reaction
            reaction_groups = {}
            for reaction_id, metabolite_id, coefficient, reversible in relationships:
                if reaction_id not in reaction_groups:
                    reaction_groups[reaction_id] = {
                        'substrates': [],
                        'products': [],
                        'reversible': reversible
                    }
                
                if coefficient < 0:
                    reaction_groups[reaction_id]['substrates'].append(metabolite_id)
                elif coefficient > 0:
                    reaction_groups[reaction_id]['products'].append(metabolite_id)
            
            # Add edges
            for reaction_id, reaction_data in reaction_groups.items():
                substrates = reaction_data['substrates']
                products = reaction_data['products']
                reversible = reaction_data['reversible']
                
                # Add forward edges
                for substrate in substrates:
                    for product in products:
                        G.add_edge(substrate, product, reaction=reaction_id, direction='forward')
                
                # Add reverse edges if reversible
                if reversible:
                    for product in products:
                        for substrate in substrates:
                            G.add_edge(product, substrate, reaction=reaction_id, direction='reverse')
            
            return G
    
    def export_to_csv(self) -> Dict[str, str]:
        """Export database to CSV files"""
        output_files = {}
        
        with sqlite3.connect(self.db_path) as conn:
            # Export AGORA2 models
            models_df = pd.read_sql_query("SELECT * FROM agora2_models", conn)
            models_file = self.output_path / "agora2_models.csv"
            models_df.to_csv(models_file, index=False)
            output_files['agora2_models'] = str(models_file)
            
            # Export reactions
            reactions_df = pd.read_sql_query("SELECT * FROM reactions", conn)
            reactions_file = self.output_path / "reactions.csv"
            reactions_df.to_csv(reactions_file, index=False)
            output_files['reactions'] = str(reactions_file)
            
            # Export metabolites
            metabolites_df = pd.read_sql_query("SELECT * FROM metabolites", conn)
            metabolites_file = self.output_path / "metabolites.csv"
            metabolites_df.to_csv(metabolites_file, index=False)
            output_files['metabolites'] = str(metabolites_file)
            
            # Export genes
            genes_df = pd.read_sql_query("SELECT * FROM genes", conn)
            genes_file = self.output_path / "genes.csv"
            genes_df.to_csv(genes_file, index=False)
            output_files['genes'] = str(genes_file)
            
            # Export NCBI genomes
            genomes_df = pd.read_sql_query("SELECT * FROM ncbi_genomes", conn)
            genomes_file = self.output_path / "ncbi_genomes.csv"
            genomes_df.to_csv(genomes_file, index=False)
            output_files['ncbi_genomes'] = str(genomes_file)
            
            # Export model-genome associations
            associations_df = pd.read_sql_query("SELECT * FROM model_genome_associations", conn)
            associations_file = self.output_path / "model_genome_associations.csv"
            associations_df.to_csv(associations_file, index=False)
            output_files['model_genome_associations'] = str(associations_file)
            
            # Export metabolic network
            network_query = '''
                SELECT r.model_id, r.reaction_id, rm1.metabolite_id as substrate,
                       rm2.metabolite_id as product, rm1.coefficient as substrate_coeff,
                       rm2.coefficient as product_coeff, r.reversible
                FROM reactions r
                JOIN reaction_metabolites rm1 ON r.reaction_id = rm1.reaction_id 
                    AND r.model_id = rm1.model_id AND rm1.coefficient < 0
                JOIN reaction_metabolites rm2 ON r.reaction_id = rm2.reaction_id 
                    AND r.model_id = rm2.model_id AND rm2.coefficient > 0
                ORDER BY r.model_id, r.reaction_id
            '''
            network_df = pd.read_sql_query(network_query, conn)
            network_file = self.output_path / "metabolic_network.csv"
            network_df.to_csv(network_file, index=False)
            output_files['metabolic_network'] = str(network_file)
        
        return output_files
    
    def generate_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # AGORA2 model statistics
            cursor.execute("SELECT COUNT(*) FROM agora2_models")
            stats['total_models'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM reactions")
            stats['total_reactions'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM metabolites")
            stats['total_metabolites'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM genes")
            stats['total_genes'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM ncbi_genomes")
            stats['total_genomes'] = cursor.fetchone()[0]
            
            # Taxonomic distribution
            cursor.execute('''
                SELECT domain, COUNT(*) as count
                FROM agora2_models
                GROUP BY domain
                ORDER BY count DESC
            ''')
            stats['domain_distribution'] = cursor.fetchall()
            
            cursor.execute('''
                SELECT phylum, COUNT(*) as count
                FROM agora2_models
                GROUP BY phylum
                ORDER BY count DESC
                LIMIT 10
            ''')
            stats['top_phyla'] = cursor.fetchall()
            
            # Model size statistics
            cursor.execute('''
                SELECT AVG(reactions), AVG(metabolites), AVG(genes),
                       MIN(reactions), MAX(reactions),
                       MIN(metabolites), MAX(metabolites),
                       MIN(genes), MAX(genes)
                FROM agora2_models
            ''')
            size_stats = cursor.fetchone()
            stats['model_size_stats'] = {
                'avg_reactions': size_stats[0],
                'avg_metabolites': size_stats[1],
                'avg_genes': size_stats[2],
                'min_reactions': size_stats[3],
                'max_reactions': size_stats[4],
                'min_metabolites': size_stats[5],
                'max_metabolites': size_stats[6],
                'min_genes': size_stats[7],
                'max_genes': size_stats[8]
            }
            
            # Top models by size
            cursor.execute('''
                SELECT model_id, organism, reactions, metabolites, genes
                FROM agora2_models
                ORDER BY reactions DESC
                LIMIT 10
            ''')
            stats['largest_models'] = cursor.fetchall()
            
            # Reaction subsystem distribution
            cursor.execute('''
                SELECT subsystem, COUNT(*) as count
                FROM reactions
                WHERE subsystem != ''
                GROUP BY subsystem
                ORDER BY count DESC
                LIMIT 10
            ''')
            stats['top_subsystems'] = cursor.fetchall()
            
            # Quality score statistics
            cursor.execute('''
                SELECT AVG(quality_score), MIN(quality_score), MAX(quality_score)
                FROM agora2_models
                WHERE quality_score > 0
            ''')
            quality_stats = cursor.fetchone()
            if quality_stats[0] is not None:
                stats['quality_stats'] = {
                    'avg_quality': quality_stats[0],
                    'min_quality': quality_stats[1],
                    'max_quality': quality_stats[2]
                }
            
            stats['timestamp'] = datetime.now(timezone.utc).isoformat()
            
        return stats

class NCBIAgoraIntegration:
    """Main class for comprehensive NCBI/AGORA2 integration"""
    
    def __init__(self, output_path: str = "data"):
        self.output_path = Path(output_path)
        self.agora2_downloader = AGORA2Downloader()
        self.ncbi_downloader = NCBIGenomeDownloader()
        self.processor = MetabolicModelProcessor(str(self.output_path / "processed/agora2"))
        self.progress_file = self.output_path / "raw/agora2/progress.json"
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _load_progress(self) -> Dict[str, Any]:
        """Load download progress"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            'downloaded_models': [],
            'processed_models': [],
            'downloaded_genomes': [],
            'total_models': 0,
            'total_genomes': 0
        }
    
    def _save_progress(self, progress: Dict[str, Any]):
        """Save download progress"""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving progress: {e}")
    
    async def download_agora2_models(self, max_models: Optional[int] = None) -> Dict[str, Any]:
        """Download AGORA2 models with progress tracking"""
        logger.info("Starting AGORA2 model download")
        
        progress = self._load_progress()
        
        # Get model list
        model_list = await self.agora2_downloader.fetch_agora2_model_list()
        logger.info(f"Found {len(model_list)} AGORA2 models")
        
        if max_models:
            model_list = model_list[:max_models]
            logger.info(f"Limited to {max_models} models for testing")
        
        progress['total_models'] = len(model_list)
        
        # Download and process models
        downloaded_count = 0
        processed_count = 0
        failed_count = 0
        
        for i, model_info in enumerate(model_list):
            model_id = model_info['model_id']
            model_file = model_info['model_file']
            
            # Skip if already downloaded
            if model_id in progress['downloaded_models']:
                continue
            
            try:
                logger.info(f"Downloading model {i+1}/{len(model_list)}: {model_id}")
                
                # Download model file
                file_path = await self.agora2_downloader.download_agora2_model(model_id, model_file)
                
                if file_path:
                    # Create AGORA2Model object
                    model = AGORA2Model(
                        model_id=model_id,
                        organism=model_info['organism'],
                        strain=model_info['strain'],
                        taxonomy=model_info['taxonomy'],
                        domain=model_info['domain'],
                        phylum=model_info['phylum'],
                        class_name=model_info['class'],
                        order=model_info['order'],
                        family=model_info['family'],
                        genus=model_info['genus'],
                        species=model_info['species'],
                        reactions=model_info['reactions'],
                        metabolites=model_info['metabolites'],
                        genes=model_info['genes'],
                        biomass_reaction=model_info['biomass_reaction'],
                        growth_medium=model_info['growth_medium'],
                        model_file=model_file
                    )
                    
                    # Store model metadata
                    self.processor.store_agora2_model(model, file_path)
                    progress['downloaded_models'].append(model_id)
                    downloaded_count += 1
                    
                    # Parse and store detailed model information
                    if model_id not in progress['processed_models']:
                        model_details = self.processor.parse_sbml_model(file_path)
                        if model_details:
                            self.processor.store_model_details(model_id, model_details)
                            progress['processed_models'].append(model_id)
                            processed_count += 1
                        else:
                            logger.warning(f"Failed to parse model: {model_id}")
                    
                    # Save progress every 10 models
                    if downloaded_count % 10 == 0:
                        self._save_progress(progress)
                        logger.info(f"Progress saved: {downloaded_count} models downloaded, {processed_count} processed")
                else:
                    failed_count += 1
                    logger.warning(f"Failed to download model: {model_id}")
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"Error downloading model {model_id}: {e}")
        
        # Final progress save
        self._save_progress(progress)
        
        logger.info(f"AGORA2 download complete: {downloaded_count} downloaded, {processed_count} processed, {failed_count} failed")
        
        return {
            'downloaded_models': downloaded_count,
            'processed_models': processed_count,
            'failed_models': failed_count,
            'total_models': len(model_list)
        }
    
    async def download_ncbi_genomes(self, domain: str = "bacteria", max_genomes: Optional[int] = None) -> Dict[str, Any]:
        """Download NCBI genome data"""
        logger.info(f"Starting NCBI {domain} genome download")
        
        progress = self._load_progress()
        
        # Get genome list
        genome_list = await self.ncbi_downloader.fetch_assembly_summary(domain)
        logger.info(f"Found {len(genome_list)} {domain} genomes")
        
        if max_genomes:
            genome_list = genome_list[:max_genomes]
            logger.info(f"Limited to {max_genomes} genomes for testing")
        
        progress['total_genomes'] = len(genome_list)
        
        # Download genome data
        downloaded_count = 0
        failed_count = 0
        
        for i, genome_info in enumerate(genome_list):
            assembly_accession = genome_info['assembly_accession']
            
            # Skip if already downloaded
            if assembly_accession in progress['downloaded_genomes']:
                continue
            
            try:
                logger.info(f"Processing genome {i+1}/{len(genome_list)}: {assembly_accession}")
                
                # Create NCBIGenome object
                genome = NCBIGenome(
                    assembly_accession=assembly_accession,
                    organism_name=genome_info.get('organism_name', ''),
                    strain=genome_info.get('strain', ''),
                    taxid=int(genome_info.get('taxid', 0)),
                    assembly_level=genome_info.get('assembly_level', ''),
                    annotation_provider=genome_info.get('annotation_provider', ''),
                    annotation_date=genome_info.get('annotation_date', ''),
                    ftp_path=genome_info.get('ftp_path', '')
                )
                
                # Download annotation and sequence files (optional, can be large)
                annotation_file = ""
                sequence_file = ""
                
                if genome.ftp_path:
                    # Download annotation (smaller file)
                    annotation_file = await self.ncbi_downloader.download_genome_annotation(
                        genome.ftp_path, assembly_accession
                    )
                    
                    # Optionally download sequence file (commented out for demo)
                    # sequence_file = await self.ncbi_downloader.download_genome_sequence(
                    #     genome.ftp_path, assembly_accession
                    # )
                
                # Store genome metadata
                self.processor.store_ncbi_genome(genome, sequence_file, annotation_file or "")
                progress['downloaded_genomes'].append(assembly_accession)
                downloaded_count += 1
                
                # Save progress every 20 genomes
                if downloaded_count % 20 == 0:
                    self._save_progress(progress)
                    logger.info(f"Progress saved: {downloaded_count} genomes downloaded")
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"Error downloading genome {assembly_accession}: {e}")
        
        # Final progress save
        self._save_progress(progress)
        
        logger.info(f"NCBI {domain} download complete: {downloaded_count} downloaded, {failed_count} failed")
        
        return {
            'downloaded_genomes': downloaded_count,
            'failed_genomes': failed_count,
            'total_genomes': len(genome_list)
        }
    
    def associate_models_genomes(self) -> Dict[str, Any]:
        """Associate AGORA2 models with NCBI genomes based on taxonomy"""
        logger.info("Creating model-genome associations")
        
        with sqlite3.connect(self.processor.db_path) as conn:
            cursor = conn.cursor()
            
            # Get all models and genomes
            cursor.execute('''
                SELECT model_id, organism, strain, species, genus
                FROM agora2_models
            ''')
            models = cursor.fetchall()
            
            cursor.execute('''
                SELECT assembly_accession, organism_name, strain
                FROM ncbi_genomes
            ''')
            genomes = cursor.fetchall()
            
            associations = 0
            
            # Simple matching based on organism names
            for model_id, organism, strain, species, genus in models:
                for assembly_accession, organism_name, genome_strain in genomes:
                    confidence = 0.0
                    method = "none"
                    
                    # Exact organism match
                    if organism.lower() == organism_name.lower():
                        confidence = 0.9
                        method = "exact_organism"
                    # Species match
                    elif species and species.lower() in organism_name.lower():
                        confidence = 0.8
                        method = "species_match"
                    # Genus match
                    elif genus and genus.lower() in organism_name.lower():
                        confidence = 0.6
                        method = "genus_match"
                    
                    # Store association if confidence > 0.5
                    if confidence > 0.5:
                        self.processor.associate_model_genome(
                            model_id, assembly_accession, confidence, method
                        )
                        associations += 1
                        break  # One association per model
            
            logger.info(f"Created {associations} model-genome associations")
            
            return {
                'total_associations': associations,
                'total_models': len(models),
                'total_genomes': len(genomes)
            }
    
    def export_datasets(self) -> Dict[str, Any]:
        """Export all processed datasets"""
        logger.info("Exporting NCBI/AGORA2 datasets")
        
        output_files = self.processor.export_to_csv()
        statistics = self.processor.generate_statistics()
        
        # Save statistics
        stats_file = self.output_path / "processed/agora2/ncbi_agora2_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(statistics, f, indent=2)
        
        output_files['statistics'] = str(stats_file)
        
        return {
            'output_files': output_files,
            'statistics': statistics
        }
    
    async def run_full_integration(self, max_models: Optional[int] = None, max_genomes: Optional[int] = None) -> Dict[str, Any]:
        """Run complete NCBI/AGORA2 integration pipeline"""
        try:
            logger.info("Starting full NCBI/AGORA2 integration")
            
            # Download AGORA2 models
            agora2_results = await self.download_agora2_models(max_models)
            
            # Download NCBI genomes
            ncbi_results = await self.download_ncbi_genomes("bacteria", max_genomes)
            
            # Associate models with genomes
            association_results = self.associate_models_genomes()
            
            # Export datasets
            export_results = self.export_datasets()
            
            # Generate final report
            report = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'agora2_results': agora2_results,
                'ncbi_results': ncbi_results,
                'association_results': association_results,
                'export_results': export_results,
                'status': 'completed'
            }
            
            # Save report
            report_file = self.output_path / "processed/agora2/integration_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            return report
            
        except Exception as e:
            logger.error(f"Error in full integration: {e}")
            raise
        finally:
            await self.agora2_downloader.close()
            await self.ncbi_downloader.close()

# Main execution
async def main():
    """Main execution function"""
    integration = NCBIAgoraIntegration()
    
    # Run full integration with sample data for demonstration
    # Use max_models=50, max_genomes=100 for testing
    report = await integration.run_full_integration(max_models=50, max_genomes=100)
    
    logger.info("NCBI/AGORA2 integration completed successfully")
    logger.info(f"Report: {report}")
    
    return report

if __name__ == "__main__":
    # Run the integration
    report = asyncio.run(main())
    print(f"Integration completed: {report}") 
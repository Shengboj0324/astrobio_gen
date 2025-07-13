#!/usr/bin/env python3
"""
Comprehensive Metadata and Annotation System
============================================

Advanced metadata management system for astrobiology genomics research:
- Rich metadata capture and storage
- Semantic annotations with ontologies
- Standardized documentation
- Cross-reference mapping
- Provenance tracking
- FAIR data principles implementation
- Automated metadata extraction
- Quality annotations

Supports all data types: KEGG, NCBI, AGORA2, genomic, metabolic, and environmental data.
"""

import os
import json
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Set, Union
import logging
import sqlite3
import hashlib
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import re
from urllib.parse import urlparse
import requests
from collections import defaultdict
import xml.etree.ElementTree as ET
from functools import lru_cache
import threading
from threading import Lock
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetadataType(Enum):
    """Types of metadata"""
    DESCRIPTIVE = "descriptive"
    STRUCTURAL = "structural"
    ADMINISTRATIVE = "administrative"
    TECHNICAL = "technical"
    PRESERVATION = "preservation"
    PROVENANCE = "provenance"
    QUALITY = "quality"
    SEMANTIC = "semantic"

class AnnotationType(Enum):
    """Types of annotations"""
    ONTOLOGY = "ontology"
    TAXONOMY = "taxonomy"
    FUNCTIONAL = "functional"
    QUALITY = "quality"
    RELATIONSHIP = "relationship"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    STATISTICAL = "statistical"

class DataStandard(Enum):
    """Data standards and formats"""
    DUBLIN_CORE = "dublin_core"
    DATACITE = "datacite"
    DCAT = "dcat"
    BIOSCHEMAS = "bioschemas"
    FAIR = "fair"
    MIAME = "miame"
    MINSEQE = "minseqe"
    MIGS = "migs"
    KEGG = "kegg"
    NCBI = "ncbi"
    AGORA = "agora"

@dataclass
class Annotation:
    """Semantic annotation structure"""
    annotation_id: str
    annotation_type: AnnotationType
    value: str
    ontology: str = ""
    ontology_id: str = ""
    ontology_version: str = ""
    confidence: float = 1.0
    source: str = ""
    created_by: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CrossReference:
    """Cross-reference to external databases"""
    xref_id: str
    database: str
    identifier: str
    url: str = ""
    relationship: str = "exact_match"
    confidence: float = 1.0
    verified: bool = False
    verified_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Provenance:
    """Data provenance information"""
    provenance_id: str
    source: str
    creation_method: str
    created_by: str
    created_at: datetime
    version: str = "1.0"
    parent_ids: List[str] = field(default_factory=list)
    transformations: List[str] = field(default_factory=list)
    quality_checks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QualityAnnotation:
    """Quality-related annotations"""
    quality_id: str
    metric: str
    value: float
    threshold: float
    status: str  # 'pass', 'fail', 'warning'
    description: str
    assessed_by: str
    assessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MetadataRecord:
    """Comprehensive metadata record"""
    record_id: str
    data_source: str
    data_type: str
    title: str
    description: str
    keywords: List[str] = field(default_factory=list)
    creators: List[str] = field(default_factory=list)
    contributors: List[str] = field(default_factory=list)
    publisher: str = ""
    publication_date: Optional[datetime] = None
    language: str = "en"
    format: str = ""
    size: int = 0
    checksum: str = ""
    license: str = ""
    access_rights: str = ""
    
    # Technical metadata
    schema_version: str = "1.0"
    encoding: str = "utf-8"
    mime_type: str = ""
    
    # Semantic metadata
    annotations: List[Annotation] = field(default_factory=list)
    cross_references: List[CrossReference] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    
    # Quality metadata
    quality_annotations: List[QualityAnnotation] = field(default_factory=list)
    quality_score: float = 0.0
    
    # Provenance
    provenance: Optional[Provenance] = None
    
    # Temporal metadata
    temporal_coverage: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    modified_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Custom metadata
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

class OntologyManager:
    """Ontology and vocabulary management"""
    
    def __init__(self, cache_dir: str = "data/metadata/ontologies"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ontologies = {}
        self.lock = Lock()
        self._load_ontologies()
    
    def _load_ontologies(self):
        """Load ontology mappings"""
        self.ontologies = {
            'GO': {
                'name': 'Gene Ontology',
                'url': 'http://geneontology.org/',
                'api_url': 'https://api.geneontology.org/',
                'pattern': r'^GO:\d{7}$',
                'namespaces': ['biological_process', 'molecular_function', 'cellular_component']
            },
            'KEGG': {
                'name': 'Kyoto Encyclopedia of Genes and Genomes',
                'url': 'https://www.genome.jp/kegg/',
                'api_url': 'https://rest.kegg.jp/',
                'pattern': r'^(map\d{5}|R\d{5}|C\d{5}|ec:\d+\.\d+\.\d+\.\d+)$',
                'namespaces': ['pathway', 'reaction', 'compound', 'enzyme']
            },
            'CHEBI': {
                'name': 'Chemical Entities of Biological Interest',
                'url': 'https://www.ebi.ac.uk/chebi/',
                'api_url': 'https://www.ebi.ac.uk/webservices/chebi/',
                'pattern': r'^CHEBI:\d+$',
                'namespaces': ['chemical']
            },
            'NCBI_TAXONOMY': {
                'name': 'NCBI Taxonomy',
                'url': 'https://www.ncbi.nlm.nih.gov/taxonomy/',
                'api_url': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/',
                'pattern': r'^\d+$',
                'namespaces': ['taxonomy']
            },
            'UNIPROT': {
                'name': 'Universal Protein Resource',
                'url': 'https://www.uniprot.org/',
                'api_url': 'https://rest.uniprot.org/',
                'pattern': r'^[A-Z0-9]{6,10}$',
                'namespaces': ['protein']
            },
            'BIGG': {
                'name': 'BiGG Models',
                'url': 'http://bigg.ucsd.edu/',
                'api_url': 'http://bigg.ucsd.edu/api/v2/',
                'pattern': r'^[a-zA-Z0-9_]+$',
                'namespaces': ['reaction', 'metabolite', 'model']
            },
            'METACYC': {
                'name': 'MetaCyc',
                'url': 'https://metacyc.org/',
                'api_url': 'https://websvc.biocyc.org/apikey',
                'pattern': r'^[A-Z0-9\-]+$',
                'namespaces': ['pathway', 'reaction', 'compound']
            }
        }
    
    @lru_cache(maxsize=1000)
    def resolve_ontology_term(self, term: str, ontology: str) -> Dict[str, Any]:
        """Resolve ontology term to full metadata"""
        if ontology not in self.ontologies:
            return {}
        
        ont_config = self.ontologies[ontology]
        
        # Check pattern match
        if not re.match(ont_config['pattern'], term):
            return {}
        
        # Try to fetch from cache first
        cache_file = self.cache_dir / f"{ontology}_{term.replace(':', '_')}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        # Fetch from API
        try:
            term_data = self._fetch_ontology_term(term, ontology)
            if term_data:
                # Cache the result
                with open(cache_file, 'w') as f:
                    json.dump(term_data, f, indent=2)
                return term_data
        except Exception as e:
            logger.error(f"Error resolving ontology term {term} in {ontology}: {e}")
        
        return {}
    
    def _fetch_ontology_term(self, term: str, ontology: str) -> Dict[str, Any]:
        """Fetch ontology term from API"""
        ont_config = self.ontologies[ontology]
        
        try:
            if ontology == 'GO':
                # Gene Ontology API
                url = f"{ont_config['api_url']}ontology/term/{term}"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    return {
                        'term': term,
                        'name': data.get('label', ''),
                        'definition': data.get('definition', ''),
                        'synonyms': data.get('synonyms', []),
                        'namespace': data.get('namespace', ''),
                        'obsolete': data.get('obsolete', False)
                    }
            
            elif ontology == 'KEGG':
                # KEGG API
                url = f"{ont_config['api_url']}get/{term}"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    content = response.text
                    return self._parse_kegg_entry(content, term)
            
            elif ontology == 'CHEBI':
                # ChEBI API
                url = f"{ont_config['api_url']}getCompleteEntity"
                params = {'chebiId': term}
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    # Parse XML response
                    root = ET.fromstring(response.content)
                    return self._parse_chebi_entry(root, term)
            
            elif ontology == 'NCBI_TAXONOMY':
                # NCBI Taxonomy API
                url = f"{ont_config['api_url']}efetch.fcgi"
                params = {
                    'db': 'taxonomy',
                    'id': term,
                    'retmode': 'xml'
                }
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    root = ET.fromstring(response.content)
                    return self._parse_ncbi_taxonomy(root, term)
            
        except Exception as e:
            logger.error(f"Error fetching {ontology} term {term}: {e}")
        
        return {}
    
    def _parse_kegg_entry(self, content: str, term: str) -> Dict[str, Any]:
        """Parse KEGG entry format"""
        data = {'term': term}
        
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('NAME'):
                data['name'] = line[4:].strip()
            elif line.startswith('DEFINITION'):
                data['definition'] = line[10:].strip()
            elif line.startswith('CLASS'):
                data['class'] = line[5:].strip()
            elif line.startswith('PATHWAY'):
                data['pathway'] = line[7:].strip()
            elif line.startswith('ENZYME'):
                data['enzyme'] = line[6:].strip()
            elif line.startswith('REACTION'):
                data['reaction'] = line[8:].strip()
        
        return data
    
    def _parse_chebi_entry(self, root: ET.Element, term: str) -> Dict[str, Any]:
        """Parse ChEBI XML entry"""
        data = {'term': term}
        
        # Extract basic information
        for elem in root.iter():
            if elem.tag.endswith('chebiAsciiName'):
                data['name'] = elem.text
            elif elem.tag.endswith('definition'):
                data['definition'] = elem.text
            elif elem.tag.endswith('chebiId'):
                data['id'] = elem.text
            elif elem.tag.endswith('synonyms'):
                data['synonyms'] = [syn.text for syn in elem.iter() if syn.text]
        
        return data
    
    def _parse_ncbi_taxonomy(self, root: ET.Element, term: str) -> Dict[str, Any]:
        """Parse NCBI Taxonomy XML"""
        data = {'term': term}
        
        for taxon in root.iter('Taxon'):
            data['name'] = taxon.find('ScientificName').text if taxon.find('ScientificName') is not None else ''
            data['rank'] = taxon.find('Rank').text if taxon.find('Rank') is not None else ''
            data['division'] = taxon.find('Division').text if taxon.find('Division') is not None else ''
            
            # Extract lineage
            lineage = []
            for lineage_elem in taxon.iter('LineageEx'):
                for taxon_elem in lineage_elem.iter('Taxon'):
                    sci_name = taxon_elem.find('ScientificName')
                    rank = taxon_elem.find('Rank')
                    if sci_name is not None and rank is not None:
                        lineage.append(f"{rank.text}: {sci_name.text}")
            data['lineage'] = lineage
        
        return data
    
    def suggest_annotations(self, text: str, data_type: str = None) -> List[Annotation]:
        """Suggest annotations based on text analysis"""
        suggestions = []
        
        # Simple pattern matching for common identifiers
        patterns = {
            'GO': (r'\bGO:\d{7}\b', 'GO'),
            'KEGG_PATHWAY': (r'\bmap\d{5}\b', 'KEGG'),
            'KEGG_REACTION': (r'\bR\d{5}\b', 'KEGG'),
            'KEGG_COMPOUND': (r'\bC\d{5}\b', 'KEGG'),
            'CHEBI': (r'\bCHEBI:\d+\b', 'CHEBI'),
            'NCBI_TAXON': (r'\btaxid:\d+\b', 'NCBI_TAXONOMY'),
            'UNIPROT': (r'\b[A-Z0-9]{6,10}\b', 'UNIPROT')
        }
        
        for pattern_name, (pattern, ontology) in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                term_data = self.resolve_ontology_term(match, ontology)
                if term_data:
                    suggestions.append(Annotation(
                        annotation_id=str(uuid.uuid4()),
                        annotation_type=AnnotationType.ONTOLOGY,
                        value=term_data.get('name', match),
                        ontology=ontology,
                        ontology_id=match,
                        confidence=0.8,
                        source='auto_suggestion'
                    ))
        
        return suggestions

class MetadataExtractor:
    """Extract metadata from various data sources"""
    
    def __init__(self, ontology_manager: OntologyManager):
        self.ontology_manager = ontology_manager
    
    def extract_from_dataframe(self, df: pd.DataFrame, data_source: str, data_type: str) -> MetadataRecord:
        """Extract metadata from pandas DataFrame"""
        record_id = f"{data_source}_{data_type}_{int(datetime.now().timestamp())}"
        
        # Basic metadata
        metadata = MetadataRecord(
            record_id=record_id,
            data_source=data_source,
            data_type=data_type,
            title=f"{data_source} {data_type} dataset",
            description=f"Dataset containing {len(df)} records with {len(df.columns)} columns",
            format="pandas_dataframe",
            size=df.memory_usage(deep=True).sum(),
            checksum=self._calculate_dataframe_checksum(df)
        )
        
        # Technical metadata
        metadata.custom_metadata.update({
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'unique_counts': df.nunique().to_dict()
        })
        
        # Extract annotations from column names and data
        annotations = []
        for column in df.columns:
            column_annotations = self.ontology_manager.suggest_annotations(column, data_type)
            annotations.extend(column_annotations)
        
        # Extract annotations from data values (sample)
        if len(df) > 0:
            sample_data = df.head(10).astype(str)
            for column in sample_data.columns:
                for value in sample_data[column]:
                    value_annotations = self.ontology_manager.suggest_annotations(str(value), data_type)
                    annotations.extend(value_annotations[:3])  # Limit to prevent too many
        
        metadata.annotations = annotations
        
        return metadata
    
    def extract_from_file(self, file_path: Path, data_source: str, data_type: str) -> MetadataRecord:
        """Extract metadata from file"""
        record_id = f"{data_source}_{data_type}_{file_path.stem}"
        
        # Basic file metadata
        stat = file_path.stat()
        metadata = MetadataRecord(
            record_id=record_id,
            data_source=data_source,
            data_type=data_type,
            title=file_path.name,
            description=f"File: {file_path.name}",
            format=file_path.suffix.lstrip('.'),
            size=stat.st_size,
            checksum=self._calculate_file_checksum(file_path),
            created_at=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        )
        
        # Technical metadata
        metadata.custom_metadata.update({
            'file_path': str(file_path),
            'file_size_bytes': stat.st_size,
            'file_extension': file_path.suffix,
            'access_time': datetime.fromtimestamp(stat.st_atime, tz=timezone.utc).isoformat(),
            'mime_type': self._guess_mime_type(file_path)
        })
        
        # Extract content-based metadata
        if file_path.suffix.lower() in ['.csv', '.tsv', '.txt']:
            try:
                # Try to read as CSV for additional metadata
                df = pd.read_csv(file_path, nrows=100)  # Sample only
                content_metadata = self.extract_from_dataframe(df, data_source, data_type)
                metadata.annotations.extend(content_metadata.annotations)
                metadata.custom_metadata.update(content_metadata.custom_metadata)
            except Exception as e:
                logger.warning(f"Could not extract content metadata from {file_path}: {e}")
        
        return metadata
    
    def extract_from_kegg_pathway(self, pathway_data: Dict[str, Any]) -> MetadataRecord:
        """Extract metadata from KEGG pathway data"""
        pathway_id = pathway_data.get('pathway_id', '')
        record_id = f"kegg_pathway_{pathway_id}"
        
        metadata = MetadataRecord(
            record_id=record_id,
            data_source='kegg',
            data_type='pathway',
            title=pathway_data.get('name', ''),
            description=pathway_data.get('description', ''),
            keywords=[pathway_id, 'pathway', 'metabolic', 'kegg'],
            creators=['KEGG Database'],
            publisher='Kyoto Encyclopedia of Genes and Genomes',
            format='kegg_pathway',
        )
        
        # Add KEGG-specific annotations
        annotations = [
            Annotation(
                annotation_id=str(uuid.uuid4()),
                annotation_type=AnnotationType.ONTOLOGY,
                value=pathway_data.get('name', ''),
                ontology='KEGG',
                ontology_id=pathway_id,
                confidence=1.0,
                source='kegg_database'
            )
        ]
        
        # Add cross-references
        cross_refs = [
            CrossReference(
                xref_id=str(uuid.uuid4()),
                database='KEGG',
                identifier=pathway_id,
                url=f"https://www.genome.jp/kegg-bin/show_pathway?{pathway_id}",
                relationship='exact_match',
                confidence=1.0,
                verified=True
            )
        ]
        
        metadata.annotations = annotations
        metadata.cross_references = cross_refs
        
        return metadata
    
    def extract_from_ncbi_genome(self, genome_data: Dict[str, Any]) -> MetadataRecord:
        """Extract metadata from NCBI genome data"""
        accession = genome_data.get('assembly_accession', '')
        record_id = f"ncbi_genome_{accession}"
        
        metadata = MetadataRecord(
            record_id=record_id,
            data_source='ncbi',
            data_type='genome',
            title=f"Genome assembly {accession}",
            description=f"Genome assembly for {genome_data.get('organism_name', '')}",
            keywords=['genome', 'assembly', 'ncbi', genome_data.get('organism_name', '')],
            creators=['NCBI'],
            publisher='National Center for Biotechnology Information',
            format='ncbi_genome',
        )
        
        # Add taxonomic annotations
        if 'taxid' in genome_data:
            taxon_data = self.ontology_manager.resolve_ontology_term(
                str(genome_data['taxid']), 'NCBI_TAXONOMY'
            )
            if taxon_data:
                annotations = [
                    Annotation(
                        annotation_id=str(uuid.uuid4()),
                        annotation_type=AnnotationType.TAXONOMY,
                        value=taxon_data.get('name', ''),
                        ontology='NCBI_TAXONOMY',
                        ontology_id=str(genome_data['taxid']),
                        confidence=1.0,
                        source='ncbi_taxonomy'
                    )
                ]
                metadata.annotations = annotations
        
        # Add cross-references
        cross_refs = [
            CrossReference(
                xref_id=str(uuid.uuid4()),
                database='NCBI',
                identifier=accession,
                url=f"https://www.ncbi.nlm.nih.gov/assembly/{accession}",
                relationship='exact_match',
                confidence=1.0,
                verified=True
            )
        ]
        
        metadata.cross_references = cross_refs
        
        return metadata
    
    def extract_from_agora2_model(self, model_data: Dict[str, Any]) -> MetadataRecord:
        """Extract metadata from AGORA2 model data"""
        model_id = model_data.get('model_id', '')
        record_id = f"agora2_model_{model_id}"
        
        metadata = MetadataRecord(
            record_id=record_id,
            data_source='agora2',
            data_type='metabolic_model',
            title=f"AGORA2 model {model_id}",
            description=f"Metabolic model for {model_data.get('organism', '')}",
            keywords=['metabolic_model', 'agora2', 'constraint_based', model_data.get('organism', '')],
            creators=['AGORA2 Consortium'],
            publisher='Virtual Metabolic Human',
            format='sbml',
        )
        
        # Add taxonomic annotations
        if 'taxonomy' in model_data:
            taxonomy_parts = model_data['taxonomy'].split(';')
            for part in taxonomy_parts:
                if ':' in part:
                    rank, name = part.split(':', 1)
                    annotations = [
                        Annotation(
                            annotation_id=str(uuid.uuid4()),
                            annotation_type=AnnotationType.TAXONOMY,
                            value=name.strip(),
                            ontology='NCBI_TAXONOMY',
                            confidence=0.9,
                            source='agora2_taxonomy'
                        )
                    ]
                    metadata.annotations.extend(annotations)
        
        # Add model-specific metadata
        metadata.custom_metadata.update({
            'reactions': model_data.get('reactions', 0),
            'metabolites': model_data.get('metabolites', 0),
            'genes': model_data.get('genes', 0),
            'biomass_reaction': model_data.get('biomass_reaction', ''),
            'growth_medium': model_data.get('growth_medium', ''),
            'domain': model_data.get('domain', ''),
            'phylum': model_data.get('phylum', ''),
            'class': model_data.get('class', ''),
            'order': model_data.get('order', ''),
            'family': model_data.get('family', ''),
            'genus': model_data.get('genus', ''),
            'species': model_data.get('species', '')
        })
        
        return metadata
    
    def _calculate_dataframe_checksum(self, df: pd.DataFrame) -> str:
        """Calculate checksum for DataFrame"""
        content = df.to_json(sort_keys=True, orient='records')
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate file checksum"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _guess_mime_type(self, file_path: Path) -> str:
        """Guess MIME type from file extension"""
        extension_map = {
            '.csv': 'text/csv',
            '.tsv': 'text/tab-separated-values',
            '.txt': 'text/plain',
            '.json': 'application/json',
            '.xml': 'application/xml',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.zip': 'application/zip',
            '.gz': 'application/gzip',
            '.fasta': 'application/x-fasta',
            '.fastq': 'application/x-fastq',
            '.sam': 'application/x-sam',
            '.bam': 'application/x-bam',
            '.vcf': 'application/x-vcf',
            '.gff': 'application/x-gff',
            '.gtf': 'application/x-gtf',
            '.bed': 'application/x-bed'
        }
        
        return extension_map.get(file_path.suffix.lower(), 'application/octet-stream')

class MetadataManager:
    """Comprehensive metadata management system"""
    
    def __init__(self, db_path: str = "data/metadata/metadata.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ontology_manager = OntologyManager()
        self.extractor = MetadataExtractor(self.ontology_manager)
        self.lock = Lock()
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize metadata database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Metadata records table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata_records (
                    record_id TEXT PRIMARY KEY,
                    data_source TEXT,
                    data_type TEXT,
                    title TEXT,
                    description TEXT,
                    keywords TEXT,
                    creators TEXT,
                    contributors TEXT,
                    publisher TEXT,
                    publication_date TIMESTAMP,
                    language TEXT,
                    format TEXT,
                    size INTEGER,
                    checksum TEXT,
                    license TEXT,
                    access_rights TEXT,
                    schema_version TEXT,
                    encoding TEXT,
                    mime_type TEXT,
                    quality_score REAL,
                    temporal_coverage TEXT,
                    created_at TIMESTAMP,
                    modified_at TIMESTAMP,
                    custom_metadata TEXT
                )
            ''')
            
            # Annotations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS annotations (
                    annotation_id TEXT PRIMARY KEY,
                    record_id TEXT,
                    annotation_type TEXT,
                    value TEXT,
                    ontology TEXT,
                    ontology_id TEXT,
                    ontology_version TEXT,
                    confidence REAL,
                    source TEXT,
                    created_by TEXT,
                    created_at TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (record_id) REFERENCES metadata_records(record_id)
                )
            ''')
            
            # Cross-references table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cross_references (
                    xref_id TEXT PRIMARY KEY,
                    record_id TEXT,
                    database TEXT,
                    identifier TEXT,
                    url TEXT,
                    relationship TEXT,
                    confidence REAL,
                    verified BOOLEAN,
                    verified_at TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (record_id) REFERENCES metadata_records(record_id)
                )
            ''')
            
            # Provenance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS provenance (
                    provenance_id TEXT PRIMARY KEY,
                    record_id TEXT,
                    source TEXT,
                    creation_method TEXT,
                    created_by TEXT,
                    created_at TIMESTAMP,
                    version TEXT,
                    parent_ids TEXT,
                    transformations TEXT,
                    quality_checks TEXT,
                    metadata TEXT,
                    FOREIGN KEY (record_id) REFERENCES metadata_records(record_id)
                )
            ''')
            
            # Quality annotations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS quality_annotations (
                    quality_id TEXT PRIMARY KEY,
                    record_id TEXT,
                    metric TEXT,
                    value REAL,
                    threshold REAL,
                    status TEXT,
                    description TEXT,
                    assessed_by TEXT,
                    assessed_at TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (record_id) REFERENCES metadata_records(record_id)
                )
            ''')
            
            # Relationships table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS relationships (
                    relationship_id TEXT PRIMARY KEY,
                    source_record_id TEXT,
                    target_record_id TEXT,
                    relationship_type TEXT,
                    confidence REAL,
                    created_at TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (source_record_id) REFERENCES metadata_records(record_id),
                    FOREIGN KEY (target_record_id) REFERENCES metadata_records(record_id)
                )
            ''')
            
            conn.commit()
    
    def store_metadata(self, metadata: MetadataRecord):
        """Store metadata record in database"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Store main metadata record
                cursor.execute('''
                    INSERT OR REPLACE INTO metadata_records 
                    (record_id, data_source, data_type, title, description, keywords, creators,
                     contributors, publisher, publication_date, language, format, size, checksum,
                     license, access_rights, schema_version, encoding, mime_type, quality_score,
                     temporal_coverage, created_at, modified_at, custom_metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metadata.record_id,
                    metadata.data_source,
                    metadata.data_type,
                    metadata.title,
                    metadata.description,
                    json.dumps(metadata.keywords),
                    json.dumps(metadata.creators),
                    json.dumps(metadata.contributors),
                    metadata.publisher,
                    metadata.publication_date,
                    metadata.language,
                    metadata.format,
                    metadata.size,
                    metadata.checksum,
                    metadata.license,
                    metadata.access_rights,
                    metadata.schema_version,
                    metadata.encoding,
                    metadata.mime_type,
                    metadata.quality_score,
                    json.dumps(metadata.temporal_coverage),
                    metadata.created_at,
                    metadata.modified_at,
                    json.dumps(metadata.custom_metadata)
                ))
                
                # Store annotations
                for annotation in metadata.annotations:
                    cursor.execute('''
                        INSERT OR REPLACE INTO annotations 
                        (annotation_id, record_id, annotation_type, value, ontology, ontology_id,
                         ontology_version, confidence, source, created_by, created_at, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        annotation.annotation_id,
                        metadata.record_id,
                        annotation.annotation_type.value,
                        annotation.value,
                        annotation.ontology,
                        annotation.ontology_id,
                        annotation.ontology_version,
                        annotation.confidence,
                        annotation.source,
                        annotation.created_by,
                        annotation.created_at,
                        json.dumps(annotation.metadata)
                    ))
                
                # Store cross-references
                for xref in metadata.cross_references:
                    cursor.execute('''
                        INSERT OR REPLACE INTO cross_references 
                        (xref_id, record_id, database, identifier, url, relationship, confidence,
                         verified, verified_at, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        xref.xref_id,
                        metadata.record_id,
                        xref.database,
                        xref.identifier,
                        xref.url,
                        xref.relationship,
                        xref.confidence,
                        xref.verified,
                        xref.verified_at,
                        json.dumps(xref.metadata)
                    ))
                
                # Store provenance
                if metadata.provenance:
                    cursor.execute('''
                        INSERT OR REPLACE INTO provenance 
                        (provenance_id, record_id, source, creation_method, created_by, created_at,
                         version, parent_ids, transformations, quality_checks, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        metadata.provenance.provenance_id,
                        metadata.record_id,
                        metadata.provenance.source,
                        metadata.provenance.creation_method,
                        metadata.provenance.created_by,
                        metadata.provenance.created_at,
                        metadata.provenance.version,
                        json.dumps(metadata.provenance.parent_ids),
                        json.dumps(metadata.provenance.transformations),
                        json.dumps(metadata.provenance.quality_checks),
                        json.dumps(metadata.provenance.metadata)
                    ))
                
                # Store quality annotations
                for quality_annotation in metadata.quality_annotations:
                    cursor.execute('''
                        INSERT OR REPLACE INTO quality_annotations 
                        (quality_id, record_id, metric, value, threshold, status, description,
                         assessed_by, assessed_at, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        quality_annotation.quality_id,
                        metadata.record_id,
                        quality_annotation.metric,
                        quality_annotation.value,
                        quality_annotation.threshold,
                        quality_annotation.status,
                        quality_annotation.description,
                        quality_annotation.assessed_by,
                        quality_annotation.assessed_at,
                        json.dumps(quality_annotation.metadata)
                    ))
                
                conn.commit()
    
    def get_metadata(self, record_id: str) -> Optional[MetadataRecord]:
        """Retrieve metadata record by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get main record
            cursor.execute('SELECT * FROM metadata_records WHERE record_id = ?', (record_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Parse main record
            metadata = MetadataRecord(
                record_id=row[0],
                data_source=row[1],
                data_type=row[2],
                title=row[3],
                description=row[4],
                keywords=json.loads(row[5]) if row[5] else [],
                creators=json.loads(row[6]) if row[6] else [],
                contributors=json.loads(row[7]) if row[7] else [],
                publisher=row[8] or "",
                publication_date=row[9],
                language=row[10] or "en",
                format=row[11] or "",
                size=row[12] or 0,
                checksum=row[13] or "",
                license=row[14] or "",
                access_rights=row[15] or "",
                schema_version=row[16] or "1.0",
                encoding=row[17] or "utf-8",
                mime_type=row[18] or "",
                quality_score=row[19] or 0.0,
                temporal_coverage=json.loads(row[20]) if row[20] else {},
                created_at=row[21],
                modified_at=row[22],
                custom_metadata=json.loads(row[23]) if row[23] else {}
            )
            
            # Get annotations
            cursor.execute('SELECT * FROM annotations WHERE record_id = ?', (record_id,))
            for ann_row in cursor.fetchall():
                annotation = Annotation(
                    annotation_id=ann_row[0],
                    annotation_type=AnnotationType(ann_row[2]),
                    value=ann_row[3],
                    ontology=ann_row[4] or "",
                    ontology_id=ann_row[5] or "",
                    ontology_version=ann_row[6] or "",
                    confidence=ann_row[7] or 1.0,
                    source=ann_row[8] or "",
                    created_by=ann_row[9] or "",
                    created_at=ann_row[10],
                    metadata=json.loads(ann_row[11]) if ann_row[11] else {}
                )
                metadata.annotations.append(annotation)
            
            # Get cross-references
            cursor.execute('SELECT * FROM cross_references WHERE record_id = ?', (record_id,))
            for xref_row in cursor.fetchall():
                xref = CrossReference(
                    xref_id=xref_row[0],
                    database=xref_row[2],
                    identifier=xref_row[3],
                    url=xref_row[4] or "",
                    relationship=xref_row[5] or "exact_match",
                    confidence=xref_row[6] or 1.0,
                    verified=xref_row[7] or False,
                    verified_at=xref_row[8],
                    metadata=json.loads(xref_row[9]) if xref_row[9] else {}
                )
                metadata.cross_references.append(xref)
            
            # Get quality annotations
            cursor.execute('SELECT * FROM quality_annotations WHERE record_id = ?', (record_id,))
            for qa_row in cursor.fetchall():
                quality_annotation = QualityAnnotation(
                    quality_id=qa_row[0],
                    metric=qa_row[2],
                    value=qa_row[3],
                    threshold=qa_row[4],
                    status=qa_row[5],
                    description=qa_row[6],
                    assessed_by=qa_row[7],
                    assessed_at=qa_row[8],
                    metadata=json.loads(qa_row[9]) if qa_row[9] else {}
                )
                metadata.quality_annotations.append(quality_annotation)
            
            return metadata
    
    def search_metadata(self, query: str, data_type: str = None, data_source: str = None) -> List[MetadataRecord]:
        """Search metadata records"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Build search query
            sql = '''
                SELECT DISTINCT record_id FROM metadata_records 
                WHERE (title LIKE ? OR description LIKE ? OR keywords LIKE ?)
            '''
            params = [f'%{query}%', f'%{query}%', f'%{query}%']
            
            if data_type:
                sql += ' AND data_type = ?'
                params.append(data_type)
            
            if data_source:
                sql += ' AND data_source = ?'
                params.append(data_source)
            
            cursor.execute(sql, params)
            
            results = []
            for row in cursor.fetchall():
                metadata = self.get_metadata(row[0])
                if metadata:
                    results.append(metadata)
            
            return results
    
    def export_metadata(self, output_format: str = "json", output_path: str = None) -> str:
        """Export all metadata to various formats"""
        if output_path is None:
            output_path = f"data/metadata/metadata_export_{int(datetime.now().timestamp())}.{output_format}"
        
        with sqlite3.connect(self.db_path) as conn:
            if output_format == "json":
                # Export to JSON
                cursor = conn.cursor()
                cursor.execute('SELECT record_id FROM metadata_records')
                
                all_metadata = []
                for row in cursor.fetchall():
                    metadata = self.get_metadata(row[0])
                    if metadata:
                        all_metadata.append(asdict(metadata))
                
                with open(output_path, 'w') as f:
                    json.dump(all_metadata, f, indent=2, default=str)
            
            elif output_format == "csv":
                # Export to CSV
                df = pd.read_sql_query('SELECT * FROM metadata_records', conn)
                df.to_csv(output_path, index=False)
            
            elif output_format == "yaml":
                # Export to YAML
                cursor = conn.cursor()
                cursor.execute('SELECT record_id FROM metadata_records')
                
                all_metadata = []
                for row in cursor.fetchall():
                    metadata = self.get_metadata(row[0])
                    if metadata:
                        all_metadata.append(asdict(metadata))
                
                with open(output_path, 'w') as f:
                    yaml.dump(all_metadata, f, default_flow_style=False)
        
        return output_path
    
    def generate_metadata_report(self) -> Dict[str, Any]:
        """Generate comprehensive metadata report"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            report = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'summary': {},
                'data_sources': {},
                'data_types': {},
                'annotations': {},
                'quality': {}
            }
            
            # Summary statistics
            cursor.execute('SELECT COUNT(*) FROM metadata_records')
            report['summary']['total_records'] = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM annotations')
            report['summary']['total_annotations'] = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM cross_references')
            report['summary']['total_cross_references'] = cursor.fetchone()[0]
            
            # Data sources
            cursor.execute('''
                SELECT data_source, COUNT(*) as count
                FROM metadata_records
                GROUP BY data_source
                ORDER BY count DESC
            ''')
            report['data_sources'] = dict(cursor.fetchall())
            
            # Data types
            cursor.execute('''
                SELECT data_type, COUNT(*) as count
                FROM metadata_records
                GROUP BY data_type
                ORDER BY count DESC
            ''')
            report['data_types'] = dict(cursor.fetchall())
            
            # Annotation statistics
            cursor.execute('''
                SELECT ontology, COUNT(*) as count
                FROM annotations
                GROUP BY ontology
                ORDER BY count DESC
            ''')
            report['annotations']['by_ontology'] = dict(cursor.fetchall())
            
            cursor.execute('''
                SELECT annotation_type, COUNT(*) as count
                FROM annotations
                GROUP BY annotation_type
                ORDER BY count DESC
            ''')
            report['annotations']['by_type'] = dict(cursor.fetchall())
            
            # Quality statistics
            cursor.execute('''
                SELECT AVG(quality_score) as avg_score,
                       MIN(quality_score) as min_score,
                       MAX(quality_score) as max_score
                FROM metadata_records
                WHERE quality_score > 0
            ''')
            quality_stats = cursor.fetchone()
            if quality_stats[0] is not None:
                report['quality'] = {
                    'average_score': quality_stats[0],
                    'minimum_score': quality_stats[1],
                    'maximum_score': quality_stats[2]
                }
            
            return report

# Main execution function
def main():
    """Main execution function for metadata system"""
    # Initialize metadata manager
    metadata_manager = MetadataManager()
    
    # Example usage
    sample_data = pd.DataFrame({
        'pathway_id': ['map00010', 'map00020', 'map00030'],
        'name': ['Glycolysis', 'TCA Cycle', 'Pentose Phosphate'],
        'reaction_count': [10, 8, 7],
        'compound_count': [12, 10, 9]
    })
    
    # Extract and store metadata
    metadata = metadata_manager.extractor.extract_from_dataframe(
        sample_data, 'kegg', 'pathway'
    )
    
    metadata_manager.store_metadata(metadata)
    
    # Generate report
    report = metadata_manager.generate_metadata_report()
    print(f"Metadata Report:")
    print(f"Total Records: {report['summary']['total_records']}")
    print(f"Total Annotations: {report['summary']['total_annotations']}")
    print(f"Data Sources: {report['data_sources']}")
    
    # Export metadata
    export_path = metadata_manager.export_metadata("json")
    print(f"Metadata exported to: {export_path}")
    
    return metadata_manager

if __name__ == "__main__":
    metadata_manager = main() 
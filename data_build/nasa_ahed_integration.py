#!/usr/bin/env python3
"""
NASA AHED (Astrobiology Habitable Environments Database) Integration System
===========================================================================

Advanced system for downloading and integrating NASA's official astrobiology datasets:
- Environmental datasets for habitability studies
- Biosignature detection data
- Field site characterization data
- Instrument calibration datasets
- Multi-mission astrobiology data
- Environmental extremophile studies

Based on ahed.nasa.gov and NASA's astrobiology research programs.

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
import gzip
import hashlib
import time
import zipfile
from dataclasses import dataclass, field
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import ftplib
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AHEDDataset:
    """AHED dataset data structure"""
    dataset_id: str
    title: str
    description: str
    # Classification
    theme: str = ""  # Abiotic Building Blocks, Habitable Worlds, etc.
    keywords: List[str] = field(default_factory=list)
    # Authorship and provenance
    authors: List[str] = field(default_factory=list)
    institution: str = ""
    contact_email: str = ""
    # Temporal information
    collection_start_date: str = ""
    collection_end_date: str = ""
    publication_date: str = ""
    last_updated: str = ""
    # Geographic information
    location: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    elevation: float = 0.0
    site_type: str = ""  # field site, laboratory, etc.
    # Environmental context
    environment_type: str = ""
    ecosystem: str = ""
    temperature_range: str = ""
    ph_range: str = ""
    salinity: str = ""
    radiation_environment: str = ""
    # Data characteristics
    data_types: List[str] = field(default_factory=list)
    file_formats: List[str] = field(default_factory=list)
    data_size: int = 0
    # Instrument information
    instruments: List[str] = field(default_factory=list)
    missions: List[str] = field(default_factory=list)
    # Quality and standards
    quality_level: str = ""
    processing_level: str = ""
    calibration_status: str = ""
    validation_status: str = ""
    # Access information
    access_level: str = ""  # public, restricted, etc.
    download_url: str = ""
    doi: str = ""
    # File information
    files: List[Dict[str, Any]] = field(default_factory=list)
    metadata_file: str = ""
    documentation_file: str = ""
    # Additional metadata
    funding_agency: str = ""
    grants: List[str] = field(default_factory=list)
    related_publications: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FieldSite:
    """Field site information"""
    site_id: str
    site_name: str
    description: str
    # Geographic information
    latitude: float = 0.0
    longitude: float = 0.0
    elevation: float = 0.0
    country: str = ""
    region: str = ""
    # Environmental characteristics
    ecosystem_type: str = ""
    habitat_type: str = ""
    climate_zone: str = ""
    # Astrobiology relevance
    analog_environments: List[str] = field(default_factory=list)  # Mars, Europa, etc.
    extremophile_types: List[str] = field(default_factory=list)
    biosignature_potential: str = ""
    # Site characteristics
    geological_setting: str = ""
    hydrology: str = ""
    chemistry: Dict[str, Any] = field(default_factory=dict)
    # Research activities
    active_studies: List[str] = field(default_factory=list)
    available_datasets: List[str] = field(default_factory=list)
    # Access information
    accessibility: str = ""
    permits_required: bool = False
    contact_info: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class AHEDDownloader:
    """NASA AHED data downloader with enterprise URL management"""
    
    def __init__(self, base_url: str = None):
        # Enterprise URL system integration
        self.url_system = None
        self.base_url = base_url or "https://ahed.nasa.gov/"
        self.api_base = "https://ahed.nasa.gov/api/"  # Will be managed dynamically
        self.cache_path = Path("data/raw/nasa_ahed/cache")
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.session = None
        self.rate_limit_delay = 1.0  # 1 second between requests (NASA rate limiting)
        
        # Initialize enterprise URL system
        self._initialize_url_system()
    
    def _initialize_url_system(self):
        """Initialize enterprise URL management system"""
        try:
            # Import here to avoid circular imports
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent))
            from utils.integrated_url_system import get_integrated_url_system
            from utils.autonomous_data_acquisition import DataPriority
            
            self.url_system = get_integrated_url_system()
            self.data_priority = DataPriority.HIGH  # NASA AHED is high priority for astrobiology
            logging.info("Enterprise URL system initialized for NASA AHED integration")
        except ImportError as e:
            logging.warning(f"Enterprise URL system not available, using fallback: {e}")
            self.url_system = None
    
    async def _get_managed_base_url(self) -> str:
        """Get managed base URL using enterprise system"""
        if self.url_system:
            try:
                # Use enterprise URL system to get optimal NASA AHED endpoint
                managed_url = await self.url_system.get_url(
                    "https://ahed.nasa.gov/", 
                    self.data_priority
                )
                if managed_url:
                    return managed_url if managed_url.endswith('/') else managed_url + '/'
            except Exception as e:
                logging.warning(f"Error getting managed URL, using fallback: {e}")
        
        # Fallback to configured base URL
        return self.base_url
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=1800)  # 30 minutes for large files
            connector = aiohttp.TCPConnector(limit=5)  # Conservative limit for NASA
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={'User-Agent': 'Astrobiology-Research-Platform/1.0'}
            )
        return self.session
    
    def _get_cache_path(self, filename: str) -> Path:
        """Get cache file path"""
        return self.cache_path / filename
    
    def _is_cache_valid(self, cache_file: Path, max_age_days: int = 7) -> bool:
        """Check if cache file is valid and recent"""
        if not cache_file.exists():
            return False
        try:
            file_time = datetime.fromtimestamp(cache_file.stat().st_mtime, tz=timezone.utc)
            return (datetime.now(timezone.utc) - file_time).days < max_age_days
        except Exception:
            return False
    
    async def search_datasets(self, theme: Optional[str] = None, 
                            keywords: Optional[List[str]] = None,
                            max_results: int = 100) -> List[Dict[str, Any]]:
        """Search AHED datasets"""
        session = await self._get_session()
        datasets = []
        
        try:
            # Try API search first
            search_params = {
                'format': 'json',
                'limit': max_results
            }
            
            if theme:
                search_params['theme'] = theme
            if keywords:
                search_params['keywords'] = ','.join(keywords)
            
            api_url = urljoin(self.api_base, "search")
            
            await asyncio.sleep(self.rate_limit_delay)
            
            async with session.get(api_url, params=search_params) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'datasets' in data:
                        datasets.extend(data['datasets'])
                        logger.info(f"Found {len(data['datasets'])} datasets via API")
                        return datasets
                else:
                    logger.warning(f"API search failed: HTTP {response.status}")
            
            # Fallback to web scraping
            datasets = await self._scrape_dataset_listings(theme, keywords, max_results)
            
        except Exception as e:
            logger.error(f"Error searching datasets: {e}")
            # Try web scraping as fallback
            datasets = await self._scrape_dataset_listings(theme, keywords, max_results)
        
        return datasets
    
    async def _scrape_dataset_listings(self, theme: Optional[str] = None,
                                     keywords: Optional[List[str]] = None,
                                     max_results: int = 100) -> List[Dict[str, Any]]:
        """Scrape dataset listings from AHED website"""
        session = await self._get_session()
        datasets = []
        
        try:
            # Get main datasets page
            datasets_url = urljoin(self.base_url, "datasets")
            
            await asyncio.sleep(self.rate_limit_delay)
            
            async with session.get(datasets_url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Look for dataset links and metadata
                    dataset_links = soup.find_all('a', href=True)
                    
                    for link in dataset_links:
                        href = link.get('href', '')
                        if '/dataset/' in href or '/data/' in href:
                            dataset_url = urljoin(self.base_url, href)
                            
                            # Extract basic info from link context
                            title = link.get_text(strip=True)
                            if title:
                                dataset_info = {
                                    'id': href.split('/')[-1],
                                    'title': title,
                                    'url': dataset_url,
                                    'description': '',
                                    'theme': theme or '',
                                    'keywords': keywords or []
                                }
                                datasets.append(dataset_info)
                                
                                if len(datasets) >= max_results:
                                    break
                    
                    logger.info(f"Scraped {len(datasets)} dataset links")
                    
                    # Get additional metadata for each dataset
                    enhanced_datasets = await self._enhance_dataset_metadata(datasets[:20])  # Limit for rate limiting
                    return enhanced_datasets
                    
        except Exception as e:
            logger.error(f"Error scraping dataset listings: {e}")
        
        return datasets
    
    async def _enhance_dataset_metadata(self, datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance dataset metadata by fetching individual pages"""
        session = await self._get_session()
        enhanced = []
        
        for dataset in datasets:
            try:
                await asyncio.sleep(self.rate_limit_delay)
                
                async with session.get(dataset['url']) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Extract metadata from dataset page
                        enhanced_dataset = dataset.copy()
                        
                        # Look for description
                        desc_elem = soup.find('div', class_=['description', 'abstract', 'summary'])
                        if desc_elem:
                            enhanced_dataset['description'] = desc_elem.get_text(strip=True)
                        
                        # Look for authors
                        authors_elem = soup.find('div', class_=['authors', 'contributors'])
                        if authors_elem:
                            enhanced_dataset['authors'] = [a.strip() for a in authors_elem.get_text().split(',')]
                        
                        # Look for download links
                        download_links = soup.find_all('a', href=True)
                        files = []
                        for link in download_links:
                            href = link.get('href', '')
                            if any(ext in href.lower() for ext in ['.csv', '.json', '.xml', '.zip', '.gz', '.nc', '.hdf']):
                                files.append({
                                    'url': urljoin(dataset['url'], href),
                                    'name': link.get_text(strip=True) or href.split('/')[-1],
                                    'type': href.split('.')[-1].lower()
                                })
                        enhanced_dataset['files'] = files
                        
                        # Look for geographic info
                        geo_elem = soup.find('div', class_=['location', 'geographic', 'coordinates'])
                        if geo_elem:
                            geo_text = geo_elem.get_text()
                            # Try to extract coordinates
                            lat_match = re.search(r'lat(?:itude)?[:\s]*([+-]?\d+\.?\d*)', geo_text, re.I)
                            lon_match = re.search(r'lon(?:gitude)?[:\s]*([+-]?\d+\.?\d*)', geo_text, re.I)
                            if lat_match:
                                enhanced_dataset['latitude'] = float(lat_match.group(1))
                            if lon_match:
                                enhanced_dataset['longitude'] = float(lon_match.group(1))
                        
                        enhanced.append(enhanced_dataset)
                        
            except Exception as e:
                logger.warning(f"Error enhancing dataset {dataset.get('id', 'unknown')}: {e}")
                enhanced.append(dataset)
        
        return enhanced
    
    async def download_dataset_files(self, dataset: Dict[str, Any], max_files: Optional[int] = None) -> Dict[str, str]:
        """Download files for a specific dataset"""
        session = await self._get_session()
        downloaded_files = {}
        
        try:
            files = dataset.get('files', [])
            if max_files:
                files = files[:max_files]
            
            for file_info in files:
                file_url = file_info.get('url', '')
                file_name = file_info.get('name', file_url.split('/')[-1])
                
                if not file_url:
                    continue
                
                # Create safe filename
                safe_filename = re.sub(r'[^\w\-_.]', '_', file_name)
                cache_file = self._get_cache_path(f"{dataset.get('id', 'unknown')}_{safe_filename}")
                
                if self._is_cache_valid(cache_file, max_age_days=30):
                    downloaded_files[file_name] = str(cache_file)
                    continue
                
                try:
                    await asyncio.sleep(self.rate_limit_delay)
                    
                    async with session.get(file_url) as response:
                        if response.status == 200:
                            content = await response.read()
                            
                            cache_file.parent.mkdir(parents=True, exist_ok=True)
                            with open(cache_file, 'wb') as f:
                                f.write(content)
                            
                            downloaded_files[file_name] = str(cache_file)
                            logger.info(f"Downloaded {file_name}")
                        else:
                            logger.warning(f"Failed to download {file_name}: HTTP {response.status}")
                            
                except Exception as e:
                    logger.warning(f"Error downloading {file_name}: {e}")
            
            return downloaded_files
            
        except Exception as e:
            logger.error(f"Error downloading dataset files: {e}")
            return {}
    
    async def get_field_sites(self) -> List[Dict[str, Any]]:
        """Get field site information"""
        session = await self._get_session()
        sites = []
        
        try:
            # Try API endpoint
            sites_url = urljoin(self.api_base, "sites")
            
            await asyncio.sleep(self.rate_limit_delay)
            
            async with session.get(sites_url) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'sites' in data:
                        return data['sites']
            
            # Fallback to scraping
            sites_page_url = urljoin(self.base_url, "sites")
            
            async with session.get(sites_page_url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Look for site information
                    site_elements = soup.find_all(['div', 'section'], class_=re.compile('site'))
                    
                    for site_elem in site_elements:
                        site_info = {
                            'id': '',
                            'name': '',
                            'description': '',
                            'latitude': 0.0,
                            'longitude': 0.0,
                            'ecosystem': '',
                            'analog_environments': []
                        }
                        
                        # Extract site name
                        name_elem = site_elem.find(['h1', 'h2', 'h3'])
                        if name_elem:
                            site_info['name'] = name_elem.get_text(strip=True)
                            site_info['id'] = re.sub(r'[^\w\-_]', '_', site_info['name'].lower())
                        
                        # Extract description
                        desc_elem = site_elem.find('p')
                        if desc_elem:
                            site_info['description'] = desc_elem.get_text(strip=True)
                        
                        sites.append(site_info)
            
            logger.info(f"Found {len(sites)} field sites")
            return sites
            
        except Exception as e:
            logger.error(f"Error getting field sites: {e}")
            return []
    
    async def get_database_info(self) -> Dict[str, Any]:
        """Get AHED database information"""
        session = await self._get_session()
        
        try:
            # Get main page info
            async with session.get(self.base_url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    info = {
                        'name': 'NASA Astrobiology Habitable Environments Database (AHED)',
                        'url': self.base_url,
                        'description': 'NASA\'s repository for astrobiology and habitability research data',
                        'themes': self.themes,
                        'last_checked': datetime.now(timezone.utc).isoformat()
                    }
                    
                    # Try to extract statistics
                    stats_elem = soup.find('div', class_=['stats', 'statistics'])
                    if stats_elem:
                        stats_text = stats_elem.get_text()
                        # Extract numbers
                        dataset_match = re.search(r'(\d+)\s*datasets?', stats_text, re.I)
                        if dataset_match:
                            info['total_datasets'] = int(dataset_match.group(1))
                    
                    return info
                    
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
        
        return {
            'name': 'NASA AHED',
            'url': self.base_url,
            'description': 'NASA Astrobiology Habitable Environments Database',
            'themes': self.themes,
            'last_checked': datetime.now(timezone.utc).isoformat()
        }
    
    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()

class AHEDParser:
    """Parser for AHED data"""
    
    def __init__(self, output_path: str = "data/processed/nasa_ahed"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.output_path / "ahed.db"
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database for AHED data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Datasets table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    title TEXT,
                    description TEXT,
                    theme TEXT,
                    keywords TEXT,
                    authors TEXT,
                    institution TEXT,
                    contact_email TEXT,
                    collection_start_date TEXT,
                    collection_end_date TEXT,
                    publication_date TEXT,
                    last_updated TEXT,
                    location TEXT,
                    latitude REAL,
                    longitude REAL,
                    elevation REAL,
                    site_type TEXT,
                    environment_type TEXT,
                    ecosystem TEXT,
                    temperature_range TEXT,
                    ph_range TEXT,
                    salinity TEXT,
                    radiation_environment TEXT,
                    data_types TEXT,
                    file_formats TEXT,
                    data_size INTEGER,
                    instruments TEXT,
                    missions TEXT,
                    quality_level TEXT,
                    processing_level TEXT,
                    calibration_status TEXT,
                    validation_status TEXT,
                    access_level TEXT,
                    download_url TEXT,
                    doi TEXT,
                    funding_agency TEXT,
                    grants TEXT,
                    related_publications TEXT,
                    metadata TEXT
                )
            ''')
            
            # Field sites table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS field_sites (
                    site_id TEXT PRIMARY KEY,
                    site_name TEXT,
                    description TEXT,
                    latitude REAL,
                    longitude REAL,
                    elevation REAL,
                    country TEXT,
                    region TEXT,
                    ecosystem_type TEXT,
                    habitat_type TEXT,
                    climate_zone TEXT,
                    analog_environments TEXT,
                    extremophile_types TEXT,
                    biosignature_potential TEXT,
                    geological_setting TEXT,
                    hydrology TEXT,
                    chemistry TEXT,
                    active_studies TEXT,
                    available_datasets TEXT,
                    accessibility TEXT,
                    permits_required BOOLEAN,
                    contact_info TEXT,
                    metadata TEXT
                )
            ''')
            
            # Files table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS dataset_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id TEXT,
                    file_name TEXT,
                    file_path TEXT,
                    file_type TEXT,
                    file_size INTEGER,
                    download_url TEXT,
                    checksum TEXT,
                    FOREIGN KEY (dataset_id) REFERENCES datasets (dataset_id)
                )
            ''')
            
            # Create indices
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_dataset_theme ON datasets(theme)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_dataset_environment ON datasets(environment_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_dataset_location ON datasets(latitude, longitude)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_site_ecosystem ON field_sites(ecosystem_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_site_location ON field_sites(latitude, longitude)')
            
            conn.commit()
    
    def parse_datasets(self, datasets: List[Dict[str, Any]]) -> List[AHEDDataset]:
        """Parse dataset dictionaries into AHEDDataset objects"""
        parsed_datasets = []
        
        for dataset in datasets:
            try:
                ahed_dataset = AHEDDataset(
                    dataset_id=str(dataset.get('id', '')),
                    title=str(dataset.get('title', '')),
                    description=str(dataset.get('description', '')),
                    theme=str(dataset.get('theme', '')),
                    keywords=dataset.get('keywords', []),
                    authors=dataset.get('authors', []),
                    institution=str(dataset.get('institution', '')),
                    contact_email=str(dataset.get('contact_email', '')),
                    location=str(dataset.get('location', '')),
                    latitude=float(dataset.get('latitude', 0)),
                    longitude=float(dataset.get('longitude', 0)),
                    elevation=float(dataset.get('elevation', 0)),
                    environment_type=str(dataset.get('environment_type', '')),
                    ecosystem=str(dataset.get('ecosystem', '')),
                    data_types=dataset.get('data_types', []),
                    file_formats=dataset.get('file_formats', []),
                    instruments=dataset.get('instruments', []),
                    missions=dataset.get('missions', []),
                    download_url=str(dataset.get('url', '')),
                    files=dataset.get('files', [])
                )
                
                parsed_datasets.append(ahed_dataset)
                
            except Exception as e:
                logger.warning(f"Error parsing dataset {dataset.get('id', 'unknown')}: {e}")
        
        return parsed_datasets
    
    def parse_field_sites(self, sites: List[Dict[str, Any]]) -> List[FieldSite]:
        """Parse field site dictionaries into FieldSite objects"""
        parsed_sites = []
        
        for site in sites:
            try:
                field_site = FieldSite(
                    site_id=str(site.get('id', '')),
                    site_name=str(site.get('name', '')),
                    description=str(site.get('description', '')),
                    latitude=float(site.get('latitude', 0)),
                    longitude=float(site.get('longitude', 0)),
                    elevation=float(site.get('elevation', 0)),
                    country=str(site.get('country', '')),
                    region=str(site.get('region', '')),
                    ecosystem_type=str(site.get('ecosystem', '')),
                    habitat_type=str(site.get('habitat_type', '')),
                    analog_environments=site.get('analog_environments', []),
                    extremophile_types=site.get('extremophile_types', []),
                    biosignature_potential=str(site.get('biosignature_potential', ''))
                )
                
                parsed_sites.append(field_site)
                
            except Exception as e:
                logger.warning(f"Error parsing field site {site.get('id', 'unknown')}: {e}")
        
        return parsed_sites
    
    def store_datasets(self, datasets: List[AHEDDataset]):
        """Store datasets in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for dataset in datasets:
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO datasets (
                            dataset_id, title, description, theme, keywords, authors,
                            institution, contact_email, collection_start_date,
                            collection_end_date, publication_date, last_updated,
                            location, latitude, longitude, elevation, site_type,
                            environment_type, ecosystem, temperature_range, ph_range,
                            salinity, radiation_environment, data_types, file_formats,
                            data_size, instruments, missions, quality_level,
                            processing_level, calibration_status, validation_status,
                            access_level, download_url, doi, funding_agency, grants,
                            related_publications, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        dataset.dataset_id, dataset.title, dataset.description,
                        dataset.theme, json.dumps(dataset.keywords),
                        json.dumps(dataset.authors), dataset.institution,
                        dataset.contact_email, dataset.collection_start_date,
                        dataset.collection_end_date, dataset.publication_date,
                        dataset.last_updated, dataset.location, dataset.latitude,
                        dataset.longitude, dataset.elevation, dataset.site_type,
                        dataset.environment_type, dataset.ecosystem,
                        dataset.temperature_range, dataset.ph_range, dataset.salinity,
                        dataset.radiation_environment, json.dumps(dataset.data_types),
                        json.dumps(dataset.file_formats), dataset.data_size,
                        json.dumps(dataset.instruments), json.dumps(dataset.missions),
                        dataset.quality_level, dataset.processing_level,
                        dataset.calibration_status, dataset.validation_status,
                        dataset.access_level, dataset.download_url, dataset.doi,
                        dataset.funding_agency, json.dumps(dataset.grants),
                        json.dumps(dataset.related_publications),
                        json.dumps(dataset.metadata)
                    ))
                    
                    # Store files
                    for file_info in dataset.files:
                        cursor.execute('''
                            INSERT OR REPLACE INTO dataset_files (
                                dataset_id, file_name, file_path, file_type, download_url
                            ) VALUES (?, ?, ?, ?, ?)
                        ''', (
                            dataset.dataset_id, file_info.get('name', ''),
                            file_info.get('path', ''), file_info.get('type', ''),
                            file_info.get('url', '')
                        ))
                        
                except Exception as e:
                    logger.error(f"Error storing dataset {dataset.dataset_id}: {e}")
            
            conn.commit()
    
    def store_field_sites(self, sites: List[FieldSite]):
        """Store field sites in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for site in sites:
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO field_sites (
                            site_id, site_name, description, latitude, longitude,
                            elevation, country, region, ecosystem_type, habitat_type,
                            climate_zone, analog_environments, extremophile_types,
                            biosignature_potential, geological_setting, hydrology,
                            chemistry, active_studies, available_datasets,
                            accessibility, permits_required, contact_info, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        site.site_id, site.site_name, site.description,
                        site.latitude, site.longitude, site.elevation,
                        site.country, site.region, site.ecosystem_type,
                        site.habitat_type, site.climate_zone,
                        json.dumps(site.analog_environments),
                        json.dumps(site.extremophile_types),
                        site.biosignature_potential, site.geological_setting,
                        site.hydrology, json.dumps(site.chemistry),
                        json.dumps(site.active_studies),
                        json.dumps(site.available_datasets), site.accessibility,
                        site.permits_required, site.contact_info,
                        json.dumps(site.metadata)
                    ))
                except Exception as e:
                    logger.error(f"Error storing field site {site.site_id}: {e}")
            
            conn.commit()
    
    def export_to_csv(self) -> Dict[str, str]:
        """Export data to CSV files"""
        output_files = {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Export datasets
                df_datasets = pd.read_sql_query('''
                    SELECT dataset_id, title, description, theme, authors,
                           institution, location, latitude, longitude, environment_type,
                           ecosystem, data_types, instruments, missions, access_level,
                           download_url, doi
                    FROM datasets
                ''', conn)
                
                if not df_datasets.empty:
                    datasets_file = self.output_path / "nasa_ahed_datasets.csv"
                    df_datasets.to_csv(datasets_file, index=False)
                    output_files['datasets'] = str(datasets_file)
                
                # Export field sites
                df_sites = pd.read_sql_query('''
                    SELECT site_id, site_name, description, latitude, longitude,
                           elevation, country, region, ecosystem_type, habitat_type,
                           analog_environments, extremophile_types, biosignature_potential
                    FROM field_sites
                ''', conn)
                
                if not df_sites.empty:
                    sites_file = self.output_path / "nasa_ahed_field_sites.csv"
                    df_sites.to_csv(sites_file, index=False)
                    output_files['field_sites'] = str(sites_file)
                
                # Export by theme
                themes = df_datasets['theme'].unique() if not df_datasets.empty else []
                for theme in themes:
                    if theme and theme.strip():
                        safe_theme = re.sub(r'[^\w\-_]', '_', theme.lower())
                        df_theme = df_datasets[df_datasets['theme'] == theme]
                        theme_file = self.output_path / f"nasa_ahed_{safe_theme}.csv"
                        df_theme.to_csv(theme_file, index=False)
                        output_files[f'theme_{safe_theme}'] = str(theme_file)
                
                logger.info(f"Exported NASA AHED data to {len(output_files)} CSV files")
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
                
                # Dataset statistics
                cursor.execute("SELECT COUNT(*) FROM datasets")
                stats['total_datasets'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM field_sites")
                stats['total_field_sites'] = cursor.fetchone()[0]
                
                # Theme distribution
                cursor.execute('''
                    SELECT theme, COUNT(*) 
                    FROM datasets 
                    WHERE theme != ''
                    GROUP BY theme
                ''')
                theme_dist = dict(cursor.fetchall())
                stats['theme_distribution'] = theme_dist
                
                # Environment distribution
                cursor.execute('''
                    SELECT environment_type, COUNT(*) 
                    FROM datasets 
                    WHERE environment_type != ''
                    GROUP BY environment_type 
                    ORDER BY COUNT(*) DESC 
                    LIMIT 10
                ''')
                env_dist = dict(cursor.fetchall())
                stats['environment_distribution'] = env_dist
                
                # Geographic distribution
                cursor.execute('''
                    SELECT COUNT(*) 
                    FROM datasets 
                    WHERE latitude != 0 AND longitude != 0
                ''')
                stats['georeferenced_datasets'] = cursor.fetchone()[0]
                
                # Data access levels
                cursor.execute('''
                    SELECT access_level, COUNT(*) 
                    FROM datasets 
                    WHERE access_level != ''
                    GROUP BY access_level
                ''')
                access_dist = dict(cursor.fetchall())
                stats['access_level_distribution'] = access_dist
                
                return stats
                
        except Exception as e:
            logger.error(f"Error generating statistics: {e}")
            return {}

class NASAAHEDIntegration:
    """Main integration class for NASA AHED data"""
    
    def __init__(self, output_path: str = "data"):
        self.output_path = Path(output_path)
        self.downloader = AHEDDownloader()
        self.parser = AHEDParser(str(self.output_path / "processed" / "nasa_ahed"))
        
        # Progress tracking
        self.progress_file = self.output_path / "interim" / "nasa_ahed_progress.json"
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _load_progress(self) -> Dict[str, Any]:
        """Load integration progress"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading progress: {e}")
        return {"searches": {}, "downloads": {}, "last_updated": None}
    
    def _save_progress(self, progress: Dict[str, Any]):
        """Save integration progress"""
        try:
            progress["last_updated"] = datetime.now(timezone.utc).isoformat()
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving progress: {e}")
    
    async def search_and_download_data(self, themes: Optional[List[str]] = None,
                                     max_datasets_per_theme: int = 50,
                                     download_files: bool = False) -> Dict[str, Any]:
        """Search and download AHED data"""
        progress = self._load_progress()
        results = {"searches": {}, "downloads": {}, "processing": {}}
        
        try:
            if themes is None:
                themes = self.downloader.themes
            
            all_datasets = []
            
            for theme in themes:
                logger.info(f"Searching datasets for theme: {theme}")
                
                # Search datasets
                datasets = await self.downloader.search_datasets(
                    theme=theme,
                    max_results=max_datasets_per_theme
                )
                
                if datasets:
                    progress["searches"][theme] = len(datasets)
                    results["searches"][theme] = len(datasets)
                    all_datasets.extend(datasets)
                    
                    # Download files if requested
                    if download_files:
                        downloaded_files = {}
                        for dataset in datasets[:5]:  # Limit downloads for rate limiting
                            files = await self.downloader.download_dataset_files(
                                dataset, max_files=3
                            )
                            if files:
                                downloaded_files[dataset.get('id', 'unknown')] = files
                        
                        progress["downloads"][theme] = downloaded_files
                        results["downloads"][theme] = downloaded_files
                
                # Save progress after each theme
                self._save_progress(progress)
                await asyncio.sleep(2)  # Rate limiting
            
            # Parse and store datasets
            if all_datasets:
                parsed_datasets = self.parser.parse_datasets(all_datasets)
                self.parser.store_datasets(parsed_datasets)
                results["processing"]["datasets"] = len(parsed_datasets)
            
            # Get and store field sites
            field_sites = await self.downloader.get_field_sites()
            if field_sites:
                parsed_sites = self.parser.parse_field_sites(field_sites)
                self.parser.store_field_sites(parsed_sites)
                results["processing"]["field_sites"] = len(parsed_sites)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in search and download: {e}")
            return results
        finally:
            await self.downloader.close()
    
    def export_datasets(self) -> Dict[str, str]:
        """Export all datasets to CSV format"""
        return self.parser.export_to_csv()
    
    async def run_full_integration(self, themes: Optional[List[str]] = None,
                                 max_datasets_per_theme: int = 50,
                                 download_files: bool = False) -> Dict[str, Any]:
        """Run complete NASA AHED integration"""
        logger.info("Starting comprehensive NASA AHED integration")
        
        start_time = time.time()
        results = {
            "status": "started",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "database_info": {},
            "searches": {},
            "downloads": {},
            "processing": {},
            "exports": {},
            "statistics": {},
            "errors": []
        }
        
        try:
            # Get database information
            db_info = await self.downloader.get_database_info()
            results["database_info"] = db_info
            
            # Search and download data
            search_results = await self.search_and_download_data(
                themes=themes,
                max_datasets_per_theme=max_datasets_per_theme,
                download_files=download_files
            )
            results.update(search_results)
            
            # Export to CSV
            export_files = self.export_datasets()
            results["exports"] = export_files
            
            # Generate statistics
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
            
            logger.info(f"NASA AHED integration completed in {end_time - start_time:.2f} seconds")
            logger.info(f"Total datasets processed: {results.get('statistics', {}).get('total_datasets', 0)}")
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
    integration = NASAAHEDIntegration()
    
    # Test with limited data
    results = await integration.run_full_integration(
        themes=["Abiotic Building Blocks of Life", "Characterizing Environments for Habitability and Biosignatures"],
        max_datasets_per_theme=20,
        download_files=False
    )
    
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    asyncio.run(main()) 
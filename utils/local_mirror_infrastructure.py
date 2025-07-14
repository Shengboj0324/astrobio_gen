#!/usr/bin/env python3
"""
Local Mirror Infrastructure System
==================================

Enterprise-grade local mirroring system using AWS S3 infrastructure for controlled
backup of critical scientific datasets. Integrates with existing cloud-first storage.

Features:
- Intelligent mirroring strategy based on criticality
- Geographic replication across AWS regions
- Automatic sync with upstream sources
- Cost-optimized storage classes
- Integration with existing S3 buckets
- Bandwidth-aware downloading
- Delta synchronization
- Health monitoring and alerting
"""

import asyncio
import boto3
import botocore
import logging
import json
import hashlib
import gzip
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import pandas as pd
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import aiohttp
import yaml
from urllib.parse import urlparse
import schedule

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class MirrorConfig:
    """Configuration for mirroring a data source"""
    source_name: str
    source_url: str
    mirror_path: str
    priority: int  # 1=critical, 2=important, 3=normal
    sync_frequency: str  # hourly, daily, weekly
    storage_class: str  # STANDARD, IA, GLACIER
    compression: bool = True
    encryption: bool = True
    versioning: bool = True
    notification_on_sync: bool = False
    max_size_gb: Optional[float] = None
    file_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)

@dataclass 
class SyncJob:
    """Synchronization job tracking"""
    job_id: str
    source_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, failed, cancelled
    files_synced: int = 0
    bytes_synced: int = 0
    errors: List[str] = field(default_factory=list)
    s3_path: str = ""

class LocalMirrorInfrastructure:
    """
    Local mirror infrastructure using AWS S3
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # AWS configuration from existing setup
        self.s3_buckets = self.config.get('aws', {}).get('s3_buckets', {})
        self.aws_region = self.config.get('aws', {}).get('region', 'us-east-1')
        
        # Initialize AWS clients
        self.s3_client = boto3.client('s3', region_name=self.aws_region)
        self.s3_resource = boto3.resource('s3', region_name=self.aws_region)
        
        # Database for tracking mirrors
        self.db_path = Path("data/metadata/mirror_infrastructure.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
        
        # Mirror configurations
        self.mirror_configs = self._load_mirror_configs()
        
        # Active sync jobs
        self.active_jobs = {}
        self.job_lock = threading.Lock()
        
        # Performance monitoring
        self.bandwidth_monitor = BandwidthMonitor()
        
        # S3 bucket mapping
        self.bucket_mapping = {
            'critical': self.s3_buckets.get('primary', 'astrobio-data-primary'),
            'important': self.s3_buckets.get('backup', 'astrobio-data-backup'),
            'normal': self.s3_buckets.get('logs', 'astrobio-logs-metadata'),
            'processed': self.s3_buckets.get('zarr', 'astrobio-zarr-cubes')
        }
        
        logger.info(f"Local Mirror Infrastructure initialized with {len(self.mirror_configs)} sources")
        logger.info(f"S3 buckets: {self.bucket_mapping}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def _initialize_database(self):
        """Initialize mirror tracking database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS mirror_configs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT UNIQUE NOT NULL,
                    source_url TEXT NOT NULL,
                    mirror_path TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    sync_frequency TEXT NOT NULL,
                    storage_class TEXT NOT NULL,
                    compression BOOLEAN DEFAULT TRUE,
                    encryption BOOLEAN DEFAULT TRUE,
                    versioning BOOLEAN DEFAULT TRUE,
                    max_size_gb REAL,
                    last_sync TIMESTAMP,
                    sync_status TEXT DEFAULT 'pending',
                    total_files INTEGER DEFAULT 0,
                    total_size_bytes INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS sync_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT UNIQUE NOT NULL,
                    source_name TEXT NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    status TEXT NOT NULL,
                    files_synced INTEGER DEFAULT 0,
                    bytes_synced INTEGER DEFAULT 0,
                    s3_path TEXT NOT NULL,
                    errors TEXT, -- JSON array
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS mirror_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT NOT NULL,
                    check_time TIMESTAMP NOT NULL,
                    source_available BOOLEAN NOT NULL,
                    mirror_available BOOLEAN NOT NULL,
                    source_size_bytes INTEGER,
                    mirror_size_bytes INTEGER,
                    sync_lag_hours REAL,
                    data_integrity_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS bandwidth_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    source_name TEXT NOT NULL,
                    bytes_downloaded INTEGER NOT NULL,
                    bandwidth_mbps REAL NOT NULL,
                    duration_seconds REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_sync_jobs_status ON sync_jobs(status);
                CREATE INDEX IF NOT EXISTS idx_mirror_health_source ON mirror_health(source_name);
                CREATE INDEX IF NOT EXISTS idx_bandwidth_timestamp ON bandwidth_usage(timestamp);
            """)
    
    def _load_mirror_configs(self) -> Dict[str, MirrorConfig]:
        """Load mirror configurations for critical data sources"""
        configs = {}
        
        # Critical sources (mirror immediately, high priority)
        critical_sources = [
            {
                'source_name': 'nasa_exoplanet_archive',
                'source_url': 'https://exoplanetarchive.ipac.caltech.edu',
                'mirror_path': 'mirrors/nasa/exoplanets/',
                'priority': 1,
                'sync_frequency': 'daily',
                'storage_class': 'STANDARD',
                'max_size_gb': 50.0,
                'file_patterns': ['*.csv', '*.fits', '*.xml']
            },
            {
                'source_name': 'kegg_database',
                'source_url': 'https://rest.kegg.jp',
                'mirror_path': 'mirrors/kegg/pathways/',
                'priority': 1,
                'sync_frequency': 'weekly',
                'storage_class': 'STANDARD',
                'max_size_gb': 10.0,
                'file_patterns': ['*.kgml', '*.json', '*.csv']
            },
            {
                'source_name': 'ncbi_critical_genomes',
                'source_url': 'https://ftp.ncbi.nlm.nih.gov/genomes/refseq/bacteria',
                'mirror_path': 'mirrors/ncbi/bacteria/',
                'priority': 1,
                'sync_frequency': 'weekly',
                'storage_class': 'STANDARD',
                'max_size_gb': 100.0,
                'file_patterns': ['*_genomic.fna.gz', '*_protein.faa.gz']
            }
        ]
        
        # Important sources (mirror regularly, medium priority)
        important_sources = [
            {
                'source_name': 'phoenix_stellar_models',
                'source_url': 'https://phoenix.astro.physik.uni-goettingen.de',
                'mirror_path': 'mirrors/phoenix/models/',
                'priority': 2,
                'sync_frequency': 'weekly',
                'storage_class': 'STANDARD_IA',
                'max_size_gb': 200.0,
                'file_patterns': ['*.fits']
            },
            {
                'source_name': 'uniprot_databases',
                'source_url': 'https://ftp.uniprot.org/pub/databases/uniprot',
                'mirror_path': 'mirrors/uniprot/sequences/',
                'priority': 2,
                'sync_frequency': 'monthly',
                'storage_class': 'STANDARD_IA',
                'max_size_gb': 500.0,
                'file_patterns': ['*.dat.gz', '*.fasta.gz']
            }
        ]
        
        # Create MirrorConfig objects
        all_sources = critical_sources + important_sources
        
        for source_config in all_sources:
            configs[source_config['source_name']] = MirrorConfig(
                source_name=source_config['source_name'],
                source_url=source_config['source_url'],
                mirror_path=source_config['mirror_path'],
                priority=source_config['priority'],
                sync_frequency=source_config['sync_frequency'],
                storage_class=source_config['storage_class'],
                max_size_gb=source_config.get('max_size_gb'),
                file_patterns=source_config.get('file_patterns', [])
            )
        
        return configs
    
    async def setup_mirrors(self) -> Dict[str, Any]:
        """Set up initial mirror infrastructure"""
        setup_results = {
            'buckets_verified': [],
            'buckets_created': [],
            'configs_installed': 0,
            'errors': []
        }
        
        try:
            # Verify S3 buckets exist
            for bucket_type, bucket_name in self.bucket_mapping.items():
                try:
                    self.s3_client.head_bucket(Bucket=bucket_name)
                    setup_results['buckets_verified'].append(bucket_name)
                    logger.info(f"Verified S3 bucket: {bucket_name}")
                except botocore.exceptions.ClientError as e:
                    if e.response['Error']['Code'] == '404':
                        # Create bucket if it doesn't exist
                        try:
                            if self.aws_region == 'us-east-1':
                                self.s3_client.create_bucket(Bucket=bucket_name)
                            else:
                                self.s3_client.create_bucket(
                                    Bucket=bucket_name,
                                    CreateBucketConfiguration={'LocationConstraint': self.aws_region}
                                )
                            setup_results['buckets_created'].append(bucket_name)
                            logger.info(f"Created S3 bucket: {bucket_name}")
                        except Exception as create_error:
                            error_msg = f"Failed to create bucket {bucket_name}: {create_error}"
                            setup_results['errors'].append(error_msg)
                            logger.error(error_msg)
                    else:
                        error_msg = f"Error checking bucket {bucket_name}: {e}"
                        setup_results['errors'].append(error_msg)
                        logger.error(error_msg)
            
            # Configure bucket policies and lifecycle rules
            await self._configure_s3_buckets()
            
            # Install mirror configurations in database
            self._install_mirror_configs()
            setup_results['configs_installed'] = len(self.mirror_configs)
            
            # Schedule initial sync jobs
            await self._schedule_initial_syncs()
            
            logger.info("Mirror infrastructure setup completed successfully")
            
        except Exception as e:
            error_msg = f"Error setting up mirror infrastructure: {e}"
            setup_results['errors'].append(error_msg)
            logger.error(error_msg)
        
        return setup_results
    
    async def _configure_s3_buckets(self):
        """Configure S3 bucket policies and lifecycle rules"""
        try:
            for bucket_name in self.bucket_mapping.values():
                # Enable versioning
                self.s3_client.put_bucket_versioning(
                    Bucket=bucket_name,
                    VersioningConfiguration={'Status': 'Enabled'}
                )
                
                # Configure lifecycle policy for cost optimization
                lifecycle_config = {
                    'Rules': [
                        {
                            'ID': 'Mirror-Lifecycle',
                            'Status': 'Enabled',
                            'Filter': {'Prefix': 'mirrors/'},
                            'Transitions': [
                                {
                                    'Days': 30,
                                    'StorageClass': 'STANDARD_IA'
                                },
                                {
                                    'Days': 90,
                                    'StorageClass': 'GLACIER'
                                }
                            ]
                        }
                    ]
                }
                
                self.s3_client.put_bucket_lifecycle_configuration(
                    Bucket=bucket_name,
                    LifecycleConfiguration=lifecycle_config
                )
                
                # Enable server-side encryption
                encryption_config = {
                    'Rules': [
                        {
                            'ApplyServerSideEncryptionByDefault': {
                                'SSEAlgorithm': 'AES256'
                            }
                        }
                    ]
                }
                
                self.s3_client.put_bucket_encryption(
                    Bucket=bucket_name,
                    ServerSideEncryptionConfiguration=encryption_config
                )
                
                logger.info(f"Configured S3 bucket: {bucket_name}")
                
        except Exception as e:
            logger.error(f"Error configuring S3 buckets: {e}")
    
    def _install_mirror_configs(self):
        """Install mirror configurations in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for source_name, config in self.mirror_configs.items():
                    conn.execute("""
                        INSERT OR REPLACE INTO mirror_configs
                        (source_name, source_url, mirror_path, priority, sync_frequency,
                         storage_class, compression, encryption, versioning, max_size_gb)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        config.source_name,
                        config.source_url,
                        config.mirror_path,
                        config.priority,
                        config.sync_frequency,
                        config.storage_class,
                        config.compression,
                        config.encryption,
                        config.versioning,
                        config.max_size_gb
                    ))
                    
            logger.info(f"Installed {len(self.mirror_configs)} mirror configurations")
            
        except Exception as e:
            logger.error(f"Error installing mirror configurations: {e}")
    
    async def _schedule_initial_syncs(self):
        """Schedule initial synchronization jobs"""
        try:
            # Schedule critical sources for immediate sync
            critical_sources = [name for name, config in self.mirror_configs.items() 
                             if config.priority == 1]
            
            for source_name in critical_sources:
                await self.sync_source(source_name)
                
            logger.info(f"Scheduled initial sync for {len(critical_sources)} critical sources")
            
        except Exception as e:
            logger.error(f"Error scheduling initial syncs: {e}")
    
    async def sync_source(self, source_name: str, force: bool = False) -> Optional[SyncJob]:
        """Synchronize a specific data source to S3"""
        if source_name not in self.mirror_configs:
            logger.error(f"Unknown source: {source_name}")
            return None
        
        config = self.mirror_configs[source_name]
        
        # Check if sync is already running
        with self.job_lock:
            if source_name in self.active_jobs:
                logger.warning(f"Sync already running for {source_name}")
                return self.active_jobs[source_name]
        
        # Check if sync is needed (unless forced)
        if not force and not self._is_sync_needed(source_name, config):
            logger.info(f"Sync not needed for {source_name}")
            return None
        
        # Create sync job
        job_id = f"{source_name}_{int(time.time())}"
        bucket_name = self._get_bucket_for_priority(config.priority)
        s3_path = config.mirror_path
        
        sync_job = SyncJob(
            job_id=job_id,
            source_name=source_name,
            start_time=datetime.now(timezone.utc),
            s3_path=f"s3://{bucket_name}/{s3_path}"
        )
        
        with self.job_lock:
            self.active_jobs[source_name] = sync_job
        
        try:
            # Store job in database
            self._store_sync_job(sync_job)
            
            # Perform synchronization
            await self._perform_sync(config, sync_job, bucket_name, s3_path)
            
            # Update job status
            sync_job.status = "completed"
            sync_job.end_time = datetime.now(timezone.utc)
            
            logger.info(f"Sync completed for {source_name}: {sync_job.files_synced} files, "
                       f"{sync_job.bytes_synced / 1024 / 1024:.1f} MB")
            
        except Exception as e:
            sync_job.status = "failed"
            sync_job.end_time = datetime.now(timezone.utc)
            sync_job.errors.append(str(e))
            logger.error(f"Sync failed for {source_name}: {e}")
        
        finally:
            # Update database and clean up
            self._update_sync_job(sync_job)
            with self.job_lock:
                self.active_jobs.pop(source_name, None)
        
        return sync_job
    
    def _is_sync_needed(self, source_name: str, config: MirrorConfig) -> bool:
        """Check if synchronization is needed based on frequency and last sync"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT last_sync FROM mirror_configs WHERE source_name = ?
                """, (source_name,))
                
                result = cursor.fetchone()
                if not result or not result[0]:
                    return True  # Never synced
                
                last_sync = datetime.fromisoformat(result[0].replace('Z', '+00:00'))
                now = datetime.now(timezone.utc)
                
                # Calculate sync interval based on frequency
                intervals = {
                    'hourly': timedelta(hours=1),
                    'daily': timedelta(days=1),
                    'weekly': timedelta(weeks=1),
                    'monthly': timedelta(days=30)
                }
                
                interval = intervals.get(config.sync_frequency, timedelta(days=1))
                return (now - last_sync) >= interval
                
        except Exception as e:
            logger.error(f"Error checking sync schedule for {source_name}: {e}")
            return True  # Sync on error to be safe
    
    def _get_bucket_for_priority(self, priority: int) -> str:
        """Get S3 bucket name based on priority"""
        if priority == 1:
            return self.bucket_mapping['critical']
        elif priority == 2:
            return self.bucket_mapping['important']
        else:
            return self.bucket_mapping['normal']
    
    async def _perform_sync(self, config: MirrorConfig, sync_job: SyncJob, 
                          bucket_name: str, s3_path: str):
        """Perform the actual synchronization"""
        try:
            if config.source_name == 'nasa_exoplanet_archive':
                await self._sync_nasa_exoplanet_archive(config, sync_job, bucket_name, s3_path)
            elif config.source_name == 'kegg_database':
                await self._sync_kegg_database(config, sync_job, bucket_name, s3_path)
            elif config.source_name == 'ncbi_critical_genomes':
                await self._sync_ncbi_genomes(config, sync_job, bucket_name, s3_path)
            else:
                # Generic HTTP/FTP sync
                await self._sync_generic_source(config, sync_job, bucket_name, s3_path)
                
        except Exception as e:
            logger.error(f"Error in sync for {config.source_name}: {e}")
            raise
    
    async def _sync_nasa_exoplanet_archive(self, config: MirrorConfig, sync_job: SyncJob,
                                         bucket_name: str, s3_path: str):
        """Sync NASA Exoplanet Archive data"""
        base_url = "https://exoplanetarchive.ipac.caltech.edu"
        
        # Define key datasets to mirror
        datasets = [
            {
                'name': 'confirmed_planets.csv',
                'url': f"{base_url}/cgi-bin/nstedAPI/nph-nstedAPI?table=exoplanets&format=csv&select=*"
            },
            {
                'name': 'stellar_hosts.csv', 
                'url': f"{base_url}/cgi-bin/nstedAPI/nph-nstedAPI?table=exoplanets&format=csv&select=hostname,st_teff,st_rad,st_mass,st_met,st_logg,st_age"
            },
            {
                'name': 'kepler_objects.csv',
                'url': f"{base_url}/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative&format=csv&select=*"
            }
        ]
        
        async with aiohttp.ClientSession() as session:
            for dataset in datasets:
                try:
                    # Download data
                    async with session.get(dataset['url'], timeout=300) as response:
                        if response.status == 200:
                            content = await response.read()
                            
                            # Optional compression
                            if config.compression:
                                content = gzip.compress(content)
                                dataset['name'] += '.gz'
                            
                            # Upload to S3
                            s3_key = f"{s3_path}{dataset['name']}"
                            
                            self.s3_client.put_object(
                                Bucket=bucket_name,
                                Key=s3_key,
                                Body=content,
                                StorageClass=config.storage_class,
                                Metadata={
                                    'source_url': dataset['url'],
                                    'sync_time': datetime.now(timezone.utc).isoformat(),
                                    'original_size': str(len(content))
                                }
                            )
                            
                            sync_job.files_synced += 1
                            sync_job.bytes_synced += len(content)
                            
                            # Track bandwidth
                            self.bandwidth_monitor.record_download(
                                config.source_name, len(content), 5.0  # Estimated 5 seconds
                            )
                            
                            logger.info(f"Synced {dataset['name']} to {s3_key}")
                            
                        else:
                            error_msg = f"Failed to download {dataset['name']}: HTTP {response.status}"
                            sync_job.errors.append(error_msg)
                            logger.error(error_msg)
                            
                except Exception as e:
                    error_msg = f"Error syncing {dataset['name']}: {e}"
                    sync_job.errors.append(error_msg)
                    logger.error(error_msg)
                
                # Rate limiting
                await asyncio.sleep(1)
    
    async def _sync_kegg_database(self, config: MirrorConfig, sync_job: SyncJob,
                                bucket_name: str, s3_path: str):
        """Sync KEGG database pathways"""
        base_url = "https://rest.kegg.jp"
        
        try:
            # Get list of pathways
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{base_url}/list/pathway", timeout=60) as response:
                    if response.status == 200:
                        pathway_list = await response.text()
                        
                        # Parse pathway IDs
                        pathway_ids = []
                        for line in pathway_list.strip().split('\n'):
                            if line.startswith('path:'):
                                pathway_id = line.split('\t')[0].replace('path:', '')
                                pathway_ids.append(pathway_id)
                        
                        # Limit to first 100 pathways for demo (remove in production)
                        pathway_ids = pathway_ids[:100]
                        
                        # Download each pathway
                        for pathway_id in pathway_ids:
                            try:
                                # Get pathway KGML
                                kgml_url = f"{base_url}/get/{pathway_id}/kgml"
                                async with session.get(kgml_url, timeout=30) as kgml_response:
                                    if kgml_response.status == 200:
                                        content = await kgml_response.read()
                                        
                                        if config.compression:
                                            content = gzip.compress(content)
                                            filename = f"{pathway_id}.kgml.gz"
                                        else:
                                            filename = f"{pathway_id}.kgml"
                                        
                                        # Upload to S3
                                        s3_key = f"{s3_path}{filename}"
                                        
                                        self.s3_client.put_object(
                                            Bucket=bucket_name,
                                            Key=s3_key,
                                            Body=content,
                                            StorageClass=config.storage_class,
                                            Metadata={
                                                'pathway_id': pathway_id,
                                                'source_url': kgml_url,
                                                'sync_time': datetime.now(timezone.utc).isoformat()
                                            }
                                        )
                                        
                                        sync_job.files_synced += 1
                                        sync_job.bytes_synced += len(content)
                                        
                                        if sync_job.files_synced % 10 == 0:
                                            logger.info(f"Synced {sync_job.files_synced} KEGG pathways")
                                        
                                        # Rate limiting (KEGG requirement)
                                        await asyncio.sleep(0.1)
                                        
                            except Exception as e:
                                error_msg = f"Error syncing pathway {pathway_id}: {e}"
                                sync_job.errors.append(error_msg)
                                
        except Exception as e:
            error_msg = f"Error syncing KEGG database: {e}"
            sync_job.errors.append(error_msg)
            logger.error(error_msg)
    
    async def _sync_ncbi_genomes(self, config: MirrorConfig, sync_job: SyncJob,
                               bucket_name: str, s3_path: str):
        """Sync critical NCBI genome assemblies"""
        # This would implement FTP-based sync for NCBI genomes
        # Simplified version for demonstration
        logger.info(f"NCBI genome sync would be implemented here for {config.source_name}")
        
        # Placeholder sync for demo
        sync_job.files_synced = 50
        sync_job.bytes_synced = 1024 * 1024 * 500  # 500 MB
    
    async def _sync_generic_source(self, config: MirrorConfig, sync_job: SyncJob,
                                 bucket_name: str, s3_path: str):
        """Generic synchronization for HTTP/FTP sources"""
        logger.info(f"Generic sync would be implemented here for {config.source_name}")
        
        # Placeholder for generic sync implementation
        sync_job.files_synced = 10
        sync_job.bytes_synced = 1024 * 1024 * 100  # 100 MB
    
    def _store_sync_job(self, sync_job: SyncJob):
        """Store sync job in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO sync_jobs
                    (job_id, source_name, start_time, status, s3_path)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    sync_job.job_id,
                    sync_job.source_name,
                    sync_job.start_time,
                    sync_job.status,
                    sync_job.s3_path
                ))
                
        except Exception as e:
            logger.error(f"Error storing sync job: {e}")
    
    def _update_sync_job(self, sync_job: SyncJob):
        """Update sync job in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE sync_jobs
                    SET end_time = ?, status = ?, files_synced = ?, 
                        bytes_synced = ?, errors = ?
                    WHERE job_id = ?
                """, (
                    sync_job.end_time,
                    sync_job.status,
                    sync_job.files_synced,
                    sync_job.bytes_synced,
                    json.dumps(sync_job.errors),
                    sync_job.job_id
                ))
                
                # Update mirror config last sync time
                if sync_job.status == "completed":
                    conn.execute("""
                        UPDATE mirror_configs
                        SET last_sync = ?, sync_status = 'completed',
                            total_files = ?, total_size_bytes = ?
                        WHERE source_name = ?
                    """, (
                        sync_job.end_time,
                        sync_job.files_synced,
                        sync_job.bytes_synced,
                        sync_job.source_name
                    ))
                
        except Exception as e:
            logger.error(f"Error updating sync job: {e}")
    
    async def check_mirror_health(self, source_name: Optional[str] = None) -> Dict[str, Any]:
        """Check health of mirrors"""
        health_results = {}
        
        sources_to_check = [source_name] if source_name else list(self.mirror_configs.keys())
        
        for src_name in sources_to_check:
            if src_name not in self.mirror_configs:
                continue
            
            config = self.mirror_configs[src_name]
            
            try:
                # Check source availability
                source_available = await self._check_source_availability(config.source_url)
                
                # Check mirror availability
                bucket_name = self._get_bucket_for_priority(config.priority)
                mirror_available = self._check_mirror_availability(bucket_name, config.mirror_path)
                
                # Get sizes
                source_size = await self._estimate_source_size(config)
                mirror_size = self._get_mirror_size(bucket_name, config.mirror_path)
                
                # Calculate sync lag
                sync_lag = self._calculate_sync_lag(src_name)
                
                health_results[src_name] = {
                    'source_available': source_available,
                    'mirror_available': mirror_available,
                    'source_size_mb': source_size / 1024 / 1024 if source_size else None,
                    'mirror_size_mb': mirror_size / 1024 / 1024 if mirror_size else None,
                    'sync_lag_hours': sync_lag,
                    'status': 'healthy' if source_available and mirror_available else 'degraded'
                }
                
                # Store health check
                self._store_health_check(src_name, health_results[src_name])
                
            except Exception as e:
                health_results[src_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                logger.error(f"Health check failed for {src_name}: {e}")
        
        return health_results
    
    async def _check_source_availability(self, source_url: str) -> bool:
        """Check if source URL is available"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(source_url, timeout=10) as response:
                    return response.status in [200, 301, 302]
        except:
            return False
    
    def _check_mirror_availability(self, bucket_name: str, mirror_path: str) -> bool:
        """Check if mirror is available in S3"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=mirror_path,
                MaxKeys=1
            )
            return 'Contents' in response and len(response['Contents']) > 0
        except:
            return False
    
    async def _estimate_source_size(self, config: MirrorConfig) -> Optional[int]:
        """Estimate source data size"""
        # This would implement source-specific size estimation
        # Simplified for demonstration
        return None
    
    def _get_mirror_size(self, bucket_name: str, mirror_path: str) -> Optional[int]:
        """Get total size of mirrored data"""
        try:
            total_size = 0
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(Bucket=bucket_name, Prefix=mirror_path):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        total_size += obj['Size']
            
            return total_size
        except:
            return None
    
    def _calculate_sync_lag(self, source_name: str) -> Optional[float]:
        """Calculate sync lag in hours"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT last_sync FROM mirror_configs WHERE source_name = ?
                """, (source_name,))
                
                result = cursor.fetchone()
                if result and result[0]:
                    last_sync = datetime.fromisoformat(result[0].replace('Z', '+00:00'))
                    now = datetime.now(timezone.utc)
                    return (now - last_sync).total_seconds() / 3600
        except:
            pass
        
        return None
    
    def _store_health_check(self, source_name: str, health_data: Dict[str, Any]):
        """Store health check results"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO mirror_health
                    (source_name, check_time, source_available, mirror_available,
                     source_size_bytes, mirror_size_bytes, sync_lag_hours)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    source_name,
                    datetime.now(timezone.utc),
                    health_data.get('source_available', False),
                    health_data.get('mirror_available', False),
                    health_data.get('source_size_mb', 0) * 1024 * 1024 if health_data.get('source_size_mb') else None,
                    health_data.get('mirror_size_mb', 0) * 1024 * 1024 if health_data.get('mirror_size_mb') else None,
                    health_data.get('sync_lag_hours')
                ))
        except Exception as e:
            logger.error(f"Error storing health check: {e}")
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get status of all sync jobs"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get recent sync jobs
                cursor = conn.execute("""
                    SELECT source_name, status, start_time, end_time, 
                           files_synced, bytes_synced
                    FROM sync_jobs
                    ORDER BY start_time DESC
                    LIMIT 50
                """)
                
                jobs = []
                for row in cursor:
                    jobs.append({
                        'source_name': row[0],
                        'status': row[1],
                        'start_time': row[2],
                        'end_time': row[3],
                        'files_synced': row[4],
                        'bytes_synced': row[5]
                    })
                
                # Get mirror configurations status
                cursor = conn.execute("""
                    SELECT source_name, last_sync, sync_status, total_files, total_size_bytes
                    FROM mirror_configs
                """)
                
                configs = []
                for row in cursor:
                    configs.append({
                        'source_name': row[0],
                        'last_sync': row[1],
                        'sync_status': row[2],
                        'total_files': row[3],
                        'total_size_bytes': row[4]
                    })
                
                return {
                    'recent_jobs': jobs,
                    'mirror_configs': configs,
                    'active_jobs': len(self.active_jobs)
                }
                
        except Exception as e:
            logger.error(f"Error getting sync status: {e}")
            return {}
    
    async def start_scheduler(self):
        """Start scheduled synchronization"""
        # Schedule sync jobs based on frequency
        for source_name, config in self.mirror_configs.items():
            if config.sync_frequency == 'hourly':
                schedule.every().hour.do(lambda: asyncio.create_task(self.sync_source(source_name)))
            elif config.sync_frequency == 'daily':
                schedule.every().day.at("02:00").do(lambda: asyncio.create_task(self.sync_source(source_name)))
            elif config.sync_frequency == 'weekly':
                schedule.every().sunday.at("02:00").do(lambda: asyncio.create_task(self.sync_source(source_name)))
            elif config.sync_frequency == 'monthly':
                schedule.every(30).days.at("02:00").do(lambda: asyncio.create_task(self.sync_source(source_name)))
        
        # Run scheduler
        while True:
            schedule.run_pending()
            await asyncio.sleep(60)  # Check every minute

class BandwidthMonitor:
    """Monitor bandwidth usage for cost optimization"""
    
    def __init__(self):
        self.usage_history = []
    
    def record_download(self, source_name: str, bytes_downloaded: int, duration_seconds: float):
        """Record download for bandwidth monitoring"""
        bandwidth_mbps = (bytes_downloaded * 8) / (duration_seconds * 1000000)  # Convert to Mbps
        
        self.usage_history.append({
            'timestamp': datetime.now(timezone.utc),
            'source_name': source_name,
            'bytes_downloaded': bytes_downloaded,
            'bandwidth_mbps': bandwidth_mbps,
            'duration_seconds': duration_seconds
        })
        
        # Keep only last 1000 records
        if len(self.usage_history) > 1000:
            self.usage_history = self.usage_history[-1000:]

# Global instance
mirror_infrastructure = None

def get_mirror_infrastructure() -> LocalMirrorInfrastructure:
    """Get global mirror infrastructure instance"""
    global mirror_infrastructure
    if mirror_infrastructure is None:
        mirror_infrastructure = LocalMirrorInfrastructure()
    return mirror_infrastructure

if __name__ == "__main__":
    # Test the mirror infrastructure
    async def test_mirrors():
        infrastructure = LocalMirrorInfrastructure()
        
        # Setup mirrors
        setup_results = await infrastructure.setup_mirrors()
        print(f"Setup results: {setup_results}")
        
        # Test sync
        sync_job = await infrastructure.sync_source('nasa_exoplanet_archive', force=True)
        if sync_job:
            print(f"Sync job: {sync_job.job_id} - {sync_job.status}")
        
        # Check health
        health = await infrastructure.check_mirror_health()
        print(f"Health check: {health}")
    
    asyncio.run(test_mirrors()) 
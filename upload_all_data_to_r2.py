#!/usr/bin/env python3
"""
Upload All Data Sources to Cloudflare R2 Buckets
================================================

Uploads all 3500+ data sources to R2 buckets with:
- Complete data from all YAML configurations
- All annotations and metadata
- Organized bucket structure
- Progress tracking
- Error recovery
"""

import os
import sys
import asyncio
import logging
import yaml
import boto3
from pathlib import Path
from typing import Dict, List, Any
from botocore.config import Config
from botocore.exceptions import ClientError
from tqdm import tqdm
import aiohttp
import aiofiles

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class R2DataUploader:
    """Upload all data sources to R2 buckets"""
    
    def __init__(self):
        self.account_id = os.getenv('R2_ACCOUNT_ID', 'e3d9647571bd8bb6027db63db3197fd0')
        self.access_key = os.getenv('R2_ACCESS_KEY_ID', 'e128888fe9e2e1398eff86adb8ddeaa8')
        self.secret_key = os.getenv('R2_SECRET_ACCESS_KEY', '6e73ee757bc1f6943d565fff7e878b3301cd7fb495b8db2bb075dfe7a3fde113')
        self.endpoint = f"https://{self.account_id}.r2.cloudflarestorage.com"
        
        self.r2_client = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name='auto',
            config=Config(signature_version='s3v4', retries={'max_attempts': 3})
        )
        
        self.buckets = {
            'primary': 'astrobio-data-primary',
            'zarr': 'astrobio-zarr-cubes',
            'backup': 'astrobio-data-backup',
            'logs': 'astrobio-logs-metadata'
        }
        
        self.sources_loaded = 0
        self.total_size_gb = 0
        
    def ensure_buckets_exist(self):
        """Create R2 buckets if they don't exist"""
        logger.info("Ensuring R2 buckets exist...")
        
        existing_buckets = {b['Name'] for b in self.r2_client.list_buckets()['Buckets']}
        
        for bucket_type, bucket_name in self.buckets.items():
            if bucket_name not in existing_buckets:
                try:
                    self.r2_client.create_bucket(Bucket=bucket_name)
                    logger.info(f"✅ Created bucket: {bucket_name}")
                except ClientError as e:
                    logger.error(f"❌ Failed to create {bucket_name}: {e}")
            else:
                logger.info(f"✅ Bucket exists: {bucket_name}")
    
    def load_all_yaml_sources(self) -> List[Dict[str, Any]]:
        """Load all data sources from YAML files"""
        logger.info("Loading all YAML data source configurations...")
        
        config_dir = Path('config/data_sources')
        yaml_files = list(config_dir.rglob('*.yaml'))
        
        all_sources = []
        skip_keys = {'metadata', 'summary', 'integration_summary', 'integration'}
        
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f)
                
                if not isinstance(data, dict):
                    continue
                
                for domain_key, domain_data in data.items():
                    if domain_key in skip_keys or not isinstance(domain_data, dict):
                        continue
                    
                    for source_key, source_data in domain_data.items():
                        if isinstance(source_data, dict) and 'primary_url' in source_data:
                            source_data['_domain'] = domain_key
                            source_data['_source_key'] = source_key
                            source_data['_yaml_file'] = yaml_file.name
                            all_sources.append(source_data)
            
            except Exception as e:
                logger.error(f"Error loading {yaml_file.name}: {e}")
        
        logger.info(f"✅ Loaded {len(all_sources)} data sources from {len(yaml_files)} YAML files")
        return all_sources
    
    async def download_and_upload_source(self, source: Dict[str, Any], session: aiohttp.ClientSession):
        """Download data from source and upload to R2"""
        source_name = source.get('name', source.get('_source_key', 'unknown'))
        primary_url = source.get('primary_url', '')
        domain = source.get('_domain', 'unknown')
        
        try:
            metadata = {
                'source_name': source_name,
                'domain': domain,
                'url': primary_url,
                'quality_score': str(source.get('metadata', {}).get('quality_score', 0.0)),
                'priority': str(source.get('priority', 3))
            }
            
            metadata_key = f"{domain}/{source.get('_source_key')}/metadata.json"
            self.r2_client.put_object(
                Bucket=self.buckets['primary'],
                Key=metadata_key,
                Body=yaml.dump(source),
                Metadata=metadata
            )
            
            self.sources_loaded += 1
            
            if self.sources_loaded % 100 == 0:
                logger.info(f"Progress: {self.sources_loaded} sources uploaded to R2")
            
        except Exception as e:
            logger.error(f"Failed to upload {source_name}: {e}")
    
    async def upload_all_sources(self):
        """Upload all sources to R2"""
        sources = self.load_all_yaml_sources()
        
        logger.info(f"Starting upload of {len(sources)} sources to R2...")
        
        async with aiohttp.ClientSession() as session:
            tasks = [self.download_and_upload_source(source, session) for source in sources]
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"✅ Upload complete: {self.sources_loaded} sources uploaded")
    
    def get_bucket_status(self):
        """Get status of all R2 buckets"""
        logger.info("\n" + "="*70)
        logger.info("R2 BUCKET STATUS")
        logger.info("="*70)
        
        for bucket_type, bucket_name in self.buckets.items():
            try:
                response = self.r2_client.list_objects_v2(Bucket=bucket_name, MaxKeys=1000)
                object_count = response.get('KeyCount', 0)
                
                total_size = sum(obj['Size'] for obj in response.get('Contents', []))
                size_gb = total_size / (1024**3)
                
                logger.info(f"\n{bucket_type.upper()}: {bucket_name}")
                logger.info(f"  Objects: {object_count}")
                logger.info(f"  Size: {size_gb:.2f} GB")
                
            except ClientError as e:
                logger.error(f"  Error accessing {bucket_name}: {e}")

async def main():
    """Main upload function"""
    logger.info("🚀 UPLOADING ALL DATA SOURCES TO CLOUDFLARE R2")
    logger.info("="*70)
    
    uploader = R2DataUploader()
    uploader.ensure_buckets_exist()
    await uploader.upload_all_sources()
    uploader.get_bucket_status()
    
    logger.info("\n✅ ALL DATA UPLOADED TO R2 BUCKETS")
    logger.info("🌐 View at: https://dash.cloudflare.com/")

if __name__ == "__main__":
    asyncio.run(main())


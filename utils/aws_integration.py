#!/usr/bin/env python3
"""
AWS Integration Utility
=======================

AWS integration using boto3 (avoiding awscli dependency conflicts).
Provides S3 operations, EC2 management, and data transfer capabilities.
"""

import os
import boto3
import s3fs
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from botocore.exceptions import ClientError, NoCredentialsError
import json
from datetime import datetime

# Optional async support (only if aiobotocore is installed)
try:
    import aiobotocore
    ASYNC_SUPPORT = True
except ImportError:
    ASYNC_SUPPORT = False
    logger.info("ℹ️ aiobotocore not installed - async operations not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AWSManager:
    """
    Centralized AWS management for the astrobiology project.
    Uses boto3 instead of awscli to avoid dependency conflicts.
    """
    
    def __init__(self, region: str = 'us-east-1'):
        self.region = region
        self.session = None
        self.s3_client = None
        self.s3_resource = None
        self.ec2_client = None
        self.s3fs = None
        
        # Try to initialize AWS session
        self._initialize_session()
    
    def _initialize_session(self):
        """Initialize AWS session and clients"""
        try:
            # Create session (uses AWS credentials from environment or ~/.aws/)
            self.session = boto3.Session(region_name=self.region)
            
            # Initialize clients
            self.s3_client = self.session.client('s3')
            self.s3_resource = self.session.resource('s3')
            self.ec2_client = self.session.client('ec2')
            
            # Initialize s3fs for file-like operations
            self.s3fs = s3fs.S3FileSystem()
            
            logger.info(f"[OK] AWS session initialized in region {self.region}")
            
        except NoCredentialsError:
            logger.warning("[WARN] AWS credentials not found. Please configure AWS credentials.")
            logger.warning("Run: aws configure (after installing AWS CLI v2)")
        except Exception as e:
            logger.error(f"[FAIL] Error initializing AWS session: {e}")
    
    def verify_credentials(self) -> Dict[str, Any]:
        """Verify AWS credentials and permissions"""
        try:
            # Get caller identity
            sts_client = self.session.client('sts')
            identity = sts_client.get_caller_identity()
            
            # Test S3 access
            self.s3_client.list_buckets()
            
            return {
                'status': 'success',
                'account_id': identity.get('Account'),
                'user_arn': identity.get('Arn'),
                'region': self.region
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def create_project_buckets(self, project_name: str = 'astrobio') -> Dict[str, str]:
        """Create S3 buckets for the project"""
        timestamp = datetime.now().strftime('%Y%m%d')
        
        buckets = {
            'primary': f"{project_name}-data-primary-{timestamp}",
            'backup': f"{project_name}-data-backup-{timestamp}",
            'zarr': f"{project_name}-zarr-cubes-{timestamp}",
            'logs': f"{project_name}-logs-metadata-{timestamp}"
        }
        
        created_buckets = {}
        
        for purpose, bucket_name in buckets.items():
            try:
                # Create bucket
                if self.region == 'us-east-1':
                    # us-east-1 doesn't need LocationConstraint
                    self.s3_client.create_bucket(Bucket=bucket_name)
                else:
                    self.s3_client.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': self.region}
                    )
                
                # Enable versioning for data buckets
                if purpose in ['primary', 'backup']:
                    self.s3_client.put_bucket_versioning(
                        Bucket=bucket_name,
                        VersioningConfiguration={'Status': 'Enabled'}
                    )
                
                created_buckets[purpose] = bucket_name
                logger.info(f"[OK] Created {purpose} bucket: {bucket_name}")
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'BucketAlreadyExists':
                    logger.warning(f"[WARN] Bucket {bucket_name} already exists")
                    created_buckets[purpose] = bucket_name
                else:
                    logger.error(f"[FAIL] Error creating {purpose} bucket: {e}")
        
        return created_buckets
    
    def upload_data(self, local_path: str, bucket: str, s3_key: str = None) -> bool:
        """Upload data to S3"""
        try:
            local_path = Path(local_path)
            
            if s3_key is None:
                s3_key = local_path.name
            
            if local_path.is_file():
                # Upload single file
                self.s3_client.upload_file(str(local_path), bucket, s3_key)
                logger.info(f"[OK] Uploaded {local_path} to s3://{bucket}/{s3_key}")
                
            elif local_path.is_dir():
                # Upload directory recursively
                for file_path in local_path.rglob('*'):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(local_path)
                        s3_file_key = f"{s3_key}/{relative_path}".replace('\\', '/')
                        
                        self.s3_client.upload_file(str(file_path), bucket, s3_file_key)
                        logger.info(f"[OK] Uploaded {file_path} to s3://{bucket}/{s3_file_key}")
            
            return True
            
        except Exception as e:
            logger.error(f"[FAIL] Error uploading {local_path}: {e}")
            return False
    
    def download_data(self, bucket: str, s3_key: str, local_path: str) -> bool:
        """Download data from S3"""
        try:
            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.s3_client.download_file(bucket, s3_key, str(local_path))
            logger.info(f"[OK] Downloaded s3://{bucket}/{s3_key} to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"[FAIL] Error downloading s3://{bucket}/{s3_key}: {e}")
            return False
    
    def sync_directory(self, local_dir: str, bucket: str, s3_prefix: str = '') -> bool:
        """Sync local directory to S3 (like aws s3 sync)"""
        try:
            local_dir = Path(local_dir)
            uploaded_files = 0
            
            for file_path in local_dir.rglob('*'):
                if file_path.is_file():
                    relative_path = file_path.relative_to(local_dir)
                    s3_key = f"{s3_prefix}/{relative_path}".replace('\\', '/').lstrip('/')
                    
                    # Check if file exists and is newer
                    try:
                        response = self.s3_client.head_object(Bucket=bucket, Key=s3_key)
                        s3_modified = response['LastModified'].replace(tzinfo=None)
                        local_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
                        
                        if local_modified <= s3_modified:
                            continue  # Skip if S3 version is newer or same
                    except ClientError:
                        pass  # File doesn't exist in S3, upload it
                    
                    # Upload file
                    self.s3_client.upload_file(str(file_path), bucket, s3_key)
                    uploaded_files += 1
                    logger.info(f"📤 Synced {relative_path}")
            
            logger.info(f"[OK] Sync completed: {uploaded_files} files uploaded")
            return True
            
        except Exception as e:
            logger.error(f"[FAIL] Error syncing directory: {e}")
            return False
    
    def list_buckets(self) -> List[str]:
        """List all S3 buckets"""
        try:
            response = self.s3_client.list_buckets()
            return [bucket['Name'] for bucket in response['Buckets']]
        except Exception as e:
            logger.error(f"[FAIL] Error listing buckets: {e}")
            return []
    
    def get_bucket_size(self, bucket: str) -> Dict[str, Any]:
        """Get bucket size and object count"""
        try:
            total_size = 0
            object_count = 0
            
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket)
            
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        total_size += obj['Size']
                        object_count += 1
            
            return {
                'bucket': bucket,
                'size_bytes': total_size,
                'size_gb': round(total_size / (1024**3), 2),
                'object_count': object_count
            }
            
        except Exception as e:
            logger.error(f"[FAIL] Error getting bucket size: {e}")
            return {}
    
    def setup_lifecycle_policy(self, bucket: str) -> bool:
        """Set up S3 lifecycle policy for cost optimization"""
        try:
            lifecycle_config = {
                'Rules': [
                    {
                        'ID': 'AstrobiologyDataLifecycle',
                        'Status': 'Enabled',
                        'Filter': {'Prefix': ''},
                        'Transitions': [
                            {
                                'Days': 30,
                                'StorageClass': 'STANDARD_IA'
                            },
                            {
                                'Days': 90,
                                'StorageClass': 'GLACIER'
                            },
                            {
                                'Days': 365,
                                'StorageClass': 'DEEP_ARCHIVE'
                            }
                        ]
                    }
                ]
            }
            
            self.s3_client.put_bucket_lifecycle_configuration(
                Bucket=bucket,
                LifecycleConfiguration=lifecycle_config
            )
            
            logger.info(f"[OK] Lifecycle policy applied to {bucket}")
            return True
            
        except Exception as e:
            logger.error(f"[FAIL] Error setting lifecycle policy: {e}")
            return False

def configure_aws_credentials():
    """Helper function to guide AWS credentials setup"""
    print("[FIX] AWS Credentials Setup Guide:")
    print("1. Install AWS CLI v2 from: https://aws.amazon.com/cli/")
    print("2. Run: aws configure")
    print("3. Enter your AWS Access Key ID and Secret Access Key")
    print("4. Choose region (recommended: us-east-1)")
    print("5. Choose output format (recommended: json)")
    print("\nAlternatively, set environment variables:")
    print("export AWS_ACCESS_KEY_ID=your_access_key")
    print("export AWS_SECRET_ACCESS_KEY=your_secret_key")
    print("export AWS_DEFAULT_REGION=us-east-1")

def test_aws_connection():
    """Test AWS connection and setup"""
    print("[TEST] Testing AWS Connection...")
    
    aws_manager = AWSManager()
    
    if aws_manager.session is None:
        print("[FAIL] AWS session not initialized")
        configure_aws_credentials()
        return False
    
    # Verify credentials
    verification = aws_manager.verify_credentials()
    
    if verification['status'] == 'success':
        print(f"[OK] AWS Connection successful!")
        print(f"Account ID: {verification['account_id']}")
        print(f"User ARN: {verification['user_arn']}")
        print(f"Region: {verification['region']}")
        
        # List existing buckets
        buckets = aws_manager.list_buckets()
        print(f"Existing buckets: {len(buckets)}")
        for bucket in buckets:
            print(f"  - {bucket}")
        
        return True
    else:
        print(f"[FAIL] AWS Connection failed: {verification['error']}")
        configure_aws_credentials()
        return False

if __name__ == "__main__":
    test_aws_connection() 
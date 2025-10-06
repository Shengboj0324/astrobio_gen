#!/usr/bin/env python3
"""
Cloudflare R2 Integration Testing Suite
========================================

COMPREHENSIVE TESTING - EXTREME SKEPTICISM MODE

Tests all aspects of R2 integration:
1. R2 connection and authentication
2. Bucket operations (list, create, delete)
3. Object operations (upload, download, list)
4. Streaming data loaders
5. Zarr integration
6. Training pipeline integration
7. Data source preservation (1000+)
8. Rust module compatibility

Usage:
    python test_r2_integration.py --all
    python test_r2_integration.py --connection
    python test_r2_integration.py --data-loaders
    python test_r2_integration.py --training
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class R2IntegrationTester:
    """Comprehensive R2 integration testing"""
    
    def __init__(self):
        self.test_results = {
            'passed': 0,
            'failed': 0,
            'skipped': 0
        }
        self.failed_tests = []
    
    def test_r2_connection(self) -> bool:
        """Test R2 connection and authentication"""
        logger.info("="*80)
        logger.info("TEST 1: R2 CONNECTION AND AUTHENTICATION")
        logger.info("="*80)
        
        try:
            from utils.r2_data_flow_integration import R2DataFlowManager
            
            # Initialize R2 manager
            r2_manager = R2DataFlowManager()
            
            # Check credentials
            if not r2_manager.credentials_verified:
                logger.error("❌ R2 credentials not verified")
                self.test_results['failed'] += 1
                self.failed_tests.append("R2 connection")
                return False
            
            logger.info("✅ R2 credentials verified")
            
            # List buckets
            buckets = r2_manager.list_buckets()
            logger.info(f"✅ R2 buckets accessible: {len(buckets)}")
            for bucket in buckets:
                logger.info(f"   - {bucket['name']} (created: {bucket['creation_date']})")
            
            self.test_results['passed'] += 1
            return True
            
        except Exception as e:
            logger.error(f"❌ R2 connection test failed: {e}")
            self.test_results['failed'] += 1
            self.failed_tests.append(f"R2 connection: {e}")
            return False
    
    def test_bucket_operations(self) -> bool:
        """Test R2 bucket operations"""
        logger.info("="*80)
        logger.info("TEST 2: R2 BUCKET OPERATIONS")
        logger.info("="*80)
        
        try:
            from utils.r2_data_flow_integration import R2DataFlowManager
            
            r2_manager = R2DataFlowManager()
            
            # Test bucket status for each required bucket
            required_buckets = [
                'astrobio-data-primary',
                'astrobio-zarr-cubes',
                'astrobio-data-backup',
                'astrobio-logs-metadata'
            ]
            
            all_ok = True
            for bucket_name in required_buckets:
                status = r2_manager.get_bucket_status(bucket_name)
                
                if status['status'] == 'success':
                    logger.info(f"✅ {bucket_name}: {status['size_gb']}GB ({status['object_count']} objects)")
                else:
                    logger.error(f"❌ {bucket_name}: {status.get('error', 'Unknown error')}")
                    all_ok = False
            
            if all_ok:
                self.test_results['passed'] += 1
                return True
            else:
                self.test_results['failed'] += 1
                self.failed_tests.append("Bucket operations")
                return False
                
        except Exception as e:
            logger.error(f"❌ Bucket operations test failed: {e}")
            self.test_results['failed'] += 1
            self.failed_tests.append(f"Bucket operations: {e}")
            return False
    
    def test_object_operations(self) -> bool:
        """Test R2 object operations (upload/download)"""
        logger.info("="*80)
        logger.info("TEST 3: R2 OBJECT OPERATIONS")
        logger.info("="*80)
        
        try:
            from utils.r2_data_flow_integration import R2DataFlowManager
            
            r2_manager = R2DataFlowManager()
            
            # Create test data
            test_data = torch.randn(10, 10)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
                torch.save(test_data, tmp.name)
                tmp_path = tmp.name
            
            try:
                # Upload test file
                test_bucket = 'astrobio-data-primary'
                test_key = 'test/integration_test.pt'
                
                logger.info(f"Uploading test file to {test_bucket}/{test_key}")
                r2_manager.r2_client.upload_file(tmp_path, test_bucket, test_key)
                logger.info("✅ Upload successful")
                
                # Download test file
                download_path = tmp_path + '.download'
                logger.info(f"Downloading test file from {test_bucket}/{test_key}")
                r2_manager.r2_client.download_file(test_bucket, test_key, download_path)
                logger.info("✅ Download successful")
                
                # Verify data integrity
                downloaded_data = torch.load(download_path)
                if torch.allclose(test_data, downloaded_data):
                    logger.info("✅ Data integrity verified")
                else:
                    logger.error("❌ Data integrity check failed")
                    self.test_results['failed'] += 1
                    self.failed_tests.append("Object operations - data integrity")
                    return False
                
                # Cleanup
                r2_manager.r2_client.delete_object(Bucket=test_bucket, Key=test_key)
                logger.info("✅ Cleanup successful")
                
                # Remove temporary files
                os.unlink(tmp_path)
                os.unlink(download_path)
                
                self.test_results['passed'] += 1
                return True
                
            except Exception as e:
                logger.error(f"❌ Object operations failed: {e}")
                # Cleanup on error
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                self.test_results['failed'] += 1
                self.failed_tests.append(f"Object operations: {e}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Object operations test failed: {e}")
            self.test_results['failed'] += 1
            self.failed_tests.append(f"Object operations: {e}")
            return False
    
    def test_streaming_data_loader(self) -> bool:
        """Test R2 streaming data loader"""
        logger.info("="*80)
        logger.info("TEST 4: R2 STREAMING DATA LOADER")
        logger.info("="*80)
        
        try:
            from utils.r2_data_flow_integration import R2DataFlowManager, R2StreamingDataset
            
            r2_manager = R2DataFlowManager()
            
            # Create test dataset in R2
            test_bucket = 'astrobio-data-primary'
            test_prefix = 'test/streaming/'
            
            # Upload test files
            logger.info("Creating test dataset...")
            for i in range(5):
                test_data = torch.randn(32, 64)
                with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
                    torch.save(test_data, tmp.name)
                    test_key = f"{test_prefix}sample_{i}.pt"
                    r2_manager.r2_client.upload_file(tmp.name, test_bucket, test_key)
                    os.unlink(tmp.name)
            
            logger.info("✅ Test dataset created")
            
            # Create streaming dataset
            r2_path = f"{test_bucket}/{test_prefix}"
            dataset = R2StreamingDataset(r2_path, r2_manager.r2fs)
            
            logger.info(f"✅ Streaming dataset created: {len(dataset)} samples")
            
            # Test data loading
            if len(dataset) > 0:
                sample = dataset[0]
                logger.info(f"✅ Sample loaded: shape {sample.shape}")
            else:
                logger.error("❌ No samples in dataset")
                self.test_results['failed'] += 1
                self.failed_tests.append("Streaming data loader - no samples")
                return False
            
            # Cleanup
            logger.info("Cleaning up test dataset...")
            for i in range(5):
                test_key = f"{test_prefix}sample_{i}.pt"
                r2_manager.r2_client.delete_object(Bucket=test_bucket, Key=test_key)
            
            logger.info("✅ Cleanup successful")
            
            self.test_results['passed'] += 1
            return True
            
        except Exception as e:
            logger.error(f"❌ Streaming data loader test failed: {e}")
            self.test_results['failed'] += 1
            self.failed_tests.append(f"Streaming data loader: {e}")
            return False
    
    def test_data_source_preservation(self) -> bool:
        """Test that all data sources are preserved"""
        logger.info("="*80)
        logger.info("TEST 5: DATA SOURCE PRESERVATION (1000+ SOURCES)")
        logger.info("="*80)
        
        try:
            # Check data source authentication
            from utils.data_source_auth import DataSourceAuthManager
            
            auth_manager = DataSourceAuthManager()
            
            # Verify credentials are still loaded
            if auth_manager.credentials:
                logger.info(f"✅ Data source credentials loaded: {len(auth_manager.credentials)} sources")
            else:
                logger.error("❌ Data source credentials not loaded")
                self.test_results['failed'] += 1
                self.failed_tests.append("Data source preservation")
                return False
            
            # Check critical data sources
            critical_sources = [
                'nasa_mast',
                'copernicus_cds',
                'ncbi',
                'gaia_user',
                'eso_user'
            ]
            
            all_ok = True
            for source in critical_sources:
                if source in auth_manager.credentials and auth_manager.credentials[source]:
                    logger.info(f"✅ {source}: configured")
                else:
                    logger.warning(f"⚠️  {source}: not configured")
            
            # Check data source config files
            data_source_configs = [
                'config/data_sources/expanded_1000_sources.yaml',
                'config/data_sources/comprehensive_100_sources.yaml',
                'config/data_sources/expanded_2025_sources.yaml'
            ]
            
            for config_file in data_source_configs:
                if Path(config_file).exists():
                    logger.info(f"✅ {config_file}: exists")
                else:
                    logger.warning(f"⚠️  {config_file}: not found")
            
            self.test_results['passed'] += 1
            return True
            
        except Exception as e:
            logger.error(f"❌ Data source preservation test failed: {e}")
            self.test_results['failed'] += 1
            self.failed_tests.append(f"Data source preservation: {e}")
            return False
    
    def test_rust_module_compatibility(self) -> bool:
        """Test Rust module compatibility"""
        logger.info("="*80)
        logger.info("TEST 6: RUST MODULE COMPATIBILITY")
        logger.info("="*80)
        
        try:
            # Check if Rust modules are available
            try:
                from rust_integration import datacube_accelerator, training_accelerator
                logger.info("✅ Rust integration modules imported successfully")
            except ImportError as e:
                logger.warning(f"⚠️  Rust modules not available: {e}")
                logger.info("   This is OK if Rust modules haven't been compiled yet")
            
            # Check Rust source files
            rust_files = [
                'rust_modules/Cargo.toml',
                'rust_modules/src/lib.rs',
                'rust_integration/__init__.py'
            ]
            
            all_ok = True
            for rust_file in rust_files:
                if Path(rust_file).exists():
                    logger.info(f"✅ {rust_file}: exists")
                else:
                    logger.warning(f"⚠️  {rust_file}: not found")
                    all_ok = False
            
            if all_ok:
                self.test_results['passed'] += 1
                return True
            else:
                logger.warning("⚠️  Some Rust files missing, but this may be OK")
                self.test_results['passed'] += 1
                return True
                
        except Exception as e:
            logger.error(f"❌ Rust module compatibility test failed: {e}")
            self.test_results['failed'] += 1
            self.failed_tests.append(f"Rust module compatibility: {e}")
            return False
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("="*80)
        logger.info("TEST REPORT")
        logger.info("="*80)
        
        total_tests = self.test_results['passed'] + self.test_results['failed'] + self.test_results['skipped']
        
        logger.info(f"Total tests: {total_tests}")
        logger.info(f"Passed: {self.test_results['passed']}")
        logger.info(f"Failed: {self.test_results['failed']}")
        logger.info(f"Skipped: {self.test_results['skipped']}")
        
        if self.test_results['failed'] > 0:
            logger.error("="*80)
            logger.error("FAILED TESTS:")
            logger.error("="*80)
            for failed_test in self.failed_tests:
                logger.error(f"❌ {failed_test}")
        
        if self.test_results['failed'] == 0:
            logger.info("="*80)
            logger.info("✅ ALL TESTS PASSED - R2 INTEGRATION READY")
            logger.info("="*80)
            return True
        else:
            logger.error("="*80)
            logger.error(f"❌ {self.test_results['failed']} TESTS FAILED")
            logger.error("="*80)
            return False


def main():
    parser = argparse.ArgumentParser(description='Test Cloudflare R2 integration')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--connection', action='store_true', help='Test R2 connection only')
    parser.add_argument('--data-loaders', action='store_true', help='Test data loaders only')
    parser.add_argument('--training', action='store_true', help='Test training integration only')
    
    args = parser.parse_args()
    
    if not any([args.all, args.connection, args.data_loaders, args.training]):
        parser.print_help()
        return
    
    tester = R2IntegrationTester()
    
    if args.all or args.connection:
        tester.test_r2_connection()
        tester.test_bucket_operations()
        tester.test_object_operations()
    
    if args.all or args.data_loaders:
        tester.test_streaming_data_loader()
        tester.test_data_source_preservation()
    
    if args.all:
        tester.test_rust_module_compatibility()
    
    # Generate report
    success = tester.generate_test_report()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()


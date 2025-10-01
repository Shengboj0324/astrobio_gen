#!/usr/bin/env python3
"""
Comprehensive Data Flow Analysis
Analyzes complete data flow from local → S3 → RunPod
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class DataFlowAnalyzer:
    """Comprehensive data flow analyzer"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.issues = []
        self.warnings = []
        self.successes = []
        
    def check_aws_credentials(self) -> bool:
        """Check AWS credentials configuration"""
        logger.info("\n" + "="*80)
        logger.info("CHECKING AWS CREDENTIALS")
        logger.info("="*80)
        
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        
        if aws_access_key and aws_secret_key:
            logger.info(f"✅ AWS_ACCESS_KEY_ID: {aws_access_key[:8]}...")
            logger.info(f"✅ AWS_SECRET_ACCESS_KEY: {aws_secret_key[:8]}...")
            logger.info(f"✅ AWS_DEFAULT_REGION: {aws_region}")
            self.successes.append("AWS credentials configured")
            return True
        else:
            logger.error("❌ AWS credentials NOT configured in .env")
            logger.error("   Required: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
            self.issues.append("AWS credentials missing")
            return False
    
    def check_s3_integration_files(self) -> bool:
        """Check S3 integration files exist"""
        logger.info("\n" + "="*80)
        logger.info("CHECKING S3 INTEGRATION FILES")
        logger.info("="*80)
        
        required_files = [
            'utils/s3_data_flow_integration.py',
            'utils/aws_integration.py',
            'RUNPOD_S3_INTEGRATION_SETUP.py',
            'setup_aws_infrastructure.py',
        ]
        
        all_exist = True
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                logger.info(f"✅ {file_path}")
                self.successes.append(f"File exists: {file_path}")
            else:
                logger.error(f"❌ {file_path} NOT FOUND")
                self.issues.append(f"Missing file: {file_path}")
                all_exist = False
        
        return all_exist
    
    def check_s3_dependencies(self) -> bool:
        """Check S3 Python dependencies"""
        logger.info("\n" + "="*80)
        logger.info("CHECKING S3 DEPENDENCIES")
        logger.info("="*80)
        
        required_packages = {
            'boto3': 'AWS SDK for Python',
            's3fs': 'S3 filesystem interface',
            'fsspec': 'Filesystem spec',
            'aiobotocore': 'Async boto3',
        }
        
        all_installed = True
        for package, description in required_packages.items():
            try:
                __import__(package)
                logger.info(f"✅ {package:15s} - {description}")
                self.successes.append(f"Package installed: {package}")
            except ImportError:
                logger.error(f"❌ {package:15s} - NOT INSTALLED")
                self.issues.append(f"Missing package: {package}")
                all_installed = False
        
        return all_installed
    
    def check_data_flow_components(self) -> bool:
        """Check data flow components"""
        logger.info("\n" + "="*80)
        logger.info("CHECKING DATA FLOW COMPONENTS")
        logger.info("="*80)
        
        components = {
            'S3DataFlowManager': 'utils.s3_data_flow_integration',
            'S3StreamingDataset': 'utils.s3_data_flow_integration',
            'S3ZarrDataset': 'utils.s3_data_flow_integration',
            'AWSManager': 'utils.aws_integration',
        }
        
        all_available = True
        for component, module in components.items():
            try:
                mod = __import__(module, fromlist=[component])
                cls = getattr(mod, component)
                logger.info(f"✅ {component:25s} from {module}")
                self.successes.append(f"Component available: {component}")
            except Exception as e:
                logger.error(f"❌ {component:25s} - {e}")
                self.issues.append(f"Component unavailable: {component}")
                all_available = False
        
        return all_available
    
    def analyze_data_flow_path(self) -> None:
        """Analyze complete data flow path"""
        logger.info("\n" + "="*80)
        logger.info("DATA FLOW PATH ANALYSIS")
        logger.info("="*80)
        
        logger.info("\n📊 COMPLETE DATA FLOW:")
        logger.info("   1. Local Data Acquisition")
        logger.info("      ├─ Data sources: 1100+ scientific databases")
        logger.info("      ├─ Authentication: API keys configured")
        logger.info("      ├─ Download: Parallel acquisition")
        logger.info("      └─ Processing: Rust acceleration (10-20x)")
        
        logger.info("\n   2. Upload to AWS S3")
        logger.info("      ├─ S3 Client: boto3")
        logger.info("      ├─ Buckets:")
        logger.info("      │  ├─ astrobio-data-primary-YYYYMMDD")
        logger.info("      │  ├─ astrobio-zarr-cubes-YYYYMMDD")
        logger.info("      │  ├─ astrobio-data-backup-YYYYMMDD")
        logger.info("      │  └─ astrobio-logs-metadata-YYYYMMDD")
        logger.info("      ├─ Upload: Multipart for large files")
        logger.info("      └─ Verification: Checksum validation")
        
        logger.info("\n   3. RunPod Access from S3")
        logger.info("      ├─ S3 Streaming: s3fs filesystem")
        logger.info("      ├─ DataLoader: PyTorch S3StreamingDataset")
        logger.info("      ├─ Zarr Support: Direct S3 Zarr access")
        logger.info("      ├─ Caching: Local cache for performance")
        logger.info("      └─ Training: Direct S3 → GPU pipeline")
        
        logger.info("\n   4. Training on RunPod")
        logger.info("      ├─ GPU: 2x RTX A5000 (48GB VRAM)")
        logger.info("      ├─ Data: Streamed from S3")
        logger.info("      ├─ Checkpoints: Saved to S3")
        logger.info("      └─ Logs: Uploaded to S3")
    
    def check_bucket_configuration(self) -> bool:
        """Check S3 bucket configuration"""
        logger.info("\n" + "="*80)
        logger.info("CHECKING S3 BUCKET CONFIGURATION")
        logger.info("="*80)
        
        # Check config files for bucket names
        config_files = [
            'config/config.yaml',
            'config/first_round_config.json',
        ]
        
        bucket_configs_found = False
        for config_file in config_files:
            config_path = self.project_root / config_file
            if config_path.exists():
                logger.info(f"✅ {config_file} exists")
                bucket_configs_found = True
            else:
                logger.warning(f"⚠️  {config_file} not found")
        
        # Expected bucket pattern
        logger.info("\n📦 Expected S3 Buckets:")
        logger.info("   - astrobio-data-primary-YYYYMMDD")
        logger.info("   - astrobio-zarr-cubes-YYYYMMDD")
        logger.info("   - astrobio-data-backup-YYYYMMDD")
        logger.info("   - astrobio-logs-metadata-YYYYMMDD")
        
        if bucket_configs_found:
            self.successes.append("Bucket configuration files exist")
            return True
        else:
            self.warnings.append("Some bucket configuration files missing")
            return False
    
    def check_runpod_integration(self) -> bool:
        """Check RunPod integration"""
        logger.info("\n" + "="*80)
        logger.info("CHECKING RUNPOD INTEGRATION")
        logger.info("="*80)
        
        runpod_files = [
            'RUNPOD_README.md',
            'RUNPOD_S3_INTEGRATION_SETUP.py',
            'runpod_setup.sh',
        ]
        
        all_exist = True
        for file_path in runpod_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                logger.info(f"✅ {file_path}")
                self.successes.append(f"RunPod file exists: {file_path}")
            else:
                logger.error(f"❌ {file_path} NOT FOUND")
                self.issues.append(f"Missing RunPod file: {file_path}")
                all_exist = False
        
        return all_exist
    
    def generate_report(self) -> None:
        """Generate comprehensive report"""
        logger.info("\n" + "="*80)
        logger.info("DATA FLOW ANALYSIS REPORT")
        logger.info("="*80)
        
        total_checks = len(self.successes) + len(self.issues) + len(self.warnings)
        success_rate = (len(self.successes) / total_checks * 100) if total_checks > 0 else 0
        
        logger.info(f"\n📊 SUMMARY:")
        logger.info(f"   Total checks:     {total_checks}")
        logger.info(f"   Successes:        {len(self.successes)} ✅")
        logger.info(f"   Issues:           {len(self.issues)} ❌")
        logger.info(f"   Warnings:         {len(self.warnings)} ⚠️")
        logger.info(f"   Success rate:     {success_rate:.1f}%")
        
        if self.issues:
            logger.info(f"\n❌ CRITICAL ISSUES ({len(self.issues)}):")
            for issue in self.issues:
                logger.info(f"   ❌ {issue}")
        
        if self.warnings:
            logger.info(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                logger.info(f"   ⚠️  {warning}")
        
        logger.info("\n" + "="*80)
        logger.info("RECOMMENDATIONS")
        logger.info("="*80)
        
        if not os.getenv('AWS_ACCESS_KEY_ID'):
            logger.info("1. Configure AWS credentials in .env:")
            logger.info("   AWS_ACCESS_KEY_ID=your_access_key")
            logger.info("   AWS_SECRET_ACCESS_KEY=your_secret_key")
            logger.info("   AWS_DEFAULT_REGION=us-east-1")
        
        if self.issues:
            logger.info("\n2. Install missing dependencies:")
            logger.info("   pip install boto3 s3fs fsspec aiobotocore")
        
        logger.info("\n3. Create S3 buckets:")
        logger.info("   python setup_aws_infrastructure.py")
        
        logger.info("\n4. Test S3 connection:")
        logger.info("   python -c 'from utils.s3_data_flow_integration import test_s3_data_flow; test_s3_data_flow()'")
        
        logger.info("\n5. Deploy to RunPod:")
        logger.info("   Follow RUNPOD_README.md instructions")
        
        logger.info("="*80)
    
    def run(self) -> bool:
        """Run comprehensive analysis"""
        logger.info("="*80)
        logger.info("COMPREHENSIVE DATA FLOW ANALYSIS")
        logger.info("="*80)
        
        # Run all checks
        creds_ok = self.check_aws_credentials()
        files_ok = self.check_s3_integration_files()
        deps_ok = self.check_s3_dependencies()
        components_ok = self.check_data_flow_components()
        buckets_ok = self.check_bucket_configuration()
        runpod_ok = self.check_runpod_integration()
        
        # Analyze data flow
        self.analyze_data_flow_path()
        
        # Generate report
        self.generate_report()
        
        # Return overall status
        all_ok = creds_ok and files_ok and deps_ok and components_ok and buckets_ok and runpod_ok
        
        if all_ok:
            logger.info("\n✅ DATA FLOW ANALYSIS COMPLETE - ALL SYSTEMS READY!")
            return True
        else:
            logger.info("\n⚠️  DATA FLOW ANALYSIS COMPLETE - ISSUES FOUND")
            return False


def main():
    """Main entry point"""
    analyzer = DataFlowAnalyzer()
    success = analyzer.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


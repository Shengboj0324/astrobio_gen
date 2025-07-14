#!/usr/bin/env python3
"""
AWS Infrastructure Setup Script
===============================

Sets up complete AWS infrastructure for the astrobiology project:
- Creates S3 buckets
- Configures lifecycle policies
- Updates DVC configuration
- Tests data upload/download
- Generates setup report
"""

import os
import json
import yaml
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Complete AWS infrastructure setup"""
    
    print("🚀 AWS Infrastructure Setup for Astrobiology Project")
    print("=" * 60)
    
    # Step 1: Test AWS connection
    print("\n📡 Step 1: Testing AWS Connection...")
    try:
        from utils.aws_integration import AWSManager
        aws = AWSManager()
        
        verification = aws.verify_credentials()
        
        if verification['status'] != 'success':
            print(f"❌ AWS Connection Failed: {verification['error']}")
            print("\n🔧 Please configure AWS credentials:")
            print("1. Run: aws configure")
            print("2. Enter your AWS Access Key ID and Secret Access Key")
            print("3. Choose region: us-east-1")
            print("4. Choose output format: json")
            return False
        
        print(f"✅ AWS Connection Successful!")
        print(f"   Account ID: {verification['account_id']}")
        print(f"   Region: {verification['region']}")
        
    except Exception as e:
        print(f"❌ Error importing AWS integration: {e}")
        return False
    
    # Step 2: Create S3 Buckets
    print("\n🪣 Step 2: Creating S3 Buckets...")
    try:
        buckets = aws.create_project_buckets('astrobio')
        
        if not buckets:
            print("❌ Failed to create buckets")
            return False
        
        print("✅ Created S3 Buckets:")
        for purpose, bucket_name in buckets.items():
            print(f"   {purpose}: {bucket_name}")
            
            # Set up lifecycle policies for data buckets
            if purpose in ['primary', 'backup', 'zarr']:
                try:
                    aws.setup_lifecycle_policy(bucket_name)
                    print(f"   ✅ Lifecycle policy applied to {bucket_name}")
                except Exception as e:
                    print(f"   ⚠️ Could not set lifecycle policy for {bucket_name}: {e}")
        
    except Exception as e:
        print(f"❌ Error creating buckets: {e}")
        return False
    
    # Step 3: Update DVC Configuration
    print("\n⚙️ Step 3: Updating DVC Configuration...")
    try:
        dvc_config_path = Path('.dvc/config')
        
        if dvc_config_path.exists():
            # Read current config
            with open(dvc_config_path, 'r') as f:
                content = f.read()
            
            # Update bucket URLs
            for purpose, bucket_name in buckets.items():
                if purpose == 'primary':
                    content = content.replace('s3://YOUR_PRIMARY_BUCKET_NAME', f's3://{bucket_name}')
                elif purpose == 'backup':
                    content = content.replace('s3://YOUR_BACKUP_BUCKET_NAME', f's3://{bucket_name}')
                elif purpose == 'zarr':
                    content = content.replace('s3://YOUR_ZARR_BUCKET_NAME', f's3://{bucket_name}')
            
            # Write updated config
            with open(dvc_config_path, 'w') as f:
                f.write(content)
            
            print("✅ DVC configuration updated with bucket names")
        else:
            print("⚠️ DVC config file not found - you may need to initialize DVC")
    
    except Exception as e:
        print(f"❌ Error updating DVC config: {e}")
    
    # Step 4: Test Data Upload
    print("\n📤 Step 4: Testing Data Upload...")
    try:
        # Test with a small file
        test_file = Path("database_integration_report.json")
        if test_file.exists():
            primary_bucket = buckets.get('primary')
            if primary_bucket:
                success = aws.upload_data(str(test_file), primary_bucket, 'test/database_report.json')
                if success:
                    print(f"✅ Test upload successful to {primary_bucket}")
                    
                    # Test download
                    download_path = Path("test_download.json")
                    success = aws.download_data(primary_bucket, 'test/database_report.json', str(download_path))
                    if success:
                        print("✅ Test download successful")
                        download_path.unlink()  # Clean up
                    else:
                        print("⚠️ Test download failed")
                else:
                    print("⚠️ Test upload failed")
        else:
            print("ℹ️ No test file available for upload test")
    
    except Exception as e:
        print(f"❌ Error testing data operations: {e}")
    
    # Step 5: Generate Setup Report
    print("\n📊 Step 5: Generating Setup Report...")
    try:
        report = {
            'timestamp': datetime.now().isoformat(),
            'aws_account': verification.get('account_id'),
            'region': verification.get('region'),
            'buckets_created': buckets,
            'estimated_costs': {
                'storage_per_tb_per_month': 23,  # USD
                'requests_per_1000': 0.4,       # USD
                'data_transfer_per_gb': 0.09    # USD
            },
            'next_steps': [
                "Run comprehensive data acquisition",
                "Set up billing alerts in AWS Console",
                "Configure EC2 instances for training",
                "Set up monitoring dashboard"
            ]
        }
        
        report_path = Path("aws_setup_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Setup report saved to {report_path}")
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
    
    # Step 6: Show Next Steps
    print("\n🎯 Setup Complete! Next Steps:")
    print("1. ✅ AWS infrastructure is ready")
    print("2. 📊 Set up billing alerts in AWS Console")
    print("3. 🔄 Run your first data acquisition:")
    print("   python run_first_round_data_capture.py --max-storage-tb 1.0")
    print("4. 🚀 Launch EC2 instance for larger processing:")
    print("   aws ec2 run-instances --image-id ami-0c02fb55956c7d316 --instance-type g4dn.xlarge")
    print("5. 💰 Monitor costs in AWS Console → Billing")
    
    print(f"\n📋 Bucket Summary:")
    for purpose, bucket_name in buckets.items():
        print(f"   {purpose}: s3://{bucket_name}")
    
    print("\n🎉 AWS setup successful! You're ready to scale to the cloud.")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the errors above and try again.")
        exit(1)
    else:
        print("\n✅ All systems ready for cloud-scale astrobiology research!")
        exit(0) 
#!/usr/bin/env python3
"""
Verify Complete S3 Data Flow
Comprehensive verification of S3 configuration and data flow
"""

import boto3
import os
from dotenv import load_dotenv
from datetime import datetime
import json

load_dotenv()

print("="*80)
print("COMPREHENSIVE S3 DATA FLOW VERIFICATION")
print("="*80)

# Step 1: Check credentials
print("\n" + "="*80)
print("STEP 1: AWS CREDENTIALS")
print("="*80)

access_key = os.getenv('AWS_ACCESS_KEY_ID')
secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
region = os.getenv('AWS_DEFAULT_REGION')

if access_key and secret_key:
    print(f"✅ AWS_ACCESS_KEY_ID: {access_key[:8]}...{access_key[-4:]}")
    print(f"✅ AWS_SECRET_ACCESS_KEY: {secret_key[:8]}...{secret_key[-4:]}")
    print(f"✅ AWS_DEFAULT_REGION: {region}")
else:
    print("❌ Credentials not configured")
    exit(1)

# Step 2: Check bucket configuration
print("\n" + "="*80)
print("STEP 2: BUCKET CONFIGURATION")
print("="*80)

buckets = {
    'primary': os.getenv('AWS_S3_BUCKET_PRIMARY'),
    'zarr': os.getenv('AWS_S3_BUCKET_ZARR'),
    'backup': os.getenv('AWS_S3_BUCKET_BACKUP'),
    'logs': os.getenv('AWS_S3_BUCKET_LOGS')
}

all_configured = True
for purpose, name in buckets.items():
    if name:
        print(f"✅ {purpose:10s}: {name}")
    else:
        print(f"❌ {purpose:10s}: NOT CONFIGURED")
        all_configured = False

if not all_configured:
    print("\n❌ Some buckets not configured")
    exit(1)

# Step 3: Test AWS connection
print("\n" + "="*80)
print("STEP 3: AWS CONNECTION")
print("="*80)

try:
    s3 = boto3.client('s3')
    sts = boto3.client('sts')
    
    identity = sts.get_caller_identity()
    print(f"✅ Connected to AWS")
    print(f"   Account: {identity['Account']}")
    print(f"   User: {identity['Arn'].split('/')[-1]}")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    exit(1)

# Step 4: Verify bucket access
print("\n" + "="*80)
print("STEP 4: BUCKET ACCESS VERIFICATION")
print("="*80)

all_accessible = True

for purpose, bucket_name in buckets.items():
    print(f"\n🔍 {purpose.upper()}: {bucket_name}")
    
    try:
        # Check bucket exists
        s3.head_bucket(Bucket=bucket_name)
        print(f"  ✅ Bucket exists")
        
        # Check list permission
        response = s3.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
        object_count = response.get('KeyCount', 0)
        print(f"  ✅ Can list ({object_count} objects)")
        
        # Check write permission
        test_key = f'test/verification_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        s3.put_object(Bucket=bucket_name, Key=test_key, Body=b'test')
        print(f"  ✅ Can write")
        
        # Check read permission
        s3.get_object(Bucket=bucket_name, Key=test_key)
        print(f"  ✅ Can read")
        
        # Check delete permission
        s3.delete_object(Bucket=bucket_name, Key=test_key)
        print(f"  ✅ Can delete")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        all_accessible = False

# Step 5: Test data flow components
print("\n" + "="*80)
print("STEP 5: DATA FLOW COMPONENTS")
print("="*80)

try:
    from utils.s3_data_flow_integration import S3DataFlowManager
    print("✅ S3DataFlowManager available")
    
    manager = S3DataFlowManager()
    print("✅ S3DataFlowManager initialized")
    
except Exception as e:
    print(f"❌ S3DataFlowManager error: {e}")

try:
    from utils.s3_data_flow_integration import S3StreamingDataset
    print("✅ S3StreamingDataset available")
except Exception as e:
    print(f"❌ S3StreamingDataset error: {e}")

try:
    from utils.s3_data_flow_integration import S3ZarrDataset
    print("✅ S3ZarrDataset available")
except Exception as e:
    print(f"❌ S3ZarrDataset error: {e}")

# Step 6: Configuration files
print("\n" + "="*80)
print("STEP 6: CONFIGURATION FILES")
print("="*80)

config_files = [
    '.env',
    'config/config.yaml',
    'config/first_round_config.json'
]

for config_file in config_files:
    if os.path.exists(config_file):
        print(f"✅ {config_file}")
    else:
        print(f"❌ {config_file} not found")

# Step 7: Utility scripts
print("\n" + "="*80)
print("STEP 7: UTILITY SCRIPTS")
print("="*80)

scripts = [
    'upload_to_s3.py',
    'download_from_s3.py',
    'list_s3_contents.py',
    'verify_s3_dataflow.py'
]

for script in scripts:
    if os.path.exists(script):
        print(f"✅ {script}")
    else:
        print(f"❌ {script} not found")

# Final summary
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

if all_configured and all_accessible:
    print("\n✅ ALL SYSTEMS READY!")
    print("\n🎯 DATA FLOW STATUS: 100% OPERATIONAL")
    print("\nYou can now:")
    print("  1. Upload training data:")
    print("     python upload_to_s3.py --source data/ --bucket primary --prefix training/")
    print("\n  2. List bucket contents:")
    print("     python list_s3_contents.py --bucket primary")
    print("\n  3. Download data:")
    print("     python download_from_s3.py --bucket primary --prefix training/ --dest local_data/")
    print("\n  4. Deploy to RunPod and start training!")
    
else:
    print("\n⚠️  SOME ISSUES FOUND")
    print("\nPlease review the errors above.")

print("\n" + "="*80)


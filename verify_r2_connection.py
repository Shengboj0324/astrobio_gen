#!/usr/bin/env python3
"""
Verify Cloudflare R2 Connection
================================
Quick verification script for R2 credentials and bucket access.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("="*80)
print("CLOUDFLARE R2 CONNECTION VERIFICATION")
print("="*80)

# Check credentials
print("\n1. CHECKING CREDENTIALS...")
r2_access_key = os.getenv('R2_ACCESS_KEY_ID')
r2_secret_key = os.getenv('R2_SECRET_ACCESS_KEY')
r2_account_id = os.getenv('R2_ACCOUNT_ID')
r2_endpoint = os.getenv('R2_ENDPOINT_URL')

if r2_access_key:
    print(f"✅ R2_ACCESS_KEY_ID: {r2_access_key[:10]}...{r2_access_key[-4:]}")
else:
    print("❌ R2_ACCESS_KEY_ID not set")
    sys.exit(1)

if r2_secret_key:
    print(f"✅ R2_SECRET_ACCESS_KEY: {r2_secret_key[:10]}...{r2_secret_key[-4:]}")
else:
    print("❌ R2_SECRET_ACCESS_KEY not set")
    sys.exit(1)

if r2_account_id:
    print(f"✅ R2_ACCOUNT_ID: {r2_account_id}")
else:
    print("❌ R2_ACCOUNT_ID not set")
    sys.exit(1)

if r2_endpoint:
    print(f"✅ R2_ENDPOINT_URL: {r2_endpoint}")
else:
    print("❌ R2_ENDPOINT_URL not set")
    sys.exit(1)

# Test R2 connection
print("\n2. TESTING R2 CONNECTION...")
try:
    from utils.r2_data_flow_integration import R2DataFlowManager
    
    r2_manager = R2DataFlowManager()
    
    if r2_manager.credentials_verified:
        print("✅ R2 credentials verified successfully")
    else:
        print("❌ R2 credential verification failed")
        sys.exit(1)
    
    # List buckets
    print("\n3. LISTING R2 BUCKETS...")
    buckets = r2_manager.list_buckets()
    
    if buckets:
        print(f"✅ Found {len(buckets)} R2 buckets:")
        for bucket in buckets:
            print(f"   - {bucket['name']} (created: {bucket['creation_date']})")
    else:
        print("⚠️  No buckets found (this is OK if you just created them)")
    
    # Check required buckets
    print("\n4. VERIFYING REQUIRED BUCKETS...")
    required_buckets = [
        'astrobio-data-primary',
        'astrobio-zarr-cubes',
        'astrobio-data-backup',
        'astrobio-logs-metadata'
    ]
    
    bucket_names = [b['name'] for b in buckets]
    all_present = True
    
    for required in required_buckets:
        if required in bucket_names:
            print(f"✅ {required}")
        else:
            print(f"❌ {required} - NOT FOUND")
            all_present = False
    
    if all_present:
        print("\n" + "="*80)
        print("✅ R2 CONNECTION VERIFIED - ALL SYSTEMS READY")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("⚠️  SOME BUCKETS MISSING - Please create them in Cloudflare Dashboard")
        print("="*80)
        sys.exit(1)
        
except Exception as e:
    print(f"\n❌ R2 CONNECTION FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


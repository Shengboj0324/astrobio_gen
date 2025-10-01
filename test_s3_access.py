#!/usr/bin/env python3
"""
Test S3 Bucket Access and Permissions
"""

import boto3
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

print("="*80)
print("S3 BUCKET ACCESS VERIFICATION")
print("="*80)

# Get bucket names from environment
buckets = {
    'primary': os.getenv('AWS_S3_BUCKET_PRIMARY'),
    'zarr': os.getenv('AWS_S3_BUCKET_ZARR'),
    'backup': os.getenv('AWS_S3_BUCKET_BACKUP'),
    'logs': os.getenv('AWS_S3_BUCKET_LOGS')
}

print("\n📦 Configured Buckets:")
for purpose, name in buckets.items():
    print(f"  {purpose:10s}: {name}")

# Initialize S3 client
s3 = boto3.client('s3')

print("\n" + "="*80)
print("TESTING BUCKET ACCESS")
print("="*80)

results = {}

for purpose, bucket_name in buckets.items():
    print(f"\n🔍 Testing: {bucket_name}")
    
    test_results = {
        'exists': False,
        'list': False,
        'write': False,
        'read': False,
        'delete': False
    }
    
    # Test 1: Check if bucket exists
    try:
        s3.head_bucket(Bucket=bucket_name)
        test_results['exists'] = True
        print(f"  ✅ Bucket exists")
    except Exception as e:
        print(f"  ❌ Bucket does not exist: {e}")
        results[purpose] = test_results
        continue
    
    # Test 2: List objects
    try:
        response = s3.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
        test_results['list'] = True
        object_count = response.get('KeyCount', 0)
        print(f"  ✅ Can list objects ({object_count} objects)")
    except Exception as e:
        print(f"  ❌ Cannot list objects: {e}")
    
    # Test 3: Write (upload test file)
    test_key = f'test/access_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    test_content = f'Access test at {datetime.now()}'
    
    try:
        s3.put_object(
            Bucket=bucket_name,
            Key=test_key,
            Body=test_content.encode('utf-8')
        )
        test_results['write'] = True
        print(f"  ✅ Can write objects")
    except Exception as e:
        print(f"  ❌ Cannot write objects: {e}")
    
    # Test 4: Read (download test file)
    if test_results['write']:
        try:
            response = s3.get_object(Bucket=bucket_name, Key=test_key)
            content = response['Body'].read().decode('utf-8')
            if content == test_content:
                test_results['read'] = True
                print(f"  ✅ Can read objects")
            else:
                print(f"  ❌ Read content mismatch")
        except Exception as e:
            print(f"  ❌ Cannot read objects: {e}")
    
    # Test 5: Delete (cleanup test file)
    if test_results['write']:
        try:
            s3.delete_object(Bucket=bucket_name, Key=test_key)
            test_results['delete'] = True
            print(f"  ✅ Can delete objects")
        except Exception as e:
            print(f"  ❌ Cannot delete objects: {e}")
    
    results[purpose] = test_results

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

all_passed = True
for purpose, test_results in results.items():
    bucket_name = buckets[purpose]
    passed = all(test_results.values())
    
    if passed:
        print(f"\n✅ {purpose.upper()}: {bucket_name}")
        print(f"   All tests passed!")
    else:
        print(f"\n❌ {purpose.upper()}: {bucket_name}")
        print(f"   Failed tests:")
        for test, result in test_results.items():
            if not result:
                print(f"     - {test}")
        all_passed = False

print("\n" + "="*80)
if all_passed:
    print("✅ ALL BUCKETS READY FOR USE!")
    print("\nYou can now:")
    print("  - Upload training data")
    print("  - Stream data for training")
    print("  - Save checkpoints")
    print("  - Upload logs")
else:
    print("⚠️  SOME BUCKETS HAVE ISSUES")
    print("\nPlease check the errors above.")

print("="*80)


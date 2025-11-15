#!/usr/bin/env python3
"""Quick R2 bucket status verification"""
import boto3
from botocore.config import Config

client = boto3.client(
    's3',
    endpoint_url='https://e3d9647571bd8bb6027db63db3197fd0.r2.cloudflarestorage.com',
    aws_access_key_id='e128888fe9e2e1398eff86adb8ddeaa8',
    aws_secret_access_key='6e73ee757bc1f6943d565fff7e878b3301cd7fb495b8db2bb075dfe7a3fde113',
    region_name='auto',
    config=Config(signature_version='s3v4')
)

print("="*70)
print("R2 BUCKET STATUS VERIFICATION")
print("="*70)

buckets = client.list_buckets()['Buckets']
print(f"\nBuckets found: {len(buckets)}")
for b in buckets:
    print(f"  ✅ {b['Name']}")

print("\nPrimary bucket contents:")
resp = client.list_objects_v2(Bucket='astrobio-data-primary', MaxKeys=10)
print(f"  Objects: {resp.get('KeyCount', 0)}")
if 'Contents' in resp:
    print("  Sample objects:")
    for obj in resp['Contents'][:5]:
        print(f"    - {obj['Key']}")

print("\n✅ R2 INTEGRATION VERIFIED AND OPERATIONAL")
print("="*70)


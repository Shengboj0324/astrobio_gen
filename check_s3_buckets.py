#!/usr/bin/env python3
"""
Check S3 Buckets - Detailed Analysis
"""

import boto3
from datetime import datetime

print("="*80)
print("S3 BUCKET ANALYSIS")
print("="*80)

try:
    # Initialize S3 client
    s3 = boto3.client('s3')
    
    # Get all buckets
    response = s3.list_buckets()
    buckets = response['Buckets']
    
    print(f"\n✅ Successfully connected to AWS S3")
    print(f"📊 Total buckets found: {len(buckets)}")
    
    if len(buckets) == 0:
        print("\n⚠️  NO BUCKETS FOUND!")
        print("\nPossible reasons:")
        print("1. Buckets were created in a different AWS account")
        print("2. Buckets were deleted")
        print("3. You're using different credentials than when you created them")
        print("4. Buckets are in a different region (S3 buckets are global but list should show all)")
        
    else:
        print("\n" + "="*80)
        print("ALL BUCKETS IN YOUR AWS ACCOUNT:")
        print("="*80)
        
        for i, bucket in enumerate(buckets, 1):
            name = bucket['Name']
            created = bucket['CreationDate']
            
            print(f"\n{i}. {name}")
            print(f"   Created: {created}")
            
            # Try to get bucket location
            try:
                location = s3.get_bucket_location(Bucket=name)
                region = location['LocationConstraint'] or 'us-east-1'
                print(f"   Region: {region}")
            except Exception as e:
                print(f"   Region: Unable to determine ({e})")
            
            # Try to get bucket size (count objects)
            try:
                objects = s3.list_objects_v2(Bucket=name, MaxKeys=1)
                if 'Contents' in objects:
                    # Get total count
                    paginator = s3.get_paginator('list_objects_v2')
                    pages = paginator.paginate(Bucket=name)
                    total_objects = sum(1 for page in pages for _ in page.get('Contents', []))
                    print(f"   Objects: {total_objects}")
                else:
                    print(f"   Objects: 0 (empty)")
            except Exception as e:
                print(f"   Objects: Unable to count ({e})")
        
        # Check for astrobio buckets specifically
        print("\n" + "="*80)
        print("ASTROBIO BUCKETS:")
        print("="*80)
        
        astrobio_buckets = [b for b in buckets if 'astrobio' in b['Name'].lower()]
        
        if astrobio_buckets:
            print(f"\nFound {len(astrobio_buckets)} astrobio bucket(s):")
            for bucket in astrobio_buckets:
                print(f"  ✅ {bucket['Name']}")
        else:
            print("\n❌ NO ASTROBIO BUCKETS FOUND!")
            print("\nYour buckets might have different names.")
            print("Check the list above for any buckets you recognize.")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nPossible issues:")
    print("1. AWS credentials not configured correctly")
    print("2. No internet connection")
    print("3. AWS service issue")

print("\n" + "="*80)
print("RECOMMENDATIONS:")
print("="*80)

print("\n1. If you see 12 buckets above, those are YOUR buckets!")
print("   They might not show in S3 Console if you're not signed in as root user.")

print("\n2. To access S3 Console with your root user:")
print("   a. Go to: https://s3.console.aws.amazon.com/s3/buckets")
print("   b. Make sure you're signed in as ROOT USER (not IAM user)")
print("   c. Check the region dropdown (top-right)")

print("\n3. If you see 0 buckets, create new ones:")
print("   python setup_aws_infrastructure.py")

print("\n" + "="*80)


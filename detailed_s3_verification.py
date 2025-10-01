#!/usr/bin/env python3
"""
DETAILED S3 BUCKET VERIFICATION
Comprehensive check with all possible details
"""

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import json
from datetime import datetime

print("="*80)
print("COMPREHENSIVE S3 BUCKET VERIFICATION")
print("="*80)

# Step 1: Verify credentials are loaded
print("\n" + "="*80)
print("STEP 1: VERIFYING AWS CREDENTIALS")
print("="*80)

try:
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    
    if access_key and secret_key:
        print(f"✅ AWS_ACCESS_KEY_ID: {access_key[:8]}...{access_key[-4:]}")
        print(f"✅ AWS_SECRET_ACCESS_KEY: {secret_key[:8]}...{secret_key[-4:]}")
        print(f"✅ AWS_DEFAULT_REGION: {region}")
    else:
        print("❌ Credentials not found in .env file")
        exit(1)
        
except Exception as e:
    print(f"❌ Error loading credentials: {e}")
    exit(1)

# Step 2: Test AWS connection
print("\n" + "="*80)
print("STEP 2: TESTING AWS CONNECTION")
print("="*80)

try:
    # Create S3 client
    s3 = boto3.client('s3', region_name=region)
    
    # Test connection by getting caller identity
    sts = boto3.client('sts', region_name=region)
    identity = sts.get_caller_identity()
    
    print(f"✅ Successfully connected to AWS")
    print(f"   Account ID: {identity['Account']}")
    print(f"   User ARN: {identity['Arn']}")
    print(f"   User ID: {identity['UserId']}")
    
    # Check if this is root user
    if ':root' in identity['Arn']:
        print(f"   🔑 USER TYPE: ROOT USER")
    elif ':user/' in identity['Arn']:
        print(f"   👤 USER TYPE: IAM USER")
    else:
        print(f"   ⚠️  USER TYPE: UNKNOWN")
    
except NoCredentialsError:
    print("❌ No AWS credentials found")
    exit(1)
except ClientError as e:
    print(f"❌ AWS connection failed: {e}")
    exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    exit(1)

# Step 3: List ALL buckets
print("\n" + "="*80)
print("STEP 3: LISTING ALL S3 BUCKETS")
print("="*80)

try:
    response = s3.list_buckets()
    buckets = response.get('Buckets', [])
    
    print(f"\n📊 TOTAL BUCKETS IN YOUR AWS ACCOUNT: {len(buckets)}")
    
    if len(buckets) == 0:
        print("\n❌ ABSOLUTELY NO BUCKETS FOUND!")
        print("\nThis means:")
        print("1. No S3 buckets exist in this AWS account")
        print("2. The buckets you created might be in a different AWS account")
        print("3. The buckets might have been deleted")
        print("\nYou need to create new buckets.")
        
    else:
        print(f"\n✅ FOUND {len(buckets)} BUCKET(S)")
        print("\nDETAILED BUCKET INFORMATION:")
        print("="*80)
        
        for i, bucket in enumerate(buckets, 1):
            name = bucket['Name']
            created = bucket['CreationDate']
            
            print(f"\n{'='*80}")
            print(f"BUCKET #{i}: {name}")
            print(f"{'='*80}")
            print(f"Created: {created}")
            print(f"Created (local time): {created.astimezone()}")
            
            # Get bucket location
            try:
                location_response = s3.get_bucket_location(Bucket=name)
                bucket_region = location_response['LocationConstraint']
                if bucket_region is None:
                    bucket_region = 'us-east-1'
                print(f"Region: {bucket_region}")
            except Exception as e:
                print(f"Region: Error - {e}")
            
            # Get bucket ACL (ownership)
            try:
                acl = s3.get_bucket_acl(Bucket=name)
                owner = acl['Owner']
                print(f"Owner ID: {owner.get('ID', 'Unknown')}")
                print(f"Owner Name: {owner.get('DisplayName', 'Unknown')}")
            except Exception as e:
                print(f"Owner: Error - {e}")
            
            # Count objects
            try:
                objects_response = s3.list_objects_v2(Bucket=name, MaxKeys=1000)
                
                if 'Contents' in objects_response:
                    object_count = len(objects_response['Contents'])
                    
                    # If there might be more, count all
                    if objects_response.get('IsTruncated', False):
                        paginator = s3.get_paginator('list_objects_v2')
                        pages = paginator.paginate(Bucket=name)
                        object_count = sum(len(page.get('Contents', [])) for page in pages)
                    
                    print(f"Objects: {object_count}")
                    
                    # Show first few objects
                    if object_count > 0:
                        print(f"First objects:")
                        for obj in objects_response['Contents'][:5]:
                            size_mb = obj['Size'] / (1024 * 1024)
                            print(f"  - {obj['Key']} ({size_mb:.2f} MB)")
                else:
                    print(f"Objects: 0 (empty)")
                    
            except Exception as e:
                print(f"Objects: Error - {e}")
            
            # Get bucket size
            try:
                cloudwatch = boto3.client('cloudwatch', region_name=bucket_region)
                
                # Get bucket size metric
                response = cloudwatch.get_metric_statistics(
                    Namespace='AWS/S3',
                    MetricName='BucketSizeBytes',
                    Dimensions=[
                        {'Name': 'BucketName', 'Value': name},
                        {'Name': 'StorageType', 'Value': 'StandardStorage'}
                    ],
                    StartTime=datetime.now().replace(hour=0, minute=0, second=0),
                    EndTime=datetime.now(),
                    Period=86400,
                    Statistics=['Average']
                )
                
                if response['Datapoints']:
                    size_bytes = response['Datapoints'][0]['Average']
                    size_gb = size_bytes / (1024**3)
                    print(f"Size: {size_gb:.4f} GB")
                else:
                    print(f"Size: 0 GB (or metrics not available yet)")
                    
            except Exception as e:
                print(f"Size: Unable to determine")
            
            # Check bucket tags
            try:
                tags_response = s3.get_bucket_tagging(Bucket=name)
                tags = tags_response.get('TagSet', [])
                if tags:
                    print(f"Tags:")
                    for tag in tags:
                        print(f"  - {tag['Key']}: {tag['Value']}")
            except ClientError as e:
                if e.response['Error']['Code'] != 'NoSuchTagSet':
                    print(f"Tags: Error - {e}")
            except Exception:
                pass

except ClientError as e:
    print(f"❌ Error listing buckets: {e}")
    print(f"\nError details:")
    print(f"  Code: {e.response['Error']['Code']}")
    print(f"  Message: {e.response['Error']['Message']}")
    exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    exit(1)

# Step 4: Check for astrobio buckets specifically
print("\n" + "="*80)
print("STEP 4: CHECKING FOR ASTROBIO BUCKETS")
print("="*80)

astrobio_buckets = [b for b in buckets if 'astrobio' in b['Name'].lower()]

if astrobio_buckets:
    print(f"\n✅ FOUND {len(astrobio_buckets)} ASTROBIO BUCKET(S):")
    for bucket in astrobio_buckets:
        print(f"  ✅ {bucket['Name']}")
else:
    print("\n❌ NO ASTROBIO BUCKETS FOUND")
    print("\nSearched for buckets containing 'astrobio' in the name.")
    print("None were found in your AWS account.")

# Step 5: Final verdict
print("\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

if len(buckets) == 0:
    print("\n❌ ABSOLUTELY NO BUCKETS EXIST IN YOUR AWS ACCOUNT")
    print("\nCONCLUSION:")
    print("  - Your AWS account has ZERO S3 buckets")
    print("  - The buckets you thought you created don't exist")
    print("  - Possible reasons:")
    print("    1. They were created in a different AWS account")
    print("    2. They were deleted")
    print("    3. You're using different credentials than when you created them")
    print("\nRECOMMENDATION:")
    print("  Stop searching AWS Console. Create new buckets:")
    print("  python setup_aws_infrastructure.py")
    
elif len(astrobio_buckets) > 0:
    print(f"\n✅ YES, YOUR BUCKETS EXIST!")
    print(f"\nCONCLUSION:")
    print(f"  - Found {len(astrobio_buckets)} astrobio bucket(s)")
    print(f"  - They are in your AWS account")
    print(f"  - They are accessible with your credentials")
    print(f"\nRECOMMENDATION:")
    print(f"  Stop searching AWS Console. Your buckets exist and work!")
    print(f"  Use them programmatically with the S3DataFlowManager.")
    
else:
    print(f"\n⚠️  BUCKETS EXIST BUT NONE ARE NAMED 'ASTROBIO'")
    print(f"\nCONCLUSION:")
    print(f"  - Found {len(buckets)} bucket(s) in your account")
    print(f"  - But none contain 'astrobio' in the name")
    print(f"  - Check the list above to see what buckets you have")
    print(f"\nRECOMMENDATION:")
    print(f"  Either use existing buckets or create new astrobio buckets:")
    print(f"  python setup_aws_infrastructure.py")

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)


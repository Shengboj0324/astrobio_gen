#!/usr/bin/env python3
"""
Download Data from S3
Simple script to download files/directories from S3 buckets
"""

import argparse
import boto3
import os
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Bucket mapping
BUCKETS = {
    'primary': os.getenv('AWS_S3_BUCKET_PRIMARY'),
    'zarr': os.getenv('AWS_S3_BUCKET_ZARR'),
    'backup': os.getenv('AWS_S3_BUCKET_BACKUP'),
    'logs': os.getenv('AWS_S3_BUCKET_LOGS')
}

def download_file(s3_client, bucket_name, s3_key, local_path):
    """Download a single file from S3"""
    try:
        # Get file size
        response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        file_size = response['ContentLength']
        
        # Create parent directory
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress bar
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Downloading {Path(s3_key).name}") as pbar:
            s3_client.download_file(
                bucket_name,
                s3_key,
                local_path,
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred)
            )
        return True
    except Exception as e:
        print(f"❌ Error downloading {s3_key}: {e}")
        return False

def download_prefix(s3_client, bucket_name, s3_prefix, local_dir):
    """Download all objects with given prefix from S3"""
    
    print(f"\n📥 Listing objects in s3://{bucket_name}/{s3_prefix}")
    
    # List all objects with prefix
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)
    
    objects = []
    for page in pages:
        if 'Contents' in page:
            objects.extend(page['Contents'])
    
    if not objects:
        print(f"⚠️  No objects found with prefix: {s3_prefix}")
        return 0, 0
    
    print(f"📦 Found {len(objects)} objects to download")
    
    success_count = 0
    fail_count = 0
    
    for obj in objects:
        s3_key = obj['Key']
        
        # Calculate local path
        rel_path = s3_key[len(s3_prefix):].lstrip('/')
        local_path = os.path.join(local_dir, rel_path)
        
        if download_file(s3_client, bucket_name, s3_key, local_path):
            success_count += 1
        else:
            fail_count += 1
    
    return success_count, fail_count

def main():
    parser = argparse.ArgumentParser(description='Download data from S3')
    parser.add_argument('--bucket', required=True, choices=['primary', 'zarr', 'backup', 'logs'],
                       help='Source bucket')
    parser.add_argument('--prefix', default='', help='S3 prefix (folder path) to download')
    parser.add_argument('--dest', required=True, help='Local destination directory')
    parser.add_argument('--key', help='Specific S3 key to download (single file)')
    
    args = parser.parse_args()
    
    bucket_name = BUCKETS[args.bucket]
    
    if not bucket_name:
        print(f"❌ Bucket '{args.bucket}' not configured in .env")
        return
    
    print("="*80)
    print("DOWNLOAD FROM S3")
    print("="*80)
    print(f"\nBucket:  {bucket_name}")
    print(f"Dest:    {args.dest}")
    
    # Initialize S3 client
    s3 = boto3.client('s3')
    
    if args.key:
        # Download single file
        print(f"Key:     {args.key}")
        local_path = os.path.join(args.dest, Path(args.key).name)
        
        print(f"\n📥 Downloading file...")
        if download_file(s3, bucket_name, args.key, local_path):
            print(f"✅ Download complete!")
            print(f"   Saved to: {local_path}")
        else:
            print(f"❌ Download failed")
            
    else:
        # Download prefix (directory)
        print(f"Prefix:  {args.prefix or '(all)'}")
        
        success, failed = download_prefix(s3, bucket_name, args.prefix, args.dest)
        
        print("\n" + "="*80)
        print("DOWNLOAD SUMMARY")
        print("="*80)
        print(f"✅ Successful: {success}")
        print(f"❌ Failed:     {failed}")
        print(f"📊 Total:      {success + failed}")
        
        if failed == 0 and success > 0:
            print(f"\n✅ All files downloaded successfully!")
        elif success == 0:
            print(f"\n⚠️  No files downloaded")
        else:
            print(f"\n⚠️  Some files failed to download")

if __name__ == '__main__':
    main()


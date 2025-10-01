#!/usr/bin/env python3
"""
Upload Data to S3
Simple script to upload files/directories to S3 buckets
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

def upload_file(s3_client, file_path, bucket_name, s3_key):
    """Upload a single file to S3"""
    try:
        file_size = os.path.getsize(file_path)
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Uploading {Path(file_path).name}") as pbar:
            s3_client.upload_file(
                file_path,
                bucket_name,
                s3_key,
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred)
            )
        return True
    except Exception as e:
        print(f"❌ Error uploading {file_path}: {e}")
        return False

def upload_directory(s3_client, local_dir, bucket_name, s3_prefix=''):
    """Upload entire directory to S3"""
    local_path = Path(local_dir)
    
    if not local_path.exists():
        print(f"❌ Directory not found: {local_dir}")
        return 0, 0
    
    # Get all files
    files = list(local_path.rglob('*'))
    files = [f for f in files if f.is_file()]
    
    print(f"\n📦 Found {len(files)} files to upload")
    print(f"📤 Uploading to: s3://{bucket_name}/{s3_prefix}")
    
    success_count = 0
    fail_count = 0
    
    for file_path in files:
        # Calculate relative path
        rel_path = file_path.relative_to(local_path)
        s3_key = str(Path(s3_prefix) / rel_path).replace('\\', '/')
        
        if upload_file(s3_client, str(file_path), bucket_name, s3_key):
            success_count += 1
        else:
            fail_count += 1
    
    return success_count, fail_count

def main():
    parser = argparse.ArgumentParser(description='Upload data to S3')
    parser.add_argument('--source', required=True, help='Local file or directory to upload')
    parser.add_argument('--bucket', required=True, choices=['primary', 'zarr', 'backup', 'logs'],
                       help='Target bucket')
    parser.add_argument('--prefix', default='', help='S3 prefix (folder path)')
    
    args = parser.parse_args()
    
    bucket_name = BUCKETS[args.bucket]
    
    if not bucket_name:
        print(f"❌ Bucket '{args.bucket}' not configured in .env")
        return
    
    print("="*80)
    print("UPLOAD TO S3")
    print("="*80)
    print(f"\nSource:  {args.source}")
    print(f"Bucket:  {bucket_name}")
    print(f"Prefix:  {args.prefix or '(root)'}")
    
    # Initialize S3 client
    s3 = boto3.client('s3')
    
    source_path = Path(args.source)
    
    if source_path.is_file():
        # Upload single file
        s3_key = str(Path(args.prefix) / source_path.name).replace('\\', '/')
        print(f"\n📤 Uploading file...")
        if upload_file(s3, str(source_path), bucket_name, s3_key):
            print(f"✅ Upload complete!")
            print(f"   S3 URI: s3://{bucket_name}/{s3_key}")
        else:
            print(f"❌ Upload failed")
            
    elif source_path.is_dir():
        # Upload directory
        success, failed = upload_directory(s3, str(source_path), bucket_name, args.prefix)
        
        print("\n" + "="*80)
        print("UPLOAD SUMMARY")
        print("="*80)
        print(f"✅ Successful: {success}")
        print(f"❌ Failed:     {failed}")
        print(f"📊 Total:      {success + failed}")
        
        if failed == 0:
            print(f"\n✅ All files uploaded successfully!")
        else:
            print(f"\n⚠️  Some files failed to upload")
            
    else:
        print(f"❌ Source not found: {args.source}")

if __name__ == '__main__':
    main()


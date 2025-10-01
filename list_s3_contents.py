#!/usr/bin/env python3
"""
List S3 Bucket Contents
Simple script to list objects in S3 buckets
"""

import argparse
import boto3
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Bucket mapping
BUCKETS = {
    'primary': os.getenv('AWS_S3_BUCKET_PRIMARY'),
    'zarr': os.getenv('AWS_S3_BUCKET_ZARR'),
    'backup': os.getenv('AWS_S3_BUCKET_BACKUP'),
    'logs': os.getenv('AWS_S3_BUCKET_LOGS')
}

def format_size(bytes):
    """Format bytes to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} PB"

def list_bucket_contents(s3_client, bucket_name, prefix='', max_keys=None):
    """List contents of S3 bucket"""
    
    print(f"\n📦 Listing: s3://{bucket_name}/{prefix}")
    
    # List objects
    paginator = s3_client.get_paginator('list_objects_v2')
    
    if max_keys:
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix, PaginationConfig={'MaxItems': max_keys})
    else:
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    
    objects = []
    for page in pages:
        if 'Contents' in page:
            objects.extend(page['Contents'])
    
    if not objects:
        print(f"⚠️  No objects found")
        return
    
    print(f"\n📊 Found {len(objects)} object(s)")
    print("\n" + "="*80)
    
    total_size = 0
    
    for i, obj in enumerate(objects, 1):
        key = obj['Key']
        size = obj['Size']
        modified = obj['LastModified']
        
        total_size += size
        
        print(f"\n{i}. {key}")
        print(f"   Size:     {format_size(size)}")
        print(f"   Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "="*80)
    print(f"📊 SUMMARY")
    print("="*80)
    print(f"Total objects: {len(objects)}")
    print(f"Total size:    {format_size(total_size)}")

def main():
    parser = argparse.ArgumentParser(description='List S3 bucket contents')
    parser.add_argument('--bucket', required=True, choices=['primary', 'zarr', 'backup', 'logs', 'all'],
                       help='Bucket to list')
    parser.add_argument('--prefix', default='', help='S3 prefix (folder path) to list')
    parser.add_argument('--max', type=int, help='Maximum number of objects to list')
    
    args = parser.parse_args()
    
    print("="*80)
    print("LIST S3 BUCKET CONTENTS")
    print("="*80)
    
    # Initialize S3 client
    s3 = boto3.client('s3')
    
    if args.bucket == 'all':
        # List all buckets
        for bucket_type, bucket_name in BUCKETS.items():
            if bucket_name:
                print(f"\n{'='*80}")
                print(f"BUCKET: {bucket_type.upper()} ({bucket_name})")
                print(f"{'='*80}")
                list_bucket_contents(s3, bucket_name, args.prefix, args.max)
    else:
        # List specific bucket
        bucket_name = BUCKETS[args.bucket]
        
        if not bucket_name:
            print(f"❌ Bucket '{args.bucket}' not configured in .env")
            return
        
        list_bucket_contents(s3, bucket_name, args.prefix, args.max)

if __name__ == '__main__':
    main()


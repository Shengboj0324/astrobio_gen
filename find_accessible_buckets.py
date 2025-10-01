#!/usr/bin/env python3
"""
Find which buckets we can actually access
"""

import boto3
from botocore.exceptions import ClientError

print("="*80)
print("FINDING ACCESSIBLE BUCKETS")
print("="*80)

s3 = boto3.client('s3')

# Get all buckets
try:
    response = s3.list_buckets()
    all_buckets = [b['Name'] for b in response['Buckets']]
    print(f"\n📊 Total buckets visible: {len(all_buckets)}")
    
    astrobio_buckets = [b for b in all_buckets if 'astrobio' in b.lower()]
    print(f"📦 Astrobio buckets visible: {len(astrobio_buckets)}")
    
    if astrobio_buckets:
        print("\nAstrobio buckets:")
        for bucket in astrobio_buckets:
            print(f"  - {bucket}")
    
    print("\n" + "="*80)
    print("TESTING ACCESS TO EACH BUCKET")
    print("="*80)
    
    accessible_buckets = []
    
    for bucket_name in astrobio_buckets:
        print(f"\n🔍 Testing: {bucket_name}")
        
        # Test head_bucket (check if we can access)
        try:
            s3.head_bucket(Bucket=bucket_name)
            print(f"  ✅ Can access bucket")
            
            # Try to list objects
            try:
                response = s3.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
                object_count = response.get('KeyCount', 0)
                print(f"  ✅ Can list objects ({object_count} objects)")
                
                # Try to write
                try:
                    s3.put_object(
                        Bucket=bucket_name,
                        Key='test/access_test.txt',
                        Body=b'test'
                    )
                    print(f"  ✅ Can write objects")
                    
                    # Cleanup
                    s3.delete_object(Bucket=bucket_name, Key='test/access_test.txt')
                    print(f"  ✅ Can delete objects")
                    
                    accessible_buckets.append(bucket_name)
                    
                except ClientError as e:
                    print(f"  ❌ Cannot write: {e.response['Error']['Code']}")
                    
            except ClientError as e:
                print(f"  ❌ Cannot list: {e.response['Error']['Code']}")
                
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '403':
                print(f"  ❌ Access Forbidden (403) - bucket owned by different account/user")
            elif error_code == '404':
                print(f"  ❌ Bucket not found (404)")
            else:
                print(f"  ❌ Error: {error_code}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if accessible_buckets:
        print(f"\n✅ ACCESSIBLE BUCKETS ({len(accessible_buckets)}):")
        for bucket in accessible_buckets:
            print(f"  ✅ {bucket}")
        
        print("\n💡 RECOMMENDATION:")
        print("Use these buckets for your training:")
        
        # Find most recent accessible buckets
        primary_buckets = [b for b in accessible_buckets if 'primary' in b]
        zarr_buckets = [b for b in accessible_buckets if 'zarr' in b]
        backup_buckets = [b for b in accessible_buckets if 'backup' in b]
        logs_buckets = [b for b in accessible_buckets if 'logs' in b]
        
        if primary_buckets:
            print(f"  Primary: {sorted(primary_buckets)[-1]}")
        if zarr_buckets:
            print(f"  Zarr:    {sorted(zarr_buckets)[-1]}")
        if backup_buckets:
            print(f"  Backup:  {sorted(backup_buckets)[-1]}")
        if logs_buckets:
            print(f"  Logs:    {sorted(logs_buckets)[-1]}")
            
    else:
        print("\n❌ NO ACCESSIBLE BUCKETS FOUND!")
        print("\nThis means:")
        print("  - The buckets you created are owned by a different AWS account")
        print("  - Or they were created with different credentials")
        print("  - Or there's a permissions issue")
        
        print("\n💡 SOLUTION:")
        print("  Create new buckets with your current credentials:")
        print("  python setup_aws_infrastructure.py")
    
    print("\n" + "="*80)
    
except Exception as e:
    print(f"\n❌ Error: {e}")


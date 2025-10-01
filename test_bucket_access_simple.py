import boto3
from botocore.exceptions import ClientError

s3 = boto3.client('s3')

buckets_to_test = [
    'astrobio-data-primary-20250717',
    'astrobio-zarr-cubes-20250717',
    'astrobio-data-backup-20250717',
    'astrobio-logs-metadata-20250717'
]

print("Testing bucket access:\n")

for bucket in buckets_to_test:
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"✅ {bucket}")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        print(f"❌ {bucket} - Error: {error_code}")
    except Exception as e:
        print(f"❌ {bucket} - Error: {str(e)}")


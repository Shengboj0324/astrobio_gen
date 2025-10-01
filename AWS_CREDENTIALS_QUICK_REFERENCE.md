# 🔑 AWS CREDENTIALS QUICK REFERENCE

## What You Need

```
AWS_ACCESS_KEY_ID          = 20-character string (starts with AKIA)
AWS_SECRET_ACCESS_KEY      = 40-character string
AWS_DEFAULT_REGION         = us-east-1 (default)
```

## Quick Setup (5 Minutes)

### 1. Get Credentials
```
1. Go to: https://console.aws.amazon.com/iam/
2. Click: Users → Create user
3. Name: astrobio-s3-user
4. Attach policy: AmazonS3FullAccess
5. Create access key → Application outside AWS
6. ⚠️ SAVE BOTH KEYS! (only shown once)
```

### 2. Configure in Project
```bash
# Edit .env file (line 73-88)
AWS_ACCESS_KEY_ID=AKIA...your_key_here...
AWS_SECRET_ACCESS_KEY=wJal...your_secret_here...
AWS_DEFAULT_REGION=us-east-1
```

### 3. Test
```bash
python -c "import boto3; s3 = boto3.client('s3'); print('✅ Valid')"
```

### 4. Create Buckets
```bash
python setup_aws_infrastructure.py
```

## Example Credentials Format

```bash
# ✅ CORRECT FORMAT:
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_DEFAULT_REGION=us-east-1

# ❌ WRONG (don't use quotes, spaces, or comments on same line):
AWS_ACCESS_KEY_ID = "AKIA..."  # my key
```

## Troubleshooting

| Error | Fix |
|-------|-----|
| NoCredentialsError | Check .env file exists and has values |
| InvalidAccessKeyId | Verify Access Key ID (20 chars, starts AKIA) |
| SignatureDoesNotMatch | Verify Secret Access Key (40 chars) |
| AccessDenied | Attach AmazonS3FullAccess policy to IAM user |

## Security Checklist

- ✅ Use IAM user (not root account)
- ✅ Only grant S3 access (AmazonS3FullAccess)
- ✅ Store in .env file (already in .gitignore)
- ❌ Never commit to git
- ❌ Never share via email/chat

## Cost Estimate

- **Free Tier:** 5 GB storage, 20K GET, 2K PUT (first 12 months)
- **After:** ~$0.023/GB/month storage
- **37 TB:** ~$400-900/month (with Intelligent Tiering)

## Next Steps After Setup

```bash
# 1. Create buckets
python setup_aws_infrastructure.py

# 2. Test S3 connection
python -c "from utils.s3_data_flow_integration import test_s3_data_flow; test_s3_data_flow()"

# 3. Upload data
python -c "from utils.s3_data_flow_integration import S3DataFlowManager; s3 = S3DataFlowManager(); s3.upload_directory('data/', 's3://bucket/')"

# 4. Deploy to RunPod
# Follow RUNPOD_README.md
```

## Links

- **AWS Console:** https://console.aws.amazon.com/
- **IAM Console:** https://console.aws.amazon.com/iam/
- **Create Account:** https://aws.amazon.com/
- **Full Guide:** See AWS_CREDENTIALS_SETUP_GUIDE.md

---

**Quick Reference Card** | **Version 1.0** | **2025-10-01**


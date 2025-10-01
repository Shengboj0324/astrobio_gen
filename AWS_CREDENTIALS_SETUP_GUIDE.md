# 🔑 AWS CREDENTIALS SETUP GUIDE
# Complete Guide to Getting AWS Credentials for S3 Access

**Date:** 2025-10-01  
**Purpose:** Configure AWS credentials for AstroBio-Gen S3 data flow  
**Required Time:** 10-15 minutes

---

## 🎯 WHAT YOU NEED

You need **3 specific credentials** to enable S3 data flow:

### **1. AWS Access Key ID**
- **Format:** 20-character alphanumeric string
- **Example:** `AKIAIOSFODNN7EXAMPLE`
- **Purpose:** Identifies your AWS account/user
- **Used by:** boto3, s3fs, all AWS SDK operations

### **2. AWS Secret Access Key**
- **Format:** 40-character alphanumeric string
- **Example:** `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY`
- **Purpose:** Password for your AWS account/user
- **Security:** ⚠️ **NEVER share or commit to git!**
- **Used by:** boto3, s3fs, all AWS SDK operations

### **3. AWS Region** (Optional - has default)
- **Format:** Region code
- **Example:** `us-east-1`
- **Purpose:** Specifies which AWS region to use
- **Default:** `us-east-1` (already configured)
- **Used by:** S3 bucket creation and access

---

## 📋 STEP-BY-STEP GUIDE

### **Option 1: Create New IAM User (RECOMMENDED)** ✅

This is the **safest and most recommended** approach for production use.

#### **Step 1: Log into AWS Console**
1. Go to: https://console.aws.amazon.com/
2. Sign in with your AWS account
3. If you don't have an AWS account:
   - Go to: https://aws.amazon.com/
   - Click "Create an AWS Account"
   - Follow the registration process (requires credit card)

#### **Step 2: Navigate to IAM**
1. In AWS Console, search for "IAM" in the top search bar
2. Click on "IAM" (Identity and Access Management)
3. Or go directly to: https://console.aws.amazon.com/iam/

#### **Step 3: Create New IAM User**
1. In left sidebar, click **"Users"**
2. Click **"Create user"** button (top right)
3. Enter user name: `astrobio-s3-user` (or any name you prefer)
4. Click **"Next"**

#### **Step 4: Set Permissions**
1. Select **"Attach policies directly"**
2. Search for and select: **`AmazonS3FullAccess`**
   - This gives full S3 access (read, write, delete)
   - For production, you may want more restrictive policies
3. Click **"Next"**
4. Review and click **"Create user"**

#### **Step 5: Create Access Keys**
1. Click on the newly created user (`astrobio-s3-user`)
2. Click on **"Security credentials"** tab
3. Scroll down to **"Access keys"** section
4. Click **"Create access key"**
5. Select use case: **"Application running outside AWS"**
6. Click **"Next"**
7. (Optional) Add description tag: `AstroBio-Gen S3 Access`
8. Click **"Create access key"**

#### **Step 6: Save Your Credentials** ⚠️ **CRITICAL**
You will see a screen with:
- **Access key ID:** `AKIA...` (20 characters)
- **Secret access key:** `wJal...` (40 characters)

**⚠️ IMPORTANT:**
- **This is the ONLY time you can see the secret access key!**
- Click **"Download .csv file"** to save credentials
- Or copy both values to a secure location
- **DO NOT close this window until you've saved the credentials!**

---

### **Option 2: Use Existing IAM User** ✅

If you already have an IAM user with S3 access:

#### **Step 1: Find Your User**
1. Go to IAM Console: https://console.aws.amazon.com/iam/
2. Click **"Users"** in left sidebar
3. Click on your existing user

#### **Step 2: Verify S3 Permissions**
1. Click **"Permissions"** tab
2. Verify user has `AmazonS3FullAccess` or similar S3 policy
3. If not, click **"Add permissions"** and attach `AmazonS3FullAccess`

#### **Step 3: Create New Access Key**
1. Click **"Security credentials"** tab
2. Scroll to **"Access keys"** section
3. Click **"Create access key"**
4. Follow steps 5-6 from Option 1 above

---

### **Option 3: Use Root Account Credentials** ⚠️ **NOT RECOMMENDED**

**⚠️ WARNING:** Using root account credentials is a security risk!

If you must use root credentials:
1. Go to: https://console.aws.amazon.com/iam/home#/security_credentials
2. Expand **"Access keys"** section
3. Click **"Create access key"**
4. Confirm the warning
5. Save the credentials

**⚠️ STRONGLY RECOMMENDED:** Create an IAM user instead (Option 1)

---

## 🔧 CONFIGURE CREDENTIALS IN PROJECT

Once you have your credentials, configure them in the project:

### **Method 1: Update .env File** (RECOMMENDED)

1. Open `.env` file in project root
2. Find the AWS section (around line 73)
3. Replace the placeholder values:

```bash
# Before:
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
AWS_DEFAULT_REGION=us-east-1

# After (with your actual credentials):
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_DEFAULT_REGION=us-east-1
```

4. Save the file
5. **⚠️ NEVER commit .env to git!** (already in .gitignore)

### **Method 2: Environment Variables** (Alternative)

On Windows (PowerShell):
```powershell
$env:AWS_ACCESS_KEY_ID="AKIAIOSFODNN7EXAMPLE"
$env:AWS_SECRET_ACCESS_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
$env:AWS_DEFAULT_REGION="us-east-1"
```

On Linux/Mac (bash):
```bash
export AWS_ACCESS_KEY_ID="AKIAIOSFODNN7EXAMPLE"
export AWS_SECRET_ACCESS_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
export AWS_DEFAULT_REGION="us-east-1"
```

### **Method 3: AWS CLI Configuration** (Alternative)

If you have AWS CLI installed:
```bash
aws configure
# Enter Access Key ID: AKIAIOSFODNN7EXAMPLE
# Enter Secret Access Key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
# Enter Default region: us-east-1
# Enter Default output format: json
```

This creates `~/.aws/credentials` and `~/.aws/config` files that boto3 will automatically use.

---

## ✅ VERIFY CREDENTIALS

After configuring credentials, verify they work:

### **Test 1: Basic Connection**
```bash
python -c "import boto3; s3 = boto3.client('s3'); print('✅ Credentials valid'); print('Buckets:', [b['Name'] for b in s3.list_buckets()['Buckets']])"
```

**Expected output:**
```
✅ Credentials valid
Buckets: [list of your S3 buckets]
```

### **Test 2: S3 Data Flow Integration**
```bash
python -c "from utils.s3_data_flow_integration import S3DataFlowManager; s3 = S3DataFlowManager(); print('✅ S3 Data Flow Ready' if s3.credentials_verified else '❌ Credentials Failed')"
```

**Expected output:**
```
✅ S3 connections initialized successfully
✅ Credentials verified
✅ S3 Data Flow Ready
```

### **Test 3: Full Data Flow Analysis**
```bash
python analyze_data_flow.py
```

**Expected output:**
```
✅ AWS_ACCESS_KEY_ID: AKIA...
✅ AWS_SECRET_ACCESS_KEY: wJal...
✅ AWS_DEFAULT_REGION: us-east-1
...
✅ DATA FLOW ANALYSIS COMPLETE - ALL SYSTEMS READY!
```

---

## 🔒 SECURITY BEST PRACTICES

### **DO:**
- ✅ Create dedicated IAM user for this project
- ✅ Use least-privilege permissions (only S3 access needed)
- ✅ Store credentials in `.env` file (already in .gitignore)
- ✅ Rotate access keys regularly (every 90 days)
- ✅ Enable MFA on your AWS account
- ✅ Monitor AWS CloudTrail for suspicious activity

### **DON'T:**
- ❌ Use root account credentials
- ❌ Commit credentials to git
- ❌ Share credentials via email/chat
- ❌ Use same credentials across multiple projects
- ❌ Give broader permissions than needed
- ❌ Leave unused access keys active

---

## 💰 AWS COSTS

### **Free Tier (First 12 Months)**
- **S3 Storage:** 5 GB free
- **S3 Requests:** 20,000 GET, 2,000 PUT
- **Data Transfer:** 100 GB out free

### **After Free Tier**
- **S3 Storage:** $0.023/GB/month (Standard)
- **S3 Requests:** $0.0004/1000 GET, $0.005/1000 PUT
- **Data Transfer:** $0.09/GB (first 10 TB)

### **Estimated Costs for AstroBio-Gen**
- **37 TB Storage:** ~$850/month (Standard)
- **With Intelligent Tiering:** ~$400/month (estimated)
- **Data Transfer:** ~$50/month
- **Total:** ~$450-900/month

**💡 TIP:** Start with small dataset to test, then scale up!

---

## 🚀 NEXT STEPS AFTER CONFIGURATION

Once credentials are configured:

1. ✅ **Create S3 Buckets:**
   ```bash
   python setup_aws_infrastructure.py
   ```

2. ✅ **Test S3 Connection:**
   ```bash
   python -c "from utils.s3_data_flow_integration import test_s3_data_flow; test_s3_data_flow()"
   ```

3. ✅ **Upload Data to S3:**
   ```bash
   python -c "
   from utils.s3_data_flow_integration import S3DataFlowManager
   s3 = S3DataFlowManager()
   s3.upload_directory('data/', 's3://astrobio-data-primary-YYYYMMDD/')
   "
   ```

4. ✅ **Deploy to RunPod:**
   - Follow `RUNPOD_README.md`
   - Configure same AWS credentials on RunPod
   - Test S3 streaming from RunPod

---

## 🆘 TROUBLESHOOTING

### **Error: "NoCredentialsError"**
- **Cause:** Credentials not found
- **Fix:** Verify `.env` file has correct values, or set environment variables

### **Error: "InvalidAccessKeyId"**
- **Cause:** Access Key ID is incorrect
- **Fix:** Double-check the Access Key ID (20 characters, starts with AKIA)

### **Error: "SignatureDoesNotMatch"**
- **Cause:** Secret Access Key is incorrect
- **Fix:** Double-check the Secret Access Key (40 characters)

### **Error: "AccessDenied"**
- **Cause:** IAM user doesn't have S3 permissions
- **Fix:** Attach `AmazonS3FullAccess` policy to IAM user

### **Error: "BucketAlreadyExists"**
- **Cause:** Bucket name is taken globally
- **Fix:** Bucket names are created with timestamp, should be unique

---

## 📋 SUMMARY

**What you need:**
1. ✅ **AWS Access Key ID** (20 characters, starts with AKIA)
2. ✅ **AWS Secret Access Key** (40 characters)
3. ✅ **AWS Region** (default: us-east-1)

**How to get them:**
1. Create IAM user in AWS Console
2. Attach `AmazonS3FullAccess` policy
3. Create access key
4. Save credentials securely

**How to configure:**
1. Update `.env` file with credentials
2. Verify with test commands
3. Create S3 buckets
4. Start using S3 data flow!

---

**Guide Created:** 2025-10-01  
**Status:** ✅ **COMPLETE**  
**Next Action:** **Get AWS credentials and configure in .env**

---

**END OF AWS CREDENTIALS SETUP GUIDE**


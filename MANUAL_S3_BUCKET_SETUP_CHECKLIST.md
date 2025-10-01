# 📋 MANUAL S3 BUCKET SETUP CHECKLIST
# Create Buckets on AWS Console → Configure Locally

**Perfect approach!** You create the buckets manually, I'll configure everything locally.

---

## 🪣 **STEP 1: CREATE BUCKETS ON AWS CONSOLE**

### **Buckets to Create (4 total):**

Create these **4 buckets** in AWS S3 Console:

1. **`astrobio-data-primary-20251001`**
   - Purpose: Main training data storage
   - Region: **us-east-1** (US East - N. Virginia)

2. **`astrobio-zarr-cubes-20251001`**
   - Purpose: Processed Zarr datacubes
   - Region: **us-east-1** (US East - N. Virginia)

3. **`astrobio-data-backup-20251001`**
   - Purpose: Backup storage
   - Region: **us-east-1** (US East - N. Virginia)

4. **`astrobio-logs-metadata-20251001`**
   - Purpose: Training logs and metadata
   - Region: **us-east-1** (US East - N. Virginia)

---

## 🔧 **HOW TO CREATE EACH BUCKET**

### **For Each Bucket:**

1. **Go to S3 Console:**
   - https://s3.console.aws.amazon.com/s3/buckets
   - Make sure you're signed in as **ROOT USER**

2. **Click "Create bucket"** button (orange button, top-right)

3. **Bucket Settings:**

   **General Configuration:**
   - **Bucket name:** (use exact names from list above)
   - **AWS Region:** `US East (N. Virginia) us-east-1`

   **Object Ownership:**
   - ✅ Select: **ACLs disabled (recommended)**

   **Block Public Access settings:**
   - ✅ Keep: **Block all public access** (checked)
   - This is correct - we want private buckets

   **Bucket Versioning:**
   - For `astrobio-data-primary-*` and `astrobio-data-backup-*`:
     - ✅ Select: **Enable**
   - For `astrobio-zarr-cubes-*` and `astrobio-logs-metadata-*`:
     - ⚪ Select: **Disable** (or leave default)

   **Tags (Optional but Recommended):**
   - Add tag: `Project` = `AstroBio-Gen`
   - Add tag: `Environment` = `Production`

   **Default encryption:**
   - ✅ Select: **Server-side encryption with Amazon S3 managed keys (SSE-S3)**
   - This is free and automatic

   **Advanced settings:**
   - Leave defaults (Object Lock disabled)

4. **Click "Create bucket"** button at bottom

5. **Repeat for all 4 buckets**

---

## 📝 **INFORMATION I NEED FROM YOU**

After you create the buckets, give me this information:

### **Required Information:**

```
BUCKET 1:
  Name: astrobio-data-primary-20251001
  Region: us-east-1
  Status: Created ✅ or ❌

BUCKET 2:
  Name: astrobio-zarr-cubes-20251001
  Region: us-east-1
  Status: Created ✅ or ❌

BUCKET 3:
  Name: astrobio-data-backup-20251001
  Region: us-east-1
  Status: Created ✅ or ❌

BUCKET 4:
  Name: astrobio-logs-metadata-20251001
  Region: us-east-1
  Status: Created ✅ or ❌
```

### **Optional (but helpful):**
- Screenshot of S3 Console showing all 4 buckets
- Any error messages if bucket creation failed

---

## ✅ **WHAT I WILL CONFIGURE LOCALLY**

Once you confirm the buckets are created, I will:

### **1. Update Configuration Files:**
- ✅ Update `.env` with bucket names
- ✅ Update `config/config.yaml` with bucket names
- ✅ Update `config/first_round_config.json` with bucket names

### **2. Verify Connection:**
- ✅ Test S3 connection with your credentials
- ✅ Verify access to all 4 buckets
- ✅ Test read/write permissions

### **3. Configure Data Flow:**
- ✅ Set up S3DataFlowManager with correct bucket names
- ✅ Configure upload paths
- ✅ Configure download paths
- ✅ Set up streaming data loaders

### **4. Test Complete Pipeline:**
- ✅ Test data upload to S3
- ✅ Test data download from S3
- ✅ Test S3 streaming for training
- ✅ Verify checkpoint saving to S3
- ✅ Verify log uploading to S3

### **5. Create Usage Scripts:**
- ✅ Script to upload training data
- ✅ Script to download data
- ✅ Script to list bucket contents
- ✅ Script to verify data flow

---

## 🎯 **GUARANTEED SMOOTH DATA FLOW CHECKLIST**

To guarantee everything works perfectly, I need:

### **✅ Bucket Information:**
- [x] AWS credentials configured (already done)
- [ ] 4 bucket names confirmed
- [ ] All buckets in us-east-1 region
- [ ] All buckets created successfully

### **✅ Access Verification:**
- [ ] Can list buckets
- [ ] Can upload to buckets
- [ ] Can download from buckets
- [ ] Can delete from buckets (for cleanup)

### **✅ Data Flow Components:**
- [x] S3DataFlowManager ready
- [x] S3StreamingDataset ready
- [x] S3ZarrDataset ready
- [ ] Bucket names configured

### **✅ Training Integration:**
- [ ] Data upload path configured
- [ ] Training data streaming configured
- [ ] Checkpoint save path configured
- [ ] Log upload path configured

---

## 🚀 **AFTER YOU CREATE BUCKETS**

### **Just tell me:**

```
"I created all 4 buckets:
- astrobio-data-primary-20251001 ✅
- astrobio-zarr-cubes-20251001 ✅
- astrobio-data-backup-20251001 ✅
- astrobio-logs-metadata-20251001 ✅

All in us-east-1 region."
```

### **Then I will:**

1. ✅ Update all configuration files
2. ✅ Test connection to all buckets
3. ✅ Verify read/write access
4. ✅ Configure complete data flow
5. ✅ Create upload/download scripts
6. ✅ Test end-to-end pipeline
7. ✅ Give you final verification report

---

## 📊 **WHAT YOU'LL GET**

After I configure everything, you'll have:

### **✅ Ready-to-Use Commands:**

```bash
# Upload training data
python upload_to_s3.py --source data/ --bucket primary

# Download data
python download_from_s3.py --bucket primary --dest local_data/

# List bucket contents
python list_s3_contents.py --bucket primary

# Verify data flow
python verify_s3_dataflow.py

# Start training with S3 data
python train_with_s3_data.py
```

### **✅ Verified Data Flow:**

```
Local Data → S3 Upload → S3 Storage → S3 Streaming → RunPod Training
     ✅           ✅           ✅            ✅              ✅
```

### **✅ Complete Documentation:**
- Bucket configuration details
- Upload/download procedures
- Training integration guide
- Troubleshooting guide

---

## 🎯 **SUMMARY**

### **You Do:**
1. Create 4 buckets in AWS S3 Console (exact names above)
2. All in region: us-east-1
3. Tell me when done

### **I Do:**
1. Configure all local files
2. Test all connections
3. Verify complete data flow
4. Create usage scripts
5. Give you final report

### **Result:**
✅ **100% guaranteed smooth data flow for training**

---

## 📝 **QUICK REFERENCE**

**Bucket Names (copy-paste these):**
```
astrobio-data-primary-20251001
astrobio-zarr-cubes-20251001
astrobio-data-backup-20251001
astrobio-logs-metadata-20251001
```

**Region:**
```
us-east-1
```

**Settings:**
- Block all public access: ✅ YES
- Versioning: Enable for primary/backup, disable for others
- Encryption: SSE-S3 (default)

---

**Ready when you are! Just create the buckets and let me know!** 🚀


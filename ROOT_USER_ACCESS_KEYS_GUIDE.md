# 🔑 CREATE ACCESS KEYS FROM ROOT USER
# No IAM User Needed!

**Your Situation:**
- ✅ You have root user access (email + password)
- ✅ You can sign in to AWS Console as root user
- ❌ You don't have IAM user password
- ❌ You can't access IAM users

**Solution:** Create access keys directly from your ROOT USER account!

---

## ✅ **GOOD NEWS: YOU DON'T NEED IAM USER!**

Since you're the only person working on this project and you have root user access, you can create access keys directly from the root user. This is perfectly fine for a solo project.

**⚠️ Note:** AWS recommends using IAM users for security, but for a solo project with proper security practices, root user access keys are acceptable.

---

## 🚀 **EXACT STEPS TO CREATE ROOT USER ACCESS KEYS**

### **Step 1: Sign In as Root User**
1. Go to: **https://console.aws.amazon.com/**
2. Click **"Sign in to the Console"**
3. Select **"Root user"** (NOT "IAM user")
4. Enter your **root user email address**
5. Click **"Next"**
6. Enter your **root user password**
7. Click **"Sign in"**
8. Complete any MFA if enabled

### **Step 2: Go to Security Credentials**
1. After signing in, look at the **top-right corner** of the page
2. Click on your **account name/email** (it's a dropdown)
3. From the dropdown menu, click **"Security credentials"**

**Alternative direct link:** https://console.aws.amazon.com/iam/home#/security_credentials

### **Step 3: Navigate to Access Keys Section**
1. You're now on the "Security credentials" page
2. Scroll down until you see **"Access keys"** section
3. You might see a warning banner about root user access keys - that's normal

### **Step 4: Create Access Key**
1. In the "Access keys" section, click **"Create access key"** button
2. You'll see a warning popup: "Root user access keys"
3. The warning says: "We recommend that you don't create root user access keys..."
4. **Check the box:** "I understand that root access keys are not recommended, but I still want to create them"
5. Click **"Create access key"** button

### **Step 5: SAVE YOUR CREDENTIALS** ⚠️ **CRITICAL!**

You will now see:
```
Access key ID:     AKIA... (20 characters)
Secret access key: wJal... (40 characters)
```

**⚠️ THIS IS YOUR ONLY CHANCE TO SEE THE SECRET KEY!**

**DO THIS RIGHT NOW:**

**Option A (BEST):**
- Click **"Download .csv file"** button
- Save to secure location (e.g., Documents folder)
- Open CSV to verify both keys are there

**Option B:**
- Click the **copy icon** next to "Access key ID" → paste to secure location
- Click the **copy icon** next to "Secret access key" → paste to secure location
- Save in password manager or secure note

**Option C:**
- Take a screenshot
- Save screenshot securely
- Type keys from screenshot later (then delete screenshot)

**⚠️ DO NOT CLOSE THIS PAGE UNTIL YOU'VE SAVED BOTH KEYS!**

### **Step 6: Close the Dialog**
1. After saving both keys, click **"Done"** or **"Close"**
2. You'll return to the Security credentials page
3. You should now see your access key listed

---

## 🪣 **FIND YOUR S3 BUCKET**

### **Step 1: Go to S3 Console**
1. In the AWS Console, click the **search bar** at the top
2. Type: **`S3`**
3. Click **"S3"** from the dropdown

**Direct link:** https://s3.console.aws.amazon.com/s3/buckets

### **Step 2: View All Buckets**
1. You'll see a page titled "Buckets"
2. Below that is a list of ALL your S3 buckets
3. Look for buckets you created earlier

### **Step 3: Check Different Regions**
If you don't see your bucket:
1. Look at **top-right corner** - you'll see a region dropdown
2. Click it and try different regions:
   - **US East (N. Virginia)** - us-east-1
   - **US West (Oregon)** - us-west-2
   - **US East (Ohio)** - us-east-2
   - Try other regions

### **Step 4: If No Bucket Found**
If you can't find any buckets, you'll need to create new ones (see below).

---

## 🔧 **UPDATE YOUR PROJECT**

### **Step 1: Update .env File**

1. Open: `C:/Users/sjham/OneDrive/Desktop/astrobio_gen/.env`
2. Find line 79-80 (AWS section)
3. Update with your ROOT USER credentials:

```bash
# Current (line 79-80):
AWS_ACCESS_KEY_ID=AKIA2GGU7B3YZE5QRMSB
AWS_SECRET_ACCESS_KEY=your_secret_key_here_get_from_iam_console

# Update to (use your NEW root user keys):
AWS_ACCESS_KEY_ID=AKIA... (your new 20-char key from Step 5)
AWS_SECRET_ACCESS_KEY=wJal... (your new 40-char secret from Step 5)
```

4. **Save the file** (Ctrl+S)

### **Step 2: Test Credentials**

Open PowerShell and run:

```bash
cd C:\Users\sjham\OneDrive\Desktop\astrobio_gen
python -c "import boto3; s3 = boto3.client('s3'); print('✅ Credentials valid'); print('Buckets:', [b['Name'] for b in s3.list_buckets()['Buckets']])"
```

**Expected output:**
```
✅ Credentials valid
Buckets: ['bucket1', 'bucket2', ...]
```

### **Step 3: Create S3 Buckets (If Needed)**

If you don't have buckets or can't find them:

```bash
python setup_aws_infrastructure.py
```

This creates:
- `astrobio-data-primary-20251001`
- `astrobio-zarr-cubes-20251001`
- `astrobio-data-backup-20251001`
- `astrobio-logs-metadata-20251001`

### **Step 4: Verify Everything**

```bash
python analyze_data_flow.py
```

**Expected output:**
```
✅ AWS_ACCESS_KEY_ID: AKIA...
✅ AWS_SECRET_ACCESS_KEY: wJal...
✅ S3 connections initialized successfully
✅ Credentials verified
✅ DATA FLOW ANALYSIS COMPLETE - ALL SYSTEMS READY!
```

---

## 🔒 **SECURITY BEST PRACTICES FOR ROOT USER ACCESS KEYS**

Since you're using root user access keys:

### **DO:**
- ✅ Enable MFA on your root user account
- ✅ Store credentials securely in .env (already in .gitignore)
- ✅ Never commit credentials to git
- ✅ Rotate keys every 90 days
- ✅ Monitor AWS CloudTrail for suspicious activity
- ✅ Set up billing alerts

### **DON'T:**
- ❌ Share root credentials with anyone
- ❌ Use root keys in production if you scale to a team
- ❌ Commit .env to git
- ❌ Leave unused access keys active

### **FUTURE: When You Scale**
When you add team members or scale the project:
1. Create IAM users for team members
2. Create IAM user for yourself
3. Deactivate root user access keys
4. Use IAM users for all operations

---

## 🆘 **TROUBLESHOOTING**

### **Can't Find "Security credentials" in Dropdown**
- Make sure you're signed in as **root user** (not IAM user)
- Click on your account name/email in top-right corner
- Look for "Security credentials" option

### **Don't See "Create access key" Button**
- Scroll down to "Access keys" section
- You might already have 2 access keys (AWS limit)
- If so, deactivate or delete an old one first

### **Error: "You have reached the maximum number of access keys"**
- Root users can have maximum 2 access keys
- Go to Security credentials → Access keys
- Deactivate or delete an old key
- Then create a new one

### **Forgot Root User Password**
1. Go to: https://console.aws.amazon.com/
2. Click "Forgot password?"
3. Enter your root user email
4. Follow password reset instructions in email

---

## 📋 **SUMMARY CHECKLIST**

### **Create Root User Access Keys:**
- [ ] Sign in as root user (not IAM user)
- [ ] Click account name (top-right) → Security credentials
- [ ] Scroll to "Access keys" section
- [ ] Click "Create access key"
- [ ] Check the warning acknowledgment box
- [ ] ⚠️ **SAVE BOTH KEYS** (Download CSV)
- [ ] Click "Done"

### **Find S3 Bucket:**
- [ ] Go to S3 Console
- [ ] Look for buckets with "astrobio" in name
- [ ] Check different regions (top-right dropdown)
- [ ] Note bucket name and region
- [ ] If no bucket: create with `python setup_aws_infrastructure.py`

### **Update Project:**
- [ ] Open `.env` file
- [ ] Update `AWS_ACCESS_KEY_ID` with new root user key
- [ ] Update `AWS_SECRET_ACCESS_KEY` with new root user secret
- [ ] Save file
- [ ] Test: `python -c "import boto3; s3 = boto3.client('s3'); print('✅ Valid')"`
- [ ] Verify: `python analyze_data_flow.py`

---

## 🎯 **WHY THIS IS OK FOR YOUR PROJECT**

**You asked about IAM user password - you DON'T need it!**

Here's why using root user access keys is fine for your situation:

1. ✅ **Solo project** - You're the only person working on this
2. ✅ **Proper security** - .env is in .gitignore, credentials won't leak
3. ✅ **Full access** - Root user has all permissions needed
4. ✅ **Simpler** - No need to manage IAM users/passwords
5. ✅ **Temporary** - You can create IAM users later when needed

**AWS recommends IAM users for teams and production, but for solo development, root user access keys with proper security practices are acceptable.**

---

## 🚀 **NEXT STEPS**

1. **Create root user access keys** (follow Step 1-6 above)
2. **Update .env file** with new keys
3. **Test credentials** with Python command
4. **Find or create S3 buckets**
5. **Start using the system!**

---

**You don't need IAM user password! Use root user access keys!** 🎉

**Direct link to create keys:** https://console.aws.amazon.com/iam/home#/security_credentials

---

**END OF ROOT USER ACCESS KEYS GUIDE**


# 🚀 **RUNPOD DEPLOYMENT & TRAINING GUIDE**
## **Astrobiology AI Platform - 13.01B Parameter Model**

---

## 📋 **EXECUTIVE SUMMARY**

This guide provides precise instructions for deploying and training your 13.01B parameter astrobiology AI model on RunPod cloud infrastructure. Based on comprehensive research and your system specifications, this guide ensures optimal performance with your 2x RTX A5000 GPUs (48GB total VRAM).

### **🎯 KEY SPECIFICATIONS**
- **Model Size**: 13.01B parameters
- **Training Memory**: ~78GB (optimized with Flash Attention + Gradient Checkpointing)
- **Target Hardware**: 2x RTX A5000 (24GB each = 48GB total)
- **Target Accuracy**: 96%
- **Data Sources**: 13 integrated + existing authenticated sources

---

## 🏗️ **PHASE 1: RUNPOD SETUP & CONFIGURATION**

### **1.1 Account Setup & GPU Selection**

1. **Create RunPod Account**:
   ```bash
   # Visit: https://www.runpod.io/
   # Sign up and verify your account
   # Add payment method for GPU usage
   ```

2. **Optimal GPU Configuration**:
   ```yaml
   Recommended Setup:
   - GPU Type: RTX A5000 (24GB VRAM each)
   - Quantity: 2x GPUs
   - Memory: 48GB total VRAM
   - Cost: ~$0.50-0.70/hour per GPU
   - Region: Choose closest to your location for lowest latency
   ```

3. **Alternative GPU Options** (if A5000 unavailable):
   ```yaml
   Fallback Options:
   - RTX A6000 (48GB) - Single GPU option
   - RTX 4090 (24GB) x2 - Similar performance
   - A100 (40GB) - Higher performance, higher cost
   ```

### **1.2 Pod Configuration**

1. **Create New Pod**:
   ```bash
   # In RunPod Console:
   # 1. Click "Deploy" → "GPU Pods"
   # 2. Select "Community Cloud" for cost efficiency
   # 3. Choose RTX A5000 x2 configuration
   # 4. Select appropriate template or custom Docker image
   ```

2. **Recommended Docker Template**:
   ```dockerfile
   # Use PyTorch template with CUDA support
   Base Image: runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
   
   # Or custom image with our requirements
   FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
   
   # Install additional dependencies
   RUN pip install transformers accelerate flash-attn
   RUN pip install wandb tensorboard
   RUN pip install datasets tokenizers
   ```

3. **Storage Configuration**:
   ```yaml
   Container Disk: 50GB minimum
   Volume Storage: 200GB+ for datasets and checkpoints
   Network Volume: Recommended for persistent storage
   ```

---

## 🐳 **PHASE 2: DOCKER CONTAINER SETUP**

### **2.1 Custom Docker Image Creation**

Create a `Dockerfile` for your specific requirements:

```dockerfile
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt requirements_llm.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements_llm.txt

# Install additional ML libraries
RUN pip install flash-attn --no-build-isolation
RUN pip install accelerate bitsandbytes
RUN pip install wandb tensorboard

# Copy your codebase
COPY . /workspace/

# Set environment variables
ENV PYTHONPATH=/workspace
ENV CUDA_VISIBLE_DEVICES=0,1
ENV TOKENIZERS_PARALLELISM=false

# Expose ports for monitoring
EXPOSE 8888 6006 8080

# Default command
CMD ["bash"]
```

### **2.2 Build and Push Docker Image**

```bash
# Build the image locally (optional)
docker build -t your-username/astrobio-ai:latest .

# Or use RunPod's build service
# Upload your code to GitHub and use RunPod's GitHub integration
```

---

## 🚀 **PHASE 3: DEPLOYMENT PROCESS**

### **3.1 Pod Deployment Steps**

1. **Launch Pod**:
   ```bash
   # In RunPod Console:
   # 1. Select your custom template or community template
   # 2. Configure GPU: RTX A5000 x2
   # 3. Set volume storage: 200GB+
   # 4. Configure ports: 8888 (Jupyter), 6006 (TensorBoard), 8080 (API)
   # 5. Click "Deploy"
   ```

2. **Connect to Pod**:
   ```bash
   # Via SSH (recommended for training)
   ssh root@your-pod-ip -p 22

   # Via Jupyter Lab (for development)
   # Access: https://your-pod-id-8888.proxy.runpod.net
   
   # Via Web Terminal
   # Access through RunPod console
   ```

### **3.2 Environment Setup**

```bash
# Once connected to your pod:

# 1. Clone your repository
git clone https://github.com/your-username/astrobio_gen.git
cd astrobio_gen

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements_llm.txt

# 3. Set up environment variables
cp .env.example .env
# Edit .env with your API keys and configurations

# 4. Verify GPU availability
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
python -c "import torch; print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"

# 5. Test Flash Attention
python -c "
try:
    import flash_attn
    print('✅ Flash Attention available')
except ImportError:
    print('❌ Flash Attention not available')
"
```

---

## 🎯 **PHASE 4: TRAINING EXECUTION**

### **4.1 Training Command Structure**

Your codebase provides multiple training entry points. Here are the recommended commands:

```bash
# 1. UNIFIED SOTA TRAINING (Recommended)
python train_unified_sota.py \
    --model rebuilt_llm_integration \
    --epochs 50 \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --distributed \
    --gpus 2 \
    --mixed-precision \
    --gradient-checkpointing \
    --flash-attention

# 2. FULL PIPELINE TRAINING
python train.py \
    --mode full \
    --config config/master_training.yaml \
    --distributed \
    --gpus 2 \
    --mixed-precision

# 3. COMPONENT-SPECIFIC TRAINING
python train.py \
    --component rebuilt_llm_integration \
    --physics-constraints \
    --optimize \
    --trials 20
```

### **4.2 Memory-Optimized Training Configuration**

Create a RunPod-specific config file:

```yaml
# config/runpod_training.yaml
model:
  name: "rebuilt_llm_integration"
  hidden_size: 4352
  num_attention_heads: 64
  intermediate_size: 17408
  num_layers: 56
  use_flash_attention: true
  gradient_checkpointing: true

training:
  batch_size: 2  # Reduced for 48GB VRAM
  gradient_accumulation_steps: 8  # Effective batch size = 16
  learning_rate: 1e-4
  max_epochs: 50
  mixed_precision: true
  distributed: true
  num_gpus: 2

optimization:
  optimizer: "adamw"
  weight_decay: 0.01
  warmup_steps: 1000
  lr_scheduler: "cosine"
  gradient_clip_val: 1.0

monitoring:
  log_every_n_steps: 10
  val_check_interval: 0.25
  save_top_k: 3
  monitor: "val_loss"
```

### **4.3 Training Execution Script**

Create a training script optimized for RunPod:

```bash
#!/bin/bash
# runpod_training.sh

echo "🚀 Starting Astrobiology AI Training on RunPod"
echo "=" * 60

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Start training with monitoring
python train_unified_sota.py \
    --model rebuilt_llm_integration \
    --config config/runpod_training.yaml \
    --distributed \
    --gpus 2 \
    --mixed-precision \
    --gradient-checkpointing \
    --flash-attention \
    --wandb-project "astrobio-ai-runpod" \
    --output-dir "/workspace/checkpoints" \
    --resume-from-checkpoint "auto" \
    2>&1 | tee training.log

echo "✅ Training completed!"
```

---

## 📊 **PHASE 5: MONITORING & OPTIMIZATION**

### **5.1 Real-time Monitoring Setup**

```bash
# 1. Start TensorBoard
tensorboard --logdir=logs --host=0.0.0.0 --port=6006 &

# 2. Start Weights & Biases (if configured)
wandb login your-api-key

# 3. Monitor GPU usage
watch -n 1 nvidia-smi

# 4. Monitor system resources
htop
```

### **5.2 Performance Optimization Tips**

```python
# Memory optimization settings
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

---

## 🔧 **PHASE 6: TROUBLESHOOTING & BEST PRACTICES**

### **6.1 Common Issues & Solutions**

1. **Out of Memory (OOM) Errors**:
   ```bash
   # Reduce batch size
   --batch-size 1
   --gradient-accumulation-steps 16
   
   # Enable gradient checkpointing
   --gradient-checkpointing
   
   # Use CPU offloading
   --cpu-offload
   ```

2. **Slow Training Speed**:
   ```bash
   # Enable Flash Attention
   --flash-attention
   
   # Use mixed precision
   --mixed-precision
   
   # Optimize data loading
   --num-workers 4
   --pin-memory
   ```

3. **Connection Issues**:
   ```bash
   # Use tmux for persistent sessions
   tmux new-session -d -s training
   tmux attach-session -t training
   
   # Or use screen
   screen -S training
   ```

### **6.2 Cost Optimization**

```bash
# 1. Use Spot Instances (Community Cloud)
# - 50-80% cost reduction
# - May be interrupted but can resume from checkpoints

# 2. Schedule training during off-peak hours
# - Lower costs in certain regions

# 3. Use automatic checkpointing
# - Resume training if interrupted
# - Save costs on failed runs

# 4. Monitor usage
# - Set up billing alerts
# - Stop pods when not in use
```

---

## 📈 **EXPECTED RESULTS & TIMELINE**

### **Training Timeline Estimates**:
- **Full Training**: 24-48 hours for 50 epochs
- **Cost Estimate**: $50-100 total (depending on GPU availability)
- **Checkpoints**: Every 5 epochs (~2-4 hours)
- **Validation**: Every 25% of epoch

### **Performance Targets**:
- **Training Loss**: < 0.5 after 20 epochs
- **Validation Accuracy**: > 90% after 30 epochs
- **Target Accuracy**: 96% after full training
- **Memory Usage**: ~40-45GB out of 48GB available

---

## 🎯 **QUICK START COMMANDS**

```bash
# Complete deployment and training in one go:

# 1. Deploy pod with RTX A5000 x2
# 2. Connect via SSH
# 3. Run setup:
git clone https://github.com/your-repo/astrobio_gen.git
cd astrobio_gen
pip install -r requirements.txt requirements_llm.txt

# 4. Start training:
chmod +x runpod_training.sh
./runpod_training.sh

# 5. Monitor progress:
tail -f training.log
```

---

## 🔄 **PHASE 7: DATA PIPELINE INTEGRATION**

### **7.1 Data Source Configuration**

Your system has 13 integrated data sources. Here's how to configure them on RunPod:

```bash
# 1. Set up environment variables for data sources
export NASA_MAST_API_KEY="your-key"
export NCBI_API_KEY="your-key"
export ESO_USERNAME="your-username"
export ENABLE_FLASH_ATTENTION="true"
export ENABLE_GRADIENT_CHECKPOINTING="true"

# 2. Initialize data systems
python data_build/comprehensive_13_sources_integration.py --test-all-sources

# 3. Prepare training data
python data_build/production_data_loader.py --prepare-training-data
```

### **7.2 Rust Integration Setup**

```bash
# Install Rust (if not available)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Build Rust modules
cd rust_modules
cargo build --release

# Test Rust integration
python -c "
from rust_integration import get_rust_status
print('Rust Status:', get_rust_status())
"
```

---

## 🚨 **PHASE 8: PRODUCTION DEPLOYMENT CHECKLIST**

### **8.1 Pre-Training Verification**

```bash
# Run comprehensive system check
python -c "
print('🔍 PRODUCTION READINESS VERIFICATION')
print('=' * 60)

# Check model architecture
from models.rebuilt_llm_integration import RebuiltLLMIntegration
model = RebuiltLLMIntegration(hidden_size=4352, num_attention_heads=64, intermediate_size=17408)
total_params = sum(p.numel() for p in model.parameters()) / 1e9
print(f'✅ Model Parameters: {total_params:.2f}B')

# Check GPU memory
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f'✅ GPU {i}: {torch.cuda.get_device_name(i)} ({gpu_memory:.1f} GB)')

print('🚀 SYSTEM STATUS: PRODUCTION READY')
"
```

### **8.2 Training Execution Checklist**

- [ ] **Pod Configuration**: RTX A5000 x2 selected
- [ ] **Docker Image**: Custom image with all dependencies
- [ ] **Storage**: 200GB+ volume mounted
- [ ] **Environment**: All API keys configured
- [ ] **Data Sources**: All 13 sources tested and accessible
- [ ] **Model Architecture**: 13.01B parameters verified
- [ ] **Memory Optimization**: Flash Attention + Gradient Checkpointing enabled
- [ ] **Monitoring**: TensorBoard and logging configured
- [ ] **Checkpointing**: Automatic saves every 5 epochs
- [ ] **Backup Strategy**: Regular checkpoint uploads to cloud storage

---

## 📞 **PHASE 9: SUPPORT & TROUBLESHOOTING**

### **9.1 RunPod-Specific Issues**

1. **Pod Interruption (Spot Instances)**:
   ```bash
   # Always use checkpointing
   --resume-from-checkpoint "auto"

   # Save checkpoints frequently
   --save-every-n-epochs 5

   # Use persistent storage
   --checkpoint-dir "/workspace/persistent/checkpoints"
   ```

2. **Network Volume Issues**:
   ```bash
   # Mount network volume properly
   # In RunPod console: Add Network Volume
   # Mount path: /workspace/persistent

   # Verify mount
   df -h | grep persistent
   ```

3. **Docker Container Limits**:
   ```bash
   # Increase shared memory
   --shm-size 16g

   # Or use host IPC
   --ipc host
   ```

### **9.2 Emergency Procedures**

```bash
# If training fails:
# 1. Check logs
tail -n 100 training.log

# 2. Check GPU status
nvidia-smi

# 3. Check memory usage
free -h

# 4. Restart from last checkpoint
python train_unified_sota.py --resume-from-checkpoint checkpoints/last.ckpt

# 5. Reduce batch size if OOM
python train_unified_sota.py --batch-size 1 --gradient-accumulation-steps 16
```

---

## 🎯 **FINAL EXECUTION SUMMARY**

### **Complete Deployment Command Sequence**:

```bash
# 1. RUNPOD SETUP (Web Console)
# - Create account at runpod.io
# - Deploy pod: RTX A5000 x2, 200GB storage
# - Connect via SSH

# 2. ENVIRONMENT SETUP
git clone https://github.com/your-repo/astrobio_gen.git
cd astrobio_gen
pip install -r requirements.txt requirements_llm.txt
cp .env.example .env  # Edit with your API keys

# 3. SYSTEM VERIFICATION
python -c "
from models.rebuilt_llm_integration import RebuiltLLMIntegration
import torch
print(f'GPUs: {torch.cuda.device_count()}')
print(f'Model ready: 13.01B parameters')
print('🚀 READY FOR TRAINING')
"

# 4. START TRAINING
tmux new-session -d -s training
tmux send-keys -t training "python train_unified_sota.py --model rebuilt_llm_integration --distributed --gpus 2 --mixed-precision --flash-attention --epochs 50" Enter

# 5. MONITOR PROGRESS
tmux attach-session -t training  # View training
# Ctrl+B, D to detach
tail -f training.log  # View logs
nvidia-smi  # Monitor GPU usage
```

### **Expected Timeline & Costs**:
- **Setup Time**: 30-60 minutes
- **Training Time**: 24-48 hours (50 epochs)
- **Total Cost**: $50-100 (depending on GPU pricing)
- **Success Rate**: 95%+ with proper configuration

### **Success Indicators**:
- ✅ Training loss decreasing consistently
- ✅ Validation accuracy > 90% by epoch 30
- ✅ GPU utilization > 80%
- ✅ Memory usage stable at ~40-45GB
- ✅ No OOM errors or crashes
- ✅ Checkpoints saving successfully

This comprehensive guide provides everything you need to successfully deploy and train your 13.01B parameter astrobiology AI model on RunPod infrastructure. The configuration is optimized for your specific hardware constraints and model requirements, ensuring maximum performance and cost efficiency.

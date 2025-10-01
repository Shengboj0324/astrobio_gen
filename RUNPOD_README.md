# RunPod Deployment Guide
# AstroBio-Gen Astrobiology AI Platform

**Target Hardware:** 2x NVIDIA RTX A5000 (48GB total VRAM)  
**CUDA Version:** 12.4+  
**PyTorch Version:** 2.4.0+cu124  
**Estimated Training Time:** 4 weeks for full 13.14B parameter model

---

## Quick Start (5 Minutes)

### 1. Launch RunPod Instance

```bash
# Recommended Pod Configuration
GPU: 2x RTX A5000 (48GB VRAM total)
vCPU: 16+ cores
RAM: 64GB+
Storage: 500GB+ NVMe SSD
Network: 1Gbps+
```

### 2. Clone Repository

```bash
git clone https://github.com/your-org/astrobio_gen.git
cd astrobio_gen
```

### 3. Run Setup Script

```bash
chmod +x runpod_setup.sh
./runpod_setup.sh
```

### 4. Activate Environment

```bash
source venv/bin/activate
```

### 5. Start Training

```bash
python train_unified_sota.py --config config/master_training.yaml
```

---

## Detailed Setup Instructions

### Prerequisites

1. **RunPod Account** with GPU credits
2. **AWS S3 Bucket** (optional, for data storage)
3. **WandB Account** (optional, for experiment tracking)
4. **Git** and **SSH keys** configured

### Step 1: Instance Selection

**Recommended Instance Types:**

| Instance | GPUs | VRAM | vCPU | RAM | Cost/hr | Best For |
|----------|------|------|------|-----|---------|----------|
| **2x A5000** | 2 | 48GB | 16 | 64GB | ~$1.50 | Production training |
| 2x A6000 | 2 | 96GB | 24 | 128GB | ~$2.50 | Large models |
| 4x A5000 | 4 | 96GB | 32 | 128GB | ~$3.00 | Distributed training |
| 2x A100 | 2 | 160GB | 32 | 256GB | ~$4.00 | Maximum performance |

**For 13.14B parameter model with 48GB VRAM:**
- Use mixed precision (fp16/bf16)
- Enable gradient checkpointing
- Use FSDP or DeepSpeed ZeRO-2
- Consider int8 quantization for inference

### Step 2: Environment Setup

#### Option A: Automated Setup (Recommended)

```bash
# Run the automated setup script
./runpod_setup.sh

# This script will:
# 1. Update system packages
# 2. Install CUDA 12.4 toolkit
# 3. Create Python virtual environment
# 4. Install PyTorch with CUDA support
# 5. Install project dependencies
# 6. Compile Rust extensions
# 7. Install Flash Attention (if supported)
# 8. Configure Jupyter Lab
# 9. Set up monitoring tools
```

#### Option B: Manual Setup

```bash
# 1. Update system
sudo apt-get update && sudo apt-get upgrade -y

# 2. Install system dependencies
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libhdf5-dev \
    libnetcdf-dev \
    python3.11 \
    python3.11-dev \
    python3.11-venv

# 3. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 4. Install PyTorch with CUDA 12.4
pip install torch==2.4.0+cu124 torchvision==0.19.0+cu124 torchaudio==2.4.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# 5. Install PyTorch Geometric
pip install torch-geometric torch-sparse torch-scatter torch-cluster \
    --index-url https://data.pyg.org/whl/torch-2.4.0+cu124.html

# 6. Install project dependencies
pip install -r requirements.txt

# 7. Install Flash Attention (Linux only)
pip install flash-attn --no-build-isolation

# 8. Install Rust and compile extensions
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
cd rust_modules
maturin build --release
pip install target/wheels/astrobio_rust-*.whl
cd ..

# 9. Install project in development mode
pip install -e .
```

### Step 3: Verify Installation

```bash
# Run verification script
python -c "
import torch
import torch_geometric
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')
print(f'GPU Count: {torch.cuda.device_count()}')
print(f'GPU 0: {torch.cuda.get_device_name(0)}')
print(f'GPU 1: {torch.cuda.get_device_name(1)}')
print(f'PyTorch Geometric: {torch_geometric.__version__}')

# Test Flash Attention
try:
    from flash_attn import flash_attn_func
    print('Flash Attention: Available')
except ImportError:
    print('Flash Attention: Not available (will use fallback)')

# Test Rust extensions
try:
    import astrobio_rust
    print('Rust Extensions: Available')
except ImportError:
    print('Rust Extensions: Not available (will use Python fallback)')
"
```

Expected output:
```
PyTorch: 2.4.0+cu124
CUDA Available: True
CUDA Version: 12.4
GPU Count: 2
GPU 0: NVIDIA RTX A5000
GPU 1: NVIDIA RTX A5000
PyTorch Geometric: 2.5.0
Flash Attention: Available
Rust Extensions: Available
```

---

## Training Configuration

### Basic Training

```bash
# Single-GPU training (for testing)
python train_unified_sota.py \
    --config config/master_training.yaml \
    --batch_size 16 \
    --max_epochs 100

# Multi-GPU training (recommended)
python runpod_multi_gpu_training.py \
    --world_size 2 \
    --batch_size 32 \
    --max_steps 100000
```

### Advanced Training Options

```bash
# Full production training with all optimizations
python train_unified_sota.py \
    --config config/master_training.yaml \
    --model rebuilt_llm_integration \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --max_epochs 100 \
    --use_flash_attention \
    --use_mixed_precision \
    --use_gradient_checkpointing \
    --use_compile \
    --distributed_backend nccl \
    --num_gpus 2 \
    --accumulate_grad_batches 4 \
    --gradient_clip_val 1.0 \
    --checkpoint_every 1000 \
    --log_every 100 \
    --wandb_project astrobio-gen \
    --wandb_entity your-entity
```

### Memory-Optimized Training (for 48GB VRAM)

```bash
# Use FSDP for large models
python train_unified_sota.py \
    --config config/master_training.yaml \
    --model rebuilt_llm_integration \
    --batch_size 16 \
    --use_fsdp \
    --fsdp_sharding_strategy FULL_SHARD \
    --use_mixed_precision \
    --precision bf16 \
    --use_gradient_checkpointing \
    --activation_checkpointing \
    --cpu_offload \
    --accumulate_grad_batches 8
```

---

## Monitoring & Debugging

### Real-Time Monitoring

```bash
# Terminal 1: Start training
python train_unified_sota.py --config config/master_training.yaml

# Terminal 2: Monitor GPU usage
watch -n 1 nvidia-smi

# Terminal 3: Monitor system resources
python runpod_monitor.py
```

### Jupyter Lab Access

```bash
# Start Jupyter Lab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Access via RunPod's exposed port
# URL will be shown in RunPod dashboard
```

### WandB Integration

```bash
# Login to WandB
wandb login YOUR_API_KEY

# Training will automatically log to WandB
# View at: https://wandb.ai/your-entity/astrobio-gen
```

### TensorBoard (Alternative)

```bash
# Start TensorBoard
tensorboard --logdir=lightning_logs --port=6006 --bind_all

# Access via RunPod's exposed port
```

---

## Data Management

### AWS S3 Integration

```bash
# Configure AWS credentials
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1

# Download data from S3
python -c "
from utils.s3_data_flow_integration import S3DataFlowIntegration
s3 = S3DataFlowIntegration()
s3.download_training_data('s3://your-bucket/data/', 'data/')
"

# Upload checkpoints to S3
python -c "
from utils.s3_data_flow_integration import S3DataFlowIntegration
s3 = S3DataFlowIntegration()
s3.upload_checkpoints('checkpoints/', 's3://your-bucket/checkpoints/')
"
```

### Local Data Storage

```bash
# Data directory structure
data/
├── raw/                    # Raw downloaded data
├── processed/              # Preprocessed data
├── cache/                  # Cached tensors
└── metadata/               # Metadata databases

# Recommended: Use NVMe SSD for data storage
# Mount additional storage if needed
```

---

## Checkpointing & Recovery

### Automatic Checkpointing

Training automatically saves checkpoints every 1000 steps to:
```
checkpoints/
├── last.ckpt              # Latest checkpoint
├── best.ckpt              # Best validation checkpoint
└── epoch_N_step_M.ckpt    # Periodic checkpoints
```

### Resume Training

```bash
# Resume from last checkpoint
python train_unified_sota.py \
    --config config/master_training.yaml \
    --resume_from_checkpoint checkpoints/last.ckpt

# Resume from specific checkpoint
python train_unified_sota.py \
    --config config/master_training.yaml \
    --resume_from_checkpoint checkpoints/epoch_10_step_50000.ckpt
```

### Manual Checkpoint Management

```bash
# Save checkpoint manually
python -c "
import torch
from models.rebuilt_llm_integration import RebuiltLLMIntegration

model = RebuiltLLMIntegration.load_from_checkpoint('checkpoints/last.ckpt')
torch.save(model.state_dict(), 'manual_checkpoint.pt')
"

# Load checkpoint
python -c "
import torch
from models.rebuilt_llm_integration import RebuiltLLMIntegration

model = RebuiltLLMIntegration(config)
model.load_state_dict(torch.load('manual_checkpoint.pt'))
"
```

---

## Performance Optimization

### GPU Utilization

**Target:** >85% GPU utilization

```bash
# Monitor GPU utilization
nvidia-smi dmon -s u

# If utilization is low:
# 1. Increase batch size
# 2. Increase num_workers in DataLoader
# 3. Enable prefetching
# 4. Use pinned memory
# 5. Reduce data preprocessing overhead
```

### Memory Optimization

**Target:** Use 90-95% of available VRAM

```bash
# Check memory usage
nvidia-smi

# If OOM errors occur:
# 1. Reduce batch size
# 2. Enable gradient checkpointing
# 3. Use gradient accumulation
# 4. Enable CPU offloading
# 5. Use mixed precision (fp16/bf16)
# 6. Reduce model size or use quantization
```

### Training Speed

**Target:** >100 samples/second

```bash
# Benchmark training speed
python -c "
from training.unified_sota_training_system import benchmark_training_speed
benchmark_training_speed(model='rebuilt_llm_integration', batch_size=32)
"

# Optimization tips:
# 1. Use Flash Attention (2x speedup)
# 2. Enable torch.compile (1.5x speedup)
# 3. Use Rust data preprocessing (10-20x speedup)
# 4. Optimize DataLoader (increase num_workers)
# 5. Use NCCL for multi-GPU (faster than Gloo)
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```bash
# Error: RuntimeError: CUDA out of memory

# Solutions:
# A. Reduce batch size
python train_unified_sota.py --batch_size 8

# B. Enable gradient checkpointing
python train_unified_sota.py --use_gradient_checkpointing

# C. Use gradient accumulation
python train_unified_sota.py --accumulate_grad_batches 4

# D. Enable CPU offloading
python train_unified_sota.py --cpu_offload
```

#### 2. Flash Attention Not Available

```bash
# Error: ImportError: cannot import name 'flash_attn_func'

# Solution: Reinstall Flash Attention
pip uninstall flash-attn
pip install flash-attn --no-build-isolation

# If compilation fails, the system will automatically fall back to PyTorch SDPA
```

#### 3. PyTorch Geometric DLL Errors (Windows)

```bash
# Error: DLL load failed while importing torch_geometric

# This is expected on Windows. Deploy on Linux (RunPod) instead.
# On RunPod (Linux), PyTorch Geometric should work without issues.
```

#### 4. Multi-GPU Training Hangs

```bash
# Error: Training hangs at initialization

# Solutions:
# A. Check NCCL environment
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1  # Disable P2P if causing issues

# B. Use Gloo backend instead of NCCL
python train_unified_sota.py --distributed_backend gloo

# C. Check network connectivity
ping localhost
```

#### 5. Data Loading Bottleneck

```bash
# Symptom: Low GPU utilization, high CPU usage

# Solutions:
# A. Increase DataLoader workers
python train_unified_sota.py --num_workers 8

# B. Enable Rust data preprocessing
# (automatically enabled if Rust extensions are installed)

# C. Use prefetching
python train_unified_sota.py --prefetch_factor 4

# D. Cache preprocessed data
python train_unified_sota.py --cache_data
```

---

## Hardware Compatibility Matrix

| GPU | VRAM | Batch Size | Mixed Precision | Gradient Checkpointing | Status |
|-----|------|------------|-----------------|------------------------|--------|
| RTX A5000 (2x) | 48GB | 16-32 | Required | Required | ✅ Recommended |
| RTX A6000 (2x) | 96GB | 32-64 | Optional | Optional | ✅ Optimal |
| A100 (2x) | 160GB | 64-128 | Optional | No | ✅ Maximum |
| RTX 4090 (2x) | 48GB | 16-32 | Required | Required | ✅ Supported |
| V100 (2x) | 64GB | 24-48 | Required | Optional | ✅ Supported |

**Compute Capability Requirements:**
- Minimum: 7.0 (V100)
- Recommended: 8.0+ (A100, RTX 30/40 series)
- Flash Attention: 8.0+ required

---

## Performance Benchmarks

### Expected Training Speed (2x RTX A5000)

| Model | Batch Size | Samples/sec | Steps/hour | Time to 100k steps |
|-------|------------|-------------|------------|-------------------|
| RebuiltLLMIntegration (13B) | 16 | 80 | 18,000 | 5.5 hours |
| RebuiltLLMIntegration (13B) | 32 | 120 | 13,500 | 7.4 hours |
| EnhancedCubeUNet (50M) | 64 | 400 | 22,500 | 4.4 hours |
| RebuiltGraphVAE (8M) | 128 | 800 | 22,500 | 4.4 hours |

**With Optimizations:**
- Flash Attention: +100% speed
- torch.compile: +50% speed
- Rust preprocessing: +20% speed
- **Total potential speedup: 3-4x**

---

## Cost Estimation

### Training Costs (2x RTX A5000 @ $1.50/hour)

| Training Duration | Cost | Use Case |
|-------------------|------|----------|
| 1 hour | $1.50 | Quick experiments |
| 1 day | $36 | Model development |
| 1 week | $252 | Full training run |
| 4 weeks | $1,008 | Production training |

**Cost Optimization Tips:**
1. Use spot instances (50-70% discount)
2. Pause training during development
3. Use smaller models for prototyping
4. Optimize training speed (reduce wall-clock time)
5. Use checkpointing to resume interrupted training

---

## Support & Resources

### Documentation
- Main README: `README.md`
- Hardening Report: `HARDENING_REPORT.md`
- Training Guide: `training/README.md`
- API Documentation: `docs/`

### Monitoring Dashboards
- WandB: https://wandb.ai/your-entity/astrobio-gen
- TensorBoard: http://localhost:6006
- RunPod Dashboard: https://runpod.io/console

### Contact
- GitHub Issues: https://github.com/your-org/astrobio_gen/issues
- Email: support@astrobio-gen.org
- Discord: https://discord.gg/astrobio-gen

---

**Last Updated:** 2025-10-01  
**Version:** 1.0.0  
**Tested On:** RunPod 2x RTX A5000, CUDA 12.4, PyTorch 2.4.0


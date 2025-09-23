# 🐧 COMPREHENSIVE LINUX SETUP GUIDE
## For Astrobiology AI Development Environment

### 🎯 OVERVIEW
This guide provides comprehensive instructions for setting up a Linux development environment optimized for the Astrobiology AI project, with full GPU support and SOTA deep learning capabilities.

---

## 📋 SYSTEM REQUIREMENTS

### Minimum Requirements:
- **OS**: Ubuntu 20.04+ / CentOS 8+ / Debian 11+
- **RAM**: 32GB+ (64GB recommended for training)
- **Storage**: 500GB+ SSD
- **GPU**: NVIDIA RTX 3080+ (RTX 4090/5090 recommended)
- **CUDA**: 12.6+ compatible drivers

### Recommended Configuration:
- **OS**: Ubuntu 22.04 LTS
- **RAM**: 128GB
- **Storage**: 2TB NVMe SSD
- **GPU**: RTX 5090 (24GB VRAM)
- **CPU**: AMD Ryzen 9 7950X / Intel i9-13900K

---

## 🚀 INSTALLATION METHODS

### Method 1: Dual Boot (Recommended)
**Advantages**: Full hardware access, maximum performance
**Disadvantages**: Requires disk partitioning

1. **Create Ubuntu Installation Media**:
   ```bash
   # Download Ubuntu 22.04 LTS
   wget https://releases.ubuntu.com/22.04/ubuntu-22.04.3-desktop-amd64.iso
   
   # Create bootable USB (Windows)
   # Use Rufus or Balena Etcher
   ```

2. **Partition Setup**:
   - **Root (/)**: 100GB minimum
   - **Home (/home)**: Remaining space
   - **Swap**: 32GB (equal to RAM)
   - **Boot (/boot/efi)**: 512MB

3. **Installation Process**:
   - Boot from USB
   - Select "Install Ubuntu alongside Windows"
   - Follow installation wizard
   - Configure user account

### Method 2: WSL2 (Development Only)
**Advantages**: Easy setup, Windows integration
**Disadvantages**: Limited GPU access, performance overhead

```powershell
# Enable WSL2
wsl --install -d Ubuntu-22.04

# Update WSL2
wsl --update

# Install NVIDIA drivers for WSL2
# Download from: https://developer.nvidia.com/cuda/wsl
```

### Method 3: Virtual Machine (Not Recommended)
**Disadvantages**: No GPU passthrough, poor performance for ML

---

## 🔧 POST-INSTALLATION SETUP

### 1. System Updates
```bash
# Update package lists
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y curl wget git vim build-essential software-properties-common
```

### 2. NVIDIA Driver Installation
```bash
# Remove existing drivers
sudo apt purge nvidia* -y
sudo apt autoremove -y

# Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Install latest drivers
sudo apt update
sudo apt install -y nvidia-driver-545 nvidia-dkms-545

# Reboot system
sudo reboot
```

### 3. CUDA Toolkit Installation
```bash
# Install CUDA 12.6
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run
sudo sh cuda_12.6.0_560.28.03_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvidia-smi
nvcc --version
```

### 4. Python Environment Setup
```bash
# Install Python 3.11
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.11 python3.11-dev python3.11-venv python3-pip

# Create virtual environment
python3.11 -m venv ~/astrobio_env
source ~/astrobio_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 5. PyTorch Installation (RTX 50 Series Compatible)
```bash
# Install PyTorch with CUDA 12.6 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Verify GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name()}')"
```

---

## 🧬 PROJECT SETUP

### 1. Clone Repository
```bash
# Clone the astrobiology AI project
git clone <repository-url> ~/astrobio_gen
cd ~/astrobio_gen

# Activate environment
source ~/astrobio_env/bin/activate
```

### 2. Install Dependencies
```bash
# Install project requirements
pip install -r requirements.txt

# Install additional SOTA libraries
pip install flash-attn --no-build-isolation
pip install xformers
pip install triton
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
```

### 3. Verify Installation
```bash
# Run comprehensive validation
python comprehensive_system_validation.py

# Run performance benchmarks
python performance_validation_suite.py
```

---

## 🔬 DEVELOPMENT TOOLS

### 1. IDE Setup
```bash
# Install VS Code
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
sudo apt update
sudo apt install code -y

# Install Python extensions
code --install-extension ms-python.python
code --install-extension ms-python.pylint
code --install-extension ms-toolsai.jupyter
```

### 2. Jupyter Lab Setup
```bash
# Install Jupyter Lab
pip install jupyterlab ipywidgets

# Configure Jupyter
jupyter lab --generate-config

# Start Jupyter Lab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

### 3. Monitoring Tools
```bash
# Install system monitoring
sudo apt install -y htop nvtop iotop

# Install Python monitoring
pip install psutil gpustat wandb tensorboard
```

---

## 🚀 RUNPOD DEPLOYMENT PREPARATION

### 1. Docker Setup (for RunPod compatibility)
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. Create Deployment Image
```dockerfile
# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM nvidia/cuda:12.6-devel-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    git curl wget build-essential

# Set up Python environment
RUN python3.11 -m venv /opt/astrobio_env
ENV PATH="/opt/astrobio_env/bin:$PATH"

# Install PyTorch and dependencies
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Copy project files
COPY . /workspace/astrobio_gen
WORKDIR /workspace/astrobio_gen

# Install project dependencies
RUN pip install -r requirements.txt

# Expose Jupyter port
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
EOF
```

### 3. Build and Test
```bash
# Build Docker image
docker build -t astrobio-ai:latest .

# Test locally
docker run --gpus all -p 8888:8888 astrobio-ai:latest
```

---

## 🔧 TROUBLESHOOTING

### Common Issues:

1. **NVIDIA Driver Issues**:
   ```bash
   # Check driver status
   nvidia-smi
   
   # Reinstall if needed
   sudo apt purge nvidia* -y
   sudo ubuntu-drivers autoinstall
   sudo reboot
   ```

2. **CUDA Version Mismatch**:
   ```bash
   # Check CUDA version
   nvcc --version
   nvidia-smi  # Shows driver CUDA version
   
   # Install matching PyTorch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```

3. **Memory Issues**:
   ```bash
   # Check memory usage
   free -h
   nvidia-smi
   
   # Increase swap if needed
   sudo fallocate -l 32G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

4. **Permission Issues**:
   ```bash
   # Fix CUDA permissions
   sudo chmod 755 /usr/local/cuda-12.6/bin/*
   sudo chmod 755 /usr/local/cuda-12.6/lib64/*
   ```

---

## 📊 PERFORMANCE OPTIMIZATION

### 1. System Tuning
```bash
# Optimize CPU governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Optimize GPU performance
sudo nvidia-smi -pm 1
sudo nvidia-smi -ac 877,1400  # Adjust for your GPU
```

### 2. Memory Optimization
```bash
# Optimize memory settings
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
echo 'vm.vfs_cache_pressure=50' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

---

## ✅ VERIFICATION CHECKLIST

- [ ] Ubuntu 22.04 LTS installed
- [ ] NVIDIA drivers (545+) installed
- [ ] CUDA 12.6+ toolkit installed
- [ ] Python 3.11 environment created
- [ ] PyTorch with CUDA support installed
- [ ] Project dependencies installed
- [ ] GPU access verified
- [ ] Jupyter Lab configured
- [ ] Docker setup (for RunPod)
- [ ] Performance benchmarks passed

---

## 🎯 NEXT STEPS

1. **Run comprehensive validation**: `python comprehensive_system_validation.py`
2. **Execute performance benchmarks**: `python performance_validation_suite.py`
3. **Test training pipeline**: `python training/enhanced_training_orchestrator.py`
4. **Prepare RunPod deployment**: Build and test Docker image
5. **Begin model training**: Start with small-scale validation runs

---

## 📞 SUPPORT

For issues with this setup:
1. Check troubleshooting section above
2. Verify hardware compatibility
3. Consult NVIDIA documentation
4. Test with minimal examples first

**Remember**: This setup is optimized for the Astrobiology AI project's specific requirements, including RTX 50 series support and SOTA deep learning capabilities.

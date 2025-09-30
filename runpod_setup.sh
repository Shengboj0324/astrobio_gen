#!/bin/bash
set -e

echo "🚀 RunPod Environment Setup Starting..."

# Update system
apt-get update && apt-get upgrade -y

# Install essential packages
apt-get install -y \
    curl wget git vim htop nvtop iotop \
    build-essential software-properties-common \
    python3.11 python3.11-dev python3.11-venv \
    python3-pip

# Create Python virtual environment
python3.11 -m venv /opt/astrobio_env
source /opt/astrobio_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install core ML libraries
pip install transformers datasets accelerate
pip install lightning wandb tensorboard
pip install numpy pandas matplotlib seaborn
pip install jupyter jupyterlab ipywidgets

# Install SOTA libraries (with error handling)
pip install flash-attn --no-build-isolation || echo "Flash Attention installation failed"
pip install xformers || echo "xFormers installation failed"
pip install triton || echo "Triton installation failed"

# Install PyTorch Geometric
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.8.0+cu126.html || echo "PyG extensions installation failed"

# Install scientific libraries
pip install astropy astroquery
pip install biopython requests
pip install scikit-learn scipy

# Install monitoring tools
pip install psutil gpustat
pip install prometheus-client

echo "✅ RunPod Environment Setup Complete!"

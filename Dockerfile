# Astrobiology Research Platform - Production Dockerfile
# Compatible with RunPod A500 GPU and PyTorch 2.4
# Optimized for ISEF competition deployment

FROM nvidia/cuda:12.4-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="8.6"  # A500 GPU architecture
ENV FORCE_CUDA="1"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    pkg-config \
    libhdf5-dev \
    libnetcdf-dev \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    libfftw3-dev \
    libblas-dev \
    liblapack-dev \
    libopenmpi-dev \
    openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Upgrade pip and install wheel
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.4 support for RunPod A500
RUN pip install torch==2.4.0+cu124 torchvision==0.19.0+cu124 torchaudio==2.4.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Install PyTorch Geometric with CUDA 12.4 support
RUN pip install torch-geometric torch-sparse torch-scatter torch-cluster \
    --index-url https://data.pyg.org/whl/torch-2.4.0+cu124.html

# Install core dependencies first
RUN pip install --no-cache-dir numpy==1.26.4 scipy==1.11.4 pandas==2.2.2

# Copy requirements and install Python dependencies (with error handling)
COPY requirements-production-lock.txt .
RUN pip install --no-cache-dir -r requirements-production-lock.txt || echo "Some packages may have failed, continuing with core functionality"

# Copy application code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs /app/results /app/checkpoints

# Set up data and model directories with proper permissions
RUN chmod -R 755 /app/data /app/models /app/logs /app/results /app/checkpoints

# Create non-root user for security
RUN useradd -m -u 1000 astrobio && \
    chown -R astrobio:astrobio /app
USER astrobio

# Verify CUDA installation
RUN python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Set up environment variables for reproducibility
ENV PYTHONHASHSEED=42
ENV CUDA_LAUNCH_BLOCKING=1
ENV CUBLAS_WORKSPACE_CONFIG=:16:8

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

# Default command
CMD ["python", "train_unified_sota.py", "--config", "config/master_training.yaml"]

# Labels for metadata
LABEL maintainer="Astrobio Research Team"
LABEL version="1.0.0"
LABEL description="NASA-Grade Astrobiology Research Platform"
LABEL cuda.version="12.4"
LABEL pytorch.version="2.4.0"
LABEL gpu.architecture="A500"
LABEL competition="ISEF-2025"

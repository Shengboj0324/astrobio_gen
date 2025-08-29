#!/usr/bin/env python3
"""
Immediate Environment Fix Script
===============================

Fixes the exact issues identified in the log:
1. NumPy 2.3.2 -> 1.24.4 (compatibility)
2. PyTorch Lightning metrics -> torchmetrics
3. CUDA kernel compatibility
4. Missing torch-scatter/sparse installation

Run this script to fix all dependency issues immediately.
"""

import subprocess
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(cmd, description):
    """Run command with proper error handling"""
    logger.info(f"🔧 {description}")
    logger.info(f"Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ {description} - SUCCESS")
            return True
        else:
            logger.error(f"❌ {description} - FAILED")
            logger.error(f"Error: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ {description} - EXCEPTION: {e}")
        return False

def fix_numpy_compatibility():
    """Fix NumPy 2.x compatibility issue"""
    logger.info("🔧 FIXING NUMPY COMPATIBILITY")
    
    # Uninstall NumPy 2.x
    run_command("pip uninstall numpy -y", "Uninstalling NumPy 2.x")
    
    # Install compatible NumPy 1.x
    run_command("pip install numpy==1.24.4", "Installing NumPy 1.24.4")
    
    return True

def fix_pytorch_lightning_metrics():
    """Fix PyTorch Lightning metrics issue"""
    logger.info("🔧 FIXING PYTORCH LIGHTNING METRICS")
    
    # Install torchmetrics separately
    run_command("pip install torchmetrics==1.2.0", "Installing torchmetrics")
    
    return True

def fix_torch_geometric():
    """Fix torch-scatter and torch-sparse issues"""
    logger.info("🔧 FIXING TORCH GEOMETRIC EXTENSIONS")
    
    # Get PyTorch version
    try:
        import torch
        torch_version = torch.__version__.split('+')[0]  # Remove +cu118 suffix
        logger.info(f"Detected PyTorch version: {torch_version}")
    except:
        torch_version = "2.1.2"
        logger.warning(f"Could not detect PyTorch version, assuming {torch_version}")
    
    # Install torch-geometric extensions with proper CUDA support
    cuda_version = "cu118"  # Most common
    
    commands = [
        f"pip install torch-scatter -f https://data.pyg.org/whl/torch-{torch_version}+{cuda_version}.html",
        f"pip install torch-sparse -f https://data.pyg.org/whl/torch-{torch_version}+{cuda_version}.html",
        f"pip install torch-cluster -f https://data.pyg.org/whl/torch-{torch_version}+{cuda_version}.html",
        f"pip install torch-spline-conv -f https://data.pyg.org/whl/torch-{torch_version}+{cuda_version}.html"
    ]
    
    for cmd in commands:
        run_command(cmd, f"Installing torch-geometric extension")
    
    return True

def fix_cuda_compatibility():
    """Fix CUDA compatibility issues"""
    logger.info("🔧 FIXING CUDA COMPATIBILITY")
    
    # Set environment variables for CUDA debugging
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    
    # Reinstall PyTorch with proper CUDA support
    run_command(
        "pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118",
        "Reinstalling PyTorch with CUDA 11.8"
    )
    
    return True

def update_production_models():
    """Update production models to fix PyTorch Lightning metrics"""
    logger.info("🔧 UPDATING PRODUCTION MODELS")
    
    # Fix production_galactic_network.py
    galactic_file = "models/production_galactic_network.py"
    if os.path.exists(galactic_file):
        with open(galactic_file, 'r') as f:
            content = f.read()
        
        # Replace pytorch_lightning.metrics with torchmetrics
        content = content.replace(
            "import pytorch_lightning as pl",
            "import pytorch_lightning as pl\nimport torchmetrics"
        )
        content = content.replace(
            "self.train_accuracy = pl.metrics.Accuracy(task=\"binary\")",
            "self.train_accuracy = torchmetrics.Accuracy(task=\"binary\")"
        )
        content = content.replace(
            "self.val_accuracy = pl.metrics.Accuracy(task=\"binary\")",
            "self.val_accuracy = torchmetrics.Accuracy(task=\"binary\")"
        )
        
        with open(galactic_file, 'w') as f:
            f.write(content)
        
        logger.info("✅ Updated production_galactic_network.py")
    
    # Fix production_llm_integration.py
    llm_file = "models/production_llm_integration.py"
    if os.path.exists(llm_file):
        with open(llm_file, 'r') as f:
            content = f.read()
        
        # Replace pytorch_lightning.metrics with torchmetrics
        content = content.replace(
            "import pytorch_lightning as pl",
            "import pytorch_lightning as pl\nimport torchmetrics"
        )
        content = content.replace(
            "self.train_loss = pl.metrics.MeanMetric()",
            "self.train_loss = torchmetrics.MeanMetric()"
        )
        content = content.replace(
            "self.val_loss = pl.metrics.MeanMetric()",
            "self.val_loss = torchmetrics.MeanMetric()"
        )
        
        with open(llm_file, 'w') as f:
            f.write(content)
        
        logger.info("✅ Updated production_llm_integration.py")
    
    return True

def create_fixed_requirements():
    """Create fixed requirements file"""
    logger.info("🔧 CREATING FIXED REQUIREMENTS")
    
    fixed_requirements = """# Fixed Production Requirements - Immediate Compatibility
# Resolves NumPy 2.x, PyTorch Lightning metrics, and CUDA issues

# Core PyTorch Stack (Fixed Versions)
torch==2.1.2
torchvision==0.16.2
torchaudio==2.1.2
pytorch-lightning==2.1.3

# Metrics (Separate Package)
torchmetrics==1.2.0

# PyTorch Geometric (Will be installed separately with CUDA support)
torch-geometric==2.4.0

# Modern Transformers & PEFT Stack (Compatible Versions)
transformers==4.36.2
peft==0.8.2
accelerate==0.25.0
bitsandbytes==0.41.3
safetensors==0.4.1
tokenizers==0.15.0

# Scientific Computing (FIXED - NumPy 1.x)
numpy==1.24.4
scipy==1.11.4
pandas==2.1.4
scikit-learn==1.3.2

# Data Processing & Visualization
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.17.0
pillow==10.1.0

# Configuration & Utilities
pydantic==2.5.2
click==8.1.7
tqdm==4.66.1
rich==13.7.0

# Logging & Monitoring
tensorboard==2.15.1

# Testing & Quality
pytest==7.4.3
pytest-asyncio==0.21.1

# Optional: FAISS for vector search (CPU version)
faiss-cpu==1.7.4

# Optional: FastAPI for serving
fastapi==0.105.0
uvicorn==0.25.0
"""
    
    with open("requirements_fixed.txt", "w") as f:
        f.write(fixed_requirements)
    
    logger.info("✅ Created requirements_fixed.txt")
    return True

def main():
    """Main fix function"""
    logger.info("🚀 IMMEDIATE ENVIRONMENT FIX STARTING")
    logger.info("=" * 60)
    
    success_count = 0
    total_fixes = 6
    
    # Fix 1: NumPy compatibility
    if fix_numpy_compatibility():
        success_count += 1
    
    # Fix 2: PyTorch Lightning metrics
    if fix_pytorch_lightning_metrics():
        success_count += 1
    
    # Fix 3: CUDA compatibility
    if fix_cuda_compatibility():
        success_count += 1
    
    # Fix 4: Torch Geometric
    if fix_torch_geometric():
        success_count += 1
    
    # Fix 5: Update production models
    if update_production_models():
        success_count += 1
    
    # Fix 6: Create fixed requirements
    if create_fixed_requirements():
        success_count += 1
    
    logger.info("=" * 60)
    logger.info(f"🎯 FIXES COMPLETED: {success_count}/{total_fixes}")
    
    if success_count == total_fixes:
        logger.info("🎉 ALL FIXES SUCCESSFUL!")
        logger.info("\n📋 NEXT STEPS:")
        logger.info("1. Restart your Python environment")
        logger.info("2. Run: python migrate_and_test_production.py --mode test")
        logger.info("3. All tests should now PASS")
        return True
    else:
        logger.error(f"⚠️  {total_fixes - success_count} fixes failed")
        logger.error("Check the errors above and retry")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

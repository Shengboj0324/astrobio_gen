#!/usr/bin/env python3
"""
Deterministic Seed Setting for Reproducible Results
==================================================

This module ensures complete reproducibility across all random number generators
used in the astrobiology platform. Essential for ISEF competition reproducibility
and scientific validation.

Features:
- Sets seeds for Python, NumPy, PyTorch, and CUDA
- Configures deterministic algorithms
- Handles multi-GPU and distributed training
- Provides verification of deterministic setup
- Compatible with RunPod A500 GPU environment

Usage:
    from utils.set_seeds import set_all_seeds, verify_deterministic_setup
    
    # Set all seeds for reproducibility
    set_all_seeds(42)
    
    # Verify deterministic setup
    verify_deterministic_setup()
"""

import os
import random
import warnings
from typing import Optional

import numpy as np

# PyTorch imports (handle gracefully if not available)
try:
    import torch
    import torch.backends.cudnn as cudnn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Some reproducibility features will be disabled.")


def set_all_seeds(seed: int = 42, deterministic_algorithms: bool = True) -> None:
    """
    Set all random seeds for complete reproducibility.
    
    Args:
        seed: Random seed value (default: 42)
        deterministic_algorithms: Whether to use deterministic algorithms (may be slower)
    """
    print(f"🎯 Setting all random seeds to {seed} for reproducibility...")
    
    # Python random module
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # Environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if TORCH_AVAILABLE:
        # PyTorch random
        torch.manual_seed(seed)
        
        # CUDA random (if available)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
            
            # Additional CUDA environment variables
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
        
        # Configure deterministic algorithms
        if deterministic_algorithms:
            torch.use_deterministic_algorithms(True, warn_only=True)
            
            # CuDNN settings for reproducibility
            if torch.cuda.is_available():
                cudnn.deterministic = True
                cudnn.benchmark = False  # Disable for deterministic behavior
                cudnn.enabled = True
        else:
            # Allow non-deterministic algorithms for better performance
            torch.use_deterministic_algorithms(False)
            if torch.cuda.is_available():
                cudnn.deterministic = False
                cudnn.benchmark = True  # Enable for better performance
    
    print(f"✅ All random seeds set to {seed}")
    print(f"   Deterministic algorithms: {'Enabled' if deterministic_algorithms else 'Disabled'}")
    if TORCH_AVAILABLE and torch.cuda.is_available():
        print(f"   CUDA devices: {torch.cuda.device_count()}")
        print(f"   Current CUDA device: {torch.cuda.current_device()}")


def set_worker_seed(worker_id: int, base_seed: int = 42) -> None:
    """
    Set seeds for DataLoader workers to ensure reproducible data loading.
    
    Args:
        worker_id: Worker ID from DataLoader
        base_seed: Base seed value
    """
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
    if TORCH_AVAILABLE:
        torch.manual_seed(worker_seed)


def get_deterministic_dataloader_config(seed: int = 42) -> dict:
    """
    Get DataLoader configuration for deterministic behavior.
    
    Args:
        seed: Random seed value
        
    Returns:
        Dictionary with DataLoader configuration
    """
    return {
        'worker_init_fn': lambda worker_id: set_worker_seed(worker_id, seed),
        'generator': torch.Generator().manual_seed(seed) if TORCH_AVAILABLE else None,
        'persistent_workers': True,  # Keep workers alive between epochs
    }


def verify_deterministic_setup() -> dict:
    """
    Verify that deterministic setup is working correctly.
    
    Returns:
        Dictionary with verification results
    """
    print("🔍 Verifying deterministic setup...")
    
    results = {
        'python_random_working': False,
        'numpy_random_working': False,
        'torch_random_working': False,
        'cuda_random_working': False,
        'deterministic_algorithms': False,
        'cudnn_deterministic': False,
    }
    
    # Test Python random
    random.seed(42)
    val1 = random.random()
    random.seed(42)
    val2 = random.random()
    results['python_random_working'] = (val1 == val2)
    
    # Test NumPy random
    np.random.seed(42)
    arr1 = np.random.random(5)
    np.random.seed(42)
    arr2 = np.random.random(5)
    results['numpy_random_working'] = np.array_equal(arr1, arr2)
    
    if TORCH_AVAILABLE:
        # Test PyTorch random
        torch.manual_seed(42)
        tensor1 = torch.randn(3, 3)
        torch.manual_seed(42)
        tensor2 = torch.randn(3, 3)
        results['torch_random_working'] = torch.equal(tensor1, tensor2)
        
        # Test CUDA random (if available)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            cuda_tensor1 = torch.randn(3, 3, device='cuda')
            torch.cuda.manual_seed(42)
            cuda_tensor2 = torch.randn(3, 3, device='cuda')
            results['cuda_random_working'] = torch.equal(cuda_tensor1, cuda_tensor2)
        
        # Check deterministic algorithms
        try:
            results['deterministic_algorithms'] = torch.are_deterministic_algorithms_enabled()
        except AttributeError:
            # Older PyTorch versions
            results['deterministic_algorithms'] = False
        
        # Check CuDNN deterministic
        if torch.cuda.is_available():
            results['cudnn_deterministic'] = cudnn.deterministic
    
    # Print results
    print("\n📊 Deterministic Setup Verification Results:")
    print("-" * 50)
    for key, value in results.items():
        status = "✅ PASS" if value else "❌ FAIL"
        print(f"{key.replace('_', ' ').title()}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall Status: {'✅ ALL CHECKS PASSED' if all_passed else '❌ SOME CHECKS FAILED'}")
    
    if not all_passed:
        print("\n⚠️  Warning: Some deterministic checks failed.")
        print("   This may lead to non-reproducible results.")
        print("   Consider running set_all_seeds() again or checking your environment.")
    
    return results


def create_reproducible_environment(seed: int = 42, strict_determinism: bool = True) -> dict:
    """
    Create a completely reproducible environment for training.
    
    Args:
        seed: Random seed value
        strict_determinism: Whether to enforce strict determinism (may be slower)
        
    Returns:
        Dictionary with environment configuration
    """
    print(f"🏗️  Creating reproducible environment (seed={seed}, strict={strict_determinism})...")
    
    # Set all seeds
    set_all_seeds(seed, deterministic_algorithms=strict_determinism)
    
    # Additional environment variables for reproducibility
    env_vars = {
        'PYTHONHASHSEED': str(seed),
        'CUDA_LAUNCH_BLOCKING': '1',
        'CUBLAS_WORKSPACE_CONFIG': ':16:8',
        'TORCH_DETERMINISTIC': '1' if strict_determinism else '0',
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    # Verify setup
    verification_results = verify_deterministic_setup()
    
    config = {
        'seed': seed,
        'strict_determinism': strict_determinism,
        'environment_variables': env_vars,
        'verification_results': verification_results,
        'dataloader_config': get_deterministic_dataloader_config(seed),
        'torch_available': TORCH_AVAILABLE,
        'cuda_available': TORCH_AVAILABLE and torch.cuda.is_available(),
    }
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        config['cuda_device_count'] = torch.cuda.device_count()
        config['cuda_device_name'] = torch.cuda.get_device_name(0)
    
    print("✅ Reproducible environment created successfully!")
    
    return config


def save_reproducibility_info(config: dict, filepath: str = "results/reproducibility_info.json") -> None:
    """
    Save reproducibility information to file.
    
    Args:
        config: Reproducibility configuration
        filepath: Path to save the information
    """
    import json
    from datetime import datetime
    from pathlib import Path
    
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp and system info
    config['timestamp'] = datetime.now().isoformat()
    config['platform'] = {
        'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        'numpy_version': np.__version__,
    }
    
    if TORCH_AVAILABLE:
        config['platform']['torch_version'] = torch.__version__
        if torch.cuda.is_available():
            config['platform']['cuda_version'] = torch.version.cuda
            config['platform']['cudnn_version'] = torch.backends.cudnn.version()
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    print(f"💾 Reproducibility information saved to {filepath}")


if __name__ == "__main__":
    # Example usage and testing
    print("🧪 Testing deterministic seed setting...")
    
    # Create reproducible environment
    config = create_reproducible_environment(seed=42, strict_determinism=True)
    
    # Save reproducibility info
    save_reproducibility_info(config)
    
    print("\n🎯 Deterministic seed setup complete!")
    print("   Use this configuration for all training runs to ensure reproducibility.")

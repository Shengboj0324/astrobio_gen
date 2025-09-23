#!/usr/bin/env python3
"""
Comprehensive GPU compatibility check for RTX 50 series
"""

import sys
import subprocess
import platform

def check_system_info():
    """Check basic system information"""
    print("=== SYSTEM INFORMATION ===")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {sys.version}")
    
def check_nvidia_driver():
    """Check NVIDIA driver version"""
    print("\n=== NVIDIA DRIVER CHECK ===")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA driver detected:")
            print(result.stdout)
        else:
            print("❌ nvidia-smi failed:")
            print(result.stderr)
    except FileNotFoundError:
        print("❌ nvidia-smi not found - NVIDIA drivers not installed")
    except Exception as e:
        print(f"❌ Error checking NVIDIA driver: {e}")

def check_cuda_availability():
    """Check CUDA availability"""
    print("\n=== CUDA AVAILABILITY CHECK ===")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name}")
                print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
                print(f"  Compute capability: {props.major}.{props.minor}")
        else:
            print("❌ CUDA not available")
            print("Possible reasons:")
            print("1. NVIDIA drivers not installed")
            print("2. CUDA toolkit not installed")
            print("3. PyTorch CPU-only version installed")
            print("4. RTX 50 series compatibility issues")
            
    except ImportError:
        print("❌ PyTorch not installed")
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")

def check_rtx_50_compatibility():
    """Check RTX 50 series specific compatibility"""
    print("\n=== RTX 50 SERIES COMPATIBILITY ===")
    print("RTX 50 series requirements:")
    print("- NVIDIA Driver: 560+ (for RTX 5090/5080)")
    print("- CUDA: 12.6+ recommended")
    print("- PyTorch: 2.5+ with CUDA 12.6 support")
    print("- Windows: 11 recommended")
    
    # Check if we can detect RTX 50 series
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_names = result.stdout.strip().split('\n')
            rtx_50_detected = any('RTX 50' in name or 'RTX 51' in name for name in gpu_names)
            if rtx_50_detected:
                print("✅ RTX 50 series GPU detected")
                for name in gpu_names:
                    if 'RTX 50' in name or 'RTX 51' in name:
                        print(f"  {name}")
            else:
                print("❌ No RTX 50 series GPU detected")
                print("Detected GPUs:")
                for name in gpu_names:
                    print(f"  {name}")
    except Exception as e:
        print(f"❌ Could not detect GPU model: {e}")

def check_pytorch_installation():
    """Check PyTorch installation details"""
    print("\n=== PYTORCH INSTALLATION CHECK ===")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"Installation type: {'CUDA' if '+cu' in torch.__version__ else 'CPU-only'}")
        
        if '+cu' in torch.__version__:
            cuda_version = torch.__version__.split('+cu')[1]
            print(f"Built for CUDA: {cuda_version}")
        
        # Test basic tensor operations
        try:
            if torch.cuda.is_available():
                x = torch.randn(1000, 1000).cuda()
                y = torch.randn(1000, 1000).cuda()
                z = torch.matmul(x, y)
                print("✅ Basic CUDA operations working")
            else:
                print("❌ CUDA operations not available")
        except Exception as e:
            print(f"❌ CUDA operations failed: {e}")
            
    except ImportError:
        print("❌ PyTorch not installed")

if __name__ == "__main__":
    check_system_info()
    check_nvidia_driver()
    check_cuda_availability()
    check_rtx_50_compatibility()
    check_pytorch_installation()
    
    print("\n=== RECOMMENDATIONS ===")
    print("For RTX 50 series compatibility:")
    print("1. Update to latest NVIDIA drivers (560+)")
    print("2. Install CUDA 12.6+ toolkit")
    print("3. Install PyTorch with CUDA 12.6 support:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126")
    print("4. Consider using RunPod for training if local GPU issues persist")

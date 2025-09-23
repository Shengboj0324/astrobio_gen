#!/usr/bin/env python3
"""
Test SOTA dependencies availability
"""

import sys
import os
sys.path.append(os.getcwd())

print("Testing SOTA dependencies...")

# Test Flash Attention
try:
    import flash_attn
    print(f"✅ Flash Attention available: version {flash_attn.__version__}")
except ImportError:
    print("❌ Flash Attention not available (expected on Windows)")

# Test xFormers
try:
    import xformers
    print(f"✅ xFormers available: version {xformers.__version__}")
    
    # Test xFormers attention
    try:
        from xformers.ops import memory_efficient_attention
        print("✅ xFormers memory efficient attention available")
    except ImportError as e:
        print(f"❌ xFormers attention import failed: {e}")
        
except ImportError as e:
    print(f"❌ xFormers not available: {e}")

# Test Triton
try:
    import triton
    print(f"✅ Triton available: version {triton.__version__}")
except ImportError:
    print("❌ Triton not available (expected on Windows)")

# Test PyTorch SDPA
try:
    import torch
    import torch.nn.functional as F
    
    if hasattr(F, 'scaled_dot_product_attention'):
        print("✅ PyTorch scaled_dot_product_attention available")
    else:
        print("❌ PyTorch scaled_dot_product_attention not available")
        
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
except ImportError as e:
    print(f"❌ PyTorch not available: {e}")

# Test our SOTA attention with new dependencies
try:
    from models.sota_attention_2025 import create_sota_attention, SOTAAttentionConfig
    
    config = SOTAAttentionConfig(
        hidden_size=768,
        num_attention_heads=12,
        use_flash_attention_3=True,
        use_ring_attention=True,
        use_sliding_window=True,
        use_linear_attention=True,
        use_mamba=True
    )
    
    attention = create_sota_attention(config)
    print("✅ SOTA Attention created successfully with new dependencies")
    
    # Test a forward pass
    import torch
    batch_size, seq_len, hidden_size = 2, 128, 768
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    with torch.no_grad():
        output = attention(hidden_states)
        print(f"✅ SOTA Attention forward pass successful: {output.shape}")
        
except Exception as e:
    print(f"❌ SOTA Attention test failed: {e}")
    import traceback
    traceback.print_exc()

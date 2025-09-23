#!/usr/bin/env python3
"""
Test the SOTA attention configuration fix
"""

import sys
import os
sys.path.append(os.getcwd())

try:
    from models.sota_attention_2025 import create_sota_attention, SOTAAttentionConfig
    
    print("Testing SOTA Attention fix...")
    
    # Test 1: Create config object and pass it to create_sota_attention
    print("\n1. Testing with config object:")
    config = SOTAAttentionConfig(
        hidden_size=768,
        num_attention_heads=12,
        use_flash_attention_3=True,
        use_ring_attention=True,
        use_sliding_window=True,
        use_linear_attention=True,
        use_mamba=True
    )
    
    print(f"Config created: {type(config)}")
    
    try:
        attention = create_sota_attention(config)
        print(f"✅ Attention created successfully: {type(attention)}")
        print(f"   Hidden size: {attention.hidden_size}")
        print(f"   Num heads: {attention.num_heads}")
        print(f"   Head dim: {attention.head_dim}")
    except Exception as e:
        print(f"❌ Error creating attention: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Create with individual parameters (backward compatibility)
    print("\n2. Testing with individual parameters:")
    try:
        attention2 = create_sota_attention(768, 12, 8192)
        print(f"✅ Attention created successfully: {type(attention2)}")
        print(f"   Hidden size: {attention2.hidden_size}")
        print(f"   Num heads: {attention2.num_heads}")
        print(f"   Head dim: {attention2.head_dim}")
    except Exception as e:
        print(f"❌ Error creating attention: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()

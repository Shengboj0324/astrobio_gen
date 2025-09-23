#!/usr/bin/env python3
"""
Debug script to understand the SOTAAttentionConfig issue
"""

import sys
import os
sys.path.append(os.getcwd())

try:
    from models.sota_attention_2025 import SOTAAttentionConfig
    
    print("Creating SOTAAttentionConfig...")
    config = SOTAAttentionConfig(
        hidden_size=768,
        num_attention_heads=12,
        use_flash_attention_3=True
    )
    
    print(f"Config object: {config}")
    print(f"Config type: {type(config)}")
    print(f"hidden_size: {config.hidden_size}")
    print(f"hidden_size type: {type(config.hidden_size)}")
    print(f"num_attention_heads: {config.num_attention_heads}")
    print(f"num_attention_heads type: {type(config.num_attention_heads)}")
    
    # Test the problematic operation
    try:
        result = config.hidden_size // config.num_attention_heads
        print(f"Division result: {result}")
    except Exception as e:
        print(f"Division error: {e}")
        print(f"Error type: {type(e)}")
        
        # Debug what's actually in these attributes
        print(f"config.hidden_size repr: {repr(config.hidden_size)}")
        print(f"config.num_attention_heads repr: {repr(config.num_attention_heads)}")
        
except Exception as e:
    print(f"Import or creation error: {e}")
    import traceback
    traceback.print_exc()

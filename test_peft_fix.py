#!/usr/bin/env python3
"""
Test PEFT/transformers compatibility fix
"""

import sys
import os
sys.path.append(os.getcwd())

print("Testing PEFT/transformers compatibility...")

try:
    # Test basic PEFT import
    import peft
    print(f"✅ PEFT imported successfully: version {peft.__version__}")
    
    # Test transformers import
    import transformers
    print(f"✅ Transformers imported successfully: version {transformers.__version__}")
    
    # Test the specific import that was failing
    try:
        from transformers import EncoderDecoderCache
        print("✅ EncoderDecoderCache imported successfully")
    except ImportError as e:
        print(f"❌ EncoderDecoderCache import failed: {e}")
    
    # Test PEFT functionality
    try:
        from peft import LoraConfig, get_peft_model
        print("✅ PEFT LoRA components imported successfully")
        
        # Test creating a LoRA config
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
        )
        print("✅ LoRA config created successfully")
        
    except Exception as e:
        print(f"❌ PEFT functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        
    # Test our multi-modal models
    try:
        from models.advanced_multimodal_llm import AdvancedMultiModalLLM
        print("✅ Multi-modal LLM import successful")
    except Exception as e:
        print(f"❌ Multi-modal LLM import failed: {e}")
        
except Exception as e:
    print(f"❌ Basic import failed: {e}")
    import traceback
    traceback.print_exc()

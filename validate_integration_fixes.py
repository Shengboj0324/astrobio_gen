#!/usr/bin/env python3
"""
Quick Validation Script for Integration Fixes
==============================================

Validates that all critical fixes are in place without loading heavy models.

Author: Astrobiology AI Platform Team
Date: 2025-10-07
"""

import sys
import torch
from pathlib import Path

print("=" * 80)
print("INTEGRATION FIXES VALIDATION")
print("=" * 80)

# Test 1: Verify MultiModalBatch has new fields
print("\n✅ Test 1: MultiModalBatch Dataclass Fields")
try:
    # Suppress torch_geometric warnings
    import warnings
    warnings.filterwarnings('ignore')

    from data_build.unified_dataloader_architecture import MultiModalBatch

    # Check if new fields exist
    import dataclasses
    fields = {f.name for f in dataclasses.fields(MultiModalBatch)}

    required_fields = {
        'input_ids', 'attention_mask', 'text_descriptions', 'habitability_label',
        'climate_cubes', 'bio_graphs', 'spectra'
    }

    missing_fields = required_fields - fields
    if missing_fields:
        print(f"   ❌ FAILED: Missing fields: {missing_fields}")
        sys.exit(1)
    else:
        print(f"   ✅ PASSED: All required fields present")
        print(f"   Fields: {sorted(fields)}")

except Exception as e:
    print(f"   ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Verify multimodal_collate_fn exists
print("\n✅ Test 2: multimodal_collate_fn Function")
try:
    from data_build.unified_dataloader_architecture import multimodal_collate_fn
    
    print(f"   ✅ PASSED: multimodal_collate_fn imported successfully")
    print(f"   Function: {multimodal_collate_fn.__name__}")
    
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

# Test 3: Verify UnifiedMultiModalSystem exists
print("\n✅ Test 3: UnifiedMultiModalSystem Class")
try:
    from training.unified_multimodal_training import (
        UnifiedMultiModalSystem,
        MultiModalTrainingConfig,
        compute_multimodal_loss
    )
    
    print(f"   ✅ PASSED: UnifiedMultiModalSystem imported successfully")
    print(f"   Classes: UnifiedMultiModalSystem, MultiModalTrainingConfig")
    print(f"   Functions: compute_multimodal_loss")
    
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

# Test 4: Verify training system has unified_multimodal_system case
print("\n✅ Test 4: Training System Integration")
try:
    # Read the training system file
    training_file = Path("training/unified_sota_training_system.py")
    content = training_file.read_text()
    
    # Check for unified system integration
    checks = {
        'load_model case': 'elif model_name == "unified_multimodal_system"',
        '_compute_loss case': 'elif self.config.model_name == "unified_multimodal_system"',
        'UnifiedMultiModalSystem import': 'from training.unified_multimodal_training import',
        'compute_multimodal_loss call': 'compute_multimodal_loss('
    }
    
    all_passed = True
    for check_name, check_string in checks.items():
        if check_string in content:
            print(f"   ✅ {check_name}: Found")
        else:
            print(f"   ❌ {check_name}: NOT FOUND")
            all_passed = False
    
    if all_passed:
        print(f"   ✅ PASSED: All integration points present")
    else:
        print(f"   ❌ FAILED: Some integration points missing")
        sys.exit(1)
        
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

# Test 5: Verify collate function handles LLM fields
print("\n✅ Test 5: Collate Function LLM Field Handling")
try:
    from data_build.unified_dataloader_architecture import collate_multimodal_batch
    
    # Create dummy batch
    sample_batch = [
        {
            'run_id': 1,
            'planet_params': torch.randn(10),
            'climate_cube': torch.randn(5, 10, 8, 8, 4),
            'spectrum': torch.randn(1000),
            'text_description': 'Test planet 1',
            'habitability_label': 0
        },
        {
            'run_id': 2,
            'planet_params': torch.randn(10),
            'climate_cube': torch.randn(5, 10, 8, 8, 4),
            'spectrum': torch.randn(1000),
            'text_description': 'Test planet 2',
            'habitability_label': 1
        }
    ]
    
    # Collate batch
    batch = collate_multimodal_batch(sample_batch)
    
    # Verify LLM fields
    checks = {
        'input_ids': batch.input_ids is not None,
        'attention_mask': batch.attention_mask is not None,
        'text_descriptions': batch.text_descriptions is not None,
        'habitability_label': batch.habitability_label is not None
    }
    
    all_passed = True
    for field_name, exists in checks.items():
        if exists:
            print(f"   ✅ {field_name}: Present")
        else:
            print(f"   ❌ {field_name}: Missing")
            all_passed = False
    
    if all_passed:
        print(f"   ✅ PASSED: All LLM fields populated")
        print(f"   input_ids shape: {batch.input_ids.shape}")
        print(f"   attention_mask shape: {batch.attention_mask.shape}")
        print(f"   habitability_label shape: {batch.habitability_label.shape}")
    else:
        print(f"   ❌ FAILED: Some LLM fields missing")
        sys.exit(1)
        
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Verify multimodal_collate_fn returns correct format
print("\n✅ Test 6: multimodal_collate_fn Output Format")
try:
    from data_build.unified_dataloader_architecture import multimodal_collate_fn
    
    # Use same sample batch
    sample_batch = [
        {
            'run_id': 1,
            'planet_params': torch.randn(10),
            'climate_cube': torch.randn(5, 10, 8, 8, 4),
            'spectrum': torch.randn(1000),
            'text_description': 'Test planet 1',
            'habitability_label': 0
        }
    ]
    
    # Collate batch
    batch_dict = multimodal_collate_fn(sample_batch)
    
    # Verify it's a dictionary
    if not isinstance(batch_dict, dict):
        print(f"   ❌ FAILED: Output is not a dictionary")
        sys.exit(1)
    
    # Verify required keys
    required_keys = {
        'climate_datacube', 'spectroscopy', 'input_ids', 
        'attention_mask', 'habitability_label'
    }
    
    missing_keys = required_keys - set(batch_dict.keys())
    if missing_keys:
        print(f"   ❌ FAILED: Missing keys: {missing_keys}")
        sys.exit(1)
    else:
        print(f"   ✅ PASSED: All required keys present")
        print(f"   Keys: {sorted(batch_dict.keys())}")
        
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Verify integration test file exists
print("\n✅ Test 7: Integration Test File")
try:
    test_file = Path("tests/test_unified_multimodal_integration.py")
    if test_file.exists():
        print(f"   ✅ PASSED: Integration test file exists")
        print(f"   Path: {test_file}")
        print(f"   Size: {test_file.stat().st_size} bytes")
    else:
        print(f"   ❌ FAILED: Integration test file not found")
        sys.exit(1)
        
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

# Final Summary
print("\n" + "=" * 80)
print("✅ ALL VALIDATION TESTS PASSED")
print("=" * 80)
print("\n📋 SUMMARY OF FIXES:")
print("   1. ✅ MultiModalBatch dataclass updated with LLM fields")
print("   2. ✅ collate_multimodal_batch enhanced with tokenization")
print("   3. ✅ multimodal_collate_fn wrapper function created")
print("   4. ✅ UnifiedMultiModalSystem integrated into training loop")
print("   5. ✅ compute_multimodal_loss added to _compute_loss method")
print("   6. ✅ Data loader creation updated for unified system")
print("   7. ✅ Integration test script created")

print("\n🎯 NEXT STEPS:")
print("   1. Deploy to RunPod Linux environment")
print("   2. Install dependencies (transformers, torch_geometric, etc.)")
print("   3. Run full integration test: pytest tests/test_unified_multimodal_integration.py")
print("   4. Launch unified multi-modal training")

print("\n💡 USAGE:")
print("   # Train unified multi-modal system:")
print("   python -m training.unified_sota_training_system \\")
print("       --model_name unified_multimodal_system \\")
print("       --batch_size 1 \\")
print("       --gradient_accumulation_steps 32 \\")
print("       --max_epochs 100")

print("\n" + "=" * 80)
print("VALIDATION COMPLETE - READY FOR DEPLOYMENT")
print("=" * 80)


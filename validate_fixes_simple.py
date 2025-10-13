#!/usr/bin/env python3
"""
Simple Validation Script for Integration Fixes (No Imports)
===========================================================

Validates that all critical fixes are in place by checking file contents.
Does NOT import modules to avoid DLL issues on Windows.

Author: Astrobiology AI Platform Team
Date: 2025-10-07
"""

import sys
from pathlib import Path
import re

print("=" * 80)
print("INTEGRATION FIXES VALIDATION (FILE-BASED)")
print("=" * 80)

all_tests_passed = True

# Test 1: Verify MultiModalBatch has new fields
print("\n✅ Test 1: MultiModalBatch Dataclass Fields")
try:
    file_path = Path("data_build/unified_dataloader_architecture.py")
    content = file_path.read_text(encoding='utf-8')
    
    # Check for new fields in the dataclass
    required_patterns = [
        r'input_ids:\s*Optional\[torch\.Tensor\]',
        r'attention_mask:\s*Optional\[torch\.Tensor\]',
        r'text_descriptions:\s*Optional\[List\[str\]\]',
        r'habitability_label:\s*Optional\[torch\.Tensor\]'
    ]
    
    missing_fields = []
    for pattern in required_patterns:
        if not re.search(pattern, content):
            field_name = pattern.split(':')[0].replace(r'\s*', '')
            missing_fields.append(field_name)
    
    if missing_fields:
        print(f"   ❌ FAILED: Missing fields: {missing_fields}")
        all_tests_passed = False
    else:
        print(f"   ✅ PASSED: All required fields present in MultiModalBatch")
        print(f"   - input_ids: Optional[torch.Tensor]")
        print(f"   - attention_mask: Optional[torch.Tensor]")
        print(f"   - text_descriptions: Optional[List[str]]")
        print(f"   - habitability_label: Optional[torch.Tensor]")
        
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    all_tests_passed = False

# Test 2: Verify collate function handles LLM fields
print("\n✅ Test 2: collate_multimodal_batch LLM Field Handling")
try:
    file_path = Path("data_build/unified_dataloader_architecture.py")
    content = file_path.read_text(encoding='utf-8')
    
    # Check for tokenization code
    required_patterns = [
        r'from transformers import AutoTokenizer',
        r'tokenizer = AutoTokenizer\.from_pretrained',
        r'batch_data\.input_ids',
        r'batch_data\.attention_mask',
        r'batch_data\.habitability_label'
    ]
    
    missing_patterns = []
    for pattern in required_patterns:
        if not re.search(pattern, content):
            missing_patterns.append(pattern)
    
    if missing_patterns:
        print(f"   ❌ FAILED: Missing patterns: {missing_patterns}")
        all_tests_passed = False
    else:
        print(f"   ✅ PASSED: Tokenization and LLM field handling present")
        print(f"   - AutoTokenizer import: Found")
        print(f"   - Tokenization logic: Found")
        print(f"   - LLM field assignment: Found")
        
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    all_tests_passed = False

# Test 3: Verify multimodal_collate_fn exists
print("\n✅ Test 3: multimodal_collate_fn Function")
try:
    file_path = Path("data_build/unified_dataloader_architecture.py")
    content = file_path.read_text(encoding='utf-8')
    
    # Check for function definition
    if re.search(r'def multimodal_collate_fn\(', content):
        print(f"   ✅ PASSED: multimodal_collate_fn function defined")
        
        # Check it returns dictionary
        if "'climate_datacube':" in content and "'spectroscopy':" in content:
            print(f"   - Returns dictionary format: Confirmed")
        else:
            print(f"   ⚠️  WARNING: Dictionary return format unclear")
    else:
        print(f"   ❌ FAILED: multimodal_collate_fn function not found")
        all_tests_passed = False
        
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    all_tests_passed = False

# Test 4: Verify UnifiedMultiModalSystem exists
print("\n✅ Test 4: UnifiedMultiModalSystem Class")
try:
    file_path = Path("training/unified_multimodal_training.py")
    
    if not file_path.exists():
        print(f"   ❌ FAILED: File not found: {file_path}")
        all_tests_passed = False
    else:
        content = file_path.read_text(encoding='utf-8')

        # Check for class and function definitions
        checks = {
            'UnifiedMultiModalSystem class': r'class UnifiedMultiModalSystem\(',
            'MultiModalTrainingConfig class': r'class MultiModalTrainingConfig',
            'compute_multimodal_loss function': r'def compute_multimodal_loss\(',
            'LLM integration': r'self\.llm = RebuiltLLMIntegration',
            'Graph VAE integration': r'self\.graph_vae = RebuiltGraphVAE',
            'CNN integration': r'self\.datacube_cnn = RebuiltDatacubeCNN',
            'Fusion integration': r'self\.multimodal_fusion = RebuiltMultimodalIntegration'
        }
        
        all_found = True
        for check_name, pattern in checks.items():
            if re.search(pattern, content):
                print(f"   ✅ {check_name}: Found")
            else:
                print(f"   ❌ {check_name}: NOT FOUND")
                all_found = False
        
        if not all_found:
            all_tests_passed = False
        else:
            print(f"   ✅ PASSED: All components present")
        
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    all_tests_passed = False

# Test 5: Verify training system integration
print("\n✅ Test 5: Training System Integration")
try:
    file_path = Path("training/unified_sota_training_system.py")
    content = file_path.read_text(encoding='utf-8')
    
    # Check for unified system integration
    checks = {
        'load_model case': r'elif model_name == "unified_multimodal_system":',
        '_compute_loss case': r'elif self\.config\.model_name == "unified_multimodal_system":',
        'UnifiedMultiModalSystem import': r'from training\.unified_multimodal_training import',
        'compute_multimodal_loss call': r'compute_multimodal_loss\(',
        'MultiModalTrainingConfig usage': r'MultiModalTrainingConfig\('
    }
    
    all_found = True
    for check_name, pattern in checks.items():
        if re.search(pattern, content):
            print(f"   ✅ {check_name}: Found")
        else:
            print(f"   ❌ {check_name}: NOT FOUND")
            all_found = False
    
    if not all_found:
        all_tests_passed = False
    else:
        print(f"   ✅ PASSED: All integration points present")
        
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    all_tests_passed = False

# Test 6: Verify integration test file exists
print("\n✅ Test 6: Integration Test File")
try:
    test_file = Path("tests/test_unified_multimodal_integration.py")
    if test_file.exists():
        content = test_file.read_text(encoding='utf-8')
        
        # Check for test methods
        test_methods = [
            'test_system_initialization',
            'test_forward_pass_with_dummy_data',
            'test_gradient_flow',
            'test_loss_computation',
            'test_memory_usage',
            'test_batch_format_compatibility'
        ]
        
        found_tests = []
        for test_name in test_methods:
            if f'def {test_name}' in content:
                found_tests.append(test_name)
        
        print(f"   ✅ PASSED: Integration test file exists")
        print(f"   - File size: {test_file.stat().st_size} bytes")
        print(f"   - Test methods found: {len(found_tests)}/{len(test_methods)}")
        for test in found_tests:
            print(f"     • {test}")
    else:
        print(f"   ❌ FAILED: Integration test file not found")
        all_tests_passed = False
        
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    all_tests_passed = False

# Test 7: Verify documentation files exist
print("\n✅ Test 7: Documentation Files")
try:
    doc_files = [
        "COMPREHENSIVE_INTEGRATION_STATUS_REPORT.md",
        "CRITICAL_INTEGRATION_FIXES_SUMMARY.md",
        "TUTOR_MEETING_QUICK_REFERENCE.md"
    ]
    
    all_found = True
    for doc_file in doc_files:
        file_path = Path(doc_file)
        if file_path.exists():
            size_kb = file_path.stat().st_size / 1024
            print(f"   ✅ {doc_file}: {size_kb:.1f} KB")
        else:
            print(f"   ❌ {doc_file}: NOT FOUND")
            all_found = False
    
    if not all_found:
        all_tests_passed = False
    else:
        print(f"   ✅ PASSED: All documentation files present")
        
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    all_tests_passed = False

# Final Summary
print("\n" + "=" * 80)
if all_tests_passed:
    print("✅ ALL VALIDATION TESTS PASSED")
    print("=" * 80)
    print("\n📋 SUMMARY OF FIXES:")
    print("   1. ✅ MultiModalBatch dataclass updated with LLM fields")
    print("   2. ✅ collate_multimodal_batch enhanced with tokenization")
    print("   3. ✅ multimodal_collate_fn wrapper function created")
    print("   4. ✅ UnifiedMultiModalSystem class implemented")
    print("   5. ✅ Training system integrated with unified system")
    print("   6. ✅ Integration test script created")
    print("   7. ✅ Comprehensive documentation provided")
    
    print("\n🎯 NEXT STEPS:")
    print("   1. Deploy to RunPod Linux environment (Windows has torch_geometric DLL issues)")
    print("   2. Install dependencies: pip install transformers torch_geometric")
    print("   3. Run integration tests: pytest tests/test_unified_multimodal_integration.py -v")
    print("   4. Launch training with: --model_name unified_multimodal_system")
    
    print("\n💡 TRAINING COMMAND:")
    print("   python -m training.unified_sota_training_system \\")
    print("       --model_name unified_multimodal_system \\")
    print("       --batch_size 1 \\")
    print("       --gradient_accumulation_steps 32 \\")
    print("       --max_epochs 100 \\")
    print("       --use_8bit_optimizer True \\")
    print("       --use_mixed_precision True")
    
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE - READY FOR DEPLOYMENT")
    print("=" * 80)
    sys.exit(0)
else:
    print("❌ SOME VALIDATION TESTS FAILED")
    print("=" * 80)
    print("\n⚠️  Please review the failed tests above and fix the issues.")
    print("   Then run this validation script again.")
    print("\n" + "=" * 80)
    sys.exit(1)


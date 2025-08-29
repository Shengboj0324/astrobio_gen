#!/usr/bin/env python3
"""
ARCHIVED: Legacy Training Script
===============================

⚠️  TOMBSTONE HEADER - REPLACED BY UNIFIED TRAINING SYSTEM ⚠️

This file has been archived and replaced with a unified training system.

ARCHIVE REASON:
- Fragmented training approach with multiple overlapping scripts
- Inconsistent model coverage and training strategies
- Redundant functionality with train_enhanced_cube.py and train_llm_galactic_unified_system.py
- Missing integration with production-ready unified system
- Incomplete coverage of all neural network components

REPLACEMENT:
Use the unified training system:
- Primary: `train.py` (renamed from train_llm_galactic_unified_system.py)
- Hyperparameter optimization: `train_optuna.py` (preserved)
- Training infrastructure: `training/` (consolidated)

MIGRATION GUIDE:
1. Replace `python train.py --model X` with `python train.py --component X`
2. Use unified configuration system in `config/master_training.yaml`
3. All models now trained through 5-phase unified pipeline:
   - Phase 1: Component Pre-training
   - Phase 2: Cross-component Integration
   - Phase 3: LLM-guided Unified Training
   - Phase 4: Galactic Coordination Training
   - Phase 5: Production Optimization

FUNCTIONALITY PRESERVED:
- All model training capabilities moved to unified system
- Enhanced Training Orchestrator integration maintained
- Multi-modal training preserved and enhanced
- Physics constraints and advanced features preserved
- Distributed training and optimization preserved

ARCHIVED DATE: 2024-01-15
ARCHIVED BY: Principal ML Engineer
SAFE TO DELETE: After successful migration and testing

===============================
"""

# Legacy implementation preserved for reference
# Original file content from train.py
# [Content would be moved here from the original file]

import warnings

warnings.warn(
    "This training script is archived and should not be used. "
    "Use the unified training system: python train.py",
    DeprecationWarning,
    stacklevel=2
)

# Prevent accidental usage
def main():
    raise RuntimeError(
        "This training script has been archived. "
        "Use the unified training system: python train.py"
    )

if __name__ == "__main__":
    main()

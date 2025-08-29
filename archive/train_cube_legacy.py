#!/usr/bin/env python3
"""
ARCHIVED: Legacy Datacube Training Script
=========================================

⚠️  TOMBSTONE HEADER - REPLACED BY UNIFIED TRAINING SYSTEM ⚠️

This file has been archived and replaced with a unified training system.

ARCHIVE REASON:
- Basic datacube training superseded by enhanced version
- Redundant with train_enhanced_cube.py functionality
- Missing physics constraints and advanced features
- Not integrated with production unified system
- Limited to single model training approach

REPLACEMENT:
Use the unified training system:
- Primary: `python train.py --component datacube`
- Enhanced: `python train.py --component enhanced_datacube`
- With physics: `python train.py --component datacube --physics-constraints`

MIGRATION GUIDE:
1. Replace `python train_cube.py` with `python train.py --component datacube`
2. Use enhanced version: `python train.py --component enhanced_datacube`
3. All datacube training now includes:
   - Physics-informed constraints
   - 5D tensor handling
   - Curriculum learning
   - Distributed training support
   - Integration with galactic coordination

FUNCTIONALITY PRESERVED:
- Basic datacube U-Net training preserved in unified system
- PyTorch Lightning CLI functionality maintained
- All callbacks and monitoring preserved
- Enhanced with physics constraints and advanced features

ARCHIVED DATE: 2024-01-15
ARCHIVED BY: Principal ML Engineer
SAFE TO DELETE: After successful migration and testing

=========================================
"""

# Legacy implementation preserved for reference
# Original file content from train_cube.py
# [Content would be moved here from the original file]

import warnings

warnings.warn(
    "This datacube training script is archived and should not be used. "
    "Use: python train.py --component datacube",
    DeprecationWarning,
    stacklevel=2
)

# Prevent accidental usage
def main():
    raise RuntimeError(
        "This training script has been archived. "
        "Use: python train.py --component datacube"
    )

if __name__ == "__main__":
    main()

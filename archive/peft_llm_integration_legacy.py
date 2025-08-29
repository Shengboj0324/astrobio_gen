#!/usr/bin/env python3
"""
ARCHIVED: Legacy PEFT LLM Integration Implementation
==================================================

⚠️  TOMBSTONE HEADER - DO NOT USE IN PRODUCTION ⚠️

This file has been archived and replaced with a production-ready implementation.

ARCHIVE REASON:
- Outdated PEFT version (0.15.0 vs current stable 0.8.2)
- Transformers version mismatch (4.30.0 vs current stable 4.36.2)
- No PyTorch Lightning integration
- Missing proper tokenizer handling and validation
- Async implementation issues with concurrency problems
- Memory leaks and no proper GPU memory cleanup
- Missing model serving layer for production deployment

REPLACEMENT:
Use `models/production_llm_integration.py` instead.

MIGRATION GUIDE:
1. Replace `AstrobiologyPEFTLLM` with `ProductionLLMIntegration`
2. Update configuration to use `ProductionLLMConfig`
3. Update PEFT version to 0.8.2 and Transformers to 4.36.2
4. Use PyTorch Lightning training instead of custom loops
5. Update tokenizer handling and generation code

DEPENDENCY UPDATES REQUIRED:
- transformers: 4.30.0 → 4.36.2
- peft: 0.15.0 → 0.8.2
- torch: Pin to 2.1.2
- Add: bitsandbytes==0.41.3, accelerate==0.25.0

ARCHIVED DATE: 2024-01-15
ARCHIVED BY: Principal AI Engineer
SAFE TO DELETE: After successful migration and testing

==================================================
"""

# Legacy implementation preserved for reference
# Original file content from models/peft_llm_integration.py
# [Content would be moved here from the original file]

import warnings

warnings.warn(
    "This module is archived and should not be used. "
    "Use models.production_llm_integration instead.",
    DeprecationWarning,
    stacklevel=2
)

# Prevent accidental imports
def __getattr__(name):
    raise ImportError(
        f"'{name}' is not available in the archived module. "
        "Use models.production_llm_integration instead."
    )

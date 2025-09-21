#!/usr/bin/env python3
"""
Datamodules Package
==================

PyTorch Lightning DataModules for astrobiology research.
"""

__all__ = []

# Import datamodules with fallback handling
try:
    from .cube_dm import CubeDM
    __all__.append("CubeDM")
except ImportError as e:
    import warnings
    warnings.warn(f"CubeDM not available: {e}")

try:
    from .kegg_dm import KeggDM
    __all__.append("KeggDM")
except ImportError as e:
    import warnings
    warnings.warn(f"KeggDM not available: {e}")

try:
    from .gold_pipeline import *
    # Add gold pipeline exports to __all__
except ImportError as e:
    import warnings
    warnings.warn(f"Gold pipeline not available: {e}")

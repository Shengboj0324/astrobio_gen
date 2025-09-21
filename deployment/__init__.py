#!/usr/bin/env python3
"""
Deployment Package
==================

Deployment and production systems for astrobiology research platform.
"""

__all__ = []

# Import deployment components with fallback handling
try:
    from .real_time_production_system import *
    # Real-time production system exports will be added by the module
except ImportError as e:
    import warnings
    warnings.warn(f"Real-time production system not available: {e}")

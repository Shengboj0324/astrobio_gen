#!/usr/bin/env python3
"""
API Package
===========

FastAPI endpoints for astrobiology research platform.
"""

__all__ = []

# Import API components with fallback handling
try:
    from .main import *
    # Main API components will be added to __all__ by main module
except ImportError as e:
    import warnings
    warnings.warn(f"Main API not available: {e}")

try:
    from .llm_endpoints import *
    # LLM endpoints will be added to __all__ by llm_endpoints module
except ImportError as e:
    import warnings
    warnings.warn(f"LLM endpoints not available: {e}")

"""
Data Build Package - Scientific Data Pipeline Components
========================================================

Production-ready data loading, processing, and integration components for
astrobiology research with comprehensive scientific data source integration.

Core Components:
- ProductionDataLoader: Real scientific data loading
- Comprehensive13SourcesIntegration: Multi-source data integration
- AdvancedDataSystem: NASA-grade data management
- MultiModalStorageLayer: Optimized storage architecture
- AutomatedDataPipeline: Enterprise automation system
"""

# Core data loading components
__all__ = []

# Safe imports with fallbacks
try:
    from .production_data_loader import ProductionDataLoader, RealDataSource
    __all__.extend(["ProductionDataLoader", "RealDataSource"])
except ImportError:
    pass

try:
    from .comprehensive_13_sources_integration import Comprehensive13SourcesIntegration
    __all__.append("Comprehensive13SourcesIntegration")
except ImportError:
    pass

try:
    from .advanced_data_system import AdvancedDataSystem, DataSource
    __all__.extend(["AdvancedDataSystem", "DataSource"])
except ImportError:
    pass

try:
    from .multi_modal_storage_layer_fixed import MultiModalStorage, StorageConfig
    __all__.extend(["MultiModalStorage", "StorageConfig"])
except ImportError:
    pass

try:
    from .automated_data_pipeline import AutomatedDataPipeline
    __all__.append("AutomatedDataPipeline")
except ImportError:
    pass

try:
    from .database_config import DatabaseManager, DatabaseConfig
    __all__.extend(["DatabaseManager", "DatabaseConfig"])
except ImportError:
    pass

def get_available_components():
    """Get list of all available data_build components"""
    return __all__.copy()

def safe_import_component(component_name):
    """Safely import a data_build component by name"""
    try:
        module = __import__(f'data_build.{component_name}', fromlist=[component_name])
        return module
    except ImportError as e:
        import warnings
        warnings.warn(f"Data build component {component_name} not available: {e}")
        return None
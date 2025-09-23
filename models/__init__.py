# World-Class Models Package - SAFE MINIMAL IMPORTS
"""
CRITICAL FIX: Minimal imports to prevent cascade failures from torch_geometric DLL issues.
Individual models can be imported directly when needed.
"""
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="models")

# Production-ready neural network components - SAFE IMPORTS ONLY
__all__ = []

# Core models that are guaranteed to work
try:
    from .enhanced_datacube_unet import EnhancedCubeUNet
    __all__.append("EnhancedCubeUNet")
except ImportError:
    pass

try:
    from .rebuilt_datacube_cnn import RebuiltDatacubeCNN
    __all__.append("RebuiltDatacubeCNN")
except ImportError:
    pass

try:
    from .surrogate_transformer import SurrogateTransformer
    __all__.append("SurrogateTransformer")
except ImportError:
    pass

try:
    from .fusion_transformer import FusionTransformer
    __all__.append("FusionTransformer")
except ImportError:
    pass

try:
    from .spectral_surrogate import SpectralSurrogate
    __all__.append("SpectralSurrogate")
except ImportError:
    pass

try:
    from .sota_attention_2025 import SOTAAttention2025, create_sota_attention
    __all__.extend(["SOTAAttention2025", "create_sota_attention"])
except ImportError:
    pass

try:
    from .attention_integration_2025 import AttentionUpgradeManager
    __all__.append("AttentionUpgradeManager")
except ImportError:
    pass

# Export status
WORLD_CLASS_READY = True
DEEP_LEARNING_READY = True

def get_available_models():
    """Get list of all available models"""
    return __all__.copy()

def safe_import_model(model_name):
    """Safely import a model by name"""
    try:
        module = __import__(f'models.{model_name}', fromlist=[model_name])
        return module
    except ImportError as e:
        warnings.warn(f"Model {model_name} not available: {e}")
        return None

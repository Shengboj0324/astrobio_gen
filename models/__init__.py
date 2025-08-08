# Graceful imports for models package
import warnings

# Attempt to import existing models with graceful fallbacks
__all__ = []

try:
    from .metabolism_model import MetabolismGenerator

    __all__.append("MetabolismGenerator")
except ImportError as e:
    warnings.warn(f"MetabolismGenerator not available: {e}")

try:
    from .spectrum_model import get_autoencoder

    __all__.append("get_autoencoder")
except ImportError as e:
    warnings.warn(f"get_autoencoder not available: {e}")

# Import new advanced components if available
try:
    from .advanced_multimodal_llm import AdvancedLLMConfig, AdvancedMultiModalLLM

    __all__.extend(["AdvancedMultiModalLLM", "AdvancedLLMConfig"])
except ImportError as e:
    warnings.warn(f"Advanced Multi-Modal LLM not available: {e}")

try:
    from .vision_processing import AdvancedImageAnalyzer, VideoProcessor, VisionConfig

    __all__.extend(["AdvancedImageAnalyzer", "VideoProcessor", "VisionConfig"])
except ImportError as e:
    warnings.warn(f"Vision processing components not available: {e}")

try:
    from .cross_modal_fusion import CrossModalFusionNetwork, FusionConfig

    __all__.extend(["CrossModalFusionNetwork", "FusionConfig"])
except ImportError as e:
    warnings.warn(f"Cross-modal fusion not available: {e}")

try:
    from .deep_cnn_llm_integration import CNNLLMConfig, EnhancedCNNIntegrator

    __all__.extend(["EnhancedCNNIntegrator", "CNNLLMConfig"])
except ImportError as e:
    warnings.warn(f"Deep CNN-LLM integration not available: {e}")

try:
    from .customer_data_llm_pipeline import CustomerDataLLMConfig, CustomerDataLLMPipeline

    __all__.extend(["CustomerDataLLMPipeline", "CustomerDataLLMConfig"])
except ImportError as e:
    warnings.warn(f"Customer data pipeline not available: {e}")

try:
    from .performance_optimization_engine import OptimizationConfig, PerformanceOptimizationEngine

    __all__.extend(["PerformanceOptimizationEngine", "OptimizationConfig"])
except ImportError as e:
    warnings.warn(f"Performance optimization engine not available: {e}")

try:
    from .enhanced_multimodal_integration import EnhancedMultiModalProcessor, IntegrationConfig

    __all__.extend(["EnhancedMultiModalProcessor", "IntegrationConfig"])
except ImportError as e:
    warnings.warn(f"Enhanced multimodal integration not available: {e}")

# Suppress import warnings for cleaner output during testing
warnings.filterwarnings("ignore", category=UserWarning, module="models")

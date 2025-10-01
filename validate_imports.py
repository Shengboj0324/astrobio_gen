#!/usr/bin/env python3
"""
Comprehensive Import Validation
Tests all critical imports to verify fixes
"""

import importlib
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class ImportValidator:
    """Validate all critical imports"""
    
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []
        
    def test_import(self, module_name: str, optional: bool = False) -> bool:
        """Test a single import"""
        try:
            importlib.import_module(module_name)
            self.passed.append(module_name)
            logger.info(f"✅ {module_name}")
            return True
        except ImportError as e:
            if optional:
                self.warnings.append((module_name, str(e)))
                logger.warning(f"⚠️  {module_name} (optional)")
            else:
                self.failed.append((module_name, str(e)))
                logger.error(f"❌ {module_name}: {e}")
            return False
        except Exception as e:
            self.failed.append((module_name, str(e)))
            logger.error(f"❌ {module_name}: {e}")
            return False
    
    def run_validation(self) -> None:
        """Run comprehensive validation"""
        logger.info("="*80)
        logger.info("COMPREHENSIVE IMPORT VALIDATION")
        logger.info("="*80)
        
        # Core PyTorch
        logger.info("\n📦 Core Deep Learning:")
        self.test_import('torch')
        self.test_import('torch.nn')
        self.test_import('pytorch_lightning')
        self.test_import('transformers')
        
        # Core models
        logger.info("\n🧠 Core Models:")
        self.test_import('models.sota_attention_2025')
        self.test_import('models.enhanced_datacube_unet')
        self.test_import('models.rebuilt_llm_integration')
        self.test_import('models.rebuilt_graph_vae')
        self.test_import('models.rebuilt_datacube_cnn')
        
        # Data systems
        logger.info("\n💾 Data Systems:")
        self.test_import('data_build.advanced_data_system')
        self.test_import('data_build.advanced_quality_system')
        self.test_import('data_build.data_versioning_system')
        self.test_import('data_build.planet_run_primary_key_system')
        self.test_import('data_build.metadata_annotation_system')
        
        # Chat systems
        logger.info("\n💬 Chat Systems:")
        self.test_import('chat.enhanced_tool_router')
        
        # Training systems
        logger.info("\n🏋️ Training Systems:")
        self.test_import('training.unified_sota_training_system')
        self.test_import('training.enhanced_training_orchestrator')
        
        # Rust integration
        logger.info("\n🦀 Rust Integration:")
        self.test_import('rust_integration', optional=True)
        
        # Optional packages
        logger.info("\n⚙️  Optional Packages:")
        self.test_import('flash_attn', optional=True)
        self.test_import('triton', optional=True)
        self.test_import('torch_geometric', optional=True)
        self.test_import('xformers', optional=True)
        
        # External packages
        logger.info("\n📚 External Packages:")
        self.test_import('streamlit', optional=True)
        self.test_import('selenium', optional=True)
        self.test_import('duckdb', optional=True)
        self.test_import('pendulum', optional=True)
        self.test_import('langchain', optional=True)
        
        # Astronomy packages
        logger.info("\n🔭 Astronomy Packages:")
        self.test_import('astropy')
        self.test_import('astroquery')
        self.test_import('specutils')
        
        # Data science
        logger.info("\n📊 Data Science:")
        self.test_import('numpy')
        self.test_import('pandas')
        self.test_import('scipy')
        self.test_import('sklearn')
        
        # Generate report
        self.generate_report()
    
    def generate_report(self) -> None:
        """Generate validation report"""
        logger.info("\n" + "="*80)
        logger.info("VALIDATION REPORT")
        logger.info("="*80)
        
        total = len(self.passed) + len(self.failed) + len(self.warnings)
        pass_rate = (len(self.passed) / total * 100) if total > 0 else 0
        
        logger.info(f"\n📊 SUMMARY:")
        logger.info(f"   Total tests:    {total}")
        logger.info(f"   Passed:         {len(self.passed)} ✅")
        logger.info(f"   Failed:         {len(self.failed)} ❌")
        logger.info(f"   Warnings:       {len(self.warnings)} ⚠️")
        logger.info(f"   Pass rate:      {pass_rate:.1f}%")
        
        if self.failed:
            logger.info(f"\n❌ FAILED IMPORTS ({len(self.failed)}):")
            for module, error in self.failed[:10]:
                logger.info(f"   - {module}")
                logger.info(f"     Error: {error[:100]}")
            if len(self.failed) > 10:
                logger.info(f"   ... and {len(self.failed) - 10} more")
        
        if self.warnings:
            logger.info(f"\n⚠️  OPTIONAL IMPORTS NOT AVAILABLE ({len(self.warnings)}):")
            for module, _ in self.warnings[:10]:
                logger.info(f"   - {module}")
            if len(self.warnings) > 10:
                logger.info(f"   ... and {len(self.warnings) - 10} more")
        
        logger.info("\n" + "="*80)
        
        if len(self.failed) == 0:
            logger.info("✅ ALL CRITICAL IMPORTS PASSED!")
            logger.info("="*80)
            return True
        else:
            logger.info("❌ SOME CRITICAL IMPORTS FAILED")
            logger.info("="*80)
            return False


def main():
    """Main entry point"""
    validator = ImportValidator()
    success = validator.run_validation()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


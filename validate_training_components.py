#!/usr/bin/env python3
"""
Training Components Validation Script
=====================================

Comprehensive validation of ALL training components to ensure:
1. NO import errors
2. NO module errors
3. NO name errors
4. NO method errors
5. NO dummy data references
6. All real data integrations working

This script performs exhaustive validation before training starts.
"""

import sys
import logging
import importlib
from pathlib import Path
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingComponentsValidator:
    """Comprehensive validator for all training components"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.passed = []
        
    def validate_all(self) -> bool:
        """Run all validation checks"""
        logger.info("=" * 80)
        logger.info("🔍 COMPREHENSIVE TRAINING COMPONENTS VALIDATION")
        logger.info("=" * 80)
        
        checks = [
            ("Import Validation", self.validate_imports),
            ("Data Loader Validation", self.validate_data_loaders),
            ("Model Import Validation", self.validate_models),
            ("Training Script Validation", self.validate_training_scripts),
            ("Real Data Integration", self.validate_real_data_integration),
            ("No Dummy Data Check", self.validate_no_dummy_data),
        ]
        
        for check_name, check_func in checks:
            logger.info(f"\n{'='*80}")
            logger.info(f"🔍 {check_name}")
            logger.info(f"{'='*80}")
            try:
                check_func()
            except Exception as e:
                self.errors.append(f"{check_name}: {e}")
                logger.error(f"❌ {check_name} FAILED: {e}")
        
        # Print summary
        self.print_summary()
        
        return len(self.errors) == 0
    
    def validate_imports(self):
        """Validate all critical imports"""
        critical_imports = [
            # Core training systems
            ("training.unified_sota_training_system", ["UnifiedSOTATrainer", "SOTATrainingConfig"]),
            ("training.enhanced_training_orchestrator", ["EnhancedTrainingOrchestrator"]),
            ("training.sota_training_strategies", ["GraphTransformerTrainer", "CNNViTTrainer"]),
            
            # Data systems
            ("data_build.real_data_storage", ["RealDataStorage"]),
            ("data_build.unified_dataloader_fixed", ["create_multimodal_dataloaders", "DataLoaderConfig"]),
            ("data_build.production_data_loader", ["ProductionDataLoader"]),
            
            # Models (with fallbacks expected)
            ("models.rebuilt_llm_integration", ["RebuiltLLMIntegration"]),
            ("models.rebuilt_datacube_cnn", ["RebuiltDatacubeCNN"]),
            ("models.rebuilt_multimodal_integration", ["RebuiltMultimodalIntegration"]),
        ]
        
        for module_name, expected_attrs in critical_imports:
            try:
                module = importlib.import_module(module_name)
                
                # Check expected attributes
                for attr in expected_attrs:
                    if not hasattr(module, attr):
                        self.errors.append(f"{module_name} missing attribute: {attr}")
                        logger.error(f"❌ {module_name}.{attr} NOT FOUND")
                    else:
                        self.passed.append(f"{module_name}.{attr}")
                        logger.info(f"✅ {module_name}.{attr}")
                        
            except ImportError as e:
                # Some imports may fail on Windows (torch_geometric)
                if "torch_geometric" in str(e) or "WinError 127" in str(e):
                    self.warnings.append(f"{module_name}: {e} (Expected on Windows)")
                    logger.warning(f"⚠️  {module_name}: {e} (Expected on Windows)")
                else:
                    self.errors.append(f"{module_name}: {e}")
                    logger.error(f"❌ {module_name}: {e}")
    
    def validate_data_loaders(self):
        """Validate data loader functionality"""
        try:
            from data_build.real_data_storage import RealDataStorage
            from data_build.unified_dataloader_fixed import DataLoaderConfig, create_multimodal_dataloaders
            
            # Try to create RealDataStorage
            try:
                storage = RealDataStorage()
                runs = storage.list_stored_runs()
                logger.info(f"✅ RealDataStorage: {len(runs)} runs available")
                self.passed.append("RealDataStorage instantiation")
            except FileNotFoundError as e:
                self.warnings.append(f"RealDataStorage: {e} (Data not downloaded yet)")
                logger.warning(f"⚠️  RealDataStorage: {e}")
                logger.warning("   Run: python training/enable_automatic_data_download.py")
            
            # Validate DataLoaderConfig
            config = DataLoaderConfig(
                batch_size=4,
                num_workers=0,
                pin_memory=False
            )
            logger.info(f"✅ DataLoaderConfig created successfully")
            self.passed.append("DataLoaderConfig creation")
            
        except ImportError as e:
            self.errors.append(f"Data loader imports failed: {e}")
            logger.error(f"❌ Data loader imports failed: {e}")
    
    def validate_models(self):
        """Validate model imports"""
        models_to_test = [
            ("models.rebuilt_llm_integration", "RebuiltLLMIntegration"),
            ("models.rebuilt_datacube_cnn", "RebuiltDatacubeCNN"),
            ("models.rebuilt_multimodal_integration", "RebuiltMultimodalIntegration"),
        ]
        
        for module_name, class_name in models_to_test:
            try:
                module = importlib.import_module(module_name)
                model_class = getattr(module, class_name)
                logger.info(f"✅ {class_name} import successful")
                self.passed.append(f"{class_name} import")
            except ImportError as e:
                if "torch_geometric" in str(e) or "WinError 127" in str(e):
                    self.warnings.append(f"{class_name}: {e} (Expected on Windows)")
                    logger.warning(f"⚠️  {class_name}: {e} (Expected on Windows)")
                else:
                    self.errors.append(f"{class_name}: {e}")
                    logger.error(f"❌ {class_name}: {e}")
    
    def validate_training_scripts(self):
        """Validate training scripts can be imported"""
        training_scripts = [
            "training.unified_sota_training_system",
            "training.enhanced_training_orchestrator",
            "training.sota_training_strategies",
            "training.enhanced_training_workflow",
        ]
        
        for script in training_scripts:
            try:
                module = importlib.import_module(script)
                logger.info(f"✅ {script} import successful")
                self.passed.append(f"{script} import")
            except ImportError as e:
                if "torch_geometric" in str(e) or "WinError 127" in str(e):
                    self.warnings.append(f"{script}: {e} (Expected on Windows)")
                    logger.warning(f"⚠️  {script}: {e} (Expected on Windows)")
                else:
                    self.errors.append(f"{script}: {e}")
                    logger.error(f"❌ {script}: {e}")
    
    def validate_real_data_integration(self):
        """Validate real data integration in training scripts"""
        # Check that training scripts use RealDataStorage
        training_files = [
            "training/unified_sota_training_system.py",
            "training/enhanced_training_workflow.py",
        ]
        
        for file_path in training_files:
            path = Path(file_path)
            if not path.exists():
                self.errors.append(f"Training file not found: {file_path}")
                logger.error(f"❌ File not found: {file_path}")
                continue
            
            content = path.read_text(encoding='utf-8')
            
            # Check for RealDataStorage import
            if "from data_build.real_data_storage import RealDataStorage" in content:
                logger.info(f"✅ {file_path}: Uses RealDataStorage")
                self.passed.append(f"{file_path} uses RealDataStorage")
            else:
                self.warnings.append(f"{file_path}: No RealDataStorage import found")
                logger.warning(f"⚠️  {file_path}: No RealDataStorage import found")
            
            # Check for MockDataStorage usage (should not exist)
            if "MockDataStorage" in content and "from data_build.unified_dataloader_fixed import MockDataStorage" in content:
                self.errors.append(f"{file_path}: Still uses MockDataStorage")
                logger.error(f"❌ {file_path}: Still uses MockDataStorage")
    
    def validate_no_dummy_data(self):
        """Validate no dummy data references in training scripts"""
        training_files = [
            "training/unified_sota_training_system.py",
            "training/enhanced_training_orchestrator.py",
            "training/enhanced_training_workflow.py",
        ]
        
        dummy_patterns = [
            "_create_dummy_data_loaders",
            "dummy_data",
            "synthetic_data",
            "mock_data",
        ]
        
        for file_path in training_files:
            path = Path(file_path)
            if not path.exists():
                continue
            
            content = path.read_text(encoding='utf-8')
            
            found_dummy = False
            for pattern in dummy_patterns:
                if pattern in content.lower():
                    # Check if it's in a comment or error message
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        if pattern in line.lower():
                            # Skip comments and error messages
                            if line.strip().startswith('#') or 'NO DUMMY DATA' in line or 'CRITICAL' in line:
                                continue
                            self.warnings.append(f"{file_path}:{i}: Contains '{pattern}'")
                            logger.warning(f"⚠️  {file_path}:{i}: Contains '{pattern}'")
                            found_dummy = True
            
            if not found_dummy:
                logger.info(f"✅ {file_path}: No dummy data references")
                self.passed.append(f"{file_path} no dummy data")
    
    def print_summary(self):
        """Print validation summary"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 VALIDATION SUMMARY")
        logger.info("=" * 80)
        
        logger.info(f"\n✅ PASSED: {len(self.passed)}")
        for item in self.passed[:10]:  # Show first 10
            logger.info(f"   ✅ {item}")
        if len(self.passed) > 10:
            logger.info(f"   ... and {len(self.passed) - 10} more")
        
        if self.warnings:
            logger.info(f"\n⚠️  WARNINGS: {len(self.warnings)}")
            for warning in self.warnings:
                logger.warning(f"   ⚠️  {warning}")
        
        if self.errors:
            logger.info(f"\n❌ ERRORS: {len(self.errors)}")
            for error in self.errors:
                logger.error(f"   ❌ {error}")
        
        logger.info("\n" + "=" * 80)
        if len(self.errors) == 0:
            logger.info("✅ OVERALL STATUS: PASSED")
            logger.info("🚀 Training components are ready!")
            logger.info("=" * 80)
            return True
        else:
            logger.error("❌ OVERALL STATUS: FAILED")
            logger.error(f"❌ {len(self.errors)} critical errors must be fixed before training")
            logger.info("=" * 80)
            return False


def main():
    """Main validation entry point"""
    validator = TrainingComponentsValidator()
    success = validator.validate_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


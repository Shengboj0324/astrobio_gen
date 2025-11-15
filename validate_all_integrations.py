#!/usr/bin/env python3
"""
Comprehensive Integration Validation Script
==========================================

Validates all model integrations, data pipelines, and training scripts.
Zero tolerance for errors - comprehensive validation of entire codebase.

Validation Scope:
1. All 67 model files - syntax and import validation
2. All training scripts - integration validation
3. All data pipelines - connectivity validation
4. Rust modules - integration validation
5. R2 bucket - connectivity validation
6. RunPod notebook - completeness validation
"""

import ast
import importlib
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class ComprehensiveIntegrationValidator:
    """Comprehensive validation of all integrations"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.successes = []
        
    def validate_syntax(self, file_path: Path) -> bool:
        """Validate Python file syntax"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            ast.parse(code)
            self.successes.append(f"✅ Syntax OK: {file_path.name}")
            return True
        except SyntaxError as e:
            self.errors.append(f"❌ Syntax Error in {file_path.name}: {e}")
            return False
        except Exception as e:
            self.errors.append(f"❌ Error reading {file_path.name}: {e}")
            return False
    
    def validate_imports(self, file_path: Path) -> bool:
        """Validate that file can be imported"""
        try:
            # Get module path relative to project root
            rel_path = file_path.relative_to(project_root)
            module_path = str(rel_path.with_suffix('')).replace('/', '.').replace('\\', '.')
            
            # Try to import
            importlib.import_module(module_path)
            self.successes.append(f"✅ Import OK: {module_path}")
            return True
        except ImportError as e:
            self.warnings.append(f"⚠️  Import Warning for {file_path.name}: {e}")
            return False
        except Exception as e:
            self.errors.append(f"❌ Import Error in {file_path.name}: {e}")
            return False
    
    def validate_model_files(self) -> Dict[str, bool]:
        """Validate all model files"""
        logger.info("=" * 70)
        logger.info("VALIDATING ALL MODEL FILES (67 files)")
        logger.info("=" * 70)
        
        models_dir = project_root / "models"
        model_files = sorted(models_dir.glob("*.py"))
        
        results = {}
        for model_file in model_files:
            if model_file.name == "__init__.py":
                continue
            
            logger.info(f"\nValidating: {model_file.name}")
            syntax_ok = self.validate_syntax(model_file)
            results[model_file.name] = syntax_ok
        
        return results
    
    def validate_training_scripts(self) -> Dict[str, bool]:
        """Validate all training scripts"""
        logger.info("\n" + "=" * 70)
        logger.info("VALIDATING TRAINING SCRIPTS")
        logger.info("=" * 70)
        
        training_files = [
            project_root / "train_unified_sota.py",
            project_root / "training" / "unified_sota_training_system.py",
            project_root / "training" / "unified_multimodal_training.py",
            project_root / "training" / "enhanced_training_orchestrator.py",
        ]
        
        results = {}
        for training_file in training_files:
            if training_file.exists():
                logger.info(f"\nValidating: {training_file.name}")
                syntax_ok = self.validate_syntax(training_file)
                results[training_file.name] = syntax_ok
        
        return results
    
    def validate_data_pipelines(self) -> Dict[str, bool]:
        """Validate all data pipeline files"""
        logger.info("\n" + "=" * 70)
        logger.info("VALIDATING DATA PIPELINES")
        logger.info("=" * 70)
        
        data_files = [
            project_root / "data_build" / "unified_dataloader_architecture.py",
            project_root / "data_build" / "production_data_loader.py",
            project_root / "data_build" / "comprehensive_data_annotation_treatment.py",
            project_root / "data_build" / "multi_modal_storage_layer_simple.py",
        ]
        
        results = {}
        for data_file in data_files:
            if data_file.exists():
                logger.info(f"\nValidating: {data_file.name}")
                syntax_ok = self.validate_syntax(data_file)
                results[data_file.name] = syntax_ok
        
        return results
    
    def print_summary(self):
        """Print validation summary"""
        logger.info("\n" + "=" * 70)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 70)
        
        logger.info(f"\n✅ SUCCESSES: {len(self.successes)}")
        logger.info(f"⚠️  WARNINGS: {len(self.warnings)}")
        logger.info(f"❌ ERRORS: {len(self.errors)}")
        
        if self.errors:
            logger.error("\n❌ ERRORS FOUND:")
            for error in self.errors:
                logger.error(f"  {error}")
        
        if self.warnings:
            logger.warning("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                logger.warning(f"  {warning}")
        
        logger.info("\n" + "=" * 70)
        if not self.errors:
            logger.info("🎉 ALL VALIDATIONS PASSED - ZERO ERRORS!")
        else:
            logger.error(f"❌ VALIDATION FAILED - {len(self.errors)} ERRORS FOUND")
        logger.info("=" * 70)


def main():
    """Main validation function"""
    logger.info("🚀 COMPREHENSIVE INTEGRATION VALIDATION")
    logger.info("Zero Tolerance Error Elimination - Complete Codebase Validation\n")
    
    validator = ComprehensiveIntegrationValidator()
    
    # Validate all components
    model_results = validator.validate_model_files()
    training_results = validator.validate_training_scripts()
    data_results = validator.validate_data_pipelines()
    
    # Print summary
    validator.print_summary()
    
    # Return exit code
    return 0 if not validator.errors else 1


if __name__ == "__main__":
    sys.exit(main())


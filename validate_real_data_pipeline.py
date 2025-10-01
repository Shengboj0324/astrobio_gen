#!/usr/bin/env python3
"""
Real Data Pipeline Validation
==============================

Comprehensive validation of the entire data pipeline to ensure:
1. NO dummy/mock/synthetic data exists
2. Real data is properly downloaded and accessible
3. Rust modules use real HTTP requests
4. Training pipeline is configured for real data only
5. All data sources are validated and ready

CRITICAL: Training will NOT start unless ALL validations pass.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealDataPipelineValidator:
    """Comprehensive validator for real data pipeline"""
    
    def __init__(self):
        self.base_path = Path(".")
        self.validation_results = {
            "dummy_data_check": False,
            "real_data_check": False,
            "rust_integration_check": False,
            "data_loader_check": False,
            "training_pipeline_check": False,
            "overall_status": False
        }
        self.errors = []
        self.warnings = []
    
    async def run_full_validation(self) -> Dict:
        """Run complete validation of data pipeline"""
        logger.info("="*80)
        logger.info("REAL DATA PIPELINE VALIDATION")
        logger.info("="*80)
        
        # Check 1: Verify NO dummy data exists
        logger.info("\n🔍 Check 1: Verifying NO dummy data exists...")
        self.validation_results["dummy_data_check"] = await self._check_no_dummy_data()
        
        # Check 2: Verify real data exists and is accessible
        logger.info("\n🔍 Check 2: Verifying real data exists...")
        self.validation_results["real_data_check"] = await self._check_real_data_exists()
        
        # Check 3: Verify Rust integration uses real HTTP requests
        logger.info("\n🔍 Check 3: Verifying Rust integration...")
        self.validation_results["rust_integration_check"] = await self._check_rust_integration()
        
        # Check 4: Verify data loaders use real data
        logger.info("\n🔍 Check 4: Verifying data loaders...")
        self.validation_results["data_loader_check"] = await self._check_data_loaders()
        
        # Check 5: Verify training pipeline configuration
        logger.info("\n🔍 Check 5: Verifying training pipeline...")
        self.validation_results["training_pipeline_check"] = await self._check_training_pipeline()
        
        # Determine overall status
        self.validation_results["overall_status"] = all([
            self.validation_results["dummy_data_check"],
            self.validation_results["real_data_check"],
            self.validation_results["rust_integration_check"],
            self.validation_results["data_loader_check"],
            self.validation_results["training_pipeline_check"]
        ])
        
        # Print final report
        self._print_final_report()
        
        return self.validation_results
    
    async def _check_no_dummy_data(self) -> bool:
        """Check that NO dummy/mock/synthetic data exists"""
        logger.info("Checking for dummy data files...")
        
        # List of dummy data files that should NOT exist
        dummy_files = [
            "data/pathways/dummy_metabolism.json",
            "data/dummy_planets.csv",
            "data/dummy_spectra.npz",
            "data/mock_climate.nc"
        ]
        
        found_dummy_files = []
        for dummy_file in dummy_files:
            file_path = self.base_path / dummy_file
            if file_path.exists():
                found_dummy_files.append(str(file_path))
                logger.error(f"❌ Found dummy data file: {file_path}")
        
        # Check for mock data generation in code
        mock_patterns = [
            ("data_build/unified_dataloader_standalone.py", "MockDataStorage"),
            ("data_build/unified_dataloader_fixed.py", "MockDataStorage"),
            ("utils/data_utils.py", "load_dummy_planets"),
            ("utils/data_utils.py", "load_dummy_metabolism")
        ]
        
        found_mock_code = []
        for file_path, pattern in mock_patterns:
            full_path = self.base_path / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if pattern in content and "DEPRECATED" not in content:
                            found_mock_code.append(f"{file_path}: {pattern}")
                            logger.warning(f"⚠️  Found mock code pattern: {file_path} contains {pattern}")
                except Exception as e:
                    logger.warning(f"⚠️  Could not read {file_path}: {e}")

        # Check Rust simulation code
        rust_file = self.base_path / "rust_modules/src/concurrent_data_acquisition.rs"
        if rust_file.exists():
            try:
                with open(rust_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if "simulate_data_acquisition" in content and "acquire_real_data" not in content:
                        found_mock_code.append("rust_modules: simulate_data_acquisition without acquire_real_data")
                        logger.error("❌ Rust module still using simulated data acquisition")
            except Exception as e:
                logger.warning(f"⚠️  Could not read Rust file: {e}")
        
        if found_dummy_files:
            self.errors.append(f"Found {len(found_dummy_files)} dummy data files")
            logger.error(f"❌ FAILED: Found {len(found_dummy_files)} dummy data files")
            return False
        
        if found_mock_code:
            self.warnings.append(f"Found {len(found_mock_code)} mock code patterns")
            logger.warning(f"⚠️  WARNING: Found {len(found_mock_code)} mock code patterns")
        
        logger.info("✅ PASSED: No dummy data files found")
        return True
    
    async def _check_real_data_exists(self) -> bool:
        """Check that real data exists and is accessible"""
        logger.info("Checking for real data...")
        
        required_data = [
            ("data/planets/2025-06-exoplanets.csv", "NASA Exoplanet Archive"),
            ("data/processed/kegg", "KEGG Pathways"),
            ("data/kegg_graphs", "KEGG Graphs"),
            ("data/astronomy/raw", "Astronomical Data"),
            ("data/planet_runs", "Planet Simulation Runs")
        ]
        
        missing_data = []
        found_data = []
        
        for data_path, description in required_data:
            full_path = self.base_path / data_path
            if full_path.exists():
                # Check if directory has files or file has content
                if full_path.is_dir():
                    files = list(full_path.glob("**/*"))
                    if len(files) > 0:
                        found_data.append(f"{description}: {len(files)} files")
                        logger.info(f"✅ {description}: Found {len(files)} files")
                    else:
                        missing_data.append(f"{description}: Directory empty")
                        logger.error(f"❌ {description}: Directory exists but is empty")
                else:
                    size = full_path.stat().st_size
                    if size > 0:
                        found_data.append(f"{description}: {size} bytes")
                        logger.info(f"✅ {description}: Found ({size} bytes)")
                    else:
                        missing_data.append(f"{description}: File empty")
                        logger.error(f"❌ {description}: File exists but is empty")
            else:
                missing_data.append(f"{description}: Not found")
                logger.error(f"❌ {description}: Not found at {full_path}")
        
        if missing_data:
            self.errors.append(f"Missing {len(missing_data)} required data sources")
            logger.error(f"❌ FAILED: Missing {len(missing_data)} required data sources")
            logger.error("Run: python training/enable_automatic_data_download.py")
            return False
        
        logger.info(f"✅ PASSED: All {len(required_data)} required data sources found")
        return True
    
    async def _check_rust_integration(self) -> bool:
        """Check Rust integration uses real HTTP requests"""
        logger.info("Checking Rust integration...")
        
        rust_file = self.base_path / "rust_modules/src/concurrent_data_acquisition.rs"
        if not rust_file.exists():
            self.errors.append("Rust concurrent_data_acquisition.rs not found")
            logger.error("❌ Rust concurrent_data_acquisition.rs not found")
            return False
        
        try:
            with open(rust_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            self.errors.append(f"Could not read Rust file: {e}")
            logger.error(f"❌ Could not read Rust file: {e}")
            return False

        # Check for real HTTP implementation
        checks = [
            ("acquire_real_data", "Real data acquisition function"),
            ("reqwest::Client", "HTTP client"),
            ("CRITICAL", "Critical error handling"),
            ("TRAINING CANNOT PROCEED", "Training blocker on failure")
        ]
        
        missing_checks = []
        for pattern, description in checks:
            if pattern not in content:
                missing_checks.append(description)
                logger.error(f"❌ Missing: {description} ({pattern})")
        
        if missing_checks:
            self.errors.append(f"Rust integration missing {len(missing_checks)} required features")
            logger.error(f"❌ FAILED: Rust integration incomplete")
            return False
        
        # Check Cargo.toml for reqwest dependency
        cargo_file = self.base_path / "rust_modules/Cargo.toml"
        if cargo_file.exists():
            try:
                with open(cargo_file, 'r', encoding='utf-8', errors='ignore') as f:
                    cargo_content = f.read()

                if 'reqwest' not in cargo_content:
                    self.errors.append("reqwest dependency not found in Cargo.toml")
                    logger.error("❌ reqwest dependency not found in Cargo.toml")
                    return False

                if 'optional = true' in cargo_content and 'reqwest' in cargo_content:
                    self.warnings.append("reqwest is marked as optional in Cargo.toml")
                    logger.warning("⚠️  reqwest is marked as optional - should be required")
            except Exception as e:
                logger.warning(f"⚠️  Could not read Cargo.toml: {e}")
        
        logger.info("✅ PASSED: Rust integration configured for real data")
        return True
    
    async def _check_data_loaders(self) -> bool:
        """Check data loaders use real data"""
        logger.info("Checking data loaders...")
        
        # Check production data loader
        prod_loader = self.base_path / "data_build/production_data_loader.py"
        if not prod_loader.exists():
            self.errors.append("production_data_loader.py not found")
            logger.error("❌ production_data_loader.py not found")
            return False
        
        try:
            with open(prod_loader, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            if "mock" in content.lower() or "dummy" in content.lower():
                self.warnings.append("production_data_loader.py contains mock/dummy references")
                logger.warning("⚠️  production_data_loader.py contains mock/dummy references")
        except Exception as e:
            logger.warning(f"⚠️  Could not read production_data_loader.py: {e}")
        
        # Check for RealDataStorage
        real_storage = self.base_path / "data_build/real_data_storage.py"
        if not real_storage.exists():
            self.errors.append("real_data_storage.py not found")
            logger.error("❌ real_data_storage.py not found")
            return False
        
        logger.info("✅ PASSED: Data loaders configured")
        return True
    
    async def _check_training_pipeline(self) -> bool:
        """Check training pipeline configuration"""
        logger.info("Checking training pipeline...")
        
        # Check for automatic data download enabler
        auto_download = self.base_path / "training/enable_automatic_data_download.py"
        if not auto_download.exists():
            self.errors.append("enable_automatic_data_download.py not found")
            logger.error("❌ enable_automatic_data_download.py not found")
            return False
        
        logger.info("✅ PASSED: Training pipeline configured")
        return True
    
    def _print_final_report(self):
        """Print final validation report"""
        logger.info("\n" + "="*80)
        logger.info("VALIDATION REPORT")
        logger.info("="*80)
        
        for check_name, result in self.validation_results.items():
            if check_name == "overall_status":
                continue
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{check_name}: {status}")
        
        logger.info("\n" + "-"*80)
        
        if self.errors:
            logger.error(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                logger.error(f"  - {error}")
        
        if self.warnings:
            logger.warning(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")
        
        logger.info("\n" + "="*80)
        
        if self.validation_results["overall_status"]:
            logger.info("✅ OVERALL STATUS: PASSED")
            logger.info("✅ TRAINING CAN START WITH REAL DATA")
        else:
            logger.error("❌ OVERALL STATUS: FAILED")
            logger.error("❌ TRAINING CANNOT START - FIX ERRORS FIRST")
            logger.error("\nTo fix:")
            logger.error("1. Run: python training/enable_automatic_data_download.py")
            logger.error("2. Rebuild Rust modules: cd rust_modules && maturin develop --release")
            logger.error("3. Re-run this validation")
        
        logger.info("="*80)


async def main():
    """Main entry point"""
    validator = RealDataPipelineValidator()
    results = await validator.run_full_validation()
    
    # Return exit code based on validation result
    if results["overall_status"]:
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)


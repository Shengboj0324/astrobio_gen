#!/usr/bin/env python3
"""
System Import Validation Test
============================

Test script to validate that all core system components can import
and initialize properly without optional dependencies.
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_core_imports():
    """Test core system imports"""
    try:
        logger.info("Testing core imports...")

        # Test URL management
        logger.info("Testing URL management...")
        from utils.url_management import URLManager, get_url_manager

        url_manager = get_url_manager()
        logger.info("✅ URL management imported successfully")

        # Test predictive discovery
        logger.info("Testing predictive discovery...")
        from utils.predictive_url_discovery import PredictiveURLDiscovery, get_predictive_discovery

        predictive = get_predictive_discovery()
        logger.info("✅ Predictive discovery imported successfully")

        # Test mirror infrastructure
        logger.info("Testing mirror infrastructure...")
        from utils.local_mirror_infrastructure import (
            LocalMirrorInfrastructure,
            get_mirror_infrastructure,
        )

        mirror = get_mirror_infrastructure()
        logger.info("✅ Mirror infrastructure imported successfully")

        # Test autonomous system
        logger.info("Testing autonomous system...")
        from utils.autonomous_data_acquisition import (
            AutonomousDataAcquisition,
            get_autonomous_system,
        )

        autonomous = get_autonomous_system()
        logger.info("✅ Autonomous system imported successfully")

        # Test global network
        logger.info("Testing global network...")
        from utils.global_scientific_network import GlobalScientificNetwork, get_global_network

        network = get_global_network()
        logger.info("✅ Global network imported successfully")

        # Test integrated system
        logger.info("Testing integrated system...")
        from utils.integrated_url_system import IntegratedURLSystem, get_integrated_url_system

        integrated = get_integrated_url_system()
        logger.info("✅ Integrated system imported successfully")

        logger.info("🎉 All core imports successful!")
        return True

    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False


def test_train_module():
    """Test train module imports"""
    try:
        logger.info("Testing train module...")
        import train

        logger.info("✅ Train module imported successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Train module import failed: {e}")
        return False


def test_database_paths():
    """Test that database paths can be created"""
    try:
        logger.info("Testing database paths...")

        # Create data directories
        data_dir = Path("data/metadata")
        data_dir.mkdir(parents=True, exist_ok=True)

        logger.info("✅ Database paths created successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Database path creation failed: {e}")
        return False


def main():
    """Run all validation tests"""
    logger.info("🧪 Starting System Import Validation")
    logger.info("=" * 50)

    tests = [
        ("Database Paths", test_database_paths),
        ("Core Imports", test_core_imports),
        ("Train Module", test_train_module),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} test...")
        if test_func():
            passed += 1
            logger.info(f"✅ {test_name} test passed")
        else:
            logger.error(f"❌ {test_name} test failed")

    logger.info("\n" + "=" * 50)
    logger.info(f"🏁 Test Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! System is ready.")
        return 0
    else:
        logger.error("❌ Some tests failed. Check the logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

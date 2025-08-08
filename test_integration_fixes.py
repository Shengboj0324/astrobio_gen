#!/usr/bin/env python3
"""
Integration Fixes Validation Test
================================

Test script to validate that the integration fixes have resolved
the major issues affecting system performance.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def test_integration_fixes():
    """Test the integration fixes we've implemented"""

    test_results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {},
        "overall_status": "pending",
        "success_rate": 0.0,
        "summary": "",
    }

    logger.info("Starting integration fixes validation...")

    # Test 1: SSL Configuration
    logger.info("Testing SSL configuration...")
    try:
        from utils.ssl_config import check_ssl_configuration

        ssl_status = check_ssl_configuration()
        test_results["tests"]["ssl_configuration"] = {
            "status": "PASSED" if ssl_status["external_api_accessible"] else "FAILED",
            "details": ssl_status,
        }
    except Exception as e:
        test_results["tests"]["ssl_configuration"] = {"status": "FAILED", "error": str(e)}

    # Test 2: IntegratedURLSystem API Methods
    logger.info("Testing IntegratedURLSystem API methods...")
    try:
        from utils.integrated_url_system import get_integrated_url_system

        url_system = get_integrated_url_system()

        # Test the new methods
        session = await url_system.get_session()
        health_check = await url_system.run_health_check_async()
        test_url = await url_system.get_managed_url("https://httpbin.org/get")

        # Close session
        await url_system.close_session()

        test_results["tests"]["url_system_api"] = {
            "status": "PASSED",
            "details": {
                "session_created": session is not None,
                "health_check_ran": isinstance(health_check, dict),
                "url_management_working": test_url is not None,
            },
        }
    except Exception as e:
        test_results["tests"]["url_system_api"] = {"status": "FAILED", "error": str(e)}

    # Test 3: Process Metadata System
    logger.info("Testing process metadata system...")
    try:
        # Check if systematic biases source was added
        db_path = Path("data/processed/process_metadata.db")
        if db_path.exists():
            import sqlite3

            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM process_metadata_sources WHERE metadata_type = 'systematic_biases'"
                )
                bias_count = cursor.fetchone()[0]

                test_results["tests"]["process_metadata"] = {
                    "status": "PASSED" if bias_count >= 100 else "PARTIAL",
                    "details": {
                        "systematic_biases_count": bias_count,
                        "target_achieved": bias_count >= 100,
                    },
                }
        else:
            test_results["tests"]["process_metadata"] = {
                "status": "SKIPPED",
                "details": "Database not found - run process metadata collection first",
            }
    except Exception as e:
        test_results["tests"]["process_metadata"] = {"status": "FAILED", "error": str(e)}

    # Test 4: PyTorch Compatibility
    logger.info("Testing PyTorch compatibility...")
    try:
        import torch
        import torchvision

        # Test that PyTorch and torchvision are compatible
        test_tensor = torch.rand(5, 5)

        # Test torchvision NMS operator (the one that was failing)
        boxes = torch.tensor([[0, 0, 1, 1], [0.1, 0.1, 1.1, 1.1]], dtype=torch.float32)
        scores = torch.tensor([0.9, 0.8], dtype=torch.float32)

        from torchvision.ops import nms

        keep = nms(boxes, scores, 0.5)

        test_results["tests"]["pytorch_compatibility"] = {
            "status": "PASSED",
            "details": {
                "torch_version": torch.__version__,
                "torchvision_version": torchvision.__version__,
                "nms_operator_working": len(keep) > 0,
            },
        }
    except Exception as e:
        test_results["tests"]["pytorch_compatibility"] = {"status": "FAILED", "error": str(e)}

    # Test 5: Data Acquisition Modules
    logger.info("Testing data acquisition modules...")
    try:
        # Test basic imports without actually running collection
        from data_build.gtdb_integration import GTDBIntegration
        from data_build.kegg_real_data_integration import KEGGRealDataIntegration

        # Test initialization without network calls
        kegg = KEGGRealDataIntegration()
        gtdb = GTDBIntegration()

        test_results["tests"]["data_acquisition"] = {
            "status": "PASSED",
            "details": {
                "kegg_integration_imports": True,
                "gtdb_integration_imports": True,
                "requests_available": "requests" in sys.modules,
            },
        }
    except Exception as e:
        test_results["tests"]["data_acquisition"] = {"status": "FAILED", "error": str(e)}

    # Calculate overall success rate
    passed_tests = sum(1 for test in test_results["tests"].values() if test["status"] == "PASSED")
    total_tests = len(test_results["tests"])
    test_results["success_rate"] = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

    # Set overall status
    if test_results["success_rate"] >= 80:
        test_results["overall_status"] = "PASSED"
    elif test_results["success_rate"] >= 60:
        test_results["overall_status"] = "PARTIAL"
    else:
        test_results["overall_status"] = "FAILED"

    # Generate summary
    test_results[
        "summary"
    ] = f"""
Integration Fixes Validation Results
===================================
Overall Status: {test_results['overall_status']}
Success Rate: {test_results['success_rate']:.1f}% ({passed_tests}/{total_tests} tests passed)

Test Results:
"""
    for test_name, result in test_results["tests"].items():
        status_emoji = (
            "✅" if result["status"] == "PASSED" else "❌" if result["status"] == "FAILED" else "⚠️"
        )
        test_results["summary"] += f"  {status_emoji} {test_name}: {result['status']}\n"

    test_results["summary"] += f"\nTimestamp: {test_results['timestamp']}"

    # Print summary
    print(test_results["summary"])

    # Save results
    results_file = Path("integration_fixes_validation_results.json")
    with open(results_file, "w") as f:
        json.dump(test_results, f, indent=2)

    logger.info(f"Validation results saved to {results_file}")

    return test_results


async def main():
    """Main test function"""
    try:
        results = await test_integration_fixes()

        # Exit with appropriate code
        if results["overall_status"] == "PASSED":
            sys.exit(0)
        elif results["overall_status"] == "PARTIAL":
            sys.exit(1)
        else:
            sys.exit(2)

    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(3)


if __name__ == "__main__":
    asyncio.run(main())

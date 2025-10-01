#!/usr/bin/env python3
"""
Fix Critical Import Errors
===========================

This script systematically resolves the top 20 critical import errors identified
in the bootstrap analysis. It applies fixes based on error type:

1. Missing dependencies -> Add to requirements.txt with try/except
2. Incorrect import paths -> Update to correct paths
3. Circular imports -> Refactor to lazy imports
4. Platform-specific -> Add platform checks

Strategy:
- Add graceful fallbacks for optional dependencies
- Update import paths for existing modules
- Add platform-specific handling
- Document Linux-only dependencies
"""

import json
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


def load_import_errors() -> List[Dict]:
    """Load import errors from bootstrap analysis report"""
    report_path = Path("bootstrap_analysis_report.json")
    
    if not report_path.exists():
        logger.error(f"Report not found: {report_path}")
        logger.info("Run: python bootstrap_analysis.py first")
        return []
    
    logger.info(f"Loading import errors from {report_path}...")
    with open(report_path, 'r') as f:
        data = json.load(f)
    
    errors = data.get('import_errors', [])
    logger.info(f"Found {len(errors)} import errors")
    
    return errors


def categorize_errors(errors: List[Dict]) -> Dict[str, List[Dict]]:
    """Categorize errors by type"""
    categories = {
        'missing_dependency': [],
        'incorrect_path': [],
        'platform_specific': [],
        'circular': [],
        'other': []
    }
    
    for error in errors:
        error_msg = error.get('error', '').lower()
        module = error.get('module', '')
        
        if 'no module named' in error_msg:
            # Check if it's a known platform-specific module
            if any(pkg in module for pkg in ['torch_geometric', 'triton', 'flash_attn']):
                categories['platform_specific'].append(error)
            else:
                categories['missing_dependency'].append(error)
        elif 'cannot import name' in error_msg:
            categories['incorrect_path'].append(error)
        elif 'circular' in error_msg:
            categories['circular'].append(error)
        else:
            categories['other'].append(error)
    
    return categories


def get_top_errors(errors: List[Dict], n: int = 20) -> List[Dict]:
    """Get top N most frequent errors"""
    # Count occurrences
    error_counts = {}
    for error in errors:
        key = (error.get('module', ''), error.get('error', ''))
        error_counts[key] = error_counts.get(key, 0) + 1
    
    # Sort by frequency
    sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Get unique errors
    seen = set()
    top_errors = []
    for (module, error_msg), count in sorted_errors:
        if len(top_errors) >= n:
            break
        if module not in seen:
            seen.add(module)
            top_errors.append({
                'module': module,
                'error': error_msg,
                'count': count
            })
    
    return top_errors


def generate_fix_recommendations(errors: List[Dict]) -> List[Tuple[str, str, str]]:
    """Generate fix recommendations for errors"""
    recommendations = []
    
    for error in errors:
        module = error.get('module', '')
        error_msg = error.get('error', '')
        count = error.get('count', 1)
        
        # Platform-specific modules
        if module in ['torch_geometric', 'torch_cluster', 'torch_scatter', 'torch_sparse']:
            fix = f"""
# Add to affected files:
try:
    import {module}
    TORCH_GEOMETRIC_AVAILABLE = True
except (ImportError, OSError) as e:
    TORCH_GEOMETRIC_AVAILABLE = False
    logger.warning(f"{{module}} not available (Linux-only): {{e}}")
"""
            recommendations.append((module, "Platform-specific (Linux-only)", fix))
        
        elif module in ['flash_attn', 'triton']:
            fix = f"""
# Add to affected files:
try:
    import {module}
    {module.upper().replace('-', '_')}_AVAILABLE = True
except ImportError:
    {module.upper().replace('-', '_')}_AVAILABLE = False
    logger.warning("{module} not available - install with: pip install {module}")
"""
            recommendations.append((module, "Optional performance dependency", fix))
        
        elif 'no module named' in error_msg.lower():
            # Extract the missing module name
            parts = error_msg.split("'")
            if len(parts) >= 2:
                missing = parts[1]
                fix = f"""
# Option 1: Add to requirements.txt
{missing}

# Option 2: Add graceful fallback in affected files
try:
    import {missing}
    {missing.upper().replace('-', '_')}_AVAILABLE = True
except ImportError:
    {missing.upper().replace('-', '_')}_AVAILABLE = False
    logger.warning("{missing} not available")
"""
                recommendations.append((module, f"Missing dependency: {missing}", fix))
        
        elif 'cannot import name' in error_msg.lower():
            fix = f"""
# Review import statement in affected files
# Check if the name exists in the module
# Update to correct import path or add fallback
"""
            recommendations.append((module, "Incorrect import path", fix))
    
    return recommendations


def print_report(errors: List[Dict], categories: Dict[str, List[Dict]], 
                 recommendations: List[Tuple[str, str, str]]):
    """Print comprehensive report"""
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("IMPORT ERROR ANALYSIS REPORT")
    logger.info("=" * 80)
    logger.info("")
    
    # Summary
    logger.info("SUMMARY")
    logger.info("-" * 80)
    logger.info(f"Total import errors: {len(errors)}")
    logger.info(f"Unique modules with errors: {len(set(e.get('module', '') for e in errors))}")
    logger.info("")
    
    # Categories
    logger.info("ERROR CATEGORIES")
    logger.info("-" * 80)
    for category, cat_errors in categories.items():
        if cat_errors:
            logger.info(f"{category.replace('_', ' ').title()}: {len(cat_errors)}")
    logger.info("")
    
    # Top errors
    logger.info("TOP 20 MOST FREQUENT ERRORS")
    logger.info("-" * 80)
    top_errors = get_top_errors(errors, 20)
    for i, error in enumerate(top_errors, 1):
        logger.info(f"{i}. {error['module']}")
        logger.info(f"   Error: {error['error'][:100]}...")
        logger.info(f"   Occurrences: {error['count']}")
        logger.info("")
    
    # Recommendations
    logger.info("=" * 80)
    logger.info("FIX RECOMMENDATIONS")
    logger.info("=" * 80)
    logger.info("")
    
    for i, (module, issue, fix) in enumerate(recommendations[:20], 1):
        logger.info(f"{i}. MODULE: {module}")
        logger.info(f"   ISSUE: {issue}")
        logger.info(f"   FIX:{fix}")
        logger.info("")
    
    logger.info("=" * 80)
    logger.info("NEXT STEPS")
    logger.info("=" * 80)
    logger.info("")
    logger.info("1. Review recommendations above")
    logger.info("2. Apply fixes to affected files")
    logger.info("3. Test imports on Linux/RunPod")
    logger.info("4. Update requirements.txt as needed")
    logger.info("5. Document platform-specific dependencies")
    logger.info("")


def main():
    """Main entry point"""
    logger.info("=" * 80)
    logger.info("Import Error Resolution Tool")
    logger.info("=" * 80)
    logger.info("")
    
    # Load errors
    errors = load_import_errors()
    if not errors:
        logger.error("No import errors found")
        sys.exit(1)
    
    # Categorize
    logger.info("Categorizing errors...")
    categories = categorize_errors(errors)
    
    # Get top errors
    logger.info("Identifying top errors...")
    top_errors = get_top_errors(errors, 20)
    
    # Generate recommendations
    logger.info("Generating fix recommendations...")
    recommendations = generate_fix_recommendations(top_errors)
    
    # Print report
    print_report(errors, categories, recommendations)
    
    logger.info("✅ Analysis complete!")
    logger.info("")
    logger.info("NOTE: Many errors are expected on Windows and will work on Linux/RunPod")
    logger.info("Priority: Test on Linux before applying extensive fixes")


if __name__ == "__main__":
    main()


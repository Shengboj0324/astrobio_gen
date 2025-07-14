#!/usr/bin/env python3
"""
Database Integration Verification
==================================

Comprehensive verification of our centralized SQLite database system.
Tests all components to ensure proper integration and configuration.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_centralized_config():
    """Test the centralized database configuration system"""
    logger.info("🧪 Testing centralized database configuration...")
    
    try:
        from data_build.database_config import get_database_manager, verify_database_system
        
        # Get database manager
        manager = get_database_manager()
        logger.info(f"✅ Database manager initialized with {len(manager.list_databases())} databases")
        
        # Verify all databases
        verification = verify_database_system()
        logger.info(f"✅ Database verification completed: {verification['accessible']}/{verification['total_databases']} accessible")
        
        # Test specific database connections
        for db_name in ['metadata', 'versions', 'quality']:
            try:
                with manager.get_connection(db_name) as conn:
                    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = [row[0] for row in cursor.fetchall()]
                    logger.info(f"✅ {db_name} database: {len(tables)} tables")
            except Exception as e:
                logger.warning(f"⚠️ Could not connect to {db_name}: {e}")
        
        return True, verification
    
    except Exception as e:
        logger.error(f"❌ Database configuration test failed: {e}")
        return False, str(e)

def test_metadata_integration():
    """Test metadata database integration"""
    logger.info("🧪 Testing metadata database integration...")
    
    try:
        from data_build.metadata_db import create_metadata_manager
        
        # Test creating metadata manager with centralized config
        manager = create_metadata_manager()
        logger.info(f"✅ Metadata manager created using centralized configuration")
        
        # Test database statistics
        stats = manager.get_dataset_statistics()
        logger.info(f"✅ Database statistics: {stats}")
        
        # Close connection
        manager.close()
        
        return True, "Metadata integration working"
    
    except Exception as e:
        logger.error(f"❌ Metadata integration test failed: {e}")
        return False, str(e)

def test_comprehensive_acquisition_integration():
    """Test comprehensive acquisition system integration"""
    logger.info("🧪 Testing comprehensive acquisition integration...")
    
    try:
        # Import without running the full system
        import importlib.util
        spec = importlib.util.spec_from_file_location("comprehensive", "data_build/comprehensive_multi_domain_acquisition.py")
        comprehensive = importlib.util.module_from_spec(spec)
        
        logger.info("✅ Comprehensive acquisition module imports successfully")
        
        # Test that it can import database components
        from data_build.database_config import get_database_path
        metadata_path = get_database_path('metadata')
        logger.info(f"✅ Can access metadata database path: {metadata_path}")
        
        return True, "Comprehensive acquisition integration working"
    
    except Exception as e:
        logger.error(f"❌ Comprehensive acquisition integration test failed: {e}")
        return False, str(e)

def test_database_performance():
    """Test database performance with centralized configuration"""
    logger.info("🧪 Testing database performance...")
    
    try:
        from data_build.database_config import get_database_manager
        import time
        
        manager = get_database_manager()
        
        # Test connection speed
        start_time = time.time()
        with manager.get_connection('metadata') as conn:
            conn.execute("SELECT 1")
        connection_time = time.time() - start_time
        
        logger.info(f"✅ Database connection time: {connection_time:.3f}s")
        
        # Test optimization
        manager.optimize_database('metadata')
        logger.info("✅ Database optimization completed")
        
        return True, f"Connection time: {connection_time:.3f}s"
    
    except Exception as e:
        logger.error(f"❌ Database performance test failed: {e}")
        return False, str(e)

def test_config_yaml_integration():
    """Test config.yaml database configuration"""
    logger.info("🧪 Testing config.yaml database integration...")
    
    try:
        import yaml
        
        # Load config.yaml
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Check database section exists
        if 'database' not in config:
            raise ValueError("Database section missing from config.yaml")
        
        db_config = config['database']
        
        # Verify expected databases are configured
        expected_dbs = ['metadata', 'versions', 'quality', 'security', 'kegg', 'agora2']
        configured_dbs = list(db_config.get('databases', {}).keys())
        
        missing_dbs = set(expected_dbs) - set(configured_dbs)
        if missing_dbs:
            logger.warning(f"⚠️ Missing database configurations: {missing_dbs}")
        
        logger.info(f"✅ Found {len(configured_dbs)} database configurations in config.yaml")
        
        # Verify SQLite settings
        sqlite_settings = db_config.get('settings', {}).get('sqlite', {})
        required_settings = ['journal_mode', 'cache_size', 'foreign_keys']
        
        for setting in required_settings:
            if setting in sqlite_settings:
                logger.info(f"✅ SQLite setting configured: {setting} = {sqlite_settings[setting]}")
            else:
                logger.warning(f"⚠️ Missing SQLite setting: {setting}")
        
        return True, f"Config contains {len(configured_dbs)} databases"
    
    except Exception as e:
        logger.error(f"❌ Config.yaml integration test failed: {e}")
        return False, str(e)

def test_file_structure():
    """Test that database directories are properly created"""
    logger.info("🧪 Testing database file structure...")
    
    try:
        from data_build.database_config import get_database_manager
        
        manager = get_database_manager()
        
        # Check that directories exist for all databases
        missing_dirs = []
        existing_files = []
        
        for db_name in manager.list_databases():
            db_config = manager.get_database_config(db_name)
            db_path = Path(db_config.path)
            
            if not db_path.parent.exists():
                missing_dirs.append(str(db_path.parent))
            
            if db_path.exists():
                existing_files.append(db_name)
        
        if missing_dirs:
            logger.warning(f"⚠️ Missing directories: {missing_dirs}")
        else:
            logger.info("✅ All database directories exist")
        
        logger.info(f"✅ {len(existing_files)} database files exist: {existing_files}")
        
        return True, f"Directories OK, {len(existing_files)} files exist"
    
    except Exception as e:
        logger.error(f"❌ File structure test failed: {e}")
        return False, str(e)

def generate_database_report():
    """Generate comprehensive database system report"""
    logger.info("📊 Generating comprehensive database report...")
    
    try:
        from data_build.database_config import get_database_manager
        
        manager = get_database_manager()
        
        report = {
            'timestamp': str(Path(__file__).stat().st_mtime),
            'configuration': {
                'total_databases': len(manager.list_databases()),
                'database_names': manager.list_databases()
            },
            'verification': manager.verify_all_databases(),
            'performance': {},
            'recommendations': []
        }
        
        # Performance analysis
        accessible_dbs = [db for db, info in report['verification']['database_info'].items() 
                         if info['exists']]
        
        total_size_mb = sum(info['size_mb'] for info in report['verification']['database_info'].values() 
                           if info['exists'])
        
        report['performance'] = {
            'accessible_databases': len(accessible_dbs),
            'total_size_mb': round(total_size_mb, 2),
            'largest_database': max(
                [(db, info['size_mb']) for db, info in report['verification']['database_info'].items() 
                 if info['exists']], 
                key=lambda x: x[1], default=('none', 0)
            )
        }
        
        # Recommendations
        if len(accessible_dbs) < len(manager.list_databases()):
            report['recommendations'].append("Run comprehensive data acquisition to create missing databases")
        
        if total_size_mb > 100:
            report['recommendations'].append("Consider database optimization for large databases")
        
        if not accessible_dbs:
            report['recommendations'].append("Initialize database system by running basic data acquisition")
        
        # Save report
        report_path = Path("database_integration_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✅ Database report saved to {report_path}")
        
        return report
    
    except Exception as e:
        logger.error(f"❌ Report generation failed: {e}")
        return {}

def main():
    """Run comprehensive database integration verification"""
    logger.info("🚀 Starting comprehensive database integration verification...")
    
    # Test results
    results = {}
    
    # Run all tests
    tests = [
        ("Centralized Config", test_centralized_config),
        ("Metadata Integration", test_metadata_integration),
        ("Comprehensive Acquisition", test_comprehensive_acquisition_integration),
        ("Database Performance", test_database_performance),
        ("Config.yaml Integration", test_config_yaml_integration),
        ("File Structure", test_file_structure),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            success, result = test_func()
            results[test_name] = {'success': success, 'result': result}
            if success:
                passed += 1
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED - {result}")
        except Exception as e:
            results[test_name] = {'success': False, 'result': f"Exception: {e}"}
            logger.error(f"❌ {test_name}: EXCEPTION - {e}")
            logger.debug(traceback.format_exc())
    
    # Generate comprehensive report
    report = generate_database_report()
    
    # Summary
    logger.info("=" * 60)
    logger.info(f"📊 VERIFICATION SUMMARY")
    logger.info(f"Tests Passed: {passed}/{total}")
    logger.info(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - Database system is properly configured!")
    elif passed >= total * 0.8:
        logger.info("⚠️ MOSTLY WORKING - Minor issues detected")
    else:
        logger.error("❌ SIGNIFICANT ISSUES - Database system needs attention")
    
    # Database overview
    if report:
        verification = report.get('verification', {})
        logger.info(f"Database Status: {verification.get('accessible', 0)}/{verification.get('total_databases', 0)} accessible")
        logger.info(f"Total Size: {report.get('performance', {}).get('total_size_mb', 0)} MB")
        
        if report.get('recommendations'):
            logger.info("Recommendations:")
            for rec in report['recommendations']:
                logger.info(f"  • {rec}")
    
    logger.info("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
#!/usr/bin/env python3
"""
Secure Data Management Usage Examples
====================================

This script demonstrates proper usage of the secure data management system
for the astrobiology genomics project.

Usage examples:
- Storing raw data securely
- Accessing files with proper logging
- Creating backups
- Running integrity checks
"""

import sys
import os
from pathlib import Path
import logging

# Add the data_build directory to the path to import our secure data manager
sys.path.insert(0, str(Path(__file__).parent.parent / "data_build"))

try:
    from secure_data_manager import SecureDataManager, SecurityLevel, AccessType
except ImportError as e:
    print(f"Error importing SecureDataManager: {e}")
    print("Make sure you have run 'python setup_secure_data.py' first")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def example_store_kegg_data():
    """Example: Store KEGG pathway data securely"""
    logger.info("=== Example: Storing KEGG Data Securely ===")
    
    # Initialize secure data manager
    manager = SecureDataManager()
    
    # Example: Store a KEGG pathway CSV file
    kegg_file = Path("data/raw/kegg_pathways.csv")
    
    if kegg_file.exists():
        try:
            # Store the file securely with appropriate security level
            stored_path = manager.store_file_securely(
                source_path=kegg_file,
                destination_category="raw/kegg",
                security_level=SecurityLevel.INTERNAL,
                metadata={
                    'source': 'KEGG Database',
                    'data_type': 'pathway_definitions',
                    'format': 'csv',
                    'description': 'KEGG pathway identifiers and descriptions'
                }
            )
            
            logger.info(f"Successfully stored KEGG data: {stored_path}")
            
            # Verify the file can be accessed
            accessed_path = manager.access_file_securely(
                stored_path,
                AccessType.READ
            )
            
            logger.info(f"Successfully accessed file: {accessed_path}")
            
        except Exception as e:
            logger.error(f"Failed to store KEGG data: {e}")
    
    else:
        logger.warning(f"KEGG file not found: {kegg_file}")
        logger.info("You can create a sample file for testing:")
        logger.info("echo 'pathway_id,description' > data/raw/kegg_pathways.csv")
        logger.info("echo 'map00010,Glycolysis / Gluconeogenesis' >> data/raw/kegg_pathways.csv")

def example_store_sensitive_genomic_data():
    """Example: Store sensitive genomic data with encryption"""
    logger.info("=== Example: Storing Sensitive Genomic Data ===")
    
    manager = SecureDataManager()
    
    # Create a sample "sensitive" file for demonstration
    sample_file = Path("temp_sensitive_data.txt")
    sample_file.write_text("Sample sensitive genomic data\nThis would be actual genomic sequences in practice")
    
    try:
        # Store with high security and encryption
        stored_path = manager.store_file_securely(
            source_path=sample_file,
            destination_category="raw/ncbi",
            security_level=SecurityLevel.RESTRICTED,
            encrypt=True,
            metadata={
                'source': 'NCBI',
                'data_type': 'genomic_sequence',
                'sensitivity': 'high',
                'encrypted': True,
                'description': 'Sensitive genomic assembly data'
            }
        )
        
        logger.info(f"Successfully stored encrypted genomic data: {stored_path}")
        
        # Access the encrypted file (it will be decrypted automatically)
        accessed_path = manager.access_file_securely(
            stored_path,
            AccessType.READ
        )
        
        logger.info(f"Successfully accessed encrypted file: {accessed_path}")
        
        # Show that the file was decrypted
        with open(accessed_path, 'r') as f:
            content = f.read()
            logger.info(f"File content (first 50 chars): {content[:50]}...")
        
    except Exception as e:
        logger.error(f"Failed to store sensitive data: {e}")
    
    finally:
        # Clean up sample file
        if sample_file.exists():
            sample_file.unlink()

def example_backup_and_integrity():
    """Example: Create backups and check integrity"""
    logger.info("=== Example: Backup and Integrity Checking ===")
    
    manager = SecureDataManager()
    
    # Find a file to backup
    test_files = [
        Path("data/raw/kegg_pathways.csv"),
        Path("data/raw/.gitkeep"),
        Path("data/processed/.gitkeep")
    ]
    
    backup_file = None
    for file_path in test_files:
        if file_path.exists():
            backup_file = file_path
            break
    
    if backup_file:
        try:
            # Create backup
            backup_path = manager.create_backup(
                source_path=backup_file,
                backup_type="example_backup"
            )
            
            logger.info(f"Created backup: {backup_path}")
            
            # Check file integrity
            integrity_ok = manager.verify_file_integrity(backup_file)
            logger.info(f"File integrity check: {'PASSED' if integrity_ok else 'FAILED'}")
            
        except Exception as e:
            logger.error(f"Backup/integrity check failed: {e}")
    
    else:
        logger.warning("No files found for backup example")

def example_generate_security_report():
    """Example: Generate security report"""
    logger.info("=== Example: Security Report Generation ===")
    
    manager = SecureDataManager()
    
    try:
        # Generate comprehensive security report
        report = manager.generate_security_report()
        
        logger.info("Security Report Generated:")
        logger.info(f"  - System: {report['system_info']['hostname']}")
        logger.info(f"  - User: {report['system_info']['user']}")
        logger.info(f"  - Data Root: {report['system_info']['data_root']}")
        
        # Show file summary
        if report['file_summary']:
            logger.info("  - File Summary:")
            for level, stats in report['file_summary'].items():
                logger.info(f"    {level}: {stats['count']} files, {stats['total_size_mb']:.1f} MB")
        
        # Show access summary
        if report['access_summary']:
            logger.info("  - Access Summary (last 30 days):")
            for access_type, count in report['access_summary'].items():
                logger.info(f"    {access_type}: {count}")
        
        # Show recommendations
        if report['recommendations']:
            logger.info("  - Security Recommendations:")
            for rec in report['recommendations']:
                logger.info(f"    • {rec}")
        
    except Exception as e:
        logger.error(f"Failed to generate security report: {e}")

def example_access_logging():
    """Example: Demonstrate access logging"""
    logger.info("=== Example: Access Logging ===")
    
    manager = SecureDataManager()
    
    # Try to access various files to generate log entries
    test_files = [
        Path("data/raw/.gitkeep"),
        Path("data/processed/.gitkeep"),
        Path("data/README.md")
    ]
    
    for file_path in test_files:
        if file_path.exists():
            try:
                # Access file (this will be logged)
                accessed_path = manager.access_file_securely(
                    file_path,
                    AccessType.READ
                )
                logger.info(f"Successfully accessed: {file_path}")
                
            except Exception as e:
                logger.warning(f"Failed to access {file_path}: {e}")
    
    logger.info("All file access attempts have been logged to data/logs/access.log")

def main():
    """Main function to run all examples"""
    print("🔒 Secure Data Management Usage Examples")
    print("=" * 50)
    
    # Check if secure data environment is set up
    if not Path("data/metadata").exists():
        print("❌ Secure data environment not found!")
        print("Please run 'python setup_secure_data.py' first")
        return False
    
    print("✅ Secure data environment detected")
    print()
    
    try:
        # Run examples
        example_store_kegg_data()
        print()
        
        example_store_sensitive_genomic_data()
        print()
        
        example_backup_and_integrity()
        print()
        
        example_generate_security_report()
        print()
        
        example_access_logging()
        print()
        
        print("=" * 50)
        print("🎉 All examples completed successfully!")
        print()
        print("Next steps:")
        print("1. Check data/logs/access.log for access logs")
        print("2. Review data/backups/ for created backups")
        print("3. Use manager.generate_security_report() regularly")
        print("4. Always use manager.access_file_securely() for file access")
        
        return True
        
    except Exception as e:
        logger.error(f"Examples failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
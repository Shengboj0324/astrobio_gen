#!/usr/bin/env python3
"""
Secure Data Environment Setup Script
====================================

This script initializes the secure data storage environment for the astrobiology genomics project.
It sets up proper directory structure, permissions, and security measures.

Usage:
    python setup_secure_data.py

Author: AI Assistant
Date: 2025
"""

import os
import stat
import platform
import subprocess
import getpass
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_system_requirements():
    """Check if system meets security requirements"""
    logger.info("Checking system requirements...")
    
    requirements = []
    
    # Check operating system
    system = platform.system()
    if system not in ['Linux', 'Darwin']:  # Darwin = macOS
        requirements.append(f"Warning: {system} may not support all security features")
    
    # Check if running as root (not recommended)
    if os.geteuid() == 0:
        requirements.append("Warning: Running as root is not recommended for security")
    
    # Check disk space
    data_dir = Path("data")
    if data_dir.exists():
        stat_result = os.statvfs(data_dir)
        free_space_gb = (stat_result.f_bavail * stat_result.f_frsize) / (1024**3)
        if free_space_gb < 10:
            requirements.append(f"Warning: Low disk space ({free_space_gb:.1f} GB available)")
    
    if requirements:
        logger.warning("System requirements check:")
        for req in requirements:
            logger.warning(f"  - {req}")
    else:
        logger.info("System requirements check: PASSED")
    
    return len(requirements) == 0

def create_secure_directories():
    """Create secure directory structure with proper permissions"""
    logger.info("Creating secure directory structure...")
    
    directories = {
        # Directory path: (permissions, description)
        'data': (0o755, 'Main data directory'),
        'data/raw': (0o750, 'Raw data storage (restricted)'),
        'data/raw/kegg': (0o750, 'KEGG pathway data'),
        'data/raw/ncbi': (0o750, 'NCBI genomic data'),
        'data/raw/agora2': (0o750, 'AGORA2 metabolic models'),
        'data/raw/1000g_indices': (0o750, '1000 Genomes indices'),
        'data/raw/1000g_dirlists': (0o750, '1000 Genomes directory listings'),
        'data/raw/stellar_seds': (0o750, 'Stellar spectral energy distributions'),
        'data/interim': (0o755, 'Intermediate processing'),
        'data/interim/quality_checks': (0o755, 'Quality assessment results'),
        'data/processed': (0o755, 'Final processed data'),
        'data/processed/kegg': (0o755, 'Processed KEGG data'),
        'data/processed/ncbi': (0o755, 'Processed NCBI data'),
        'data/processed/agora2': (0o755, 'Processed AGORA2 data'),
        'data/processed/quality_reports': (0o755, 'Quality reports'),
        'data/kegg_graphs': (0o750, 'KEGG network graphs'),
        'data/metadata': (0o700, 'Metadata and provenance (secure)'),
        'data/versions': (0o750, 'Data versioning'),
        'data/backups': (0o700, 'Backup storage (secure)'),
        'data/logs': (0o700, 'Access and security logs (secure)'),
        'data/temp': (0o700, 'Temporary processing (secure)'),
        'data/encrypted': (0o700, 'Encrypted sensitive data (secure)'),
    }
    
    created_dirs = []
    
    for dir_path, (permissions, description) in directories.items():
        path = Path(dir_path)
        
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(dir_path)
        
        # Set permissions
        try:
            os.chmod(path, permissions)
            logger.info(f"  ✓ {dir_path} - {description} (permissions: {oct(permissions)})")
        except OSError as e:
            logger.warning(f"  ⚠ Could not set permissions for {dir_path}: {e}")
        
        # Create .gitkeep if directory is empty
        gitkeep = path / ".gitkeep"
        if not any(path.iterdir()) and not gitkeep.exists():
            gitkeep.touch()
            os.chmod(gitkeep, 0o644)
    
    if created_dirs:
        logger.info(f"Created {len(created_dirs)} new directories")
    else:
        logger.info("All directories already exist")
    
    return True

def create_security_configs():
    """Create security configuration files"""
    logger.info("Creating security configuration files...")
    
    # Create data security config
    security_config = {
        "data_classification": {
            "raw_data": "restricted",
            "processed_data": "internal", 
            "metadata": "confidential",
            "logs": "confidential"
        },
        "access_control": {
            "raw_data_permissions": "750",
            "processed_data_permissions": "755",
            "secure_data_permissions": "700"
        },
        "backup_policy": {
            "frequency": "daily",
            "retention_days": 30,
            "compress": True,
            "encrypt_sensitive": True
        },
        "integrity_checks": {
            "enable_checksums": True,
            "verify_on_access": True,
            "hash_algorithms": ["md5", "sha256"]
        }
    }
    
    config_path = Path("data/metadata/security_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(config_path, 'w') as f:
        json.dump(security_config, f, indent=2)
    
    os.chmod(config_path, 0o600)  # Owner read/write only
    logger.info(f"  ✓ Created security config: {config_path}")
    
    return True

def install_dependencies():
    """Install required Python packages for secure data management"""
    logger.info("Checking Python dependencies...")
    
    required_packages = [
        'cryptography',  # For encryption
        'psutil',       # For system monitoring
        'pathlib2',     # Enhanced path handling (if needed)
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"  ✓ {package} - installed")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"  ⚠ {package} - missing")
    
    if missing_packages:
        logger.info("Installing missing packages...")
        try:
            import subprocess
            import sys
            
            for package in missing_packages:
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install', package
                ])
                logger.info(f"  ✓ Installed {package}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install packages: {e}")
            return False
    
    return True

def create_readme_files():
    """Create README files for key directories"""
    logger.info("Creating README files...")
    
    readme_files = {
        'data/raw/README.md': """# Raw Data Directory

**SECURITY NOTICE**: This directory contains original source data. 

## Important Rules:
- Files in this directory are READ-ONLY after download
- Never modify files directly in this directory
- All data access is logged for security
- Large files are excluded from version control

## Structure:
- `kegg/` - KEGG pathway database
- `ncbi/` - NCBI genomic assemblies
- `agora2/` - AGORA2 metabolic models
- `1000g_*` - 1000 Genomes project data
- `stellar_seds/` - Stellar spectral energy distributions
""",
        
        'data/processed/README.md': """# Processed Data Directory

This directory contains final, analysis-ready datasets.

## Quality Standards:
- All data has passed quality validation
- Checksums verified for integrity
- Standardized formats and schemas
- Comprehensive metadata included

## Structure:
- `kegg/` - Processed KEGG pathway networks
- `ncbi/` - Processed genomic data
- `agora2/` - Processed metabolic models
- `quality_reports/` - Data quality assessments
""",
    }
    
    for file_path, content in readme_files.items():
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            f.write(content)
        
        os.chmod(path, 0o644)
        logger.info(f"  ✓ Created {file_path}")
    
    return True

def verify_setup():
    """Verify the secure data environment setup"""
    logger.info("Verifying setup...")
    
    checks = []
    
    # Check critical directories exist
    critical_dirs = ['data/raw', 'data/processed', 'data/metadata', 'data/logs']
    for dir_path in critical_dirs:
        if Path(dir_path).exists():
            checks.append(f"✓ Directory {dir_path} exists")
        else:
            checks.append(f"✗ Directory {dir_path} missing")
    
    # Check permissions on secure directories
    secure_dirs = ['data/metadata', 'data/logs', 'data/backups']
    for dir_path in secure_dirs:
        path = Path(dir_path)
        if path.exists():
            permissions = oct(path.stat().st_mode)[-3:]
            if permissions == '700':
                checks.append(f"✓ Secure permissions on {dir_path}")
            else:
                checks.append(f"⚠ Incorrect permissions on {dir_path}: {permissions}")
    
    # Check .gitignore exists and is comprehensive
    gitignore = Path('.gitignore')
    if gitignore.exists():
        content = gitignore.read_text()
        if 'data/raw/**' in content and '*.db' in content:
            checks.append("✓ Comprehensive .gitignore configured")
        else:
            checks.append("⚠ .gitignore may be incomplete")
    else:
        checks.append("✗ .gitignore missing")
    
    # Display results
    success_count = sum(1 for check in checks if check.startswith('✓'))
    warning_count = sum(1 for check in checks if check.startswith('⚠'))
    error_count = sum(1 for check in checks if check.startswith('✗'))
    
    for check in checks:
        if check.startswith('✓'):
            logger.info(f"  {check}")
        elif check.startswith('⚠'):
            logger.warning(f"  {check}")
        else:
            logger.error(f"  {check}")
    
    logger.info(f"Verification complete: {success_count} passed, {warning_count} warnings, {error_count} errors")
    
    return error_count == 0

def main():
    """Main setup function"""
    print("🔒 Astrobiology Genomics - Secure Data Environment Setup")
    print("=" * 60)
    
    user = getpass.getuser()
    logger.info(f"Setting up secure data environment for user: {user}")
    
    try:
        # Step 1: Check system requirements
        if not check_system_requirements():
            logger.warning("System requirements check had warnings - proceeding with caution")
        
        # Step 2: Install dependencies
        if not install_dependencies():
            logger.error("Failed to install required dependencies")
            return False
        
        # Step 3: Create secure directories
        if not create_secure_directories():
            logger.error("Failed to create secure directories")
            return False
        
        # Step 4: Create security configs
        if not create_security_configs():
            logger.error("Failed to create security configurations")
            return False
        
        # Step 5: Create documentation
        if not create_readme_files():
            logger.error("Failed to create documentation")
            return False
        
        # Step 6: Verify setup
        if not verify_setup():
            logger.error("Setup verification failed - please review warnings and errors")
            return False
        
        print("\n" + "=" * 60)
        print("🎉 Secure data environment setup completed successfully!")
        print("\nNext steps:")
        print("1. Review the data/README.md file for usage guidelines")
        print("2. Use the SecureDataManager class for all data operations")
        print("3. Run regular integrity checks on your data")
        print("4. Set up automated backups for critical datasets")
        print("\n⚠️  Remember: Raw data should never be committed to version control!")
        
        return True
        
    except Exception as e:
        logger.error(f"Setup failed with error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 
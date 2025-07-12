#!/usr/bin/env python3
"""
Secure Data Storage Manager
===========================

NASA-grade secure data storage and management system for astrobiology research:
- Secure file permissions and access control
- Data integrity verification with checksums
- Comprehensive access logging
- Automated backup and versioning
- Encryption for sensitive data
- Compliance with data security standards

Author: AI Assistant
Date: 2025
"""

import os
import stat
import hashlib
import logging
import json
import shutil
import sqlite3
import fcntl
import tempfile
import subprocess
import platform
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from threading import Lock
import getpass
import socket
import uuid
import tarfile
import gzip
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Data security classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"

class AccessType(Enum):
    """File access types"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    MODIFY = "modify"
    COPY = "copy"

@dataclass
class FileMetadata:
    """Comprehensive file metadata for security tracking"""
    file_path: str
    original_name: str
    size_bytes: int
    checksum_md5: str
    checksum_sha256: str
    security_level: SecurityLevel
    created_by: str
    created_at: datetime
    last_accessed: datetime
    last_modified: datetime
    permissions: str
    encrypted: bool = False
    backup_copies: List[str] = field(default_factory=list)
    access_log: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AccessRecord:
    """Access logging record"""
    record_id: str
    file_path: str
    access_type: AccessType
    user: str
    ip_address: str
    hostname: str
    timestamp: datetime
    success: bool
    details: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class SecureDataManager:
    """
    Comprehensive secure data storage manager with enterprise-grade security features.
    """
    
    def __init__(self, data_root: str = "data", encryption_key: Optional[str] = None):
        self.data_root = Path(data_root)
        self.lock = Lock()
        
        # Initialize directory structure
        self._initialize_secure_directories()
        
        # Security database
        self.security_db_path = self.data_root / "metadata" / "security.db"
        self._initialize_security_database()
        
        # Access logging
        self.access_log_path = self.data_root / "logs" / "access.log"
        self._setup_access_logging()
        
        # Encryption setup
        self.encryption_key = encryption_key or self._generate_encryption_key()
        self.cipher_suite = self._initialize_encryption()
        
        # User and system info
        self.current_user = getpass.getuser()
        self.hostname = socket.gethostname()
        self.ip_address = self._get_local_ip()
        
        logger.info(f"SecureDataManager initialized for user {self.current_user}@{self.hostname}")
    
    def _initialize_secure_directories(self):
        """Initialize secure directory structure with proper permissions"""
        directories = {
            'raw': 0o750,           # rwxr-x--- (owner: rwx, group: r-x, other: ---)
            'raw/kegg': 0o750,
            'raw/ncbi': 0o750,
            'raw/agora2': 0o750,
            'raw/1000g_indices': 0o750,
            'raw/1000g_dirlists': 0o750,
            'raw/stellar_seds': 0o750,
            'interim': 0o755,       # rwxr-xr-x (more accessible for processing)
            'interim/quality_checks': 0o755,
            'processed': 0o755,
            'processed/kegg': 0o755,
            'processed/ncbi': 0o755,
            'processed/agora2': 0o755,
            'kegg_graphs': 0o750,
            'metadata': 0o700,      # rwx------ (owner only)
            'versions': 0o750,
            'backups': 0o700,       # rwx------ (highly secure)
            'logs': 0o700,          # rwx------ (sensitive access logs)
            'temp': 0o700,          # rwx------ (temporary processing)
            'encrypted': 0o700      # rwx------ (encrypted sensitive data)
        }
        
        for dir_path, permissions in directories.items():
            full_path = self.data_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            os.chmod(full_path, permissions)
            
            # Create .gitkeep files for empty directories
            gitkeep = full_path / ".gitkeep"
            if not gitkeep.exists():
                gitkeep.touch()
                os.chmod(gitkeep, 0o640)
    
    def _initialize_security_database(self):
        """Initialize security tracking database"""
        self.security_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.security_db_path) as conn:
            cursor = conn.cursor()
            
            # File metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS file_metadata (
                    file_path TEXT PRIMARY KEY,
                    original_name TEXT,
                    size_bytes INTEGER,
                    checksum_md5 TEXT,
                    checksum_sha256 TEXT,
                    security_level TEXT,
                    created_by TEXT,
                    created_at TIMESTAMP,
                    last_accessed TIMESTAMP,
                    last_modified TIMESTAMP,
                    permissions TEXT,
                    encrypted BOOLEAN DEFAULT FALSE,
                    backup_copies TEXT,
                    metadata TEXT
                )
            ''')
            
            # Access log table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS access_log (
                    record_id TEXT PRIMARY KEY,
                    file_path TEXT,
                    access_type TEXT,
                    user TEXT,
                    ip_address TEXT,
                    hostname TEXT,
                    timestamp TIMESTAMP,
                    success BOOLEAN,
                    details TEXT,
                    metadata TEXT
                )
            ''')
            
            # Security alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS security_alerts (
                    alert_id TEXT PRIMARY KEY,
                    alert_type TEXT,
                    severity TEXT,
                    description TEXT,
                    file_path TEXT,
                    user TEXT,
                    timestamp TIMESTAMP,
                    resolved BOOLEAN DEFAULT FALSE,
                    metadata TEXT
                )
            ''')
            
            conn.commit()
        
        # Set secure permissions on security database
        os.chmod(self.security_db_path, 0o600)  # rw------- (owner read/write only)
    
    def _setup_access_logging(self):
        """Setup secure access logging"""
        self.access_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create rotating log handler
        from logging.handlers import RotatingFileHandler
        
        access_logger = logging.getLogger('access_log')
        access_logger.setLevel(logging.INFO)
        
        handler = RotatingFileHandler(
            self.access_log_path,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=10
        )
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        access_logger.addHandler(handler)
        
        self.access_logger = access_logger
        
        # Set secure permissions on log files
        os.chmod(self.access_log_path, 0o600)
    
    def _generate_encryption_key(self) -> str:
        """Generate encryption key for sensitive data"""
        # In production, this should be managed by a proper key management system
        key_file = self.data_root / "metadata" / ".encryption_key"
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)  # Owner read/write only
            return key
    
    def _initialize_encryption(self) -> Fernet:
        """Initialize encryption cipher"""
        if isinstance(self.encryption_key, str):
            key = self.encryption_key.encode()
        else:
            key = self.encryption_key
        
        return Fernet(key)
    
    def _get_local_ip(self) -> str:
        """Get local IP address"""
        try:
            # Connect to a remote address to get local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
    
    def calculate_checksums(self, file_path: Path) -> Tuple[str, str]:
        """Calculate MD5 and SHA256 checksums for file integrity"""
        md5_hash = hashlib.md5()
        sha256_hash = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
                sha256_hash.update(chunk)
        
        return md5_hash.hexdigest(), sha256_hash.hexdigest()
    
    def verify_file_integrity(self, file_path: Path) -> bool:
        """Verify file integrity using stored checksums"""
        with sqlite3.connect(self.security_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT checksum_md5, checksum_sha256 FROM file_metadata WHERE file_path = ?',
                (str(file_path),)
            )
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"No stored checksums for {file_path}")
                return False
            
            stored_md5, stored_sha256 = result
            current_md5, current_sha256 = self.calculate_checksums(file_path)
            
            if stored_md5 != current_md5 or stored_sha256 != current_sha256:
                self._log_security_alert(
                    "integrity_violation",
                    "high",
                    f"File integrity check failed for {file_path}",
                    str(file_path)
                )
                return False
            
            return True
    
    def store_file_securely(
        self, 
        source_path: Union[str, Path], 
        destination_category: str,
        security_level: SecurityLevel = SecurityLevel.INTERNAL,
        encrypt: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Store file securely with proper permissions, checksums, and logging
        
        Args:
            source_path: Path to source file
            destination_category: Category (raw/kegg, raw/ncbi, etc.)
            security_level: Security classification level
            encrypt: Whether to encrypt the file
            metadata: Additional metadata
            
        Returns:
            Path to stored file
        """
        source_path = Path(source_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        # Determine destination path
        dest_dir = self.data_root / destination_category
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate secure filename if needed
        if encrypt or security_level in [SecurityLevel.CONFIDENTIAL, SecurityLevel.SECRET]:
            dest_filename = f"{uuid.uuid4()}.secure"
            dest_path = dest_dir / dest_filename
        else:
            dest_path = dest_dir / source_path.name
        
        try:
            with self.lock:
                # Calculate checksums before copying
                source_md5, source_sha256 = self.calculate_checksums(source_path)
                
                # Copy or encrypt file
                if encrypt:
                    self._encrypt_file(source_path, dest_path)
                else:
                    shutil.copy2(source_path, dest_path)
                
                # Set secure permissions
                if security_level in [SecurityLevel.CONFIDENTIAL, SecurityLevel.SECRET]:
                    os.chmod(dest_path, 0o600)  # Owner only
                elif security_level == SecurityLevel.RESTRICTED:
                    os.chmod(dest_path, 0o640)  # Owner rw, group r
                else:
                    os.chmod(dest_path, 0o644)  # Standard permissions
                
                # Store metadata
                file_metadata = FileMetadata(
                    file_path=str(dest_path),
                    original_name=source_path.name,
                    size_bytes=dest_path.stat().st_size,
                    checksum_md5=source_md5,
                    checksum_sha256=source_sha256,
                    security_level=security_level,
                    created_by=self.current_user,
                    created_at=datetime.now(timezone.utc),
                    last_accessed=datetime.now(timezone.utc),
                    last_modified=datetime.now(timezone.utc),
                    permissions=oct(dest_path.stat().st_mode)[-3:],
                    encrypted=encrypt,
                    metadata=metadata or {}
                )
                
                self._store_file_metadata(file_metadata)
                
                # Log access
                self._log_access(
                    dest_path,
                    AccessType.WRITE,
                    success=True,
                    details=f"Stored file from {source_path}"
                )
                
                logger.info(f"Securely stored file: {source_path} -> {dest_path}")
                return dest_path
                
        except Exception as e:
            self._log_access(
                dest_path,
                AccessType.WRITE,
                success=False,
                details=f"Failed to store file: {str(e)}"
            )
            raise
    
    def access_file_securely(
        self, 
        file_path: Union[str, Path],
        access_type: AccessType = AccessType.READ
    ) -> Path:
        """
        Access file with security checks and logging
        
        Args:
            file_path: Path to file
            access_type: Type of access requested
            
        Returns:
            Path to file (decrypted if necessary)
        """
        file_path = Path(file_path)
        
        try:
            with self.lock:
                # Verify file exists
                if not file_path.exists():
                    raise FileNotFoundError(f"File not found: {file_path}")
                
                # Check permissions
                if not self._check_access_permissions(file_path, access_type):
                    raise PermissionError(f"Access denied for {access_type.value} on {file_path}")
                
                # Verify file integrity
                if not self.verify_file_integrity(file_path):
                    raise ValueError(f"File integrity check failed: {file_path}")
                
                # Handle encrypted files
                metadata = self._get_file_metadata(str(file_path))
                if metadata and metadata.get('encrypted', False):
                    decrypted_path = self._decrypt_file_to_temp(file_path)
                    final_path = decrypted_path
                else:
                    final_path = file_path
                
                # Update access time
                self._update_access_time(str(file_path))
                
                # Log access
                self._log_access(
                    file_path,
                    access_type,
                    success=True,
                    details="Secure access granted"
                )
                
                return final_path
                
        except Exception as e:
            self._log_access(
                file_path,
                access_type,
                success=False,
                details=f"Access denied: {str(e)}"
            )
            raise
    
    def create_backup(self, source_path: Union[str, Path], backup_type: str = "manual") -> str:
        """Create secure backup of file or directory"""
        source_path = Path(source_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        backup_dir = self.data_root / "backups" / backup_type
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        if source_path.is_file():
            backup_name = f"{source_path.stem}_{timestamp}{source_path.suffix}.gz"
            backup_path = backup_dir / backup_name
            
            # Create compressed backup
            with open(source_path, 'rb') as f_in:
                with gzip.open(backup_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            backup_name = f"{source_path.name}_{timestamp}.tar.gz"
            backup_path = backup_dir / backup_name
            
            # Create compressed archive
            with tarfile.open(backup_path, 'w:gz') as tar:
                tar.add(source_path, arcname=source_path.name)
        
        # Set secure permissions
        os.chmod(backup_path, 0o600)
        
        logger.info(f"Created backup: {backup_path}")
        return str(backup_path)
    
    def _encrypt_file(self, source_path: Path, dest_path: Path):
        """Encrypt file to destination"""
        with open(source_path, 'rb') as f_in:
            with open(dest_path, 'wb') as f_out:
                file_data = f_in.read()
                encrypted_data = self.cipher_suite.encrypt(file_data)
                f_out.write(encrypted_data)
    
    def _decrypt_file_to_temp(self, encrypted_path: Path) -> Path:
        """Decrypt file to temporary location"""
        temp_dir = self.data_root / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        temp_path = temp_dir / f"decrypted_{uuid.uuid4()}"
        
        with open(encrypted_path, 'rb') as f_in:
            with open(temp_path, 'wb') as f_out:
                encrypted_data = f_in.read()
                decrypted_data = self.cipher_suite.decrypt(encrypted_data)
                f_out.write(decrypted_data)
        
        # Set permissions and auto-cleanup
        os.chmod(temp_path, 0o600)
        
        return temp_path
    
    def _check_access_permissions(self, file_path: Path, access_type: AccessType) -> bool:
        """Check if current user has required permissions"""
        try:
            file_stat = file_path.stat()
            file_mode = file_stat.st_mode
            
            # Check owner permissions
            if access_type == AccessType.READ:
                return bool(file_mode & stat.S_IRUSR)
            elif access_type == AccessType.WRITE:
                return bool(file_mode & stat.S_IWUSR)
            else:
                return bool(file_mode & stat.S_IRUSR)  # Default to read check
                
        except Exception:
            return False
    
    def _store_file_metadata(self, metadata: FileMetadata):
        """Store file metadata in security database"""
        with sqlite3.connect(self.security_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO file_metadata 
                (file_path, original_name, size_bytes, checksum_md5, checksum_sha256,
                 security_level, created_by, created_at, last_accessed, last_modified,
                 permissions, encrypted, backup_copies, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata.file_path,
                metadata.original_name,
                metadata.size_bytes,
                metadata.checksum_md5,
                metadata.checksum_sha256,
                metadata.security_level.value,
                metadata.created_by,
                metadata.created_at,
                metadata.last_accessed,
                metadata.last_modified,
                metadata.permissions,
                metadata.encrypted,
                json.dumps(metadata.backup_copies),
                json.dumps(metadata.metadata)
            ))
            conn.commit()
    
    def _get_file_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get file metadata from security database"""
        with sqlite3.connect(self.security_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM file_metadata WHERE file_path = ?',
                (file_path,)
            )
            result = cursor.fetchone()
            
            if result:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, result))
            return None
    
    def _update_access_time(self, file_path: str):
        """Update last access time in metadata"""
        with sqlite3.connect(self.security_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE file_metadata SET last_accessed = ? WHERE file_path = ?',
                (datetime.now(timezone.utc), file_path)
            )
            conn.commit()
    
    def _log_access(
        self, 
        file_path: Path, 
        access_type: AccessType, 
        success: bool, 
        details: str = ""
    ):
        """Log file access attempt"""
        record = AccessRecord(
            record_id=str(uuid.uuid4()),
            file_path=str(file_path),
            access_type=access_type,
            user=self.current_user,
            ip_address=self.ip_address,
            hostname=self.hostname,
            timestamp=datetime.now(timezone.utc),
            success=success,
            details=details
        )
        
        # Log to database
        with sqlite3.connect(self.security_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO access_log 
                (record_id, file_path, access_type, user, ip_address, hostname,
                 timestamp, success, details, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.record_id,
                record.file_path,
                record.access_type.value,
                record.user,
                record.ip_address,
                record.hostname,
                record.timestamp,
                record.success,
                record.details,
                json.dumps(record.metadata)
            ))
            conn.commit()
        
        # Log to file
        log_message = (
            f"ACCESS: {access_type.value} | FILE: {file_path} | "
            f"USER: {self.current_user} | SUCCESS: {success} | "
            f"DETAILS: {details}"
        )
        
        if success:
            self.access_logger.info(log_message)
        else:
            self.access_logger.warning(log_message)
    
    def _log_security_alert(
        self, 
        alert_type: str, 
        severity: str, 
        description: str, 
        file_path: str = ""
    ):
        """Log security alert"""
        alert_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.security_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO security_alerts 
                (alert_id, alert_type, severity, description, file_path, user, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert_id,
                alert_type,
                severity,
                description,
                file_path,
                self.current_user,
                datetime.now(timezone.utc),
                json.dumps({})
            ))
            conn.commit()
        
        logger.warning(f"SECURITY ALERT [{severity}]: {description}")
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        with sqlite3.connect(self.security_db_path) as conn:
            cursor = conn.cursor()
            
            report = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'system_info': {
                    'hostname': self.hostname,
                    'user': self.current_user,
                    'data_root': str(self.data_root)
                },
                'file_summary': {},
                'access_summary': {},
                'security_alerts': {},
                'recommendations': []
            }
            
            # File summary
            cursor.execute('''
                SELECT security_level, COUNT(*), SUM(size_bytes)
                FROM file_metadata
                GROUP BY security_level
            ''')
            
            for level, count, total_size in cursor.fetchall():
                report['file_summary'][level] = {
                    'count': count,
                    'total_size_mb': (total_size or 0) / (1024 * 1024)
                }
            
            # Access summary (last 30 days)
            cursor.execute('''
                SELECT access_type, success, COUNT(*)
                FROM access_log
                WHERE timestamp > datetime('now', '-30 days')
                GROUP BY access_type, success
            ''')
            
            for access_type, success, count in cursor.fetchall():
                key = f"{access_type}_{'success' if success else 'failed'}"
                report['access_summary'][key] = count
            
            # Security alerts (last 30 days)
            cursor.execute('''
                SELECT alert_type, severity, COUNT(*)
                FROM security_alerts
                WHERE timestamp > datetime('now', '-30 days')
                GROUP BY alert_type, severity
            ''')
            
            for alert_type, severity, count in cursor.fetchall():
                key = f"{alert_type}_{severity}"
                report['security_alerts'][key] = count
            
            # Generate recommendations
            total_failed_access = sum(
                count for key, count in report['access_summary'].items()
                if 'failed' in key
            )
            
            if total_failed_access > 10:
                report['recommendations'].append(
                    f"High number of failed access attempts ({total_failed_access}). Review access controls."
                )
            
            total_alerts = sum(report['security_alerts'].values())
            if total_alerts > 0:
                report['recommendations'].append(
                    f"Active security alerts ({total_alerts}). Review and resolve alerts."
                )
        
        return report


# Example usage functions
def initialize_secure_data_environment():
    """Initialize secure data environment"""
    manager = SecureDataManager()
    
    # Set up initial security policies
    logger.info("Secure data environment initialized")
    return manager

def store_kegg_data_securely(file_path: str, manager: SecureDataManager):
    """Example: Store KEGG data securely"""
    return manager.store_file_securely(
        source_path=file_path,
        destination_category="raw/kegg",
        security_level=SecurityLevel.INTERNAL,
        metadata={'source': 'KEGG Database', 'data_type': 'pathway'}
    )

def store_genomic_data_securely(file_path: str, manager: SecureDataManager):
    """Example: Store genomic data with high security"""
    return manager.store_file_securely(
        source_path=file_path,
        destination_category="raw/ncbi",
        security_level=SecurityLevel.RESTRICTED,
        encrypt=True,
        metadata={'source': 'NCBI', 'data_type': 'genome', 'sensitive': True}
    )


if __name__ == "__main__":
    # Initialize secure data manager
    manager = initialize_secure_data_environment()
    
    # Generate security report
    report = manager.generate_security_report()
    print(json.dumps(report, indent=2, default=str)) 
#!/usr/bin/env python3
"""
Centralized Database Configuration Manager
==========================================

Provides centralized SQLite database configuration and management for the
astrobiology data system. Ensures consistent database paths, settings, and
initialization across all components.
"""

import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Configuration for a single database"""

    name: str
    path: str
    description: str
    tables: List[str]


class DatabaseManager:
    """
    Centralized database configuration and connection manager.

    Features:
    - Consistent database paths across all systems
    - Connection pooling and management
    - Automatic database initialization
    - Performance optimization
    - Backup and maintenance coordination
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.databases = self._parse_database_config()
        self.connections = {}
        self.lock = threading.Lock()

        # Initialize all database directories
        self._initialize_directories()

        logger.info(f"DatabaseManager initialized with {len(self.databases)} databases")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            return config.get("database", {})
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}")
            return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Default database configuration if YAML not available"""
        return {
            "base_path": "data",
            "databases": {
                "metadata": {
                    "path": "data/metadata/metadata.db",
                    "description": "Main scientific metadata and dataset registry",
                    "tables": ["datasets", "experiments", "data_chunks"],
                },
                "versions": {
                    "path": "data/versions/versions.db",
                    "description": "Data versioning and provenance tracking",
                    "tables": ["datasets", "versions", "changes"],
                },
                "quality": {
                    "path": "data/quality/quality_monitor.db",
                    "description": "Data quality reports and monitoring",
                    "tables": ["quality_reports", "quality_issues"],
                },
                "security": {
                    "path": "data/metadata/security.db",
                    "description": "Security and access logging",
                    "tables": ["file_metadata", "access_log"],
                },
            },
            "settings": {
                "sqlite": {
                    "journal_mode": "WAL",
                    "synchronous": "NORMAL",
                    "cache_size": 10000,
                    "foreign_keys": True,
                }
            },
        }

    def _parse_database_config(self) -> Dict[str, DatabaseConfig]:
        """Parse database configurations from config"""
        databases = {}

        for name, config in self.config.get("databases", {}).items():
            databases[name] = DatabaseConfig(
                name=name,
                path=config["path"],
                description=config["description"],
                tables=config.get("tables", []),
            )

        return databases

    def _initialize_directories(self):
        """Initialize all database directories"""
        for db_config in self.databases.values():
            db_path = Path(db_config.path)
            db_path.parent.mkdir(parents=True, exist_ok=True)

    def get_database_path(self, database_name: str) -> str:
        """Get the standardized path for a database"""
        if database_name not in self.databases:
            raise ValueError(
                f"Unknown database: {database_name}. Available: {list(self.databases.keys())}"
            )

        return self.databases[database_name].path

    def get_database_config(self, database_name: str) -> DatabaseConfig:
        """Get the full configuration for a database"""
        if database_name not in self.databases:
            raise ValueError(f"Unknown database: {database_name}")

        return self.databases[database_name]

    @contextmanager
    def get_connection(self, database_name: str):
        """Get a configured database connection with proper settings"""
        db_path = self.get_database_path(database_name)

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = None
        try:
            conn = sqlite3.connect(db_path, timeout=30.0)

            # Apply SQLite configuration
            settings = self.config.get("settings", {}).get("sqlite", {})

            if settings.get("journal_mode"):
                conn.execute(f"PRAGMA journal_mode = {settings['journal_mode']}")

            if settings.get("synchronous"):
                conn.execute(f"PRAGMA synchronous = {settings['synchronous']}")

            if settings.get("cache_size"):
                conn.execute(f"PRAGMA cache_size = {settings['cache_size']}")

            if settings.get("temp_store"):
                conn.execute(f"PRAGMA temp_store = {settings['temp_store']}")

            if settings.get("foreign_keys"):
                conn.execute("PRAGMA foreign_keys = ON")

            # Enable row factory for easier access
            conn.row_factory = sqlite3.Row

            yield conn

        except Exception as e:
            logger.error(f"Database connection error for {database_name}: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def initialize_database(self, database_name: str, schema_sql: Optional[str] = None):
        """Initialize a database with proper configuration"""
        db_config = self.get_database_config(database_name)

        with self.get_connection(database_name) as conn:
            if schema_sql:
                conn.executescript(schema_sql)
                conn.commit()
                logger.info(f"Initialized database {database_name} with custom schema")
            else:
                logger.info(f"Database {database_name} connection verified")

    def optimize_database(self, database_name: str):
        """Optimize a database (VACUUM, ANALYZE, etc.)"""
        with self.get_connection(database_name) as conn:
            try:
                # Analyze query planner statistics
                conn.execute("ANALYZE")

                # Optimize database file
                conn.execute("PRAGMA optimize")

                # Check integrity
                result = conn.execute("PRAGMA integrity_check").fetchone()
                if result[0] != "ok":
                    logger.warning(f"Integrity check failed for {database_name}: {result[0]}")

                conn.commit()
                logger.info(f"Optimized database {database_name}")

            except Exception as e:
                logger.error(f"Error optimizing database {database_name}: {e}")

    def get_database_info(self, database_name: str) -> Dict[str, Any]:
        """Get information about a database"""
        db_config = self.get_database_config(database_name)
        db_path = Path(db_config.path)

        info = {
            "name": database_name,
            "path": str(db_path),
            "description": db_config.description,
            "exists": db_path.exists(),
            "size_mb": 0,
            "tables": [],
            "table_counts": {},
        }

        if db_path.exists():
            info["size_mb"] = db_path.stat().st_size / (1024 * 1024)

            try:
                with self.get_connection(database_name) as conn:
                    # Get table names
                    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = [row[0] for row in cursor.fetchall()]
                    info["tables"] = tables

                    # Get table counts
                    for table in tables:
                        try:
                            cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                            count = cursor.fetchone()[0]
                            info["table_counts"][table] = count
                        except Exception:
                            info["table_counts"][table] = "Error"

            except Exception as e:
                logger.warning(f"Could not get info for {database_name}: {e}")

        return info

    def list_databases(self) -> List[str]:
        """List all configured databases"""
        return list(self.databases.keys())

    def verify_all_databases(self) -> Dict[str, Any]:
        """Verify all databases are accessible and properly configured"""
        results = {
            "total_databases": len(self.databases),
            "accessible": 0,
            "errors": [],
            "database_info": {},
        }

        for db_name in self.databases:
            try:
                info = self.get_database_info(db_name)
                results["database_info"][db_name] = info

                if info["exists"]:
                    results["accessible"] += 1

            except Exception as e:
                error_msg = f"Error verifying {db_name}: {e}"
                results["errors"].append(error_msg)
                logger.error(error_msg)

        return results


# Global database manager instance
_db_manager = None


def get_database_manager(config_path: str = "config/config.yaml") -> DatabaseManager:
    """Get the global database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(config_path)
    return _db_manager


def get_database_path(database_name: str) -> str:
    """Convenience function to get a database path"""
    return get_database_manager().get_database_path(database_name)


def get_database_connection(database_name: str):
    """Convenience function to get a database connection"""
    return get_database_manager().get_connection(database_name)


def verify_database_system() -> Dict[str, Any]:
    """Verify the entire database system"""
    return get_database_manager().verify_all_databases()


if __name__ == "__main__":
    # Test the database manager
    print("Testing Database Manager...")

    manager = get_database_manager()

    print(f"Configured databases: {manager.list_databases()}")

    # Verify all databases
    verification = manager.verify_all_databases()
    print(f"Verification results: {verification}")

    # Test a specific database connection
    try:
        with manager.get_connection("metadata") as conn:
            print("Successfully connected to metadata database")
    except Exception as e:
        print(f"Error connecting to metadata database: {e}")

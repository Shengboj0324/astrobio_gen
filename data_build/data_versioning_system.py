#!/usr/bin/env python3
"""
Advanced Data Versioning and Provenance System
==============================================

Comprehensive data versioning system for astrobiology genomics research:
- Complete data lineage tracking
- Version control for datasets
- Change detection and diff analysis
- Reproducibility guarantees
- Provenance graph construction
- Automated backup and rollback
- Collaborative version management
- Integration with quality systems

NASA-grade data versioning with full audit traiining
"""

import asyncio
import difflib
import filecmp
import gzip
import hashlib
import json
import logging
import os
import pickle
import shutil
import sqlite3
import tarfile
import threading
import time
import uuid
import zipfile
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import git
import networkx as nx
import numpy as np
import pandas as pd
from git import GitCommandError, Repo

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Types of data changes"""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    MOVE = "move"
    COPY = "copy"
    MERGE = "merge"
    SPLIT = "split"
    TRANSFORM = "transform"


class VersionStatus(Enum):
    """Version status"""

    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    CORRUPTED = "corrupted"


class ConflictType(Enum):
    """Types of version conflicts"""

    CONTENT = "content"
    METADATA = "metadata"
    SCHEMA = "schema"
    DEPENDENCY = "dependency"


@dataclass
class DataVersion:
    """Data version information"""

    version_id: str
    dataset_id: str
    version_number: str
    parent_versions: List[str] = field(default_factory=list)
    status: VersionStatus = VersionStatus.DRAFT
    checksum: str = ""
    size: int = 0
    created_by: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Change information
    changes: List[Dict[str, Any]] = field(default_factory=list)
    files_added: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    files_deleted: List[str] = field(default_factory=list)

    # Quality information
    quality_score: float = 0.0
    quality_checks_passed: int = 0
    quality_checks_failed: int = 0

    # Storage information
    storage_path: str = ""
    compressed: bool = True
    encryption_key: str = ""


@dataclass
class ProvenanceRecord:
    """Data provenance record"""

    provenance_id: str
    entity_id: str
    entity_type: str
    activity_id: str
    activity_type: str
    agent_id: str
    agent_type: str
    generation_time: datetime
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    code_version: str = ""
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChangeRecord:
    """Detailed change record"""

    change_id: str
    version_id: str
    change_type: ChangeType
    entity_path: str
    old_value: Any = None
    new_value: Any = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    description: str = ""
    automated: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conflict:
    """Version conflict information"""

    conflict_id: str
    conflict_type: ConflictType
    base_version: str
    version_a: str
    version_b: str
    conflicting_entity: str
    description: str
    resolution_options: List[str] = field(default_factory=list)
    resolved: bool = False
    resolution: str = ""
    resolved_by: str = ""
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataDiffer:
    """Advanced data difference analysis"""

    def __init__(self):
        self.differ = difflib.unified_diff

    def compare_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, Any]:
        """Compare two DataFrames and return detailed differences"""
        diff_result = {
            "identical": False,
            "shape_changed": False,
            "columns_changed": False,
            "data_changed": False,
            "summary": {},
            "details": {
                "added_columns": [],
                "removed_columns": [],
                "modified_columns": [],
                "added_rows": [],
                "removed_rows": [],
                "modified_rows": [],
            },
        }

        # Quick identical check
        if df1.equals(df2):
            diff_result["identical"] = True
            return diff_result

        # Shape comparison
        if df1.shape != df2.shape:
            diff_result["shape_changed"] = True
            diff_result["summary"]["old_shape"] = df1.shape
            diff_result["summary"]["new_shape"] = df2.shape

        # Column comparison
        old_cols = set(df1.columns)
        new_cols = set(df2.columns)

        if old_cols != new_cols:
            diff_result["columns_changed"] = True
            diff_result["details"]["added_columns"] = list(new_cols - old_cols)
            diff_result["details"]["removed_columns"] = list(old_cols - new_cols)

        # Common columns comparison
        common_cols = old_cols.intersection(new_cols)
        if common_cols:
            # Create index-aligned comparison
            try:
                # Try to align by index
                df1_aligned = df1[list(common_cols)].sort_index()
                df2_aligned = df2[list(common_cols)].sort_index()

                # Find modified columns
                for col in common_cols:
                    if not df1_aligned[col].equals(df2_aligned[col]):
                        diff_result["details"]["modified_columns"].append(col)
                        diff_result["data_changed"] = True

                # Row-level differences (sample only for large datasets)
                if len(df1_aligned) <= 10000 and len(df2_aligned) <= 10000:
                    # Convert to string for comparison
                    df1_str = df1_aligned.astype(str)
                    df2_str = df2_aligned.astype(str)

                    # Find different rows
                    common_index = df1_str.index.intersection(df2_str.index)
                    for idx in common_index:
                        if not df1_str.loc[idx].equals(df2_str.loc[idx]):
                            diff_result["details"]["modified_rows"].append(str(idx))

                    # Find added/removed rows
                    diff_result["details"]["added_rows"] = list(
                        set(df2_str.index) - set(df1_str.index)
                    )
                    diff_result["details"]["removed_rows"] = list(
                        set(df1_str.index) - set(df2_str.index)
                    )

            except Exception as e:
                logger.warning(f"Error in detailed row comparison: {e}")
                diff_result["data_changed"] = True

        # Summary statistics
        diff_result["summary"].update(
            {
                "columns_added": len(diff_result["details"]["added_columns"]),
                "columns_removed": len(diff_result["details"]["removed_columns"]),
                "columns_modified": len(diff_result["details"]["modified_columns"]),
                "rows_added": len(diff_result["details"]["added_rows"]),
                "rows_removed": len(diff_result["details"]["removed_rows"]),
                "rows_modified": len(diff_result["details"]["modified_rows"]),
            }
        )

        return diff_result

    def compare_files(self, file1_path: Path, file2_path: Path) -> Dict[str, Any]:
        """Compare two files and return differences"""
        diff_result = {
            "identical": False,
            "size_changed": False,
            "content_changed": False,
            "details": {},
        }

        # Basic file comparison
        if filecmp.cmp(file1_path, file2_path, shallow=False):
            diff_result["identical"] = True
            return diff_result

        # Size comparison
        size1 = file1_path.stat().st_size
        size2 = file2_path.stat().st_size

        if size1 != size2:
            diff_result["size_changed"] = True
            diff_result["details"]["old_size"] = size1
            diff_result["details"]["new_size"] = size2

        # Content comparison for text files
        if file1_path.suffix.lower() in [".txt", ".csv", ".json", ".xml", ".yaml", ".yml"]:
            try:
                with open(file1_path, "r", encoding="utf-8") as f1:
                    content1 = f1.readlines()
                with open(file2_path, "r", encoding="utf-8") as f2:
                    content2 = f2.readlines()

                # Generate unified diff
                diff_lines = list(
                    difflib.unified_diff(
                        content1,
                        content2,
                        fromfile=str(file1_path),
                        tofile=str(file2_path),
                        lineterm="",
                    )
                )

                if diff_lines:
                    diff_result["content_changed"] = True
                    diff_result["details"]["diff"] = diff_lines[:100]  # Limit for storage
                    diff_result["details"]["total_diff_lines"] = len(diff_lines)

            except Exception as e:
                logger.warning(f"Error comparing text content: {e}")
                diff_result["content_changed"] = True
        else:
            # Binary file comparison
            diff_result["content_changed"] = True
            diff_result["details"]["binary_file"] = True

        return diff_result

    def compare_json(self, json1: Dict[str, Any], json2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two JSON objects"""
        diff_result = {
            "identical": False,
            "added_keys": [],
            "removed_keys": [],
            "modified_keys": [],
            "details": {},
        }

        if json1 == json2:
            diff_result["identical"] = True
            return diff_result

        def get_all_keys(obj, prefix=""):
            """Recursively get all keys from nested dict"""
            keys = set()
            if isinstance(obj, dict):
                for key, value in obj.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    keys.add(full_key)
                    if isinstance(value, dict):
                        keys.update(get_all_keys(value, full_key))
            return keys

        keys1 = get_all_keys(json1)
        keys2 = get_all_keys(json2)

        diff_result["added_keys"] = list(keys2 - keys1)
        diff_result["removed_keys"] = list(keys1 - keys2)

        # Find modified keys
        common_keys = keys1.intersection(keys2)
        for key in common_keys:
            try:
                # Navigate to nested value
                value1 = json1
                value2 = json2
                for part in key.split("."):
                    value1 = value1[part]
                    value2 = value2[part]

                if value1 != value2:
                    diff_result["modified_keys"].append(key)
                    diff_result["details"][key] = {"old_value": value1, "new_value": value2}
            except (KeyError, TypeError):
                continue

        return diff_result


class VersionStorage:
    """Advanced version storage management"""

    def __init__(self, storage_path: str = "data/versions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.compression_level = 6
        self.lock = Lock()

    def store_version(self, version: DataVersion, data: Any) -> str:
        """Store a data version"""
        with self.lock:
            version_dir = self.storage_path / version.dataset_id / version.version_id
            version_dir.mkdir(parents=True, exist_ok=True)

            # Store data
            if isinstance(data, pd.DataFrame):
                data_file = version_dir / "data.pkl.gz"
                with gzip.open(data_file, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                version.storage_path = str(data_file)
                version.size = data_file.stat().st_size
                version.compressed = True
            elif isinstance(data, dict):
                data_file = version_dir / "data.json.gz"
                with gzip.open(data_file, "wt", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                version.storage_path = str(data_file)
                version.size = data_file.stat().st_size
                version.compressed = True
            elif isinstance(data, (str, Path)):
                # File path - copy the file
                source_path = Path(data)
                if source_path.exists():
                    if source_path.is_file():
                        dest_file = version_dir / source_path.name
                        shutil.copy2(source_path, dest_file)

                        # Compress if beneficial
                        if dest_file.stat().st_size > 1024:  # 1KB threshold
                            compressed_file = Path(str(dest_file) + ".gz")
                            with open(dest_file, "rb") as f_in:
                                with gzip.open(compressed_file, "wb") as f_out:
                                    shutil.copyfileobj(f_in, f_out)
                            dest_file.unlink()  # Remove uncompressed
                            version.storage_path = str(compressed_file)
                            version.compressed = True
                        else:
                            version.storage_path = str(dest_file)
                            version.compressed = False

                        version.size = Path(version.storage_path).stat().st_size
                    else:
                        # Directory - create archive
                        archive_file = version_dir / f"{source_path.name}.tar.gz"
                        with tarfile.open(archive_file, "w:gz") as tar:
                            tar.add(source_path, arcname=source_path.name)
                        version.storage_path = str(archive_file)
                        version.size = archive_file.stat().st_size
                        version.compressed = True
            else:
                # Generic pickle storage
                data_file = version_dir / "data.pkl.gz"
                with gzip.open(data_file, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                version.storage_path = str(data_file)
                version.size = data_file.stat().st_size
                version.compressed = True

            # Calculate checksum
            version.checksum = self._calculate_checksum(version.storage_path)

            # Store version metadata
            metadata_file = version_dir / "version.json"
            with open(metadata_file, "w") as f:
                json.dump(asdict(version), f, indent=2, default=str)

            return version.storage_path

    def load_version(self, version: DataVersion) -> Any:
        """Load data from a version"""
        storage_path = Path(version.storage_path)

        if not storage_path.exists():
            raise FileNotFoundError(f"Version storage not found: {storage_path}")

        # Verify checksum
        current_checksum = self._calculate_checksum(storage_path)
        if current_checksum != version.checksum:
            logger.warning(f"Checksum mismatch for version {version.version_id}")

        # Load based on file type
        if storage_path.suffix == ".gz":
            if storage_path.name.endswith(".pkl.gz"):
                with gzip.open(storage_path, "rb") as f:
                    return pickle.load(f)
            elif storage_path.name.endswith(".json.gz"):
                with gzip.open(storage_path, "rt", encoding="utf-8") as f:
                    return json.load(f)
            elif storage_path.name.endswith(".tar.gz"):
                # Extract to temporary directory
                temp_dir = storage_path.parent / "temp_extract"
                temp_dir.mkdir(exist_ok=True)
                with tarfile.open(storage_path, "r:gz") as tar:
                    tar.extractall(temp_dir)
                return temp_dir
            else:
                # Generic compressed file
                with gzip.open(storage_path, "rb") as f:
                    return f.read()
        else:
            # Uncompressed file
            if storage_path.suffix == ".json":
                with open(storage_path, "r") as f:
                    return json.load(f)
            elif storage_path.suffix == ".pkl":
                with open(storage_path, "rb") as f:
                    return pickle.load(f)
            else:
                return storage_path

    def _calculate_checksum(self, file_path: Union[str, Path]) -> str:
        """Calculate file checksum"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def cleanup_old_versions(self, dataset_id: str, keep_versions: int = 10):
        """Clean up old versions, keeping only the most recent"""
        dataset_dir = self.storage_path / dataset_id
        if not dataset_dir.exists():
            return

        # Get all version directories
        version_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]

        # Sort by creation time
        version_dirs.sort(key=lambda x: x.stat().st_ctime, reverse=True)

        # Remove old versions
        for old_dir in version_dirs[keep_versions:]:
            try:
                shutil.rmtree(old_dir)
                logger.info(f"Cleaned up old version: {old_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up {old_dir}: {e}")


class ProvenanceTracker:
    """Advanced provenance tracking system"""

    def __init__(self):
        self.provenance_graph = nx.DiGraph()
        self.lock = Lock()

    def record_activity(
        self,
        activity_id: str,
        activity_type: str,
        inputs: List[str],
        outputs: List[str],
        agent_id: str,
        parameters: Dict[str, Any] = None,
        code_version: str = "",
        environment: Dict[str, Any] = None,
    ) -> ProvenanceRecord:
        """Record a data processing activity"""
        provenance = ProvenanceRecord(
            provenance_id=str(uuid.uuid4()),
            entity_id=outputs[0] if outputs else "",
            entity_type="dataset",
            activity_id=activity_id,
            activity_type=activity_type,
            agent_id=agent_id,
            agent_type="software",
            generation_time=datetime.now(timezone.utc),
            inputs=inputs,
            outputs=outputs,
            parameters=parameters or {},
            environment=environment or {},
            code_version=code_version,
        )

        with self.lock:
            # Add to provenance graph
            self.provenance_graph.add_node(
                activity_id,
                type="activity",
                activity_type=activity_type,
                timestamp=provenance.generation_time,
                agent=agent_id,
                parameters=parameters,
            )

            # Add input edges
            for input_id in inputs:
                self.provenance_graph.add_node(input_id, type="entity")
                self.provenance_graph.add_edge(input_id, activity_id, relation="used")

            # Add output edges
            for output_id in outputs:
                self.provenance_graph.add_node(output_id, type="entity")
                self.provenance_graph.add_edge(activity_id, output_id, relation="generated")

        return provenance

    def get_lineage(self, entity_id: str, direction: str = "backward") -> List[str]:
        """Get the lineage of an entity"""
        if entity_id not in self.provenance_graph:
            return []

        lineage = []

        if direction == "backward":
            # Get ancestors (what led to this entity)
            for predecessor in nx.ancestors(self.provenance_graph, entity_id):
                lineage.append(predecessor)
        elif direction == "forward":
            # Get descendants (what this entity led to)
            for successor in nx.descendants(self.provenance_graph, entity_id):
                lineage.append(successor)

        return lineage

    def get_provenance_path(self, start_entity: str, end_entity: str) -> List[str]:
        """Get the provenance path between two entities"""
        try:
            path = nx.shortest_path(self.provenance_graph, start_entity, end_entity)
            return path
        except nx.NetworkXNoPath:
            return []

    def export_provenance_graph(self, output_format: str = "json") -> Dict[str, Any]:
        """Export provenance graph"""
        if output_format == "json":
            return nx.node_link_data(self.provenance_graph)
        elif output_format == "dot":
            return nx.drawing.nx_pydot.to_pydot(self.provenance_graph).to_string()
        else:
            raise ValueError(f"Unsupported format: {output_format}")


class VersionManager:
    """Comprehensive version management system"""

    def __init__(self, db_path: str = "data/versions/versions.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage = VersionStorage()
        self.differ = DataDiffer()
        self.provenance = ProvenanceTracker()
        self.lock = Lock()
        self._initialize_database()

    def _initialize_database(self):
        """Initialize version management database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Datasets table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    data_type TEXT,
                    current_version TEXT,
                    created_by TEXT,
                    created_at TIMESTAMP,
                    last_modified TIMESTAMP,
                    tags TEXT,
                    metadata TEXT
                )
            """
            )

            # Versions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS versions (
                    version_id TEXT PRIMARY KEY,
                    dataset_id TEXT,
                    version_number TEXT,
                    parent_versions TEXT,
                    status TEXT,
                    checksum TEXT,
                    size INTEGER,
                    created_by TEXT,
                    created_at TIMESTAMP,
                    message TEXT,
                    tags TEXT,
                    files_added TEXT,
                    files_modified TEXT,
                    files_deleted TEXT,
                    quality_score REAL,
                    quality_checks_passed INTEGER,
                    quality_checks_failed INTEGER,
                    storage_path TEXT,
                    compressed BOOLEAN,
                    metadata TEXT,
                    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
                )
            """
            )

            # Changes table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS changes (
                    change_id TEXT PRIMARY KEY,
                    version_id TEXT,
                    change_type TEXT,
                    entity_path TEXT,
                    old_value TEXT,
                    new_value TEXT,
                    timestamp TIMESTAMP,
                    description TEXT,
                    automated BOOLEAN,
                    metadata TEXT,
                    FOREIGN KEY (version_id) REFERENCES versions(version_id)
                )
            """
            )

            # Provenance table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS provenance (
                    provenance_id TEXT PRIMARY KEY,
                    entity_id TEXT,
                    entity_type TEXT,
                    activity_id TEXT,
                    activity_type TEXT,
                    agent_id TEXT,
                    agent_type TEXT,
                    generation_time TIMESTAMP,
                    inputs TEXT,
                    outputs TEXT,
                    parameters TEXT,
                    environment TEXT,
                    code_version TEXT,
                    dependencies TEXT,
                    metadata TEXT
                )
            """
            )

            # Conflicts table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS conflicts (
                    conflict_id TEXT PRIMARY KEY,
                    conflict_type TEXT,
                    base_version TEXT,
                    version_a TEXT,
                    version_b TEXT,
                    conflicting_entity TEXT,
                    description TEXT,
                    resolution_options TEXT,
                    resolved BOOLEAN,
                    resolution TEXT,
                    resolved_by TEXT,
                    resolved_at TIMESTAMP,
                    metadata TEXT
                )
            """
            )

            # Branches table (for parallel development)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS branches (
                    branch_id TEXT PRIMARY KEY,
                    dataset_id TEXT,
                    branch_name TEXT,
                    parent_branch TEXT,
                    head_version TEXT,
                    created_by TEXT,
                    created_at TIMESTAMP,
                    description TEXT,
                    active BOOLEAN,
                    metadata TEXT,
                    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
                )
            """
            )

            conn.commit()

    def create_dataset(
        self, dataset_id: str, name: str, description: str, data_type: str, created_by: str
    ) -> str:
        """Create a new dataset"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO datasets 
                    (dataset_id, name, description, data_type, created_by, created_at, last_modified)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        dataset_id,
                        name,
                        description,
                        data_type,
                        created_by,
                        datetime.now(timezone.utc),
                        datetime.now(timezone.utc),
                    ),
                )

                conn.commit()

        return dataset_id

    def commit_version(
        self,
        dataset_id: str,
        data: Any,
        message: str,
        created_by: str,
        parent_versions: List[str] = None,
        tags: List[str] = None,
    ) -> DataVersion:
        """Commit a new version of data"""
        version_id = str(uuid.uuid4())

        # Generate version number
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT COUNT(*) FROM versions WHERE dataset_id = ?
            """,
                (dataset_id,),
            )
            version_count = cursor.fetchone()[0]
            version_number = f"v{version_count + 1}.0"

        # Create version object
        version = DataVersion(
            version_id=version_id,
            dataset_id=dataset_id,
            version_number=version_number,
            parent_versions=parent_versions or [],
            status=VersionStatus.DRAFT,
            created_by=created_by,
            message=message,
            tags=tags or [],
        )

        # Store the data
        storage_path = self.storage.store_version(version, data)

        # Detect changes if there are parent versions
        if parent_versions:
            changes = self._detect_changes(version, data, parent_versions[0])
            version.changes = changes

        # Store version in database
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO versions 
                    (version_id, dataset_id, version_number, parent_versions, status, checksum,
                     size, created_by, created_at, message, tags, files_added, files_modified,
                     files_deleted, quality_score, quality_checks_passed, quality_checks_failed,
                     storage_path, compressed, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        version.version_id,
                        version.dataset_id,
                        version.version_number,
                        json.dumps(version.parent_versions),
                        version.status.value,
                        version.checksum,
                        version.size,
                        version.created_by,
                        version.created_at,
                        version.message,
                        json.dumps(version.tags),
                        json.dumps(version.files_added),
                        json.dumps(version.files_modified),
                        json.dumps(version.files_deleted),
                        version.quality_score,
                        version.quality_checks_passed,
                        version.quality_checks_failed,
                        version.storage_path,
                        version.compressed,
                        json.dumps(version.metadata),
                    ),
                )

                # Store changes
                for change in version.changes:
                    change_record = ChangeRecord(
                        change_id=str(uuid.uuid4()),
                        version_id=version_id,
                        change_type=ChangeType(change["type"]),
                        entity_path=change["path"],
                        old_value=change.get("old_value"),
                        new_value=change.get("new_value"),
                        description=change.get("description", ""),
                        automated=True,
                        metadata=change.get("metadata", {}),
                    )

                    cursor.execute(
                        """
                        INSERT INTO changes 
                        (change_id, version_id, change_type, entity_path, old_value, new_value,
                         timestamp, description, automated, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            change_record.change_id,
                            change_record.version_id,
                            change_record.change_type.value,
                            change_record.entity_path,
                            (
                                json.dumps(change_record.old_value)
                                if change_record.old_value
                                else None
                            ),
                            (
                                json.dumps(change_record.new_value)
                                if change_record.new_value
                                else None
                            ),
                            change_record.timestamp,
                            change_record.description,
                            change_record.automated,
                            json.dumps(change_record.metadata),
                        ),
                    )

                # Update dataset current version
                cursor.execute(
                    """
                    UPDATE datasets 
                    SET current_version = ?, last_modified = ?
                    WHERE dataset_id = ?
                """,
                    (version_id, datetime.now(timezone.utc), dataset_id),
                )

                conn.commit()

        # Record provenance
        activity_id = f"commit_{version_id}"
        self.provenance.record_activity(
            activity_id=activity_id,
            activity_type="data_commit",
            inputs=parent_versions,
            outputs=[version_id],
            agent_id=created_by,
            parameters={"message": message, "tags": tags},
        )

        logger.info(f"Committed version {version_number} for dataset {dataset_id}")
        return version

    def _detect_changes(
        self, version: DataVersion, current_data: Any, parent_version_id: str
    ) -> List[Dict[str, Any]]:
        """Detect changes between current data and parent version"""
        changes = []

        try:
            # Load parent version data
            parent_version = self.get_version(parent_version_id)
            if not parent_version:
                return changes

            parent_data = self.storage.load_version(parent_version)

            # Compare based on data type
            if isinstance(current_data, pd.DataFrame) and isinstance(parent_data, pd.DataFrame):
                diff_result = self.differ.compare_dataframes(parent_data, current_data)

                if not diff_result["identical"]:
                    if diff_result["shape_changed"]:
                        changes.append(
                            {
                                "type": "update",
                                "path": "dataframe.shape",
                                "old_value": diff_result["summary"].get("old_shape"),
                                "new_value": diff_result["summary"].get("new_shape"),
                                "description": "DataFrame shape changed",
                            }
                        )

                    if diff_result["columns_changed"]:
                        for col in diff_result["details"]["added_columns"]:
                            changes.append(
                                {
                                    "type": "create",
                                    "path": f"dataframe.columns.{col}",
                                    "new_value": col,
                                    "description": f"Column {col} added",
                                }
                            )

                        for col in diff_result["details"]["removed_columns"]:
                            changes.append(
                                {
                                    "type": "delete",
                                    "path": f"dataframe.columns.{col}",
                                    "old_value": col,
                                    "description": f"Column {col} removed",
                                }
                            )

                    if diff_result["data_changed"]:
                        for col in diff_result["details"]["modified_columns"]:
                            changes.append(
                                {
                                    "type": "update",
                                    "path": f"dataframe.data.{col}",
                                    "description": f"Data in column {col} modified",
                                }
                            )

            elif isinstance(current_data, dict) and isinstance(parent_data, dict):
                diff_result = self.differ.compare_json(parent_data, current_data)

                if not diff_result["identical"]:
                    for key in diff_result["added_keys"]:
                        changes.append(
                            {
                                "type": "create",
                                "path": key,
                                "new_value": diff_result["details"].get(key, {}).get("new_value"),
                                "description": f"Key {key} added",
                            }
                        )

                    for key in diff_result["removed_keys"]:
                        changes.append(
                            {
                                "type": "delete",
                                "path": key,
                                "old_value": diff_result["details"].get(key, {}).get("old_value"),
                                "description": f"Key {key} removed",
                            }
                        )

                    for key in diff_result["modified_keys"]:
                        change_detail = diff_result["details"].get(key, {})
                        changes.append(
                            {
                                "type": "update",
                                "path": key,
                                "old_value": change_detail.get("old_value"),
                                "new_value": change_detail.get("new_value"),
                                "description": f"Value for {key} modified",
                            }
                        )

            # Update version change statistics
            version.files_modified = [
                change["path"] for change in changes if change["type"] == "update"
            ]
            version.files_added = [
                change["path"] for change in changes if change["type"] == "create"
            ]
            version.files_deleted = [
                change["path"] for change in changes if change["type"] == "delete"
            ]

        except Exception as e:
            logger.error(f"Error detecting changes: {e}")
            changes.append(
                {
                    "type": "update",
                    "path": "unknown",
                    "description": f"Change detection failed: {str(e)}",
                }
            )

        return changes

    def get_version(self, version_id: str) -> Optional[DataVersion]:
        """Get a specific version"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM versions WHERE version_id = ?", (version_id,))
            row = cursor.fetchone()

            if not row:
                return None

            return DataVersion(
                version_id=row[0],
                dataset_id=row[1],
                version_number=row[2],
                parent_versions=json.loads(row[3]) if row[3] else [],
                status=VersionStatus(row[4]),
                checksum=row[5] or "",
                size=row[6] or 0,
                created_by=row[7] or "",
                created_at=row[8],
                message=row[9] or "",
                tags=json.loads(row[10]) if row[10] else [],
                files_added=json.loads(row[11]) if row[11] else [],
                files_modified=json.loads(row[12]) if row[12] else [],
                files_deleted=json.loads(row[13]) if row[13] else [],
                quality_score=row[14] or 0.0,
                quality_checks_passed=row[15] or 0,
                quality_checks_failed=row[16] or 0,
                storage_path=row[17] or "",
                compressed=bool(row[18]),
                metadata=json.loads(row[19]) if row[19] else {},
            )

    def load_version_data(self, version_id: str) -> Any:
        """Load data from a specific version"""
        version = self.get_version(version_id)
        if not version:
            raise ValueError(f"Version {version_id} not found")

        return self.storage.load_version(version)

    def get_version_history(self, dataset_id: str) -> List[DataVersion]:
        """Get version history for a dataset"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT version_id FROM versions 
                WHERE dataset_id = ? 
                ORDER BY created_at DESC
            """,
                (dataset_id,),
            )

            versions = []
            for row in cursor.fetchall():
                version = self.get_version(row[0])
                if version:
                    versions.append(version)

            return versions

    def create_branch(
        self,
        dataset_id: str,
        branch_name: str,
        parent_branch: str,
        created_by: str,
        description: str = "",
    ) -> str:
        """Create a new branch for parallel development"""
        branch_id = str(uuid.uuid4())

        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get head version of parent branch
                if parent_branch:
                    cursor.execute(
                        """
                        SELECT head_version FROM branches 
                        WHERE branch_name = ? AND dataset_id = ?
                    """,
                        (parent_branch, dataset_id),
                    )
                    parent_head = cursor.fetchone()
                    head_version = parent_head[0] if parent_head else None
                else:
                    # Get current version of dataset
                    cursor.execute(
                        """
                        SELECT current_version FROM datasets WHERE dataset_id = ?
                    """,
                        (dataset_id,),
                    )
                    current = cursor.fetchone()
                    head_version = current[0] if current else None

                cursor.execute(
                    """
                    INSERT INTO branches 
                    (branch_id, dataset_id, branch_name, parent_branch, head_version,
                     created_by, created_at, description, active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        branch_id,
                        dataset_id,
                        branch_name,
                        parent_branch,
                        head_version,
                        created_by,
                        datetime.now(timezone.utc),
                        description,
                        True,
                    ),
                )

                conn.commit()

        return branch_id

    def merge_branches(
        self,
        dataset_id: str,
        source_branch: str,
        target_branch: str,
        merged_by: str,
        merge_message: str = "",
    ) -> DataVersion:
        """Merge one branch into another"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get head versions of both branches
            cursor.execute(
                """
                SELECT head_version FROM branches 
                WHERE branch_name = ? AND dataset_id = ?
            """,
                (source_branch, dataset_id),
            )
            source_head = cursor.fetchone()

            cursor.execute(
                """
                SELECT head_version FROM branches 
                WHERE branch_name = ? AND dataset_id = ?
            """,
                (target_branch, dataset_id),
            )
            target_head = cursor.fetchone()

            if not source_head or not target_head:
                raise ValueError("Source or target branch not found")

            # Load data from both versions
            source_data = self.load_version_data(source_head[0])
            target_data = self.load_version_data(target_head[0])

            # Simple merge strategy: use source data
            # In a real system, you'd implement sophisticated merge logic
            merged_data = source_data

            # Create merge commit
            merge_version = self.commit_version(
                dataset_id=dataset_id,
                data=merged_data,
                message=f"Merge {source_branch} into {target_branch}: {merge_message}",
                created_by=merged_by,
                parent_versions=[source_head[0], target_head[0]],
                tags=["merge"],
            )

            # Update target branch head
            cursor.execute(
                """
                UPDATE branches 
                SET head_version = ? 
                WHERE branch_name = ? AND dataset_id = ?
            """,
                (merge_version.version_id, target_branch, dataset_id),
            )

            conn.commit()

        return merge_version

    def rollback_to_version(
        self, dataset_id: str, version_id: str, rolled_by: str, rollback_message: str = ""
    ) -> DataVersion:
        """Rollback dataset to a specific version"""
        # Load the target version data
        target_data = self.load_version_data(version_id)

        # Create a new version with the rolled back data
        rollback_version = self.commit_version(
            dataset_id=dataset_id,
            data=target_data,
            message=f"Rollback to {version_id}: {rollback_message}",
            created_by=rolled_by,
            tags=["rollback"],
        )

        return rollback_version

    def export_dataset_history(self, dataset_id: str, output_format: str = "json") -> str:
        """Export complete history of a dataset"""
        versions = self.get_version_history(dataset_id)

        export_data = {
            "dataset_id": dataset_id,
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_versions": len(versions),
            "versions": [],
        }

        for version in versions:
            version_data = asdict(version)

            # Get changes for this version
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM changes WHERE version_id = ?
                """,
                    (version.version_id,),
                )

                changes = []
                for change_row in cursor.fetchall():
                    changes.append(
                        {
                            "change_id": change_row[0],
                            "change_type": change_row[2],
                            "entity_path": change_row[3],
                            "old_value": json.loads(change_row[4]) if change_row[4] else None,
                            "new_value": json.loads(change_row[5]) if change_row[5] else None,
                            "timestamp": change_row[6],
                            "description": change_row[7],
                            "automated": bool(change_row[8]),
                        }
                    )

                version_data["changes"] = changes

            export_data["versions"].append(version_data)

        # Save export
        output_path = f"data/exports/{dataset_id}_history_{int(time.time())}.{output_format}"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if output_format == "json":
            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {output_format}")

        return output_path


# Main execution function
def main():
    """Main execution function for versioning system"""
    # Initialize version manager
    version_manager = VersionManager()

    # Create a sample dataset
    dataset_id = "sample_kegg_pathways"
    version_manager.create_dataset(
        dataset_id=dataset_id,
        name="Sample KEGG Pathways",
        description="Sample dataset for testing versioning",
        data_type="kegg_pathway",
        created_by="system",
    )

    # Create initial version
    sample_data = pd.DataFrame(
        {
            "pathway_id": ["map00010", "map00020", "map00030"],
            "name": ["Glycolysis", "TCA Cycle", "Pentose Phosphate"],
            "reaction_count": [10, 8, 7],
            "compound_count": [12, 10, 9],
        }
    )

    v1 = version_manager.commit_version(
        dataset_id=dataset_id,
        data=sample_data,
        message="Initial version with 3 pathways",
        created_by="system",
        tags=["initial", "kegg"],
    )

    # Create second version with changes
    updated_data = sample_data.copy()
    updated_data.loc[0, "reaction_count"] = 12  # Update glycolysis
    updated_data.loc[len(updated_data)] = ["map00040", "Pentose Glucuronate", 6, 8]  # Add new row

    v2 = version_manager.commit_version(
        dataset_id=dataset_id,
        data=updated_data,
        message="Updated glycolysis count and added pentose glucuronate pathway",
        created_by="system",
        parent_versions=[v1.version_id],
        tags=["update"],
    )

    # Get version history
    history = version_manager.get_version_history(dataset_id)
    print(f"Version History:")
    for version in history:
        print(f"  {version.version_number}: {version.message} ({len(version.changes)} changes)")

    # Export history
    export_path = version_manager.export_dataset_history(dataset_id)
    print(f"History exported to: {export_path}")

    return version_manager


if __name__ == "__main__":
    version_manager = main()

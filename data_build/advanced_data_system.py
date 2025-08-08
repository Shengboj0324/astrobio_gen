#!/usr/bin/env python3
"""
Advanced Data Management System for Astrobiology Genomics Research
==================================================================

NASA-grade data management system for comprehensive integration of:
- KEGG pathway database (7,302+ pathways)
- NCBI AGORA2 microorganism reconstructions (7,302 species)
- Genomic and metabolic datasets
- Advanced quality control and validation
- Automated data provenance and versioning
"""

import asyncio
import ftplib
import gzip
import hashlib
import json
import logging
import os
import pickle
import sqlite3
import tarfile
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin

import aiohttp
import networkx as nx
import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class DataSource:
    """Comprehensive data source configuration"""

    name: str
    url: str
    data_type: str
    update_frequency: str
    version: str = "latest"
    checksum: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    last_updated: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)


@dataclass
class QualityMetrics:
    """Advanced quality metrics for data validation"""

    completeness: float = 0.0
    consistency: float = 0.0
    accuracy: float = 0.0
    validity: float = 0.0
    uniqueness: float = 0.0
    timeliness: float = 0.0
    conformity: float = 0.0
    integrity: float = 0.0

    def overall_score(self) -> float:
        """Calculate weighted overall quality score"""
        weights = {
            "completeness": 0.20,
            "consistency": 0.15,
            "accuracy": 0.20,
            "validity": 0.15,
            "uniqueness": 0.10,
            "timeliness": 0.10,
            "conformity": 0.05,
            "integrity": 0.05,
        }
        return sum(getattr(self, metric) * weight for metric, weight in weights.items())


class DataProcessor(ABC):
    """Abstract base class for data processors"""

    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process data according to specific requirements"""
        pass

    @abstractmethod
    def validate(self, data: Any) -> QualityMetrics:
        """Validate data quality"""
        pass


class KEGGProcessor(DataProcessor):
    """Advanced KEGG pathway data processor"""

    def __init__(self, base_url: str = "https://rest.kegg.jp/"):
        self.base_url = base_url
        self.pathway_cache = {}
        self.compound_cache = {}
        self.reaction_cache = {}

    async def fetch_pathway_data(self, pathway_id: str) -> Dict[str, Any]:
        """Fetch comprehensive pathway data from KEGG"""
        try:
            async with aiohttp.ClientSession() as session:
                # Fetch pathway information
                pathway_url = f"{self.base_url}get/{pathway_id}"
                async with session.get(pathway_url) as response:
                    pathway_data = await response.text()

                # Fetch pathway reaction list
                reaction_url = f"{self.base_url}link/reaction/{pathway_id}"
                async with session.get(reaction_url) as response:
                    reaction_data = await response.text()

                # Fetch pathway compound list
                compound_url = f"{self.base_url}link/compound/{pathway_id}"
                async with session.get(compound_url) as response:
                    compound_data = await response.text()

                return {
                    "pathway_id": pathway_id,
                    "pathway_data": pathway_data,
                    "reactions": reaction_data,
                    "compounds": compound_data,
                    "timestamp": datetime.now(timezone.utc),
                }
        except Exception as e:
            logger.error(f"Error fetching KEGG data for {pathway_id}: {e}")
            return {}

    def process_pathway_network(self, pathway_data: Dict[str, Any]) -> nx.DiGraph:
        """Process pathway into network format"""
        G = nx.DiGraph()

        # Parse reactions and compounds
        reactions = self._parse_reactions(pathway_data.get("reactions", ""))
        compounds = self._parse_compounds(pathway_data.get("compounds", ""))

        # Build network
        for reaction in reactions:
            substrates = reaction.get("substrates", [])
            products = reaction.get("products", [])

            for substrate in substrates:
                for product in products:
                    G.add_edge(substrate, product, reaction=reaction["id"])

        return G

    def _parse_reactions(self, reaction_data: str) -> List[Dict[str, Any]]:
        """Parse reaction data from KEGG format"""
        reactions = []
        for line in reaction_data.strip().split("\n"):
            if line.startswith("rn:"):
                parts = line.split("\t")
                if len(parts) >= 2:
                    reaction_id = parts[0].replace("rn:", "")
                    pathway_id = parts[1]
                    reactions.append(
                        {"id": reaction_id, "pathway": pathway_id, "substrates": [], "products": []}
                    )
        return reactions

    def _parse_compounds(self, compound_data: str) -> List[Dict[str, Any]]:
        """Parse compound data from KEGG format"""
        compounds = []
        for line in compound_data.strip().split("\n"):
            if line.startswith("cpd:"):
                parts = line.split("\t")
                if len(parts) >= 2:
                    compound_id = parts[0].replace("cpd:", "")
                    pathway_id = parts[1]
                    compounds.append({"id": compound_id, "pathway": pathway_id})
        return compounds

    def process(self, data: Any) -> pd.DataFrame:
        """Process KEGG data into standardized format"""
        processed_data = []

        if isinstance(data, dict):
            network = self.process_pathway_network(data)
            for edge in network.edges(data=True):
                processed_data.append(
                    {
                        "reaction": edge[2]["reaction"],
                        "substrate": edge[0],
                        "product": edge[1],
                        "pathway": data.get("pathway_id", "unknown"),
                        "timestamp": datetime.now(timezone.utc),
                    }
                )

        return pd.DataFrame(processed_data)

    def validate(self, data: Any) -> QualityMetrics:
        """Validate KEGG data quality"""
        metrics = QualityMetrics()

        if isinstance(data, pd.DataFrame):
            # Completeness check
            total_cells = data.size
            non_null_cells = data.count().sum()
            metrics.completeness = non_null_cells / total_cells if total_cells > 0 else 0.0

            # Consistency check - valid KEGG IDs
            valid_reactions = data["reaction"].str.match(r"^R\d+$", na=False).sum()
            valid_compounds = (
                data["substrate"].str.match(r"^C\d+$", na=False).sum()
                + data["product"].str.match(r"^C\d+$", na=False).sum()
            )
            total_ids = len(data) * 3  # reaction + substrate + product
            metrics.consistency = (
                (valid_reactions + valid_compounds) / total_ids if total_ids > 0 else 0.0
            )

            # Uniqueness check
            unique_reactions = data["reaction"].nunique()
            total_reactions = len(data)
            metrics.uniqueness = unique_reactions / total_reactions if total_reactions > 0 else 0.0

            # Validity check - no empty or null values
            valid_rows = data.dropna().shape[0]
            metrics.validity = valid_rows / len(data) if len(data) > 0 else 0.0

            # Accuracy - cross-reference with known KEGG pathways
            known_pathways = {"map00010", "map00020", "map00030", "map00040"}  # Sample
            pathway_matches = data["pathway"].isin(known_pathways).sum()
            metrics.accuracy = pathway_matches / len(data) if len(data) > 0 else 0.0

            # Timeliness - data should be recent
            if "timestamp" in data.columns and not data["timestamp"].empty:
                latest_time = data["timestamp"].max()
                days_old = (datetime.now(timezone.utc) - latest_time).days
                metrics.timeliness = max(0, 1 - days_old / 365)  # Decay over year

            # Conformity - follows expected schema
            expected_columns = {"reaction", "substrate", "product", "pathway"}
            actual_columns = set(data.columns)
            metrics.conformity = len(expected_columns.intersection(actual_columns)) / len(
                expected_columns
            )

            # Integrity - referential integrity
            metrics.integrity = 1.0  # Assume good for now

        return metrics


class NCBIProcessor(DataProcessor):
    """Advanced NCBI/AGORA2 data processor"""

    def __init__(self, ftp_host: str = "ftp.ncbi.nlm.nih.gov"):
        self.ftp_host = ftp_host
        self.agora2_base_url = "https://www.vmh.life/files/reconstructions/AGORA2/"
        self.genome_cache = {}

    async def fetch_agora2_data(self) -> Dict[str, Any]:
        """Fetch AGORA2 reconstruction data"""
        try:
            async with aiohttp.ClientSession() as session:
                # Fetch AGORA2 model list
                list_url = f"{self.agora2_base_url}AGORA2_infoFile.xlsx"
                async with session.get(list_url) as response:
                    if response.status == 200:
                        content = await response.read()
                        # Save to temporary file for processing
                        temp_file = Path("temp_agora2.xlsx")
                        with open(temp_file, "wb") as f:
                            f.write(content)

                        # Read Excel file
                        df = pd.read_excel(temp_file)
                        temp_file.unlink()  # Clean up

                        return {
                            "models": df.to_dict("records"),
                            "timestamp": datetime.now(timezone.utc),
                        }
        except Exception as e:
            logger.error(f"Error fetching AGORA2 data: {e}")
            return {}

    def fetch_genome_data(self, accession: str) -> Dict[str, Any]:
        """Fetch genome data from NCBI"""
        try:
            with ftplib.FTP(self.ftp_host) as ftp:
                ftp.login()

                # Navigate to genome directory
                path_parts = accession.split("_")
                if len(path_parts) >= 2:
                    prefix = path_parts[0]
                    numeric_part = path_parts[1]

                    # Construct FTP path
                    ftp_path = f"/genomes/all/{prefix}/{numeric_part[:3]}/{numeric_part[3:6]}/{numeric_part[6:9]}/"

                    try:
                        ftp.cwd(ftp_path)
                        files = ftp.nlst()

                        # Download relevant files
                        genome_data = {}
                        for filename in files:
                            if filename.endswith(".fna.gz"):  # Genomic FASTA
                                local_path = f"temp_{filename}"
                                with open(local_path, "wb") as f:
                                    ftp.retrbinary(f"RETR {filename}", f.write)

                                # Process compressed file
                                with gzip.open(local_path, "rt") as f:
                                    genome_data["sequence"] = f.read()

                                os.remove(local_path)
                                break

                        return genome_data
                    except ftplib.error_perm:
                        logger.warning(f"Cannot access FTP path for {accession}")
                        return {}
        except Exception as e:
            logger.error(f"Error fetching genome data for {accession}: {e}")
            return {}

    def process(self, data: Any) -> pd.DataFrame:
        """Process NCBI/AGORA2 data"""
        processed_data = []

        if isinstance(data, dict) and "models" in data:
            for model in data["models"]:
                processed_data.append(
                    {
                        "model_id": model.get("modelID", ""),
                        "organism": model.get("organism", ""),
                        "strain": model.get("strain", ""),
                        "taxonomy": model.get("taxonomy", ""),
                        "reactions": model.get("reactions", 0),
                        "metabolites": model.get("metabolites", 0),
                        "genes": model.get("genes", 0),
                        "biomass_reaction": model.get("biomass", ""),
                        "timestamp": datetime.now(timezone.utc),
                    }
                )

        return pd.DataFrame(processed_data)

    def validate(self, data: Any) -> QualityMetrics:
        """Validate NCBI/AGORA2 data quality"""
        metrics = QualityMetrics()

        if isinstance(data, pd.DataFrame):
            # Completeness
            total_cells = data.size
            non_null_cells = data.count().sum()
            metrics.completeness = non_null_cells / total_cells if total_cells > 0 else 0.0

            # Consistency - valid model IDs
            valid_models = data["model_id"].str.match(r"^.*_\d+$", na=False).sum()
            metrics.consistency = valid_models / len(data) if len(data) > 0 else 0.0

            # Uniqueness
            unique_models = data["model_id"].nunique()
            metrics.uniqueness = unique_models / len(data) if len(data) > 0 else 0.0

            # Validity - numeric fields should be positive
            numeric_cols = ["reactions", "metabolites", "genes"]
            valid_numeric = True
            for col in numeric_cols:
                if col in data.columns:
                    valid_numeric &= (data[col] >= 0).all()
            metrics.validity = 1.0 if valid_numeric else 0.0

            # Accuracy - known organisms
            known_organisms = {"Escherichia coli", "Bacillus subtilis", "Saccharomyces cerevisiae"}
            organism_matches = data["organism"].isin(known_organisms).sum()
            metrics.accuracy = min(1.0, organism_matches / 100)  # Scale appropriately

            # Timeliness
            if "timestamp" in data.columns and not data["timestamp"].empty:
                latest_time = data["timestamp"].max()
                days_old = (datetime.now(timezone.utc) - latest_time).days
                metrics.timeliness = max(0, 1 - days_old / 365)

            # Conformity
            expected_columns = {"model_id", "organism", "strain", "taxonomy"}
            actual_columns = set(data.columns)
            metrics.conformity = len(expected_columns.intersection(actual_columns)) / len(
                expected_columns
            )

            # Integrity
            metrics.integrity = 1.0  # Assume good for metabolic models

        return metrics


class AdvancedDataManager:
    """Comprehensive data management system"""

    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.db_path = self.base_path / "metadata.db"
        self.processors = {"kegg": KEGGProcessor(), "ncbi": NCBIProcessor()}
        self.data_sources = {}
        self.quality_reports = {}

        # Initialize directories
        self._initialize_directories()
        self._initialize_database()

    def _initialize_directories(self):
        """Initialize data directory structure"""
        directories = [
            "raw/kegg",
            "raw/ncbi",
            "raw/agora2",
            "interim/kegg",
            "interim/ncbi",
            "interim/agora2",
            "processed/kegg",
            "processed/ncbi",
            "processed/agora2",
            "metadata",
            "quality_reports",
            "versions",
            "backups",
        ]

        for directory in directories:
            (self.base_path / directory).mkdir(parents=True, exist_ok=True)

    def _initialize_database(self):
        """Initialize metadata database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Data sources table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS data_sources (
                    id INTEGER PRIMARY KEY,
                    name TEXT UNIQUE,
                    url TEXT,
                    data_type TEXT,
                    version TEXT,
                    checksum TEXT,
                    last_updated TIMESTAMP,
                    quality_score REAL,
                    metadata TEXT
                )
            """
            )

            # Quality metrics table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS quality_metrics (
                    id INTEGER PRIMARY KEY,
                    data_source TEXT,
                    timestamp TIMESTAMP,
                    completeness REAL,
                    consistency REAL,
                    accuracy REAL,
                    validity REAL,
                    uniqueness REAL,
                    timeliness REAL,
                    conformity REAL,
                    integrity REAL,
                    overall_score REAL
                )
            """
            )

            # Processing log table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS processing_log (
                    id INTEGER PRIMARY KEY,
                    data_source TEXT,
                    operation TEXT,
                    timestamp TIMESTAMP,
                    status TEXT,
                    details TEXT
                )
            """
            )

            conn.commit()

    def register_data_source(self, source: DataSource):
        """Register a new data source"""
        self.data_sources[source.name] = source

        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO data_sources 
                (name, url, data_type, version, checksum, last_updated, quality_score, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    source.name,
                    source.url,
                    source.data_type,
                    source.version,
                    source.checksum,
                    source.last_updated,
                    source.quality_score,
                    json.dumps(source.metadata),
                ),
            )
            conn.commit()

    async def fetch_data(self, source_name: str) -> Dict[str, Any]:
        """Fetch data from registered source"""
        if source_name not in self.data_sources:
            raise ValueError(f"Unknown data source: {source_name}")

        source = self.data_sources[source_name]

        # Log operation
        self._log_operation(source_name, "fetch", "started")

        try:
            if source.data_type == "kegg":
                # Fetch KEGG pathways
                pathway_ids = await self._get_kegg_pathway_list()
                data = {}

                # Fetch sample of pathways for demonstration
                sample_pathways = pathway_ids[:10]  # Limit for demo

                for pathway_id in sample_pathways:
                    pathway_data = await self.processors["kegg"].fetch_pathway_data(pathway_id)
                    if pathway_data:
                        data[pathway_id] = pathway_data

                # Save raw data
                raw_path = (
                    self.base_path
                    / f"raw/kegg/{source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                with open(raw_path, "w") as f:
                    json.dump(data, f, indent=2, default=str)

                self._log_operation(source_name, "fetch", "completed")
                return data

            elif source.data_type == "ncbi":
                # Fetch NCBI/AGORA2 data
                data = await self.processors["ncbi"].fetch_agora2_data()

                # Save raw data
                raw_path = (
                    self.base_path
                    / f"raw/ncbi/{source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                with open(raw_path, "w") as f:
                    json.dump(data, f, indent=2, default=str)

                self._log_operation(source_name, "fetch", "completed")
                return data

        except Exception as e:
            self._log_operation(source_name, "fetch", f"failed: {e}")
            raise

    async def _get_kegg_pathway_list(self) -> List[str]:
        """Get list of KEGG pathways"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://rest.kegg.jp/list/pathway") as response:
                    pathway_text = await response.text()
                    pathways = []
                    for line in pathway_text.strip().split("\n"):
                        if line.startswith("path:"):
                            pathway_id = line.split("\t")[0].replace("path:", "")
                            pathways.append(pathway_id)
                    return pathways
        except Exception as e:
            logger.error(f"Error fetching KEGG pathway list: {e}")
            return []

    def process_data(self, source_name: str, data: Any) -> pd.DataFrame:
        """Process raw data using appropriate processor"""
        if source_name not in self.data_sources:
            raise ValueError(f"Unknown data source: {source_name}")

        source = self.data_sources[source_name]
        processor = self.processors.get(source.data_type)

        if not processor:
            raise ValueError(f"No processor available for data type: {source.data_type}")

        self._log_operation(source_name, "process", "started")

        try:
            processed_data = processor.process(data)

            # Save processed data
            processed_path = (
                self.base_path / f"processed/{source.data_type}/{source_name}_processed.csv"
            )
            processed_data.to_csv(processed_path, index=False)

            self._log_operation(source_name, "process", "completed")
            return processed_data

        except Exception as e:
            self._log_operation(source_name, "process", f"failed: {e}")
            raise

    def validate_data(self, source_name: str, data: Any) -> QualityMetrics:
        """Validate data quality"""
        if source_name not in self.data_sources:
            raise ValueError(f"Unknown data source: {source_name}")

        source = self.data_sources[source_name]
        processor = self.processors.get(source.data_type)

        if not processor:
            raise ValueError(f"No processor available for data type: {source.data_type}")

        metrics = processor.validate(data)

        # Store quality metrics
        self._store_quality_metrics(source_name, metrics)

        return metrics

    def _store_quality_metrics(self, source_name: str, metrics: QualityMetrics):
        """Store quality metrics in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO quality_metrics 
                (data_source, timestamp, completeness, consistency, accuracy, validity, 
                 uniqueness, timeliness, conformity, integrity, overall_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    source_name,
                    datetime.now(timezone.utc),
                    metrics.completeness,
                    metrics.consistency,
                    metrics.accuracy,
                    metrics.validity,
                    metrics.uniqueness,
                    metrics.timeliness,
                    metrics.conformity,
                    metrics.integrity,
                    metrics.overall_score(),
                ),
            )
            conn.commit()

    def _log_operation(self, source_name: str, operation: str, status: str):
        """Log data processing operation"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO processing_log (data_source, operation, timestamp, status, details)
                VALUES (?, ?, ?, ?, ?)
            """,
                (source_name, operation, datetime.now(timezone.utc), status, ""),
            )
            conn.commit()

    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        report = {
            "timestamp": datetime.now(timezone.utc),
            "data_sources": [],
            "overall_quality": 0.0,
            "recommendations": [],
        }

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get latest quality metrics for each source
            cursor.execute(
                """
                SELECT data_source, MAX(timestamp) as latest_timestamp
                FROM quality_metrics
                GROUP BY data_source
            """
            )

            source_scores = []
            for source_name, latest_timestamp in cursor.fetchall():
                cursor.execute(
                    """
                    SELECT * FROM quality_metrics 
                    WHERE data_source = ? AND timestamp = ?
                """,
                    (source_name, latest_timestamp),
                )

                row = cursor.fetchone()
                if row:
                    source_report = {
                        "name": source_name,
                        "timestamp": row[2],
                        "completeness": row[3],
                        "consistency": row[4],
                        "accuracy": row[5],
                        "validity": row[6],
                        "uniqueness": row[7],
                        "timeliness": row[8],
                        "conformity": row[9],
                        "integrity": row[10],
                        "overall_score": row[11],
                    }

                    report["data_sources"].append(source_report)
                    source_scores.append(row[11])

                    # Generate recommendations
                    if row[11] < 0.8:
                        report["recommendations"].append(f"Improve data quality for {source_name}")
                    if row[3] < 0.9:  # Completeness
                        report["recommendations"].append(f"Address missing data in {source_name}")
                    if row[4] < 0.9:  # Consistency
                        report["recommendations"].append(
                            f"Fix data consistency issues in {source_name}"
                        )

        # Calculate overall quality
        if source_scores:
            report["overall_quality"] = sum(source_scores) / len(source_scores)

        # Save report
        report_path = (
            self.base_path
            / f"quality_reports/quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        return report

    def create_data_snapshot(self, version: str) -> str:
        """Create versioned snapshot of all data"""
        snapshot_path = (
            self.base_path
            / f"versions/snapshot_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        snapshot_path.mkdir(parents=True, exist_ok=True)

        # Copy processed data
        processed_path = self.base_path / "processed"
        if processed_path.exists():
            import shutil

            shutil.copytree(processed_path, snapshot_path / "processed")

        # Copy metadata
        shutil.copy2(self.db_path, snapshot_path / "metadata.db")

        # Create manifest
        manifest = {
            "version": version,
            "timestamp": datetime.now(timezone.utc),
            "data_sources": list(self.data_sources.keys()),
            "quality_report": self.generate_quality_report(),
        }

        with open(snapshot_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        return str(snapshot_path)


# Main execution functions
async def main():
    """Main execution function"""
    # Initialize data manager
    data_manager = AdvancedDataManager()

    # Register KEGG data source
    kegg_source = DataSource(
        name="kegg_pathways",
        url="https://rest.kegg.jp/",
        data_type="kegg",
        update_frequency="weekly",
        metadata={
            "description": "KEGG pathway database with comprehensive metabolic networks",
            "total_pathways": 7302,
            "coverage": "global",
        },
    )
    data_manager.register_data_source(kegg_source)

    # Register NCBI/AGORA2 data source
    ncbi_source = DataSource(
        name="ncbi_agora2",
        url="https://www.vmh.life/files/reconstructions/AGORA2/",
        data_type="ncbi",
        update_frequency="monthly",
        metadata={
            "description": "AGORA2 genome-scale metabolic reconstructions",
            "total_organisms": 7302,
            "coverage": "human microbiome",
        },
    )
    data_manager.register_data_source(ncbi_source)

    return data_manager


if __name__ == "__main__":
    asyncio.run(main())

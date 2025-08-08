#!/usr/bin/env python3
"""
Planet-Run Primary Key System
============================

Unified metadata database system that serves as the backbone for connecting
150TB+ of multi-disciplinary scientific data under consistent planet-run identifiers.

This system unifies:
- Biology (KEGG pathways, genomics, metabolism)
- Astronomy (exoplanets, stellar parameters, observations)
- Climate (ROCKE-3D simulations, atmospheric data)
- Spectroscopy (PSG spectra, JWST observations)
- Physics (radiation, magnetic fields)

Every dataset hangs off the same planet_id/run_id structure for seamless
multi-modal training and inference.

Schema:
planet_id (e.g. TOI-700 e)
   ├── run_id 0001  (orbital + stellar param set)
   │      ├─ climate_cube.zarr     (4-D GCM fields)
   │      ├─ biosphere_graph.npz   (KEGG-derived env-specific pathway)
   │      ├─ spectrum_highres.h5   (PSG R=100,000)
   │      ├─ jwst_obs.fits         (optional real data)
   │      └─ meta.json             (pointers, hashes, timestamps)
   └── run_id 0002 ...
"""

import asyncio
import hashlib
import json
import logging
import sqlite3
import threading
import uuid
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Union

import aiofiles
import numpy as np
import pandas as pd
import yaml
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

Base = declarative_base()


class DataDomain(Enum):
    """Scientific data domains unified under planet-runs"""

    BIOLOGY = "biology"  # KEGG pathways, genomics, metabolism
    ASTRONOMY = "astronomy"  # Stellar parameters, orbital mechanics
    CLIMATE = "climate"  # GCM simulations, atmospheric dynamics
    SPECTROSCOPY = "spectroscopy"  # High-res spectra, observations
    PHYSICS = "physics"  # Radiation, magnetic fields
    OBSERVATIONS = "observations"  # Real telescope data (JWST, etc.)


class RunStatus(Enum):
    """Status of planet-run simulations"""

    PENDING = "pending"  # Queued for generation
    GENERATING = "generating"  # Currently being simulated
    COMPLETE = "complete"  # All data generated and validated
    PARTIAL = "partial"  # Some data types missing
    FAILED = "failed"  # Generation failed
    ARCHIVED = "archived"  # Moved to long-term storage


class DataFormat(Enum):
    """Supported data storage formats"""

    ZARR = "zarr"  # Climate 4D datacubes
    NPZ = "npz"  # Biological networks, graphs
    HDF5 = "hdf5"  # High-resolution spectra
    FITS = "fits"  # Astronomical observations
    JSON = "json"  # Metadata and parameters
    PARQUET = "parquet"  # Tabular data
    NETCDF = "netcdf"  # Legacy climate data


# SQLAlchemy Models
class PlanetRun(Base):
    """Core planet-run table - primary key for all scientific data"""

    __tablename__ = "planet_runs"

    # Primary identification
    run_id = Column(Integer, primary_key=True, autoincrement=True)
    planet_id = Column(String(100), nullable=False, index=True)
    kepler_id = Column(String(50), index=True)  # NASA Kepler ID if available
    toi_id = Column(String(50), index=True)  # TESS TOI ID if available

    # Core planetary and stellar parameters (JSON for flexibility)
    planet_params = Column(JSON, nullable=False)
    stellar_params = Column(JSON, nullable=False)
    orbital_params = Column(JSON, nullable=False)

    # Data file paths (relative to storage root)
    climate_cube_path = Column(String(500))  # zarr store path
    biosphere_graph_path = Column(String(500))  # NPZ file path
    spectrum_path = Column(String(500))  # HDF5 file path
    observation_path = Column(String(500))  # FITS file path (if available)
    metadata_path = Column(String(500))  # JSON metadata file

    # Data status and quality
    run_status = Column(String(20), default=RunStatus.PENDING.value, index=True)
    data_completeness = Column(Float, default=0.0)  # Fraction of expected data present
    quality_score = Column(Float, default=0.0)  # Overall data quality [0-1]

    # Checksums for data integrity
    climate_checksum = Column(String(64))  # SHA256 of climate data
    biosphere_checksum = Column(String(64))  # SHA256 of biosphere data
    spectrum_checksum = Column(String(64))  # SHA256 of spectrum data

    # Temporal metadata
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, index=True)

    # Machine learning splits
    ml_split = Column(String(10), default="train", index=True)  # train, val, test

    # Relationships
    data_files = relationship("DataFile", back_populates="planet_run")
    quality_reports = relationship("QualityReport", back_populates="planet_run")

    # Indexes for performance
    __table_args__ = (
        Index("idx_planet_status", "planet_id", "run_status"),
        Index("idx_completeness", "data_completeness"),
        Index("idx_quality", "quality_score"),
        Index("idx_created", "created_at"),
        Index("idx_ml_split", "ml_split"),
    )


class DataFile(Base):
    """Individual data files associated with planet runs"""

    __tablename__ = "data_files"

    file_id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("planet_runs.run_id"), nullable=False, index=True)

    # File identification
    domain = Column(String(20), nullable=False, index=True)  # DataDomain
    data_type = Column(String(50), nullable=False)  # specific data type
    file_format = Column(String(10), nullable=False)  # DataFormat

    # File location and metadata
    file_path = Column(String(500), nullable=False)
    file_size_bytes = Column(Integer, default=0)
    checksum = Column(String(64), nullable=False)

    # Data characteristics
    dimensions = Column(JSON)  # Array shapes, variable info
    variables = Column(JSON)  # List of variables/features
    temporal_range = Column(JSON)  # Time coverage
    spatial_resolution = Column(JSON)  # Spatial grid info

    # Quality and processing
    processing_level = Column(String(20), default="raw")  # raw, interim, processed
    quality_flags = Column(JSON)  # Any quality issues
    needs_regeneration = Column(
        Boolean, default=False
    )  # Flag for corrupted files needing regeneration
    storage_tier = Column(String(10), default="hot")  # Storage tier (hot, warm, cold, frozen)

    # Temporal metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    last_accessed = Column(DateTime)

    # Relationships
    planet_run = relationship("PlanetRun", back_populates="data_files")

    __table_args__ = (
        Index("idx_domain_type", "domain", "data_type"),
        Index("idx_format", "file_format"),
        Index("idx_size", "file_size_bytes"),
    )


class QualityReport(Base):
    """Quality assessment reports for planet runs"""

    __tablename__ = "quality_reports"

    report_id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("planet_runs.run_id"), nullable=False, index=True)

    # Quality assessment
    domain = Column(String(20), nullable=False)
    quality_score = Column(Float, nullable=False)  # 0-1 score
    issues = Column(JSON)  # List of identified issues
    metrics = Column(JSON)  # Detailed quality metrics

    # Assessment metadata
    assessment_method = Column(String(50))  # Method used for assessment
    assessed_at = Column(DateTime, default=datetime.utcnow)
    assessed_by = Column(String(100))  # System/user that performed assessment

    # Relationships
    planet_run = relationship("PlanetRun", back_populates="quality_reports")


@dataclass
class PlanetParameters:
    """Standardized planet parameter structure"""

    # Basic properties
    radius_earth: float  # Planet radius in Earth radii
    mass_earth: float  # Planet mass in Earth masses

    # Orbital parameters
    orbital_period_days: float  # Orbital period in days
    semimajor_axis_au: float  # Semi-major axis in AU
    eccentricity: float  # Orbital eccentricity
    stellar_flux_earth: float  # Stellar flux in Earth units

    # Atmospheric composition (mixing ratios)
    atmosphere: Dict[str, float] = field(default_factory=dict)

    # Optional observational constraints
    transit_depth: Optional[float] = None
    rv_amplitude: Optional[float] = None


@dataclass
class StellarParameters:
    """Standardized stellar parameter structure"""

    # Basic stellar properties
    mass_solar: float  # Stellar mass in solar masses
    radius_solar: float  # Stellar radius in solar radii
    effective_temp_k: float  # Effective temperature in Kelvin
    metallicity: float  # [Fe/H] metallicity
    log_g: float  # Surface gravity

    # Activity and age
    activity_level: float = 0.0  # Stellar activity proxy
    age_gyr: Optional[float] = None  # Age in billion years


class PlanetRunManager:
    """
    Central manager for planet-run primary key system

    Provides unified interface for:
    - Creating and managing planet runs
    - Organizing multi-domain data
    - Tracking data lineage and quality
    - Facilitating ML data loading
    """

    def __init__(self, database_path: str = "data/metadata/planet_runs.db"):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        # Create database engine
        self.engine = create_engine(f"sqlite:///{self.database_path}", echo=False)
        Base.metadata.create_all(self.engine)

        # Session factory
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        # Thread safety
        self._lock = Lock()

        # Storage root paths
        self.storage_root = Path("data/planet_runs")
        self.storage_root.mkdir(parents=True, exist_ok=True)

        logger.info(f"🛢️ Planet-Run Manager initialized: {self.database_path}")

    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_planet_run(
        self,
        planet_id: str,
        planet_params: PlanetParameters,
        stellar_params: StellarParameters,
        kepler_id: Optional[str] = None,
        toi_id: Optional[str] = None,
    ) -> int:
        """
        Create new planet run with unified parameters

        Returns:
            run_id: Unique run identifier
        """
        with self._lock:
            with self.get_session() as session:
                # Create orbital parameters from planet parameters
                orbital_params = {
                    "period_days": planet_params.orbital_period_days,
                    "semimajor_axis_au": planet_params.semimajor_axis_au,
                    "eccentricity": planet_params.eccentricity,
                    "stellar_flux_earth": planet_params.stellar_flux_earth,
                }

                # Create planet run
                planet_run = PlanetRun(
                    planet_id=planet_id,
                    kepler_id=kepler_id,
                    toi_id=toi_id,
                    planet_params=asdict(planet_params),
                    stellar_params=asdict(stellar_params),
                    orbital_params=orbital_params,
                    run_status=RunStatus.PENDING.value,
                )

                session.add(planet_run)
                session.flush()  # Get the run_id

                run_id = planet_run.run_id

                # Create directory structure
                self._create_run_directory(run_id, planet_id)

                logger.info(f"[OK] Created planet run {run_id} for {planet_id}")
                return run_id

    def _create_run_directory(self, run_id: int, planet_id: str):
        """Create standardized directory structure for planet run"""
        run_dir = self.storage_root / f"planet_{planet_id.replace(' ', '_')}" / f"run_{run_id:06d}"

        # Create subdirectories for each data domain
        directories = [
            "climate",  # ZARR climate datacubes
            "biosphere",  # NPZ biological networks
            "spectroscopy",  # HDF5 high-res spectra
            "observations",  # FITS telescope data
            "metadata",  # JSON parameter files
            "quality",  # Quality assessment reports
            "derived",  # Processed/derived data products
        ]

        for subdir in directories:
            (run_dir / subdir).mkdir(parents=True, exist_ok=True)

        logger.debug(f"📁 Created directory structure: {run_dir}")

    def register_data_file(
        self,
        run_id: int,
        domain: DataDomain,
        data_type: str,
        file_path: Path,
        file_format: DataFormat,
        dimensions: Optional[Dict] = None,
        variables: Optional[List[str]] = None,
    ) -> int:
        """Register data file for planet run"""
        with self.get_session() as session:
            # Calculate file checksum
            checksum = self._calculate_checksum(file_path)

            data_file = DataFile(
                run_id=run_id,
                domain=domain.value,
                data_type=data_type,
                file_format=file_format.value,
                file_path=str(file_path.relative_to(self.storage_root)),
                file_size_bytes=file_path.stat().st_size if file_path.exists() else 0,
                checksum=checksum,
                dimensions=dimensions or {},
                variables=variables or [],
                processing_level="raw",
            )

            session.add(data_file)
            session.flush()

            file_id = data_file.file_id

            # Update planet run paths based on domain
            planet_run = session.query(PlanetRun).filter_by(run_id=run_id).first()
            if planet_run:
                if domain == DataDomain.CLIMATE:
                    planet_run.climate_cube_path = data_file.file_path
                    planet_run.climate_checksum = checksum
                elif domain == DataDomain.BIOLOGY:
                    planet_run.biosphere_graph_path = data_file.file_path
                    planet_run.biosphere_checksum = checksum
                elif domain == DataDomain.SPECTROSCOPY:
                    planet_run.spectrum_path = data_file.file_path
                    planet_run.spectrum_checksum = checksum
                elif domain == DataDomain.OBSERVATIONS:
                    planet_run.observation_path = data_file.file_path

                # Update completeness
                self._update_completeness(session, run_id)

            logger.info(f"[DOC] Registered {domain.value} file for run {run_id}: {file_path.name}")
            return file_id

    def _update_completeness(self, session, run_id: int):
        """Update data completeness for planet run"""
        planet_run = session.query(PlanetRun).filter_by(run_id=run_id).first()
        if not planet_run:
            return

        # Check which data types are present
        expected_domains = [DataDomain.CLIMATE, DataDomain.BIOLOGY, DataDomain.SPECTROSCOPY]
        present_count = 0

        if planet_run.climate_cube_path:
            present_count += 1
        if planet_run.biosphere_graph_path:
            present_count += 1
        if planet_run.spectrum_path:
            present_count += 1

        completeness = present_count / len(expected_domains)
        planet_run.data_completeness = completeness

        # Update status based on completeness
        if completeness == 1.0:
            planet_run.run_status = RunStatus.COMPLETE.value
            planet_run.completed_at = datetime.utcnow()
        elif completeness > 0:
            planet_run.run_status = RunStatus.PARTIAL.value

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum for file"""
        if not file_path.exists():
            return ""

        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def get_planet_runs(
        self,
        ml_split: Optional[str] = None,
        min_completeness: float = 0.0,
        min_quality: float = 0.0,
        status: Optional[RunStatus] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query planet runs with filtering

        Args:
            ml_split: Filter by ML split (train/val/test)
            min_completeness: Minimum data completeness
            min_quality: Minimum quality score
            status: Filter by run status

        Returns:
            List of planet run dictionaries
        """
        with self.get_session() as session:
            query = session.query(PlanetRun)

            if ml_split:
                query = query.filter(PlanetRun.ml_split == ml_split)
            if min_completeness > 0:
                query = query.filter(PlanetRun.data_completeness >= min_completeness)
            if min_quality > 0:
                query = query.filter(PlanetRun.quality_score >= min_quality)
            if status:
                query = query.filter(PlanetRun.run_status == status.value)

            runs = query.order_by(PlanetRun.created_at).all()

            return [self._planet_run_to_dict(run) for run in runs]

    def _planet_run_to_dict(self, planet_run: PlanetRun) -> Dict[str, Any]:
        """Convert PlanetRun SQLAlchemy object to dictionary"""
        return {
            "run_id": planet_run.run_id,
            "planet_id": planet_run.planet_id,
            "kepler_id": planet_run.kepler_id,
            "toi_id": planet_run.toi_id,
            "planet_params": planet_run.planet_params,
            "stellar_params": planet_run.stellar_params,
            "orbital_params": planet_run.orbital_params,
            "data_paths": {
                "climate": planet_run.climate_cube_path,
                "biosphere": planet_run.biosphere_graph_path,
                "spectrum": planet_run.spectrum_path,
                "observation": planet_run.observation_path,
                "metadata": planet_run.metadata_path,
            },
            "status": planet_run.run_status,
            "completeness": planet_run.data_completeness,
            "quality_score": planet_run.quality_score,
            "ml_split": planet_run.ml_split,
            "created_at": planet_run.created_at,
            "completed_at": planet_run.completed_at,
        }

    def assign_ml_splits(
        self,
        train_fraction: float = 0.7,
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,
        stratify_by: Optional[str] = None,
    ):
        """
        Assign ML splits to planet runs

        Args:
            train_fraction: Fraction for training
            val_fraction: Fraction for validation
            test_fraction: Fraction for testing
            stratify_by: Optional stratification criterion
        """
        assert abs(train_fraction + val_fraction + test_fraction - 1.0) < 1e-6

        with self.get_session() as session:
            # Get all complete runs
            complete_runs = (
                session.query(PlanetRun)
                .filter(PlanetRun.run_status == RunStatus.COMPLETE.value)
                .all()
            )

            if not complete_runs:
                logger.warning("No complete runs found for ML split assignment")
                return

            # Shuffle for random assignment
            np.random.seed(42)  # Reproducible splits
            indices = np.random.permutation(len(complete_runs))

            n_train = int(len(complete_runs) * train_fraction)
            n_val = int(len(complete_runs) * val_fraction)

            # Assign splits
            for i, idx in enumerate(indices):
                run = complete_runs[idx]

                if i < n_train:
                    run.ml_split = "train"
                elif i < n_train + n_val:
                    run.ml_split = "val"
                else:
                    run.ml_split = "test"

            logger.info(
                f"[OK] Assigned ML splits: {n_train} train, {n_val} val, "
                f"{len(complete_runs) - n_train - n_val} test"
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about planet runs"""
        with self.get_session() as session:
            # Basic counts
            total_runs = session.query(PlanetRun).count()
            complete_runs = (
                session.query(PlanetRun)
                .filter(PlanetRun.run_status == RunStatus.COMPLETE.value)
                .count()
            )

            # Data size statistics
            data_files = session.query(DataFile).all()
            total_size_gb = sum(df.file_size_bytes for df in data_files) / (1024**3)

            # Domain breakdown
            domain_counts = {}
            for domain in DataDomain:
                count = session.query(DataFile).filter(DataFile.domain == domain.value).count()
                domain_counts[domain.value] = count

            # Quality statistics
            avg_quality = (
                session.query(PlanetRun)
                .filter(PlanetRun.quality_score > 0)
                .with_entities(PlanetRun.quality_score)
                .all()
            )

            avg_quality_score = np.mean([q[0] for q in avg_quality]) if avg_quality else 0.0

            return {
                "total_runs": total_runs,
                "complete_runs": complete_runs,
                "completion_rate": complete_runs / max(1, total_runs),
                "total_data_size_gb": total_size_gb,
                "domain_file_counts": domain_counts,
                "average_quality_score": avg_quality_score,
                "database_path": str(self.database_path),
                "storage_root": str(self.storage_root),
            }

    def export_run_metadata(self, run_id: int) -> Dict[str, Any]:
        """Export complete metadata for a planet run"""
        with self.get_session() as session:
            planet_run = session.query(PlanetRun).filter_by(run_id=run_id).first()
            if not planet_run:
                raise ValueError(f"Planet run {run_id} not found")

            # Get associated data files
            data_files = session.query(DataFile).filter_by(run_id=run_id).all()

            # Get quality reports
            quality_reports = session.query(QualityReport).filter_by(run_id=run_id).all()

            return {
                "planet_run": self._planet_run_to_dict(planet_run),
                "data_files": [
                    {
                        "file_id": df.file_id,
                        "domain": df.domain,
                        "data_type": df.data_type,
                        "file_format": df.file_format,
                        "file_path": df.file_path,
                        "file_size_mb": df.file_size_bytes / (1024**2),
                        "checksum": df.checksum,
                        "dimensions": df.dimensions,
                        "variables": df.variables,
                        "processing_level": df.processing_level,
                    }
                    for df in data_files
                ],
                "quality_reports": [
                    {
                        "domain": qr.domain,
                        "quality_score": qr.quality_score,
                        "issues": qr.issues,
                        "metrics": qr.metrics,
                        "assessed_at": qr.assessed_at,
                    }
                    for qr in quality_reports
                ],
            }


# Convenience functions
def get_planet_run_manager(database_path: str = None) -> PlanetRunManager:
    """Get singleton planet run manager"""
    if database_path is None:
        database_path = "data/metadata/planet_runs.db"

    return PlanetRunManager(database_path)


def create_example_planet_runs(manager: PlanetRunManager, n_runs: int = 100):
    """Create example planet runs for testing"""
    logger.info(f"[TEST] Creating {n_runs} example planet runs...")

    # Well-known exoplanets for realistic examples
    example_planets = [
        ("TRAPPIST-1 e", "K2-269"),
        ("Proxima Centauri b", "Proxima Cen"),
        ("TOI-715 b", "TOI-715"),
        ("K2-18 b", "K2-18"),
        ("WASP-121 b", "WASP-121"),
        ("HD 209458 b", "HD 209458"),
        ("GJ 1214 b", "GJ 1214"),
        ("55 Cancri e", "55 Cnc"),
    ]

    run_ids = []

    for i in range(n_runs):
        # Select base planet
        base_planet, star_name = example_planets[i % len(example_planets)]
        planet_id = f"{base_planet}_var_{i:03d}"

        # Generate realistic parameters with variation
        planet_params = PlanetParameters(
            radius_earth=np.random.uniform(0.8, 2.5),
            mass_earth=np.random.uniform(0.5, 10.0),
            orbital_period_days=np.random.uniform(1.0, 100.0),
            semimajor_axis_au=np.random.uniform(0.01, 2.0),
            eccentricity=np.random.uniform(0.0, 0.3),
            stellar_flux_earth=np.random.uniform(0.1, 10.0),
            atmosphere={
                "H2O": np.random.uniform(0.0, 0.01),
                "CO2": np.random.uniform(0.0, 0.001),
                "CH4": np.random.uniform(0.0, 0.0001),
                "O2": np.random.uniform(0.0, 0.0001),
            },
        )

        stellar_params = StellarParameters(
            mass_solar=np.random.uniform(0.1, 1.5),
            radius_solar=np.random.uniform(0.1, 2.0),
            effective_temp_k=np.random.uniform(2500, 6500),
            metallicity=np.random.uniform(-1.0, 0.5),
            log_g=np.random.uniform(3.5, 5.0),
            activity_level=np.random.uniform(0.0, 1.0),
        )

        # Create planet run
        run_id = manager.create_planet_run(
            planet_id=planet_id,
            planet_params=planet_params,
            stellar_params=stellar_params,
            kepler_id=f"K2-{1000 + i}" if i % 3 == 0 else None,
            toi_id=f"TOI-{2000 + i}" if i % 5 == 0 else None,
        )

        run_ids.append(run_id)

        if (i + 1) % 20 == 0:
            logger.info(f"  Created {i + 1}/{n_runs} planet runs...")

    logger.info(f"[OK] Created {len(run_ids)} example planet runs")
    return run_ids


if __name__ == "__main__":
    # Example usage and testing
    manager = get_planet_run_manager()

    # Create example runs
    run_ids = create_example_planet_runs(manager, 50)

    # Assign ML splits
    manager.assign_ml_splits()

    # Show statistics
    stats = manager.get_statistics()
    print("\n" + "=" * 60)
    print("🛢️ PLANET-RUN PRIMARY KEY SYSTEM - STATISTICS")
    print("=" * 60)
    print(f"Total Planet Runs: {stats['total_runs']}")
    print(f"Complete Runs: {stats['complete_runs']}")
    print(f"Completion Rate: {stats['completion_rate']:.1%}")
    print(f"Total Data Size: {stats['total_data_size_gb']:.2f} GB")
    print(f"Average Quality: {stats['average_quality_score']:.3f}")
    print("\nDomain File Counts:")
    for domain, count in stats["domain_file_counts"].items():
        print(f"  {domain.title()}: {count}")
    print("=" * 60)

    # Example query
    training_runs = manager.get_planet_runs(ml_split="train", min_completeness=0.0)
    print(f"\n[TARGET] Found {len(training_runs)} training planet runs")

    if training_runs:
        example_run = training_runs[0]
        print(f"\nExample training run: {example_run['planet_id']}")
        print(f"  Run ID: {example_run['run_id']}")
        print(f"  Completeness: {example_run['completeness']:.1%}")
        print(f"  Status: {example_run['status']}")
        print(f"  Planet radius: {example_run['planet_params']['radius_earth']:.2f} R⊕")
        print(f"  Stellar flux: {example_run['orbital_params']['stellar_flux_earth']:.2f} S⊕")

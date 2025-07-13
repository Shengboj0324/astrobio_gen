#!/usr/bin/env python3
"""
Scientific Data Metadata Database System
========================================

Comprehensive metadata management for hundreds of TB of multi-domain scientific data:
- Astronomical data (stars, planets, observations)
- Exoplanet data (parameters, atmospheres, orbits)
- Environmental data (climate, atmospheric composition)
- Physics data (radiation, magnetic fields, etc.)
- Optical data (spectra, photometry)
- Space exploration data (missions, instruments, observations)
- Physiological data (biosignatures, metabolomics)

Designed for:
- Hundreds of terabytes of data
- Multi-domain scientific datasets
- Efficient querying and filtering
- Tiered storage management
- Experiment lineage tracking
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, Text, ForeignKey, Index, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.types import JSON
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class DataDomain(Enum):
    """Scientific data domains"""
    ASTRONOMICAL = "astronomical"
    EXOPLANET = "exoplanet"
    ENVIRONMENTAL = "environmental"
    PHYSICS = "physics"
    OPTICAL = "optical"
    SPACE_EXPLORATION = "space_exploration"
    PHYSIOLOGICAL = "physiological"
    BIOSIGNATURE = "biosignature"
    METABOLOMICS = "metabolomics"

class StorageTier(Enum):
    """Storage tiers for data management"""
    LOCAL_SSD = "local_ssd"
    EXTERNAL_SSD = "external_ssd"
    NAS = "nas"
    CLOUD_HOT = "cloud_hot"
    CLOUD_COLD = "cloud_cold"
    ARCHIVE = "archive"

class DataStatus(Enum):
    """Data processing status"""
    RAW = "raw"
    PROCESSING = "processing"
    PROCESSED = "processed"
    VALIDATED = "validated"
    ARCHIVED = "archived"
    ERROR = "error"

# Database Models
class Dataset(Base):
    """Main dataset table"""
    __tablename__ = 'datasets'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(200), unique=True, nullable=False)
    domain = Column(String(50), nullable=False)  # DataDomain enum
    version = Column(String(20), nullable=False)
    description = Column(Text)
    
    # Data characteristics
    size_gb = Column(Float, nullable=False)
    num_samples = Column(Integer)
    num_features = Column(Integer)
    dimensions = Column(JSON)  # Store array shapes as JSON
    
    # Storage information
    storage_tier = Column(String(50), nullable=False)  # StorageTier enum
    storage_path = Column(String(500), nullable=False)
    zarr_path = Column(String(500))
    backup_path = Column(String(500))
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status = Column(String(50), default=DataStatus.RAW.value)
    checksum = Column(String(64))  # SHA256 hash
    
    # Scientific metadata
    source = Column(String(200))  # Data source (NASA, ESA, etc.)
    instrument = Column(String(100))  # Observing instrument
    mission = Column(String(100))  # Space mission
    observation_date = Column(DateTime)
    
    # Relationships
    experiments = relationship("Experiment", back_populates="dataset")
    data_chunks = relationship("DataChunk", back_populates="dataset")
    
    # Indexes
    __table_args__ = (
        Index('idx_dataset_domain', 'domain'),
        Index('idx_dataset_status', 'status'),
        Index('idx_dataset_storage_tier', 'storage_tier'),
        Index('idx_dataset_size', 'size_gb'),
    )

class Experiment(Base):
    """Experiment/model run table"""
    __tablename__ = 'experiments'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)
    
    # Experiment parameters
    model_type = Column(String(100))  # CubeUNet, SurrogateTransformer, etc.
    hyperparameters = Column(JSON)
    
    # Execution info
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    status = Column(String(50), default='running')
    
    # Results
    accuracy = Column(Float)
    loss = Column(Float)
    r2_score = Column(Float)
    rmse = Column(Float)
    
    # System resources
    gpu_hours = Column(Float)
    memory_gb = Column(Float)
    compute_cost = Column(Float)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="experiments")
    predictions = relationship("Prediction", back_populates="experiment")
    
    # Indexes
    __table_args__ = (
        Index('idx_experiment_dataset', 'dataset_id'),
        Index('idx_experiment_status', 'status'),
        Index('idx_experiment_accuracy', 'accuracy'),
    )

class DataChunk(Base):
    """Data chunk table for large datasets"""
    __tablename__ = 'data_chunks'
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)
    chunk_id = Column(String(100), nullable=False)  # zarr chunk identifier
    
    # Chunk characteristics
    start_indices = Column(JSON)  # Start indices for each dimension
    end_indices = Column(JSON)    # End indices for each dimension
    shape = Column(JSON)          # Chunk shape
    size_mb = Column(Float)       # Chunk size in MB
    
    # Storage info
    storage_tier = Column(String(50), nullable=False)
    file_path = Column(String(500), nullable=False)
    is_cached = Column(Boolean, default=False)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    access_count = Column(Integer, default=0)
    
    # Data statistics
    min_value = Column(Float)
    max_value = Column(Float)
    mean_value = Column(Float)
    std_value = Column(Float)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="data_chunks")
    
    # Indexes
    __table_args__ = (
        Index('idx_chunk_dataset', 'dataset_id'),
        Index('idx_chunk_storage_tier', 'storage_tier'),
        Index('idx_chunk_last_accessed', 'last_accessed'),
        Index('idx_chunk_cached', 'is_cached'),
    )

class Prediction(Base):
    """Model prediction results"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey('experiments.id'), nullable=False)
    
    # Input parameters
    input_parameters = Column(JSON)
    
    # Prediction results
    prediction_data = Column(JSON)  # Store small predictions directly
    prediction_path = Column(String(500))  # Path to large prediction files
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    inference_time_ms = Column(Float)
    confidence_score = Column(Float)
    
    # SHAP explanations
    shap_values = Column(JSON)  # Store SHAP explanation data
    feature_importance = Column(JSON)
    
    # Relationships
    experiment = relationship("Experiment", back_populates="predictions")
    
    # Indexes
    __table_args__ = (
        Index('idx_prediction_experiment', 'experiment_id'),
        Index('idx_prediction_created', 'created_at'),
        Index('idx_prediction_confidence', 'confidence_score'),
    )

class AstronomicalObject(Base):
    """Astronomical objects (stars, planets, etc.)"""
    __tablename__ = 'astronomical_objects'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    object_type = Column(String(50), nullable=False)  # star, planet, system
    
    # Coordinates
    ra = Column(Float)  # Right ascension
    dec = Column(Float)  # Declination
    distance_pc = Column(Float)  # Distance in parsecs
    
    # Physical properties
    mass = Column(Float)  # Mass in solar/Earth masses
    radius = Column(Float)  # Radius in solar/Earth radii
    temperature = Column(Float)  # Temperature in Kelvin
    
    # Stellar properties (for stars)
    spectral_type = Column(String(20))
    metallicity = Column(Float)
    age_gyr = Column(Float)
    
    # Planetary properties (for planets)
    orbital_period = Column(Float)  # Days
    semi_major_axis = Column(Float)  # AU
    eccentricity = Column(Float)
    inclination = Column(Float)
    
    # Atmospheric properties
    atmosphere_pressure = Column(Float)  # Bar
    atmosphere_composition = Column(JSON)
    
    # Relationships
    datasets = relationship("Dataset", secondary="object_dataset_link")
    
    # Indexes
    __table_args__ = (
        Index('idx_astro_object_type', 'object_type'),
        Index('idx_astro_coordinates', 'ra', 'dec'),
        Index('idx_astro_distance', 'distance_pc'),
    )

class ObjectDatasetLink(Base):
    """Link table between astronomical objects and datasets"""
    __tablename__ = 'object_dataset_link'
    
    id = Column(Integer, primary_key=True)
    object_id = Column(Integer, ForeignKey('astronomical_objects.id'))
    dataset_id = Column(Integer, ForeignKey('datasets.id'))

class ValidationResult(Base):
    """Validation results for models/datasets"""
    __tablename__ = 'validation_results'
    
    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey('experiments.id'))
    dataset_id = Column(Integer, ForeignKey('datasets.id'))
    
    # Validation metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    r2_score = Column(Float)
    rmse = Column(Float)
    mae = Column(Float)
    
    # Validation details
    validation_set_size = Column(Integer)
    cross_validation_folds = Column(Integer)
    validation_date = Column(DateTime, default=datetime.utcnow)
    
    # Physics validation
    energy_conservation_error = Column(Float)
    mass_conservation_error = Column(Float)
    momentum_conservation_error = Column(Float)
    
    # Validation notes
    notes = Column(Text)
    passed_validation = Column(Boolean, default=False)

class MetadataManager:
    """Manager for scientific metadata database"""
    
    def __init__(self, db_path: str = "data/metadata.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create SQLAlchemy engine
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        # Create session
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        logger.info(f"Initialized metadata database at {self.db_path}")
    
    def register_dataset(self, dataset_info: Dict[str, Any]) -> int:
        """Register a new dataset"""
        
        # Calculate checksum if file path provided
        checksum = None
        if 'file_path' in dataset_info:
            checksum = self._calculate_checksum(dataset_info['file_path'])
        
        dataset = Dataset(
            name=dataset_info['name'],
            domain=dataset_info['domain'],
            version=dataset_info.get('version', '1.0'),
            description=dataset_info.get('description', ''),
            size_gb=dataset_info['size_gb'],
            num_samples=dataset_info.get('num_samples'),
            num_features=dataset_info.get('num_features'),
            dimensions=dataset_info.get('dimensions', {}),
            storage_tier=dataset_info['storage_tier'],
            storage_path=dataset_info['storage_path'],
            zarr_path=dataset_info.get('zarr_path'),
            backup_path=dataset_info.get('backup_path'),
            checksum=checksum,
            source=dataset_info.get('source'),
            instrument=dataset_info.get('instrument'),
            mission=dataset_info.get('mission'),
            observation_date=dataset_info.get('observation_date')
        )
        
        self.session.add(dataset)
        self.session.commit()
        
        logger.info(f"Registered dataset: {dataset.name} (ID: {dataset.id})")
        return dataset.id
    
    def register_experiment(self, experiment_info: Dict[str, Any]) -> int:
        """Register a new experiment"""
        
        experiment = Experiment(
            name=experiment_info['name'],
            dataset_id=experiment_info['dataset_id'],
            model_type=experiment_info.get('model_type'),
            hyperparameters=experiment_info.get('hyperparameters', {}),
            status=experiment_info.get('status', 'running')
        )
        
        self.session.add(experiment)
        self.session.commit()
        
        logger.info(f"Registered experiment: {experiment.name} (ID: {experiment.id})")
        return experiment.id
    
    def update_experiment_results(self, experiment_id: int, results: Dict[str, Any]):
        """Update experiment results"""
        
        experiment = self.session.query(Experiment).filter(Experiment.id == experiment_id).first()
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Update results
        for key, value in results.items():
            if hasattr(experiment, key):
                setattr(experiment, key, value)
        
        experiment.completed_at = datetime.utcnow()
        experiment.status = 'completed'
        
        self.session.commit()
        logger.info(f"Updated experiment {experiment_id} results")
    
    def register_data_chunks(self, dataset_id: int, chunk_info: List[Dict[str, Any]]):
        """Register data chunks for a dataset"""
        
        chunks = []
        for chunk_data in chunk_info:
            chunk = DataChunk(
                dataset_id=dataset_id,
                chunk_id=chunk_data['chunk_id'],
                start_indices=chunk_data['start_indices'],
                end_indices=chunk_data['end_indices'],
                shape=chunk_data['shape'],
                size_mb=chunk_data['size_mb'],
                storage_tier=chunk_data['storage_tier'],
                file_path=chunk_data['file_path'],
                min_value=chunk_data.get('min_value'),
                max_value=chunk_data.get('max_value'),
                mean_value=chunk_data.get('mean_value'),
                std_value=chunk_data.get('std_value')
            )
            chunks.append(chunk)
        
        self.session.add_all(chunks)
        self.session.commit()
        
        logger.info(f"Registered {len(chunks)} data chunks for dataset {dataset_id}")
    
    def query_datasets(self, 
                      domain: Optional[str] = None,
                      storage_tier: Optional[str] = None,
                      min_size_gb: Optional[float] = None,
                      max_size_gb: Optional[float] = None,
                      status: Optional[str] = None) -> List[Dataset]:
        """Query datasets with filters"""
        
        query = self.session.query(Dataset)
        
        if domain:
            query = query.filter(Dataset.domain == domain)
        if storage_tier:
            query = query.filter(Dataset.storage_tier == storage_tier)
        if min_size_gb:
            query = query.filter(Dataset.size_gb >= min_size_gb)
        if max_size_gb:
            query = query.filter(Dataset.size_gb <= max_size_gb)
        if status:
            query = query.filter(Dataset.status == status)
        
        return query.all()
    
    def query_experiments(self,
                         dataset_id: Optional[int] = None,
                         model_type: Optional[str] = None,
                         min_accuracy: Optional[float] = None,
                         status: Optional[str] = None) -> List[Experiment]:
        """Query experiments with filters"""
        
        query = self.session.query(Experiment)
        
        if dataset_id:
            query = query.filter(Experiment.dataset_id == dataset_id)
        if model_type:
            query = query.filter(Experiment.model_type == model_type)
        if min_accuracy:
            query = query.filter(Experiment.accuracy >= min_accuracy)
        if status:
            query = query.filter(Experiment.status == status)
        
        return query.all()
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        
        stats = {
            'total_datasets': self.session.query(Dataset).count(),
            'total_experiments': self.session.query(Experiment).count(),
            'total_data_chunks': self.session.query(DataChunk).count(),
            'total_predictions': self.session.query(Prediction).count(),
            'total_size_tb': self.session.query(func.sum(Dataset.size_gb)).scalar() / 1024 or 0
        }
        
        # Domain breakdown
        domain_stats = self.session.query(Dataset.domain, func.count(Dataset.id)).group_by(Dataset.domain).all()
        stats['domains'] = dict(domain_stats)
        
        # Storage tier breakdown
        tier_stats = self.session.query(Dataset.storage_tier, func.count(Dataset.id)).group_by(Dataset.storage_tier).all()
        stats['storage_tiers'] = dict(tier_stats)
        
        return stats
    
    def optimize_data_access(self, chunk_id: str):
        """Update data access patterns for optimization"""
        
        chunk = self.session.query(DataChunk).filter(DataChunk.chunk_id == chunk_id).first()
        if chunk:
            chunk.last_accessed = datetime.utcnow()
            chunk.access_count += 1
            self.session.commit()
    
    def get_recommended_chunks(self, dataset_id: int, limit: int = 10) -> List[DataChunk]:
        """Get recommended chunks for caching based on access patterns"""
        
        return self.session.query(DataChunk).filter(
            DataChunk.dataset_id == dataset_id,
            DataChunk.is_cached == False
        ).order_by(
            DataChunk.access_count.desc(),
            DataChunk.last_accessed.desc()
        ).limit(limit).all()
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of a file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def close(self):
        """Close database connection"""
        self.session.close()

# Utility functions
def create_metadata_manager(db_path: str = "data/metadata.db") -> MetadataManager:
    """Create a metadata manager instance"""
    return MetadataManager(db_path)

def register_astronomical_dataset(manager: MetadataManager, data_path: Path, 
                                 object_info: Dict[str, Any]) -> int:
    """Register an astronomical dataset with object information"""
    
    # Register the dataset
    dataset_id = manager.register_dataset({
        'name': f"astronomical_{object_info['name']}",
        'domain': DataDomain.ASTRONOMICAL.value,
        'storage_path': str(data_path),
        'size_gb': data_path.stat().st_size / (1024**3) if data_path.exists() else 0,
        'source': object_info.get('source', 'Unknown'),
        'instrument': object_info.get('instrument'),
        'mission': object_info.get('mission')
    })
    
    # Register the astronomical object
    astro_object = AstronomicalObject(
        name=object_info['name'],
        object_type=object_info['object_type'],
        ra=object_info.get('ra'),
        dec=object_info.get('dec'),
        distance_pc=object_info.get('distance_pc'),
        mass=object_info.get('mass'),
        radius=object_info.get('radius'),
        temperature=object_info.get('temperature'),
        spectral_type=object_info.get('spectral_type'),
        metallicity=object_info.get('metallicity'),
        age_gyr=object_info.get('age_gyr'),
        orbital_period=object_info.get('orbital_period'),
        semi_major_axis=object_info.get('semi_major_axis'),
        eccentricity=object_info.get('eccentricity'),
        inclination=object_info.get('inclination'),
        atmosphere_pressure=object_info.get('atmosphere_pressure'),
        atmosphere_composition=object_info.get('atmosphere_composition')
    )
    
    manager.session.add(astro_object)
    manager.session.commit()
    
    # Link object to dataset
    link = ObjectDatasetLink(object_id=astro_object.id, dataset_id=dataset_id)
    manager.session.add(link)
    manager.session.commit()
    
    return dataset_id

if __name__ == "__main__":
    # Example usage
    manager = create_metadata_manager()
    
    # Register a sample dataset
    dataset_id = manager.register_dataset({
        'name': 'kepler_exoplanets_v1',
        'domain': DataDomain.EXOPLANET.value,
        'storage_path': '/data/kepler/exoplanets.zarr',
        'size_gb': 150.5,
        'num_samples': 5000,
        'num_features': 20,
        'storage_tier': StorageTier.LOCAL_SSD.value,
        'source': 'NASA Kepler Mission',
        'instrument': 'Kepler Photometer',
        'mission': 'Kepler'
    })
    
    print(f"Registered dataset with ID: {dataset_id}")
    
    # Query datasets
    datasets = manager.query_datasets(domain=DataDomain.EXOPLANET.value)
    print(f"Found {len(datasets)} exoplanet datasets")
    
    # Get statistics
    stats = manager.get_dataset_statistics()
    print(f"Database statistics: {stats}")
    
    manager.close() 
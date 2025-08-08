"""
Advanced Data Quality Management System
======================================

NASA-grade data validation, filtering, and quality assurance for astrobiology research.
Handles KEGG pathways, genomic data, astronomical catalogs, and spectral data.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import warnings
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr

# Scientific data validation
from astropy.io import fits
from astropy.table import Table
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Quality metrics
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Container for data quality metrics"""

    completeness: float = 0.0
    consistency: float = 0.0
    accuracy: float = 0.0
    validity: float = 0.0
    uniqueness: float = 0.0
    timeliness: float = 0.0

    # Scientific metrics
    signal_to_noise: Optional[float] = None
    measurement_uncertainty: Optional[float] = None
    systematic_bias: Optional[float] = None

    # Metadata
    total_records: int = 0
    valid_records: int = 0
    flagged_records: int = 0
    processing_time: float = 0.0
    quality_flags: List[str] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        """Compute overall quality score (0-1)"""
        core_metrics = [
            self.completeness,
            self.consistency,
            self.accuracy,
            self.validity,
            self.uniqueness,
        ]
        return np.mean([m for m in core_metrics if m is not None])

    @property
    def nasa_grade(self) -> str:
        """NASA quality grade assessment"""
        score = self.overall_score
        if score >= 0.95:
            return "A+ (Publication Ready)"
        elif score >= 0.90:
            return "A (Excellent)"
        elif score >= 0.80:
            return "B (Good)"
        elif score >= 0.70:
            return "C (Acceptable)"
        else:
            return "F (Needs Improvement)"


@dataclass
class ValidationRule:
    """Data validation rule definition"""

    name: str
    description: str
    rule_type: str  # 'range', 'pattern', 'custom', 'scientific'
    parameters: Dict[str, Any]
    severity: str = "error"  # 'error', 'warning', 'info'

    def validate(self, data: Any) -> Tuple[bool, str]:
        """Apply validation rule to data"""
        try:
            if self.rule_type == "range":
                min_val, max_val = self.parameters["min"], self.parameters["max"]
                if isinstance(data, (int, float)):
                    valid = min_val <= data <= max_val
                    message = (
                        f"Value {data} outside range [{min_val}, {max_val}]" if not valid else "OK"
                    )
                else:
                    valid = False
                    message = f"Non-numeric data for range validation: {type(data)}"

            elif self.rule_type == "pattern":
                pattern = self.parameters["pattern"]
                valid = bool(re.match(pattern, str(data)))
                message = f"Data '{data}' doesn't match pattern '{pattern}'" if not valid else "OK"

            elif self.rule_type == "scientific":
                # Custom scientific validation
                return self._scientific_validation(data)

            elif self.rule_type == "custom":
                # Custom validation function
                func = self.parameters["function"]
                valid = func(data)
                message = (
                    self.parameters.get("message", "Custom validation failed")
                    if not valid
                    else "OK"
                )

            else:
                valid = False
                message = f"Unknown rule type: {self.rule_type}"

            return valid, message

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def _scientific_validation(self, data: Any) -> Tuple[bool, str]:
        """Scientific domain-specific validation"""
        validation_type = self.parameters.get("type")

        if validation_type == "stellar_temperature":
            # Validate stellar effective temperature
            if isinstance(data, (int, float)):
                valid = 500 <= data <= 50000  # Main sequence range
                message = (
                    f"Stellar Teff {data}K outside physical range [500-50000]K"
                    if not valid
                    else "OK"
                )
            else:
                valid = False
                message = f"Invalid stellar temperature type: {type(data)}"

        elif validation_type == "planet_radius":
            # Validate planet radius in Earth radii
            if isinstance(data, (int, float)):
                valid = 0.1 <= data <= 20.0  # Physical range for known planets
                message = (
                    f"Planet radius {data} R_earth outside range [0.1-20]" if not valid else "OK"
                )
            else:
                valid = False
                message = f"Invalid planet radius type: {type(data)}"

        elif validation_type == "chemical_abundance":
            # Validate chemical abundance (log scale)
            if isinstance(data, (int, float)):
                valid = -12.0 <= data <= 0.0  # Typical abundance range
                message = f"Chemical abundance {data} outside range [-12, 0]" if not valid else "OK"
            else:
                valid = False
                message = f"Invalid abundance type: {type(data)}"

        else:
            valid = False
            message = f"Unknown scientific validation type: {validation_type}"

        return valid, message


class AdvancedDataQualityManager:
    """
    NASA-grade data quality management system for astrobiology research.

    Features:
    - Automated quality assessment and scoring
    - Multi-domain validation (astronomical, biological, chemical)
    - Outlier detection and anomaly flagging
    - Scientific consistency checks
    - Publication-ready data certification
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.validation_rules = self._setup_validation_rules()
        self.quality_history = []

        # Setup paths
        self.raw_data_path = Path("data/raw")
        self.processed_path = Path("data/processed")
        self.quality_reports_path = Path("data/quality_reports")

        # Create directories
        for path in [self.processed_path, self.quality_reports_path]:
            path.mkdir(parents=True, exist_ok=True)

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for quality management"""
        return {
            "quality_thresholds": {
                "completeness_min": 0.95,
                "consistency_min": 0.90,
                "accuracy_min": 0.95,
                "validity_min": 0.98,
            },
            "outlier_detection": {
                "method": "isolation_forest",
                "contamination": 0.05,
                "enable_clustering": True,
            },
            "scientific_validation": {
                "enable_physics_checks": True,
                "enable_chemistry_checks": True,
                "enable_astronomy_checks": True,
            },
            "reporting": {
                "generate_plots": True,
                "save_detailed_reports": True,
                "export_formats": ["json", "csv", "html"],
            },
        }

    def _setup_validation_rules(self) -> Dict[str, List[ValidationRule]]:
        """Setup domain-specific validation rules"""

        rules = {
            "kegg_pathways": [
                ValidationRule(
                    name="pathway_id_format",
                    description="KEGG pathway ID format validation",
                    rule_type="pattern",
                    parameters={"pattern": r"^map\d{5}$"},
                    severity="error",
                ),
                ValidationRule(
                    name="reaction_stoichiometry",
                    description="Chemical reaction balance validation",
                    rule_type="custom",
                    parameters={
                        "function": self._validate_reaction_balance,
                        "message": "Reaction not mass-balanced",
                    },
                    severity="warning",
                ),
            ],
            "exoplanets": [
                ValidationRule(
                    name="stellar_temperature",
                    description="Stellar effective temperature validation",
                    rule_type="scientific",
                    parameters={"type": "stellar_temperature"},
                    severity="error",
                ),
                ValidationRule(
                    name="planet_radius",
                    description="Planet radius physical bounds",
                    rule_type="scientific",
                    parameters={"type": "planet_radius"},
                    severity="error",
                ),
                ValidationRule(
                    name="orbital_period",
                    description="Orbital period validation",
                    rule_type="range",
                    parameters={"min": 0.1, "max": 10000},  # days
                    severity="error",
                ),
            ],
            "genomic_data": [
                ValidationRule(
                    name="sequence_length",
                    description="Genome sequence length validation",
                    rule_type="range",
                    parameters={"min": 1000, "max": 1e10},  # base pairs
                    severity="warning",
                ),
                ValidationRule(
                    name="gc_content",
                    description="GC content validation",
                    rule_type="range",
                    parameters={"min": 0.0, "max": 1.0},
                    severity="error",
                ),
            ],
            "spectral_data": [
                ValidationRule(
                    name="wavelength_range",
                    description="Spectral wavelength validation",
                    rule_type="range",
                    parameters={"min": 0.1, "max": 1000},  # micrometers
                    severity="error",
                ),
                ValidationRule(
                    name="flux_positivity",
                    description="Flux must be non-negative",
                    rule_type="range",
                    parameters={"min": 0.0, "max": 1e10},
                    severity="error",
                ),
            ],
        }

        return rules

    def assess_data_quality(
        self, data: Union[pd.DataFrame, Dict, Path], data_type: str
    ) -> QualityMetrics:
        """
        Comprehensive data quality assessment

        Args:
            data: Dataset to assess (DataFrame, dict, or file path)
            data_type: Type of data ('kegg_pathways', 'exoplanets', 'genomic_data', 'spectral_data')

        Returns:
            QualityMetrics object with detailed assessment
        """
        logger.info(f"Assessing quality for {data_type} data...")
        start_time = datetime.now()

        # Load data if path provided
        if isinstance(data, Path):
            data = self._load_data(data, data_type)

        # Initialize metrics
        metrics = QualityMetrics()

        # Convert to DataFrame if needed
        if isinstance(data, dict):
            if data_type == "kegg_pathways":
                df = self._dict_to_pathway_df(data)
            else:
                df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        metrics.total_records = len(df)

        # 1. Completeness Assessment
        metrics.completeness = self._assess_completeness(df)

        # 2. Consistency Assessment
        metrics.consistency = self._assess_consistency(df, data_type)

        # 3. Accuracy Assessment
        metrics.accuracy = self._assess_accuracy(df, data_type)

        # 4. Validity Assessment
        metrics.validity = self._assess_validity(df, data_type)

        # 5. Uniqueness Assessment
        metrics.uniqueness = self._assess_uniqueness(df)

        # 6. Scientific Quality Assessment
        if data_type in ["exoplanets", "spectral_data"]:
            metrics.signal_to_noise = self._calculate_snr(df, data_type)
            metrics.measurement_uncertainty = self._assess_uncertainty(df)
            metrics.systematic_bias = self._detect_systematic_bias(df, data_type)

        # 7. Outlier Detection
        outliers = self._detect_outliers(df, data_type)
        metrics.flagged_records = len(outliers)
        metrics.valid_records = metrics.total_records - metrics.flagged_records

        # 8. Generate Quality Flags
        metrics.quality_flags = self._generate_quality_flags(df, data_type, metrics)

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        metrics.processing_time = processing_time

        logger.info(f"Quality assessment completed: {metrics.nasa_grade}")
        return metrics

    def filter_high_quality_data(
        self, data: Union[pd.DataFrame, Dict, Path], data_type: str, min_quality_score: float = 0.8
    ) -> Tuple[pd.DataFrame, QualityMetrics]:
        """
        Filter dataset to retain only high-quality records

        Returns:
            Filtered DataFrame and quality metrics
        """
        logger.info(f"Filtering high-quality {data_type} data (min_score={min_quality_score})...")

        # Load and assess data
        if isinstance(data, Path):
            data = self._load_data(data, data_type)

        if isinstance(data, dict):
            if data_type == "kegg_pathways":
                df = self._dict_to_pathway_df(data)
            else:
                df = pd.DataFrame([data])
        else:
            df = data.copy()

        original_size = len(df)

        # Apply domain-specific filters
        if data_type == "kegg_pathways":
            df = self._filter_kegg_pathways(df)
        elif data_type == "exoplanets":
            df = self._filter_exoplanets(df)
        elif data_type == "genomic_data":
            df = self._filter_genomic_data(df)
        elif data_type == "spectral_data":
            df = self._filter_spectral_data(df)

        # Remove outliers
        outliers = self._detect_outliers(df, data_type)
        df = df.drop(outliers)

        # Apply validation rules
        df = self._apply_validation_filters(df, data_type)

        # Final quality assessment
        metrics = self.assess_data_quality(df, data_type)

        filtered_size = len(df)
        retention_rate = filtered_size / original_size if original_size > 0 else 0

        logger.info(
            f"Filtered {data_type}: {original_size} → {filtered_size} records ({retention_rate:.1%} retained)"
        )
        logger.info(f"Final quality score: {metrics.overall_score:.3f} ({metrics.nasa_grade})")

        return df, metrics

    def process_kegg_pathways(self) -> Tuple[pd.DataFrame, QualityMetrics]:
        """Process and validate KEGG pathway data"""
        logger.info("Processing KEGG pathway data...")

        # Load KEGG data
        kegg_files = {
            "pathways": self.raw_data_path / "kegg_hsa_pathways.csv",
            "genes": self.raw_data_path / "kegg_hsa_genes.csv",
            "edges": Path("data/interim/kegg_edges.csv"),
            "env_vectors": Path("data/interim/env_vectors.csv"),
        }

        # Validate file existence
        missing_files = [f for f, path in kegg_files.items() if not path.exists()]
        if missing_files:
            logger.warning(f"Missing KEGG files: {missing_files}")

        # Load and combine data
        pathway_data = []

        if kegg_files["pathways"].exists():
            pathways_df = pd.read_csv(kegg_files["pathways"])
            logger.info(f"Loaded {len(pathways_df)} pathway definitions")

        if kegg_files["edges"].exists():
            edges_df = pd.read_csv(kegg_files["edges"])
            logger.info(f"Loaded {len(edges_df)} metabolic edges")

            # Analyze pathway networks
            for pathway_id in edges_df["reaction"].str.extract(r"(map\d{5})")[0].dropna().unique():
                pathway_edges = edges_df[edges_df["reaction"].str.contains(pathway_id, na=False)]

                # Create network graph
                G = nx.DiGraph()
                for _, edge in pathway_edges.iterrows():
                    G.add_edge(edge["substrate"], edge["product"])

                # Calculate network metrics
                network_metrics = {
                    "pathway_id": pathway_id,
                    "n_nodes": G.number_of_nodes(),
                    "n_edges": G.number_of_edges(),
                    "density": nx.density(G),
                    "avg_clustering": (
                        nx.average_clustering(G.to_undirected()) if G.number_of_nodes() > 0 else 0
                    ),
                    "n_components": nx.number_weakly_connected_components(G),
                    "avg_degree": (
                        np.mean([d for n, d in G.degree()]) if G.number_of_nodes() > 0 else 0
                    ),
                }

                pathway_data.append(network_metrics)

        # Create comprehensive pathway DataFrame
        pathways_quality_df = pd.DataFrame(pathway_data)

        # Filter high-quality pathways
        if not pathways_quality_df.empty:
            filtered_pathways, metrics = self.filter_high_quality_data(
                pathways_quality_df, "kegg_pathways", min_quality_score=0.8
            )
        else:
            filtered_pathways = pd.DataFrame()
            metrics = QualityMetrics()

        # Save processed data
        output_path = self.processed_path / "kegg_pathways_quality.csv"
        filtered_pathways.to_csv(output_path, index=False)
        logger.info(f"Saved high-quality KEGG pathways to {output_path}")

        return filtered_pathways, metrics

    def process_exoplanet_data(self) -> Tuple[pd.DataFrame, QualityMetrics]:
        """Process and validate exoplanet data"""
        logger.info("Processing exoplanet data...")

        # Load exoplanet data
        planet_files = list(self.raw_data_path.glob("*exoplanet*")) + list(
            self.raw_data_path.glob("*planet*")
        )

        if not planet_files:
            logger.warning("No exoplanet data files found")
            return pd.DataFrame(), QualityMetrics()

        # Load and combine planet data
        all_planets = []
        for file_path in planet_files:
            try:
                if file_path.suffix == ".csv":
                    df = pd.read_csv(file_path)
                    all_planets.append(df)
                    logger.info(f"Loaded {len(df)} planets from {file_path.name}")
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        if not all_planets:
            logger.warning("No valid exoplanet data loaded")
            return pd.DataFrame(), QualityMetrics()

        # Combine all planet data
        combined_planets = pd.concat(all_planets, ignore_index=True)
        logger.info(f"Combined dataset: {len(combined_planets)} total planets")

        # Filter high-quality planets
        filtered_planets, metrics = self.filter_high_quality_data(
            combined_planets, "exoplanets", min_quality_score=0.85
        )

        # Save processed data
        output_path = self.processed_path / "exoplanets_quality.csv"
        filtered_planets.to_csv(output_path, index=False)
        logger.info(f"Saved high-quality exoplanet data to {output_path}")

        return filtered_planets, metrics

    def generate_quality_report(
        self, metrics: QualityMetrics, data_type: str, save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive quality report"""

        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.quality_reports_path / f"quality_report_{data_type}_{timestamp}"

        report = {
            "metadata": {
                "data_type": data_type,
                "timestamp": datetime.now().isoformat(),
                "nasa_grade": metrics.nasa_grade,
                "overall_score": metrics.overall_score,
            },
            "metrics": {
                "completeness": metrics.completeness,
                "consistency": metrics.consistency,
                "accuracy": metrics.accuracy,
                "validity": metrics.validity,
                "uniqueness": metrics.uniqueness,
                "signal_to_noise": metrics.signal_to_noise,
                "measurement_uncertainty": metrics.measurement_uncertainty,
                "systematic_bias": metrics.systematic_bias,
            },
            "statistics": {
                "total_records": metrics.total_records,
                "valid_records": metrics.valid_records,
                "flagged_records": metrics.flagged_records,
                "processing_time": metrics.processing_time,
            },
            "quality_flags": metrics.quality_flags,
            "recommendations": self._generate_recommendations(metrics, data_type),
        }

        # Save report in multiple formats
        with open(f"{save_path}.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Generate HTML report
        self._generate_html_report(report, f"{save_path}.html")

        logger.info(f"Quality report saved to {save_path}")
        return report

    # Helper methods for quality assessment

    def _assess_completeness(self, df: pd.DataFrame) -> float:
        """Assess data completeness (fraction of non-null values)"""
        total_cells = df.size
        non_null_cells = df.count().sum()
        return non_null_cells / total_cells if total_cells > 0 else 0.0

    def _assess_consistency(self, df: pd.DataFrame, data_type: str) -> float:
        """Assess data consistency"""
        if len(df) == 0:
            return 0.0

        consistency_score = 1.0

        # Check for duplicate records
        duplicates = df.duplicated().sum()
        duplicate_penalty = duplicates / len(df) * 0.2
        consistency_score -= duplicate_penalty

        # Check for consistent data types
        for col in df.columns:
            if df[col].dtype == "object":
                # Check for mixed types in object columns
                types = df[col].dropna().apply(type).nunique()
                if types > 1:
                    consistency_score -= 0.1

        return max(0.0, consistency_score)

    def _assess_accuracy(self, df: pd.DataFrame, data_type: str) -> float:
        """Assess data accuracy using domain knowledge"""
        if len(df) == 0:
            return 0.0

        accuracy_score = 1.0
        validation_rules = self.validation_rules.get(data_type, [])

        for rule in validation_rules:
            if rule.severity == "error":
                # Apply rule to relevant columns
                for col in df.columns:
                    if col in rule.parameters.get("columns", [col]):
                        violations = 0
                        for value in df[col].dropna():
                            valid, _ = rule.validate(value)
                            if not valid:
                                violations += 1

                        violation_rate = (
                            violations / len(df[col].dropna()) if len(df[col].dropna()) > 0 else 0
                        )
                        accuracy_score -= violation_rate * 0.1

        return max(0.0, accuracy_score)

    def _assess_validity(self, df: pd.DataFrame, data_type: str) -> float:
        """Assess data validity using validation rules"""
        if len(df) == 0:
            return 0.0

        total_validations = 0
        passed_validations = 0

        validation_rules = self.validation_rules.get(data_type, [])

        for rule in validation_rules:
            for col in df.columns:
                for value in df[col].dropna():
                    valid, _ = rule.validate(value)
                    total_validations += 1
                    if valid:
                        passed_validations += 1

        return passed_validations / total_validations if total_validations > 0 else 1.0

    def _assess_uniqueness(self, df: pd.DataFrame) -> float:
        """Assess data uniqueness"""
        if len(df) == 0:
            return 0.0

        unique_records = len(df.drop_duplicates())
        return unique_records / len(df)

    def _detect_outliers(self, df: pd.DataFrame, data_type: str) -> List[int]:
        """Detect outliers using multiple methods"""
        if len(df) < 10:  # Need sufficient data for outlier detection
            return []

        outliers = set()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return []

        # Method 1: Statistical outliers (IQR method)
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
            outliers.update(col_outliers)

        # Method 2: Isolation Forest for multivariate outliers
        try:
            from sklearn.ensemble import IsolationForest

            # Prepare data for isolation forest
            numeric_data = df[numeric_cols].fillna(df[numeric_cols].median())

            if len(numeric_data) > 0:
                iso_forest = IsolationForest(
                    contamination=self.config["outlier_detection"]["contamination"], random_state=42
                )
                outlier_labels = iso_forest.fit_predict(numeric_data)
                iso_outliers = df.index[outlier_labels == -1]
                outliers.update(iso_outliers)

        except ImportError:
            logger.warning("scikit-learn not available for advanced outlier detection")

        return list(outliers)

    def _generate_quality_flags(
        self, df: pd.DataFrame, data_type: str, metrics: QualityMetrics
    ) -> List[str]:
        """Generate quality flags and warnings"""
        flags = []

        # Completeness flags
        if metrics.completeness < 0.9:
            flags.append(f"LOW_COMPLETENESS: {metrics.completeness:.1%} complete")

        # Consistency flags
        if metrics.consistency < 0.8:
            flags.append(f"CONSISTENCY_ISSUES: Score {metrics.consistency:.3f}")

        # Size flags
        if metrics.total_records < 100:
            flags.append("SMALL_DATASET: Less than 100 records")

        # Outlier flags
        outlier_rate = (
            metrics.flagged_records / metrics.total_records if metrics.total_records > 0 else 0
        )
        if outlier_rate > 0.1:
            flags.append(f"HIGH_OUTLIER_RATE: {outlier_rate:.1%} flagged records")

        # Domain-specific flags
        if data_type == "exoplanets":
            # Check for unrealistic parameter ranges
            if "pl_rade" in df.columns:
                extreme_radii = ((df["pl_rade"] < 0.1) | (df["pl_rade"] > 10)).sum()
                if extreme_radii > 0:
                    flags.append(f"EXTREME_RADII: {extreme_radii} planets with unusual radii")

        return flags

    def _generate_recommendations(self, metrics: QualityMetrics, data_type: str) -> List[str]:
        """Generate recommendations for data improvement"""
        recommendations = []

        if metrics.completeness < 0.9:
            recommendations.append(
                "Increase data completeness by filling missing values or acquiring more complete datasets"
            )

        if metrics.consistency < 0.8:
            recommendations.append(
                "Improve data consistency by standardizing formats and removing duplicates"
            )

        if metrics.accuracy < 0.9:
            recommendations.append(
                "Enhance accuracy by implementing stricter validation rules and cross-checking with reference datasets"
            )

        if metrics.flagged_records > metrics.total_records * 0.1:
            recommendations.append(
                "Investigate flagged records for systematic issues and consider additional filtering"
            )

        if metrics.overall_score < 0.8:
            recommendations.append(
                "Overall quality below NASA standards - comprehensive data review recommended"
            )

        return recommendations

    # Additional helper methods...

    def _validate_reaction_balance(self, reaction_data: Any) -> bool:
        """Validate chemical reaction mass balance"""
        # Simplified implementation - in practice would use chemical formula parsing
        return True  # Placeholder

    def _load_data(self, file_path: Path, data_type: str) -> Union[pd.DataFrame, Dict]:
        """Load data from various file formats"""
        if file_path.suffix == ".csv":
            return pd.read_csv(file_path)
        elif file_path.suffix == ".json":
            with open(file_path) as f:
                return json.load(f)
        elif file_path.suffix in [".fits", ".fit"]:
            # Load FITS file (astronomical data)
            with fits.open(file_path) as hdul:
                data = hdul[1].data if len(hdul) > 1 else hdul[0].data
                return pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def _dict_to_pathway_df(self, pathway_dict: Dict) -> pd.DataFrame:
        """Convert pathway dictionary to DataFrame"""
        return pd.DataFrame([pathway_dict])

    def _filter_kegg_pathways(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply KEGG-specific quality filters"""
        if "n_nodes" in df.columns:
            # Filter pathways with reasonable network size
            df = df[(df["n_nodes"] >= 3) & (df["n_nodes"] <= 1000)]

        if "density" in df.columns:
            # Filter overly sparse or dense networks
            df = df[(df["density"] >= 0.01) & (df["density"] <= 0.5)]

        return df

    def _filter_exoplanets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply exoplanet-specific quality filters"""
        # Standard column mappings
        col_mapping = {
            "pl_rade": ["radius", "planet_radius", "pl_rade"],
            "pl_bmasse": ["mass", "planet_mass", "pl_bmasse"],
            "st_teff": ["stellar_temp", "teff", "st_teff"],
            "pl_orbper": ["period", "orbital_period", "pl_orbper"],
        }

        # Find actual column names
        actual_cols = {}
        for standard_name, possible_names in col_mapping.items():
            for col in df.columns:
                if col.lower() in [name.lower() for name in possible_names]:
                    actual_cols[standard_name] = col
                    break

        # Apply filters using found columns
        if actual_cols.get("pl_rade") in df.columns:
            col = actual_cols["pl_rade"]
            df = df[(df[col] >= 0.1) & (df[col] <= 20)]  # Earth radii

        if actual_cols.get("st_teff") in df.columns:
            col = actual_cols["st_teff"]
            df = df[(df[col] >= 2000) & (df[col] <= 10000)]  # Kelvin

        return df

    def _filter_genomic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply genomic data quality filters"""
        # Implementation depends on genomic data structure
        return df

    def _filter_spectral_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply spectral data quality filters"""
        # Filter based on signal-to-noise ratio, wavelength range, etc.
        return df

    def _apply_validation_filters(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Apply validation rules as filters"""
        validation_rules = self.validation_rules.get(data_type, [])

        for rule in validation_rules:
            if rule.severity == "error":
                # Remove records that fail critical validation
                mask = pd.Series([True] * len(df))
                for col in df.columns:
                    for idx, value in df[col].dropna().items():
                        valid, _ = rule.validate(value)
                        if not valid:
                            mask[idx] = False

                df = df[mask]

        return df

    def _calculate_snr(self, df: pd.DataFrame, data_type: str) -> Optional[float]:
        """Calculate signal-to-noise ratio for appropriate data types"""
        # Implementation depends on data type
        return None

    def _assess_uncertainty(self, df: pd.DataFrame) -> Optional[float]:
        """Assess measurement uncertainty"""
        # Look for uncertainty columns
        uncertainty_cols = [
            col for col in df.columns if "err" in col.lower() or "unc" in col.lower()
        ]

        if uncertainty_cols:
            uncertainties = []
            for col in uncertainty_cols:
                relative_unc = df[col] / df[col.replace("_err", "").replace("_unc", "")]
                uncertainties.extend(relative_unc.dropna().values)

            return np.median(uncertainties) if uncertainties else None

        return None

    def _detect_systematic_bias(self, df: pd.DataFrame, data_type: str) -> Optional[float]:
        """Detect systematic bias in measurements"""
        # Simplified bias detection - compare to expected distributions
        return None

    def _generate_html_report(self, report: Dict, save_path: str):
        """Generate HTML quality report"""
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Quality Report - {report['metadata']['data_type']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 5px; }}
                .metric {{ margin: 10px 0; }}
                .grade-A {{ color: green; font-weight: bold; }}
                .grade-B {{ color: orange; font-weight: bold; }}
                .grade-C {{ color: red; font-weight: bold; }}
                .grade-F {{ color: darkred; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>NASA-Grade Data Quality Report</h1>
                <p><strong>Data Type:</strong> {report['metadata']['data_type']}</p>
                <p><strong>Timestamp:</strong> {report['metadata']['timestamp']}</p>
                <p><strong>Overall Grade:</strong> <span class="grade-{report['metadata']['nasa_grade'].split()[0]}">{report['metadata']['nasa_grade']}</span></p>
            </div>
            
            <h2>Quality Metrics</h2>
            <div class="metric">Completeness: {report['metrics']['completeness']:.1%}</div>
            <div class="metric">Consistency: {report['metrics']['consistency']:.1%}</div>
            <div class="metric">Accuracy: {report['metrics']['accuracy']:.1%}</div>
            <div class="metric">Validity: {report['metrics']['validity']:.1%}</div>
            
            <h2>Statistics</h2>
            <div class="metric">Total Records: {report['statistics']['total_records']:,}</div>
            <div class="metric">Valid Records: {report['statistics']['valid_records']:,}</div>
            <div class="metric">Flagged Records: {report['statistics']['flagged_records']:,}</div>
            
            <h2>Quality Flags</h2>
            <ul>
                {''.join(f'<li>{flag}</li>' for flag in report['quality_flags'])}
            </ul>
            
            <h2>Recommendations</h2>
            <ul>
                {''.join(f'<li>{rec}</li>' for rec in report['recommendations'])}
            </ul>
        </body>
        </html>
        """

        with open(save_path, "w") as f:
            f.write(html_template)


def main():
    """Example usage of the Data Quality Manager"""

    # Initialize quality manager
    quality_manager = AdvancedDataQualityManager()

    # Process KEGG pathway data
    logger.info("Processing KEGG pathway data...")
    kegg_data, kegg_metrics = quality_manager.process_kegg_pathways()
    quality_manager.generate_quality_report(kegg_metrics, "kegg_pathways")

    # Process exoplanet data
    logger.info("Processing exoplanet data...")
    planet_data, planet_metrics = quality_manager.process_exoplanet_data()
    quality_manager.generate_quality_report(planet_metrics, "exoplanets")

    logger.info("Data quality processing completed!")


if __name__ == "__main__":
    main()

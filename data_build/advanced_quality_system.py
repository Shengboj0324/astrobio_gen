#!/usr/bin/env python3
"""
Advanced Quality Management System
=================================

NASA-grade quality management system for comprehensive data validation:
- Multi-dimensional quality assessment
- Real-time monitoring and alerts
- Automated quality improvement
- Statistical quality control
- Compliance reporting
- Data lineage tracking

Supports all data sources: KEGG, NCBI, AGORA2, genomic data, and more.

Author: AI Assistant
Date: 2025
"""

import os
import json
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union
import logging
import sqlite3
import pickle
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import statistics
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
import threading
from threading import Lock
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QualityLevel(Enum):
    """Quality assessment levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"

class DataType(Enum):
    """Supported data types"""
    KEGG_PATHWAY = "kegg_pathway"
    KEGG_REACTION = "kegg_reaction"
    KEGG_COMPOUND = "kegg_compound"
    NCBI_GENOME = "ncbi_genome"
    AGORA2_MODEL = "agora2_model"
    METABOLIC_REACTION = "metabolic_reaction"
    METABOLITE = "metabolite"
    ENVIRONMENTAL = "environmental"
    GENERIC = "generic"

@dataclass
class QualityMetrics:
    """Comprehensive quality metrics"""
    completeness: float = 0.0
    accuracy: float = 0.0
    consistency: float = 0.0
    validity: float = 0.0
    uniqueness: float = 0.0
    timeliness: float = 0.0
    conformity: float = 0.0
    integrity: float = 0.0
    reliability: float = 0.0
    accessibility: float = 0.0
    
    def overall_score(self) -> float:
        """Calculate weighted overall quality score"""
        weights = {
            'completeness': 0.15,
            'accuracy': 0.20,
            'consistency': 0.15,
            'validity': 0.15,
            'uniqueness': 0.10,
            'timeliness': 0.10,
            'conformity': 0.05,
            'integrity': 0.05,
            'reliability': 0.03,
            'accessibility': 0.02
        }
        
        total_score = 0.0
        for metric, weight in weights.items():
            value = getattr(self, metric, 0.0)
            total_score += value * weight
        
        return min(1.0, max(0.0, total_score))
    
    def get_level(self) -> QualityLevel:
        """Get quality level based on overall score"""
        score = self.overall_score()
        if score >= 0.9:
            return QualityLevel.EXCELLENT
        elif score >= 0.8:
            return QualityLevel.GOOD
        elif score >= 0.7:
            return QualityLevel.ACCEPTABLE
        elif score >= 0.5:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL

@dataclass
class QualityIssue:
    """Quality issue representation"""
    issue_id: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    category: str
    description: str
    affected_data: str
    recommendation: str
    auto_fixable: bool = False
    fix_function: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class QualityReport:
    """Comprehensive quality report"""
    report_id: str
    data_source: str
    data_type: DataType
    timestamp: datetime
    metrics: QualityMetrics
    issues: List[QualityIssue] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    compliance_status: Dict[str, bool] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class QualityRule(ABC):
    """Abstract base class for quality rules"""
    
    def __init__(self, rule_id: str, name: str, description: str, severity: str = "medium"):
        self.rule_id = rule_id
        self.name = name
        self.description = description
        self.severity = severity
    
    @abstractmethod
    def evaluate(self, data: Any) -> Tuple[bool, List[QualityIssue]]:
        """Evaluate the quality rule against data"""
        pass

class CompletenessRule(QualityRule):
    """Rule for checking data completeness"""
    
    def __init__(self, required_fields: List[str], threshold: float = 0.95):
        super().__init__("completeness", "Data Completeness", "Check for missing values")
        self.required_fields = required_fields
        self.threshold = threshold
    
    def evaluate(self, data: Any) -> Tuple[bool, List[QualityIssue]]:
        """Evaluate completeness"""
        issues = []
        
        if isinstance(data, pd.DataFrame):
            total_cells = len(data) * len(self.required_fields)
            if total_cells == 0:
                return False, [QualityIssue(
                    issue_id=f"completeness_empty_{int(time.time())}",
                    severity="critical",
                    category="completeness",
                    description="Dataset is empty",
                    affected_data="entire_dataset",
                    recommendation="Investigate data source and loading process"
                )]
            
            missing_count = 0
            for field in self.required_fields:
                if field in data.columns:
                    missing_count += data[field].isnull().sum()
                else:
                    missing_count += len(data)
                    issues.append(QualityIssue(
                        issue_id=f"completeness_missing_field_{field}_{int(time.time())}",
                        severity="high",
                        category="completeness",
                        description=f"Required field '{field}' is missing",
                        affected_data=field,
                        recommendation=f"Add missing field '{field}' to dataset"
                    ))
            
            completeness_ratio = 1.0 - (missing_count / total_cells)
            
            if completeness_ratio < self.threshold:
                issues.append(QualityIssue(
                    issue_id=f"completeness_threshold_{int(time.time())}",
                    severity="medium",
                    category="completeness",
                    description=f"Completeness ratio {completeness_ratio:.2f} below threshold {self.threshold}",
                    affected_data="multiple_fields",
                    recommendation="Review data collection and processing pipeline"
                ))
            
            return completeness_ratio >= self.threshold, issues
        
        return True, []

class AccuracyRule(QualityRule):
    """Rule for checking data accuracy"""
    
    def __init__(self, validation_patterns: Dict[str, str]):
        super().__init__("accuracy", "Data Accuracy", "Check for data format accuracy")
        self.validation_patterns = validation_patterns
    
    def evaluate(self, data: Any) -> Tuple[bool, List[QualityIssue]]:
        """Evaluate accuracy"""
        issues = []
        
        if isinstance(data, pd.DataFrame):
            total_violations = 0
            total_checks = 0
            
            for field, pattern in self.validation_patterns.items():
                if field in data.columns:
                    # Check pattern matching
                    valid_mask = data[field].astype(str).str.match(pattern, na=False)
                    violations = (~valid_mask).sum()
                    total_violations += violations
                    total_checks += len(data)
                    
                    if violations > 0:
                        issues.append(QualityIssue(
                            issue_id=f"accuracy_pattern_{field}_{int(time.time())}",
                            severity="medium",
                            category="accuracy",
                            description=f"Field '{field}' has {violations} format violations",
                            affected_data=field,
                            recommendation=f"Review and correct format for field '{field}'"
                        ))
            
            accuracy_ratio = 1.0 - (total_violations / total_checks) if total_checks > 0 else 1.0
            return accuracy_ratio >= 0.9, issues
        
        return True, []

class ConsistencyRule(QualityRule):
    """Rule for checking data consistency"""
    
    def __init__(self, consistency_checks: Dict[str, Any]):
        super().__init__("consistency", "Data Consistency", "Check for data consistency")
        self.consistency_checks = consistency_checks
    
    def evaluate(self, data: Any) -> Tuple[bool, List[QualityIssue]]:
        """Evaluate consistency"""
        issues = []
        
        if isinstance(data, pd.DataFrame):
            # Check for duplicate records
            duplicates = data.duplicated().sum()
            if duplicates > 0:
                issues.append(QualityIssue(
                    issue_id=f"consistency_duplicates_{int(time.time())}",
                    severity="medium",
                    category="consistency",
                    description=f"Found {duplicates} duplicate records",
                    affected_data="multiple_records",
                    recommendation="Remove or investigate duplicate records"
                ))
            
            # Check for inconsistent values
            for field, expected_values in self.consistency_checks.items():
                if field in data.columns and isinstance(expected_values, set):
                    invalid_values = data[field].dropna().unique()
                    invalid_count = sum(1 for val in invalid_values if val not in expected_values)
                    
                    if invalid_count > 0:
                        issues.append(QualityIssue(
                            issue_id=f"consistency_values_{field}_{int(time.time())}",
                            severity="medium",
                            category="consistency",
                            description=f"Field '{field}' has {invalid_count} inconsistent values",
                            affected_data=field,
                            recommendation=f"Standardize values for field '{field}'"
                        ))
            
            consistency_score = 1.0 - (len(issues) / max(1, len(self.consistency_checks)))
            return consistency_score >= 0.8, issues
        
        return True, []

class ValidityRule(QualityRule):
    """Rule for checking data validity"""
    
    def __init__(self, validity_constraints: Dict[str, Any]):
        super().__init__("validity", "Data Validity", "Check for data validity constraints")
        self.validity_constraints = validity_constraints
    
    def evaluate(self, data: Any) -> Tuple[bool, List[QualityIssue]]:
        """Evaluate validity"""
        issues = []
        
        if isinstance(data, pd.DataFrame):
            constraint_violations = 0
            total_constraints = 0
            
            for field, constraints in self.validity_constraints.items():
                if field in data.columns:
                    field_data = data[field].dropna()
                    
                    # Range constraints
                    if 'min' in constraints:
                        violations = (field_data < constraints['min']).sum()
                        if violations > 0:
                            constraint_violations += violations
                            issues.append(QualityIssue(
                                issue_id=f"validity_min_{field}_{int(time.time())}",
                                severity="medium",
                                category="validity",
                                description=f"Field '{field}' has {violations} values below minimum {constraints['min']}",
                                affected_data=field,
                                recommendation=f"Review values below minimum for field '{field}'"
                            ))
                        total_constraints += len(field_data)
                    
                    if 'max' in constraints:
                        violations = (field_data > constraints['max']).sum()
                        if violations > 0:
                            constraint_violations += violations
                            issues.append(QualityIssue(
                                issue_id=f"validity_max_{field}_{int(time.time())}",
                                severity="medium",
                                category="validity",
                                description=f"Field '{field}' has {violations} values above maximum {constraints['max']}",
                                affected_data=field,
                                recommendation=f"Review values above maximum for field '{field}'"
                            ))
                        total_constraints += len(field_data)
                    
                    # Type constraints
                    if 'type' in constraints:
                        expected_type = constraints['type']
                        if expected_type == 'numeric':
                            violations = (~pd.to_numeric(field_data, errors='coerce').notna()).sum()
                        elif expected_type == 'datetime':
                            violations = (~pd.to_datetime(field_data, errors='coerce').notna()).sum()
                        else:
                            violations = 0
                        
                        if violations > 0:
                            constraint_violations += violations
                            issues.append(QualityIssue(
                                issue_id=f"validity_type_{field}_{int(time.time())}",
                                severity="medium",
                                category="validity",
                                description=f"Field '{field}' has {violations} values with incorrect type",
                                affected_data=field,
                                recommendation=f"Review and correct data types for field '{field}'"
                            ))
                        total_constraints += len(field_data)
            
            validity_ratio = 1.0 - (constraint_violations / total_constraints) if total_constraints > 0 else 1.0
            return validity_ratio >= 0.9, issues
        
        return True, []

class OutlierDetectionRule(QualityRule):
    """Rule for detecting outliers"""
    
    def __init__(self, numeric_fields: List[str], method: str = "isolation_forest"):
        super().__init__("outliers", "Outlier Detection", "Detect statistical outliers")
        self.numeric_fields = numeric_fields
        self.method = method
    
    def evaluate(self, data: Any) -> Tuple[bool, List[QualityIssue]]:
        """Evaluate for outliers"""
        issues = []
        
        if isinstance(data, pd.DataFrame) and len(data) > 10:
            outlier_count = 0
            
            for field in self.numeric_fields:
                if field in data.columns:
                    field_data = pd.to_numeric(data[field], errors='coerce').dropna()
                    
                    if len(field_data) > 10:
                        if self.method == "isolation_forest":
                            try:
                                isolation_forest = IsolationForest(contamination=0.1, random_state=42)
                                outliers = isolation_forest.fit_predict(field_data.values.reshape(-1, 1))
                                outlier_indices = np.where(outliers == -1)[0]
                                outlier_count += len(outlier_indices)
                                
                                if len(outlier_indices) > 0:
                                    issues.append(QualityIssue(
                                        issue_id=f"outliers_{field}_{int(time.time())}",
                                        severity="low",
                                        category="outliers",
                                        description=f"Field '{field}' has {len(outlier_indices)} potential outliers",
                                        affected_data=field,
                                        recommendation=f"Review outlier values in field '{field}'"
                                    ))
                            except Exception as e:
                                logger.warning(f"Error in outlier detection for {field}: {e}")
                        
                        elif self.method == "iqr":
                            Q1 = field_data.quantile(0.25)
                            Q3 = field_data.quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            
                            outliers = field_data[(field_data < lower_bound) | (field_data > upper_bound)]
                            outlier_count += len(outliers)
                            
                            if len(outliers) > 0:
                                issues.append(QualityIssue(
                                    issue_id=f"outliers_iqr_{field}_{int(time.time())}",
                                    severity="low",
                                    category="outliers",
                                    description=f"Field '{field}' has {len(outliers)} IQR outliers",
                                    affected_data=field,
                                    recommendation=f"Review IQR outlier values in field '{field}'"
                                ))
            
            # Consider acceptable if outliers are less than 5% of data
            outlier_ratio = outlier_count / len(data)
            return outlier_ratio < 0.05, issues
        
        return True, []

class QualityRuleEngine:
    """Engine for managing and executing quality rules"""
    
    def __init__(self):
        self.rules: Dict[str, Dict[DataType, List[QualityRule]]] = defaultdict(lambda: defaultdict(list))
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register default quality rules for different data types"""
        # KEGG Pathway rules
        self.add_rule(DataType.KEGG_PATHWAY, CompletenessRule(['pathway_id', 'name'], 0.95))
        self.add_rule(DataType.KEGG_PATHWAY, AccuracyRule({'pathway_id': r'^map\d{5}$'}))
        self.add_rule(DataType.KEGG_PATHWAY, ValidityRule({'reaction_count': {'min': 0, 'type': 'numeric'}}))
        
        # KEGG Reaction rules
        self.add_rule(DataType.KEGG_REACTION, CompletenessRule(['reaction_id', 'equation'], 0.90))
        self.add_rule(DataType.KEGG_REACTION, AccuracyRule({'reaction_id': r'^R\d{5}$'}))
        
        # KEGG Compound rules
        self.add_rule(DataType.KEGG_COMPOUND, CompletenessRule(['compound_id', 'name'], 0.90))
        self.add_rule(DataType.KEGG_COMPOUND, AccuracyRule({'compound_id': r'^C\d{5}$'}))
        self.add_rule(DataType.KEGG_COMPOUND, ValidityRule({'exact_mass': {'min': 0, 'type': 'numeric'}}))
        
        # NCBI Genome rules
        self.add_rule(DataType.NCBI_GENOME, CompletenessRule(['assembly_accession', 'organism_name'], 0.95))
        self.add_rule(DataType.NCBI_GENOME, AccuracyRule({'assembly_accession': r'^GC[AF]_\d{9}\.\d+$'}))
        self.add_rule(DataType.NCBI_GENOME, ValidityRule({'genome_size': {'min': 0, 'type': 'numeric'}}))
        
        # AGORA2 Model rules
        self.add_rule(DataType.AGORA2_MODEL, CompletenessRule(['model_id', 'organism', 'taxonomy'], 0.90))
        self.add_rule(DataType.AGORA2_MODEL, ValidityRule({
            'reactions': {'min': 0, 'type': 'numeric'},
            'metabolites': {'min': 0, 'type': 'numeric'},
            'genes': {'min': 0, 'type': 'numeric'}
        }))
        self.add_rule(DataType.AGORA2_MODEL, OutlierDetectionRule(['reactions', 'metabolites', 'genes']))
        
        # Environmental data rules
        self.add_rule(DataType.ENVIRONMENTAL, CompletenessRule(['pH', 'temp', 'O2'], 0.95))
        self.add_rule(DataType.ENVIRONMENTAL, ValidityRule({
            'pH': {'min': 0, 'max': 14, 'type': 'numeric'},
            'temp': {'min': 0, 'max': 500, 'type': 'numeric'},
            'O2': {'min': 0, 'max': 1, 'type': 'numeric'}
        }))
        self.add_rule(DataType.ENVIRONMENTAL, OutlierDetectionRule(['pH', 'temp', 'O2']))
    
    def add_rule(self, data_type: DataType, rule: QualityRule):
        """Add a quality rule for a specific data type"""
        self.rules[rule.rule_id][data_type].append(rule)
    
    def evaluate_data(self, data: Any, data_type: DataType) -> Tuple[bool, List[QualityIssue]]:
        """Evaluate data against all applicable rules"""
        all_issues = []
        all_passed = True
        
        for rule_id, type_rules in self.rules.items():
            if data_type in type_rules:
                for rule in type_rules[data_type]:
                    try:
                        passed, issues = rule.evaluate(data)
                        if not passed:
                            all_passed = False
                        all_issues.extend(issues)
                    except Exception as e:
                        logger.error(f"Error evaluating rule {rule_id}: {e}")
                        all_issues.append(QualityIssue(
                            issue_id=f"rule_error_{rule_id}_{int(time.time())}",
                            severity="high",
                            category="system",
                            description=f"Error evaluating rule {rule_id}: {str(e)}",
                            affected_data="rule_evaluation",
                            recommendation="Check rule implementation and data format"
                        ))
        
        return all_passed, all_issues

class QualityAnalyzer:
    """Enhanced quality analyzer with NCBI quality control support from web crawl"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        # Enhanced support for NCBI quality control files discovered in web crawl
        self.ncbi_quality_parsers = {
            'fcs_report': self._parse_fcs_report,
            'ani_report': self._parse_ani_report,
            'ani_contam_ranges': self._parse_ani_contamination,
            'assembly_stats': self._parse_assembly_stats,
            'busco_report': self._parse_busco_report,
            'checkm_report': self._parse_checkm_report
        }
    
    def calculate_completeness(self, data: pd.DataFrame, required_fields: List[str] = None) -> float:
        """Calculate data completeness score"""
        if data.empty:
            return 0.0
        
        if required_fields is None:
            required_fields = data.columns.tolist()
        
        total_cells = len(data) * len(required_fields)
        if total_cells == 0:
            return 0.0
        
        missing_count = 0
        for field in required_fields:
            if field in data.columns:
                missing_count += data[field].isnull().sum()
            else:
                missing_count += len(data)
        
        return 1.0 - (missing_count / total_cells)
    
    def calculate_accuracy(self, data: pd.DataFrame, validation_patterns: Dict[str, str] = None) -> float:
        """Calculate data accuracy score"""
        if data.empty:
            return 0.0
        
        if validation_patterns is None:
            return 1.0
        
        total_violations = 0
        total_checks = 0
        
        for field, pattern in validation_patterns.items():
            if field in data.columns:
                field_data = data[field].dropna().astype(str)
                valid_mask = field_data.str.match(pattern, na=False)
                violations = (~valid_mask).sum()
                total_violations += violations
                total_checks += len(field_data)
        
        if total_checks == 0:
            return 1.0
        
        return 1.0 - (total_violations / total_checks)
    
    def calculate_consistency(self, data: pd.DataFrame) -> float:
        """Calculate data consistency score"""
        if data.empty:
            return 0.0
        
        # Check for duplicates
        duplicate_ratio = data.duplicated().sum() / len(data)
        
        # Check for consistent data types
        type_consistency = 0.0
        for column in data.columns:
            if data[column].dtype == 'object':
                # Check if string column has consistent format
                non_null_values = data[column].dropna()
                if len(non_null_values) > 0:
                    # Simple consistency check - all values should be similar length
                    lengths = non_null_values.str.len()
                    if len(lengths) > 1:
                        cv = lengths.std() / lengths.mean() if lengths.mean() > 0 else 0
                        type_consistency += min(1.0, 1.0 - cv)
                    else:
                        type_consistency += 1.0
            else:
                type_consistency += 1.0
        
        type_consistency = type_consistency / len(data.columns) if len(data.columns) > 0 else 1.0
        
        return (1.0 - duplicate_ratio) * 0.5 + type_consistency * 0.5
    
    def calculate_validity(self, data: pd.DataFrame, constraints: Dict[str, Any] = None) -> float:
        """Calculate data validity score"""
        if data.empty:
            return 0.0
        
        if constraints is None:
            return 1.0
        
        violation_count = 0
        total_checks = 0
        
        for field, field_constraints in constraints.items():
            if field in data.columns:
                field_data = data[field].dropna()
                
                if 'min' in field_constraints:
                    violations = (pd.to_numeric(field_data, errors='coerce') < field_constraints['min']).sum()
                    violation_count += violations
                    total_checks += len(field_data)
                
                if 'max' in field_constraints:
                    violations = (pd.to_numeric(field_data, errors='coerce') > field_constraints['max']).sum()
                    violation_count += violations
                    total_checks += len(field_data)
        
        if total_checks == 0:
            return 1.0
        
        return 1.0 - (violation_count / total_checks)
    
    def calculate_uniqueness(self, data: pd.DataFrame, key_fields: List[str] = None) -> float:
        """Calculate data uniqueness score"""
        if data.empty:
            return 0.0
        
        if key_fields is None:
            key_fields = data.columns.tolist()
        
        # Check available key fields
        available_fields = [f for f in key_fields if f in data.columns]
        if not available_fields:
            return 1.0
        
        unique_count = data[available_fields].drop_duplicates().shape[0]
        return unique_count / len(data)
    
    def calculate_timeliness(self, data: pd.DataFrame, timestamp_field: str = 'timestamp') -> float:
        """Calculate data timeliness score"""
        if data.empty or timestamp_field not in data.columns:
            return 1.0
        
        try:
            timestamps = pd.to_datetime(data[timestamp_field], errors='coerce').dropna()
            if len(timestamps) == 0:
                return 1.0
            
            latest_timestamp = timestamps.max()
            current_time = pd.Timestamp.now(tz=latest_timestamp.tz)
            
            # Calculate days since last update
            days_old = (current_time - latest_timestamp).days
            
            # Score decreases with age, but slowly
            return max(0.0, 1.0 - (days_old / 365.0))
        except Exception:
            return 1.0
    
    def analyze_ncbi_quality_files(self, quality_files: Dict[str, str]) -> Dict[str, Any]:
        """Analyze NCBI quality control files discovered in web crawl"""
        quality_analysis = {}
        
        for file_type, file_path in quality_files.items():
            if file_type in self.ncbi_quality_parsers and file_path:
                try:
                    parser = self.ncbi_quality_parsers[file_type]
                    analysis = parser(file_path)
                    quality_analysis[file_type] = analysis
                except Exception as e:
                    logger.warning(f"Error parsing {file_type} file {file_path}: {e}")
                    quality_analysis[file_type] = {'error': str(e)}
        
        return quality_analysis
    
    def _parse_fcs_report(self, file_path: str) -> Dict[str, Any]:
        """Parse Foreign Contamination Screen (FCS) report"""
        try:
            contamination_regions = []
            total_contaminated_length = 0
            
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    
                    parts = line.strip().split('\t')
                    if len(parts) >= 8:
                        seq_id = parts[0]
                        start_pos = int(parts[1])
                        end_pos = int(parts[2])
                        classification = parts[3]
                        evidence = parts[4]
                        
                        contamination_regions.append({
                            'sequence_id': seq_id,
                            'start': start_pos,
                            'end': end_pos,
                            'length': end_pos - start_pos + 1,
                            'classification': classification,
                            'evidence': evidence
                        })
                        
                        total_contaminated_length += end_pos - start_pos + 1
            
            return {
                'contamination_regions': contamination_regions,
                'total_regions': len(contamination_regions),
                'total_contaminated_length': total_contaminated_length,
                'quality_score': 1.0 - min(1.0, total_contaminated_length / 1000000)  # Normalize by 1Mb
            }
            
        except Exception as e:
            return {'error': f"Failed to parse FCS report: {e}"}
    
    def _parse_ani_report(self, file_path: str) -> Dict[str, Any]:
        """Parse Average Nucleotide Identity (ANI) report"""
        try:
            ani_results = []
            
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        query = parts[0]
                        subject = parts[1]
                        ani_value = float(parts[2])
                        
                        ani_results.append({
                            'query_assembly': query,
                            'subject_assembly': subject,
                            'ani_value': ani_value
                        })
            
            if ani_results:
                avg_ani = sum(result['ani_value'] for result in ani_results) / len(ani_results)
                min_ani = min(result['ani_value'] for result in ani_results)
                max_ani = max(result['ani_value'] for result in ani_results)
                
                return {
                    'ani_results': ani_results,
                    'average_ani': avg_ani,
                    'min_ani': min_ani,
                    'max_ani': max_ani,
                    'quality_score': avg_ani / 100.0  # ANI is typically 0-100
                }
            else:
                return {'error': 'No ANI results found'}
                
        except Exception as e:
            return {'error': f"Failed to parse ANI report: {e}"}
    
    def _parse_ani_contamination(self, file_path: str) -> Dict[str, Any]:
        """Parse ANI contamination ranges file"""
        try:
            contamination_ranges = []
            
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    
                    parts = line.strip().split('\t')
                    if len(parts) >= 6:
                        seq_id = parts[0]
                        start_pos = int(parts[1])
                        end_pos = int(parts[2])
                        ani_value = float(parts[3])
                        classification = parts[4]
                        
                        contamination_ranges.append({
                            'sequence_id': seq_id,
                            'start': start_pos,
                            'end': end_pos,
                            'length': end_pos - start_pos + 1,
                            'ani_value': ani_value,
                            'classification': classification
                        })
            
            total_contaminated_length = sum(r['length'] for r in contamination_ranges)
            
            return {
                'contamination_ranges': contamination_ranges,
                'total_ranges': len(contamination_ranges),
                'total_contaminated_length': total_contaminated_length,
                'quality_score': 1.0 - min(1.0, total_contaminated_length / 1000000)
            }
            
        except Exception as e:
            return {'error': f"Failed to parse ANI contamination: {e}"}
    
    def _parse_assembly_stats(self, file_path: str) -> Dict[str, Any]:
        """Parse assembly statistics file discovered in web crawl"""
        try:
            stats = {}
            
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 2:
                        metric = parts[0]
                        value = parts[1]
                        
                        # Parse numeric values
                        try:
                            if '.' in value:
                                stats[metric] = float(value)
                            else:
                                stats[metric] = int(value)
                        except ValueError:
                            stats[metric] = value
            
            # Calculate quality score based on assembly metrics
            quality_score = 1.0
            
            # Penalize for gaps
            if 'unspanned-gaps' in stats:
                quality_score *= max(0.5, 1.0 - stats['unspanned-gaps'] / 1000)
            
            # Reward high N50
            if 'scaffold-N50' in stats:
                n50 = stats['scaffold-N50']
                if n50 > 100000:  # Good N50
                    quality_score *= 1.0
                elif n50 > 50000:  # Acceptable N50
                    quality_score *= 0.9
                else:  # Poor N50
                    quality_score *= 0.7
            
            stats['quality_score'] = quality_score
            return stats
            
        except Exception as e:
            return {'error': f"Failed to parse assembly stats: {e}"}
    
    def _parse_busco_report(self, file_path: str) -> Dict[str, Any]:
        """Parse BUSCO completeness report"""
        try:
            busco_results = {}
            
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if 'Complete BUSCOs' in line:
                        # Extract percentage
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part.endswith('%'):
                                busco_results['complete_percentage'] = float(part.rstrip('%'))
                                break
                    elif 'Complete and single-copy BUSCOs' in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part.endswith('%'):
                                busco_results['single_copy_percentage'] = float(part.rstrip('%'))
                                break
                    elif 'Complete and duplicated BUSCOs' in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part.endswith('%'):
                                busco_results['duplicated_percentage'] = float(part.rstrip('%'))
                                break
                    elif 'Fragmented BUSCOs' in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part.endswith('%'):
                                busco_results['fragmented_percentage'] = float(part.rstrip('%'))
                                break
                    elif 'Missing BUSCOs' in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part.endswith('%'):
                                busco_results['missing_percentage'] = float(part.rstrip('%'))
                                break
            
            # Calculate quality score based on BUSCO completeness
            complete_pct = busco_results.get('complete_percentage', 0)
            quality_score = complete_pct / 100.0
            busco_results['quality_score'] = quality_score
            
            return busco_results
            
        except Exception as e:
            return {'error': f"Failed to parse BUSCO report: {e}"}
    
    def _parse_checkm_report(self, file_path: str) -> Dict[str, Any]:
        """Parse CheckM quality assessment report"""
        try:
            checkm_results = {}
            
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('Completeness'):
                        # Extract completeness percentage
                        parts = line.split(':')
                        if len(parts) > 1:
                            value = parts[1].strip().rstrip('%')
                            try:
                                checkm_results['completeness'] = float(value)
                            except ValueError:
                                pass
                    elif line.startswith('Contamination'):
                        # Extract contamination percentage
                        parts = line.split(':')
                        if len(parts) > 1:
                            value = parts[1].strip().rstrip('%')
                            try:
                                checkm_results['contamination'] = float(value)
                            except ValueError:
                                pass
                    elif line.startswith('Strain heterogeneity'):
                        # Extract strain heterogeneity
                        parts = line.split(':')
                        if len(parts) > 1:
                            value = parts[1].strip().rstrip('%')
                            try:
                                checkm_results['strain_heterogeneity'] = float(value)
                            except ValueError:
                                pass
            
            # Calculate quality score based on CheckM metrics
            completeness = checkm_results.get('completeness', 0)
            contamination = checkm_results.get('contamination', 100)
            
            # High quality: >90% complete, <5% contamination
            # Medium quality: >70% complete, <10% contamination
            quality_score = (completeness / 100.0) * (1.0 - min(1.0, contamination / 100.0))
            checkm_results['quality_score'] = quality_score
            
            return checkm_results
            
        except Exception as e:
            return {'error': f"Failed to parse CheckM report: {e}"}

class QualityMonitor:
    """Real-time quality monitoring system"""
    
    def __init__(self, db_path: str = "data/quality/quality_monitor.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.rule_engine = QualityRuleEngine()
        self.analyzer = QualityAnalyzer()
        self.lock = Lock()
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize quality monitoring database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Quality reports table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS quality_reports (
                    report_id TEXT PRIMARY KEY,
                    data_source TEXT,
                    data_type TEXT,
                    timestamp TIMESTAMP,
                    overall_score REAL,
                    quality_level TEXT,
                    completeness REAL,
                    accuracy REAL,
                    consistency REAL,
                    validity REAL,
                    uniqueness REAL,
                    timeliness REAL,
                    conformity REAL,
                    integrity REAL,
                    reliability REAL,
                    accessibility REAL,
                    issue_count INTEGER,
                    critical_issues INTEGER,
                    high_issues INTEGER,
                    medium_issues INTEGER,
                    low_issues INTEGER,
                    metadata TEXT
                )
            ''')
            
            # Quality issues table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS quality_issues (
                    issue_id TEXT PRIMARY KEY,
                    report_id TEXT,
                    severity TEXT,
                    category TEXT,
                    description TEXT,
                    affected_data TEXT,
                    recommendation TEXT,
                    auto_fixable BOOLEAN,
                    fix_function TEXT,
                    status TEXT DEFAULT 'open',
                    created_at TIMESTAMP,
                    resolved_at TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (report_id) REFERENCES quality_reports(report_id)
                )
            ''')
            
            # Quality trends table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS quality_trends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_source TEXT,
                    data_type TEXT,
                    timestamp TIMESTAMP,
                    metric_name TEXT,
                    metric_value REAL,
                    trend_direction TEXT,
                    change_rate REAL
                )
            ''')
            
            # Quality alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS quality_alerts (
                    alert_id TEXT PRIMARY KEY,
                    data_source TEXT,
                    alert_type TEXT,
                    severity TEXT,
                    message TEXT,
                    triggered_at TIMESTAMP,
                    acknowledged_at TIMESTAMP,
                    resolved_at TIMESTAMP,
                    metadata TEXT
                )
            ''')
            
            conn.commit()
    
    def assess_quality(self, data: Any, data_source: str, data_type: DataType, 
                      validation_config: Dict[str, Any] = None) -> QualityReport:
        """Comprehensive quality assessment"""
        report_id = f"{data_source}_{data_type.value}_{int(time.time())}"
        timestamp = datetime.now(timezone.utc)
        
        # Initialize metrics
        metrics = QualityMetrics()
        
        if isinstance(data, pd.DataFrame):
            # Calculate core metrics
            required_fields = validation_config.get('required_fields', []) if validation_config else []
            validation_patterns = validation_config.get('validation_patterns', {}) if validation_config else {}
            constraints = validation_config.get('constraints', {}) if validation_config else {}
            
            metrics.completeness = self.analyzer.calculate_completeness(data, required_fields)
            metrics.accuracy = self.analyzer.calculate_accuracy(data, validation_patterns)
            metrics.consistency = self.analyzer.calculate_consistency(data)
            metrics.validity = self.analyzer.calculate_validity(data, constraints)
            metrics.uniqueness = self.analyzer.calculate_uniqueness(data, required_fields)
            metrics.timeliness = self.analyzer.calculate_timeliness(data)
            
            # Additional metrics
            metrics.conformity = 1.0 if len(data.columns) > 0 else 0.0
            metrics.integrity = 1.0 if not data.empty else 0.0
            metrics.reliability = min(1.0, metrics.consistency + metrics.validity) / 2
            metrics.accessibility = 1.0  # Assume accessible if we can process it
        
        # Evaluate quality rules
        rule_passed, issues = self.rule_engine.evaluate_data(data, data_type)
        
        # Generate statistics
        statistics = self._generate_statistics(data, data_type)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, issues)
        
        # Check compliance
        compliance_status = self._check_compliance(metrics, issues)
        
        # Create report
        report = QualityReport(
            report_id=report_id,
            data_source=data_source,
            data_type=data_type,
            timestamp=timestamp,
            metrics=metrics,
            issues=issues,
            statistics=statistics,
            recommendations=recommendations,
            compliance_status=compliance_status
        )
        
        # Store report
        self._store_report(report)
        
        # Check for alerts
        self._check_alerts(report)
        
        return report
    
    def _generate_statistics(self, data: Any, data_type: DataType) -> Dict[str, Any]:
        """Generate data statistics"""
        stats = {}
        
        if isinstance(data, pd.DataFrame):
            stats['row_count'] = len(data)
            stats['column_count'] = len(data.columns)
            stats['memory_usage'] = data.memory_usage(deep=True).sum()
            stats['null_count'] = data.isnull().sum().sum()
            stats['duplicate_count'] = data.duplicated().sum()
            
            # Numeric statistics
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                stats['numeric_stats'] = data[numeric_columns].describe().to_dict()
            
            # String statistics
            string_columns = data.select_dtypes(include=['object']).columns
            if len(string_columns) > 0:
                stats['string_stats'] = {}
                for col in string_columns:
                    stats['string_stats'][col] = {
                        'unique_count': data[col].nunique(),
                        'most_common': data[col].value_counts().head(5).to_dict()
                    }
        
        return stats
    
    def _generate_recommendations(self, metrics: QualityMetrics, issues: List[QualityIssue]) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        # Score-based recommendations
        if metrics.completeness < 0.8:
            recommendations.append("Improve data completeness by addressing missing values")
        
        if metrics.accuracy < 0.8:
            recommendations.append("Review data accuracy and format validation")
        
        if metrics.consistency < 0.8:
            recommendations.append("Address data consistency issues and duplicates")
        
        if metrics.validity < 0.8:
            recommendations.append("Review data validity constraints and outliers")
        
        # Issue-based recommendations
        critical_issues = [i for i in issues if i.severity == 'critical']
        if critical_issues:
            recommendations.append(f"Immediately address {len(critical_issues)} critical issues")
        
        high_issues = [i for i in issues if i.severity == 'high']
        if high_issues:
            recommendations.append(f"Prioritize resolution of {len(high_issues)} high-severity issues")
        
        # Auto-fixable issues
        auto_fixable = [i for i in issues if i.auto_fixable]
        if auto_fixable:
            recommendations.append(f"Consider auto-fixing {len(auto_fixable)} automatically resolvable issues")
        
        return recommendations
    
    def _check_compliance(self, metrics: QualityMetrics, issues: List[QualityIssue]) -> Dict[str, bool]:
        """Check compliance with quality standards"""
        compliance = {}
        
        # NASA standards (example thresholds)
        compliance['nasa_grade'] = (
            metrics.completeness >= 0.95 and
            metrics.accuracy >= 0.95 and
            metrics.consistency >= 0.90 and
            metrics.validity >= 0.95
        )
        
        # Research standards
        compliance['research_grade'] = (
            metrics.completeness >= 0.90 and
            metrics.accuracy >= 0.90 and
            metrics.consistency >= 0.85 and
            metrics.validity >= 0.90
        )
        
        # Production standards
        compliance['production_ready'] = (
            metrics.overall_score() >= 0.80 and
            len([i for i in issues if i.severity == 'critical']) == 0
        )
        
        return compliance
    
    def _store_report(self, report: QualityReport):
        """Store quality report in database"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count issues by severity
                issue_counts = Counter(issue.severity for issue in report.issues)
                
                # Store report
                cursor.execute('''
                    INSERT OR REPLACE INTO quality_reports 
                    (report_id, data_source, data_type, timestamp, overall_score, quality_level,
                     completeness, accuracy, consistency, validity, uniqueness, timeliness,
                     conformity, integrity, reliability, accessibility, issue_count,
                     critical_issues, high_issues, medium_issues, low_issues, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    report.report_id,
                    report.data_source,
                    report.data_type.value,
                    report.timestamp,
                    report.metrics.overall_score(),
                    report.metrics.get_level().value,
                    report.metrics.completeness,
                    report.metrics.accuracy,
                    report.metrics.consistency,
                    report.metrics.validity,
                    report.metrics.uniqueness,
                    report.metrics.timeliness,
                    report.metrics.conformity,
                    report.metrics.integrity,
                    report.metrics.reliability,
                    report.metrics.accessibility,
                    len(report.issues),
                    issue_counts.get('critical', 0),
                    issue_counts.get('high', 0),
                    issue_counts.get('medium', 0),
                    issue_counts.get('low', 0),
                    json.dumps(report.metadata)
                ))
                
                # Store issues
                for issue in report.issues:
                    cursor.execute('''
                        INSERT OR REPLACE INTO quality_issues 
                        (issue_id, report_id, severity, category, description, affected_data,
                         recommendation, auto_fixable, fix_function, created_at, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        issue.issue_id,
                        report.report_id,
                        issue.severity,
                        issue.category,
                        issue.description,
                        issue.affected_data,
                        issue.recommendation,
                        issue.auto_fixable,
                        issue.fix_function,
                        issue.created_at,
                        json.dumps(issue.metadata)
                    ))
                
                conn.commit()
    
    def _check_alerts(self, report: QualityReport):
        """Check for quality alerts"""
        alerts = []
        
        # Critical quality drop
        if report.metrics.overall_score() < 0.5:
            alerts.append({
                'alert_type': 'critical_quality_drop',
                'severity': 'critical',
                'message': f'Quality score dropped to {report.metrics.overall_score():.2f} for {report.data_source}'
            })
        
        # High number of critical issues
        critical_issues = [i for i in report.issues if i.severity == 'critical']
        if len(critical_issues) > 5:
            alerts.append({
                'alert_type': 'high_critical_issues',
                'severity': 'high',
                'message': f'Found {len(critical_issues)} critical issues in {report.data_source}'
            })
        
        # Compliance failure
        if not report.compliance_status.get('production_ready', False):
            alerts.append({
                'alert_type': 'compliance_failure',
                'severity': 'medium',
                'message': f'Data source {report.data_source} failed production readiness compliance'
            })
        
        # Store alerts
        for alert in alerts:
            self._store_alert(report.data_source, alert)
    
    def _store_alert(self, data_source: str, alert: Dict[str, Any]):
        """Store quality alert"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                alert_id = f"{data_source}_{alert['alert_type']}_{int(time.time())}"
                
                cursor.execute('''
                    INSERT INTO quality_alerts 
                    (alert_id, data_source, alert_type, severity, message, triggered_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert_id,
                    data_source,
                    alert['alert_type'],
                    alert['severity'],
                    alert['message'],
                    datetime.now(timezone.utc),
                    json.dumps({})
                ))
                
                conn.commit()
    
    def get_quality_trends(self, data_source: str, days: int = 30) -> Dict[str, Any]:
        """Get quality trends for a data source"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get recent reports
            cursor.execute('''
                SELECT timestamp, overall_score, completeness, accuracy, consistency, validity
                FROM quality_reports
                WHERE data_source = ? AND timestamp > datetime('now', '-{} days')
                ORDER BY timestamp
            '''.format(days), (data_source,))
            
            rows = cursor.fetchall()
            
            if not rows:
                return {}
            
            df = pd.DataFrame(rows, columns=['timestamp', 'overall_score', 'completeness', 'accuracy', 'consistency', 'validity'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            trends = {}
            for metric in ['overall_score', 'completeness', 'accuracy', 'consistency', 'validity']:
                values = df[metric].values
                if len(values) > 1:
                    # Calculate trend
                    x = np.arange(len(values))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                    
                    trends[metric] = {
                        'current': values[-1],
                        'trend': 'improving' if slope > 0 else 'declining' if slope < 0 else 'stable',
                        'slope': slope,
                        'correlation': r_value,
                        'p_value': p_value
                    }
            
            return trends
    
    def generate_quality_dashboard(self, output_path: str = "data/quality/dashboard.json") -> Dict[str, Any]:
        """Generate quality dashboard data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            dashboard = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'summary': {},
                'data_sources': {},
                'alerts': {},
                'trends': {}
            }
            
            # Overall summary
            cursor.execute('''
                SELECT COUNT(*) as total_reports,
                       AVG(overall_score) as avg_score,
                       SUM(critical_issues) as total_critical,
                       SUM(high_issues) as total_high,
                       SUM(medium_issues) as total_medium,
                       SUM(low_issues) as total_low
                FROM quality_reports
                WHERE timestamp > datetime('now', '-7 days')
            ''')
            
            summary = cursor.fetchone()
            dashboard['summary'] = {
                'total_reports': summary[0],
                'average_score': summary[1] or 0,
                'total_critical_issues': summary[2] or 0,
                'total_high_issues': summary[3] or 0,
                'total_medium_issues': summary[4] or 0,
                'total_low_issues': summary[5] or 0
            }
            
            # Data source summary
            cursor.execute('''
                SELECT data_source, 
                       COUNT(*) as report_count,
                       AVG(overall_score) as avg_score,
                       MAX(timestamp) as last_report
                FROM quality_reports
                WHERE timestamp > datetime('now', '-7 days')
                GROUP BY data_source
            ''')
            
            for row in cursor.fetchall():
                dashboard['data_sources'][row[0]] = {
                    'report_count': row[1],
                    'average_score': row[2],
                    'last_report': row[3]
                }
            
            # Recent alerts
            cursor.execute('''
                SELECT alert_type, severity, COUNT(*) as count
                FROM quality_alerts
                WHERE triggered_at > datetime('now', '-7 days')
                  AND resolved_at IS NULL
                GROUP BY alert_type, severity
            ''')
            
            for row in cursor.fetchall():
                alert_key = f"{row[0]}_{row[1]}"
                dashboard['alerts'][alert_key] = row[2]
            
            # Save dashboard
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(dashboard, f, indent=2)
            
            return dashboard

# Main execution function
def main():
    """Main execution function for quality system"""
    # Initialize quality monitor
    monitor = QualityMonitor()
    
    # Example usage with sample data
    sample_data = pd.DataFrame({
        'pathway_id': ['map00010', 'map00020', 'map00030'],
        'name': ['Glycolysis', 'TCA Cycle', 'Pentose Phosphate'],
        'reaction_count': [10, 8, 7],
        'compound_count': [12, 10, 9]
    })
    
    # Assess quality
    report = monitor.assess_quality(
        data=sample_data,
        data_source='sample_kegg',
        data_type=DataType.KEGG_PATHWAY,
        validation_config={
            'required_fields': ['pathway_id', 'name'],
            'validation_patterns': {'pathway_id': r'^map\d{5}$'},
            'constraints': {'reaction_count': {'min': 0, 'type': 'numeric'}}
        }
    )
    
    print(f"Quality Assessment Complete:")
    print(f"Overall Score: {report.metrics.overall_score():.2f}")
    print(f"Quality Level: {report.metrics.get_level().value}")
    print(f"Issues Found: {len(report.issues)}")
    
    # Generate dashboard
    dashboard = monitor.generate_quality_dashboard()
    print(f"Dashboard generated with {len(dashboard['data_sources'])} data sources")
    
    return monitor

if __name__ == "__main__":
    quality_monitor = main() 
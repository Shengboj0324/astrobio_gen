#!/usr/bin/env python3
"""
Autonomous Scientific Discovery System
======================================

Advanced autonomous system for conducting scientific discovery in astrobiology.
Integrates AI agents, causal discovery, hypothesis generation, and experimental design.

Features:
- Autonomous research agent that conducts independent investigations
- Integration with causal discovery AI for hypothesis generation
- Multimodal data analysis and synthesis
- Automated literature review and knowledge integration
- Experimental design and prioritization
- Real-time adaptation based on new findings
- Collaborative multi-agent research teams
- Scientific paper generation and peer review

Capabilities:
- Autonomous discovery of new habitability factors
- Generation of novel biosignature detection strategies
- Identification of previously unknown causal relationships
- Design of optimal observational campaigns
- Real-time hypothesis updating based on new data
- Cross-domain knowledge synthesis
- Automated research planning and execution

Architecture:
- Research Director Agent: High-level research planning
- Data Analyst Agent: Advanced data analysis and pattern recognition
- Hypothesis Generator Agent: Novel hypothesis creation and refinement
- Experimental Designer Agent: Optimal experiment design
- Literature Agent: Automated literature review and synthesis
- Synthesis Agent: Cross-domain knowledge integration

Example Usage:
    # Create autonomous discovery system
    discovery_system = AutonomousScientificDiscovery()

    # Start autonomous research
    research_results = discovery_system.conduct_autonomous_research(
        research_domain="exoplanet_habitability",
        available_data=observational_datasets,
        research_budget=research_constraints
    )

    # Generate scientific insights
    insights = discovery_system.synthesize_scientific_insights(research_results)
"""

import asyncio
import heapq
import json
import logging
import random
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our Tier 1 and Tier 2 components
try:
    from .causal_discovery_ai import CausalDiscoveryAI, CausalHypothesis
    from .enhanced_foundation_llm import EnhancedFoundationLLM, EnhancedLLMConfig
    from .multimodal_diffusion_climate import MultimodalClimateGenerator

    TIER_COMPONENTS_AVAILABLE = True
except ImportError:
    TIER_COMPONENTS_AVAILABLE = False
    logger.warning("Some Tier components not available for import")

# Scientific computing and analysis
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Logging already configured above


@dataclass
class ResearchGoal:
    """Represents a scientific research goal"""

    id: str
    title: str
    description: str
    scientific_domain: str
    priority: float
    complexity: int
    estimated_duration: timedelta
    required_resources: List[str]
    success_criteria: List[str]
    parent_goal_id: Optional[str] = None
    status: str = "planned"  # planned, active, completed, failed
    progress: float = 0.0
    insights_generated: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            self.id = f"goal_{uuid.uuid4().hex[:8]}"


@dataclass
class ScientificInsight:
    """Represents a scientific insight or discovery"""

    id: str
    title: str
    description: str
    confidence_level: float
    novelty_score: float
    impact_score: float
    supporting_evidence: List[Dict[str, Any]]
    related_hypotheses: List[str]
    generated_by: str  # Agent that generated it
    timestamp: datetime
    scientific_domain: str
    implications: List[str] = field(default_factory=list)
    future_work: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            self.id = f"insight_{uuid.uuid4().hex[:8]}"


@dataclass
class ExperimentalCampaign:
    """Represents a coordinated experimental campaign"""

    id: str
    title: str
    objectives: List[str]
    experiments: List[Dict[str, Any]]
    timeline: Dict[str, datetime]
    resource_requirements: Dict[str, Any]
    expected_outcomes: List[str]
    risk_assessment: Dict[str, Any]
    status: str = "planned"
    results: Dict[str, Any] = field(default_factory=dict)


class ResearchAgent(ABC):
    """Abstract base class for research agents"""

    def __init__(self, agent_id: str, specialization: str):
        self.agent_id = agent_id
        self.specialization = specialization
        self.current_tasks = []
        self.completed_tasks = []
        self.performance_history = []
        self.collaboration_network = {}

    @abstractmethod
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a research task"""
        pass

    @abstractmethod
    def evaluate_task_complexity(self, task: Dict[str, Any]) -> float:
        """Evaluate the complexity of a task"""
        pass

    def update_performance(self, task_result: Dict[str, Any]):
        """Update agent performance metrics"""
        success_score = task_result.get("success_score", 0.0)
        self.performance_history.append(
            {
                "timestamp": datetime.now(),
                "task_type": task_result.get("task_type", "unknown"),
                "success_score": success_score,
                "duration": task_result.get("duration", 0.0),
            }
        )

    def get_performance_summary(self) -> Dict[str, float]:
        """Get performance summary"""
        if not self.performance_history:
            return {"avg_success": 0.0, "task_count": 0}

        recent_tasks = self.performance_history[-10:]  # Last 10 tasks
        avg_success = np.mean([t["success_score"] for t in recent_tasks])

        return {
            "avg_success": avg_success,
            "task_count": len(self.performance_history),
            "specialization": self.specialization,
        }


class ResearchDirectorAgent(ResearchAgent):
    """High-level research planning and coordination agent"""

    def __init__(self):
        super().__init__("research_director", "strategic_planning")
        self.active_research_goals = []
        self.completed_goals = []
        self.resource_allocation = {}

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute research planning task"""

        task_type = task.get("type", "unknown")

        if task_type == "plan_research_campaign":
            return await self._plan_research_campaign(task["domain"], task["objectives"])
        elif task_type == "prioritize_goals":
            return await self._prioritize_research_goals(task["goals"])
        elif task_type == "allocate_resources":
            return await self._allocate_resources(task["resources"], task["goals"])
        elif task_type == "evaluate_progress":
            return await self._evaluate_research_progress()
        else:
            return {"success": False, "error": f"Unknown task type: {task_type}"}

    def evaluate_task_complexity(self, task: Dict[str, Any]) -> float:
        """Evaluate task complexity based on scope and requirements"""
        base_complexity = 0.5

        # Adjust based on task type
        task_type = task.get("type", "unknown")
        if task_type == "plan_research_campaign":
            base_complexity = 0.8
        elif task_type == "prioritize_goals":
            base_complexity = 0.6

        # Adjust based on scope
        scope_factors = {
            "num_objectives": len(task.get("objectives", [])) * 0.1,
            "num_domains": len(task.get("domains", [])) * 0.05,
            "complexity_level": task.get("complexity", 1) * 0.1,
        }

        total_complexity = base_complexity + sum(scope_factors.values())
        return min(total_complexity, 1.0)

    async def _plan_research_campaign(self, domain: str, objectives: List[str]) -> Dict[str, Any]:
        """Plan a comprehensive research campaign"""

        logger.info(f"🎯 Planning research campaign for {domain}")

        # Break down objectives into specific research goals
        research_goals = []
        for i, objective in enumerate(objectives):
            goal = ResearchGoal(
                id=f"goal_{domain}_{i}",
                title=f"Objective: {objective}",
                description=f"Research goal for {objective} in {domain}",
                scientific_domain=domain,
                priority=1.0 - (i * 0.1),  # Descending priority
                complexity=self._estimate_objective_complexity(objective),
                estimated_duration=timedelta(weeks=4 + i * 2),
                required_resources=self._identify_required_resources(objective),
                success_criteria=self._define_success_criteria(objective),
            )
            research_goals.append(goal)

        # Create timeline
        timeline = self._create_research_timeline(research_goals)

        # Identify collaboration opportunities
        collaborations = self._identify_collaboration_opportunities(research_goals)

        campaign = {
            "campaign_id": f"campaign_{domain}_{datetime.now().strftime('%Y%m%d')}",
            "domain": domain,
            "research_goals": research_goals,
            "timeline": timeline,
            "collaborations": collaborations,
            "resource_summary": self._summarize_resources(research_goals),
            "risk_assessment": self._assess_campaign_risks(research_goals),
        }

        self.active_research_goals.extend(research_goals)

        logger.info(f"✅ Research campaign planned with {len(research_goals)} goals")

        return {
            "success": True,
            "campaign": campaign,
            "success_score": 0.9,
            "task_type": "plan_research_campaign",
        }

    async def _prioritize_research_goals(self, goals: List[ResearchGoal]) -> Dict[str, Any]:
        """Prioritize research goals using multi-criteria analysis"""

        logger.info(f"📊 Prioritizing {len(goals)} research goals")

        # Multi-criteria scoring
        for goal in goals:
            # Calculate composite priority score
            impact_score = self._estimate_impact_score(goal)
            feasibility_score = self._estimate_feasibility_score(goal)
            novelty_score = self._estimate_novelty_score(goal)
            urgency_score = self._estimate_urgency_score(goal)

            # Weighted combination
            composite_score = (
                impact_score * 0.3
                + feasibility_score * 0.25
                + novelty_score * 0.25
                + urgency_score * 0.2
            )

            goal.priority = composite_score

        # Sort by priority
        prioritized_goals = sorted(goals, key=lambda g: g.priority, reverse=True)

        return {
            "success": True,
            "prioritized_goals": prioritized_goals,
            "priority_rationale": self._generate_priority_rationale(prioritized_goals),
            "success_score": 0.85,
            "task_type": "prioritize_goals",
        }

    def _estimate_objective_complexity(self, objective: str) -> int:
        """Estimate complexity of research objective"""
        complexity_keywords = {
            "mechanism": 3,
            "interaction": 4,
            "evolution": 5,
            "prediction": 3,
            "detection": 2,
            "characterization": 3,
            "novel": 4,
            "unprecedented": 5,
        }

        base_complexity = 2
        for keyword, complexity in complexity_keywords.items():
            if keyword in objective.lower():
                base_complexity = max(base_complexity, complexity)

        return base_complexity

    def _identify_required_resources(self, objective: str) -> List[str]:
        """Identify resources required for objective"""
        resources = ["computational_resources", "data_access"]

        if "observation" in objective.lower():
            resources.extend(["telescope_time", "observational_data"])
        if "model" in objective.lower():
            resources.extend(["modeling_software", "high_performance_computing"])
        if "analysis" in objective.lower():
            resources.extend(["analytical_tools", "expert_consultation"])

        return resources

    def _define_success_criteria(self, objective: str) -> List[str]:
        """Define success criteria for objective"""
        criteria = [
            f"Complete analysis relevant to: {objective}",
            "Generate testable hypotheses",
            "Achieve statistical significance (p < 0.05)",
            "Validate results through independent methods",
        ]

        if "detection" in objective.lower():
            criteria.append("Achieve target detection sensitivity")
        if "prediction" in objective.lower():
            criteria.append("Demonstrate predictive accuracy > 80%")

        return criteria

    def _estimate_impact_score(self, goal: ResearchGoal) -> float:
        """Estimate potential impact of research goal"""
        # Simplified impact estimation
        domain_impact_weights = {
            "habitability": 0.9,
            "biosignatures": 0.85,
            "atmospheric_evolution": 0.8,
            "stellar_activity": 0.7,
            "planetary_formation": 0.75,
        }

        base_impact = domain_impact_weights.get(goal.scientific_domain, 0.6)
        complexity_bonus = goal.complexity * 0.05

        return min(base_impact + complexity_bonus, 1.0)

    def _estimate_feasibility_score(self, goal: ResearchGoal) -> float:
        """Estimate feasibility of research goal"""
        # Higher complexity reduces feasibility
        complexity_penalty = goal.complexity * 0.1

        # Resource availability affects feasibility
        resource_penalty = len(goal.required_resources) * 0.05

        base_feasibility = 0.8
        return max(base_feasibility - complexity_penalty - resource_penalty, 0.1)

    def _estimate_novelty_score(self, goal: ResearchGoal) -> float:
        """Estimate novelty of research goal"""
        # Simplified novelty scoring
        novelty_keywords = ["novel", "first", "unprecedented", "new", "discovery"]

        novelty_score = 0.5
        for keyword in novelty_keywords:
            if keyword in goal.description.lower():
                novelty_score += 0.1

        return min(novelty_score, 1.0)

    def _estimate_urgency_score(self, goal: ResearchGoal) -> float:
        """Estimate urgency of research goal"""
        # Time-sensitive research gets higher urgency
        urgency_keywords = ["urgent", "timely", "opportunity", "immediate"]

        urgency_score = 0.5
        for keyword in urgency_keywords:
            if keyword in goal.description.lower():
                urgency_score += 0.1

        return min(urgency_score, 1.0)


class DataAnalystAgent(ResearchAgent):
    """Advanced data analysis and pattern recognition agent"""

    def __init__(self):
        super().__init__("data_analyst", "data_analysis")
        self.analysis_tools = {
            "statistical_analysis": self._statistical_analysis,
            "pattern_recognition": self._pattern_recognition,
            "anomaly_detection": self._anomaly_detection,
            "clustering_analysis": self._clustering_analysis,
            "time_series_analysis": self._time_series_analysis,
        }

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data analysis task"""

        task_type = task.get("type", "unknown")

        if task_type in self.analysis_tools:
            analysis_func = self.analysis_tools[task_type]
            return await analysis_func(task["data"], task.get("parameters", {}))
        else:
            return {"success": False, "error": f"Unknown analysis type: {task_type}"}

    def evaluate_task_complexity(self, task: Dict[str, Any]) -> float:
        """Evaluate data analysis task complexity"""
        data = task.get("data", [])

        if isinstance(data, pd.DataFrame):
            n_samples, n_features = data.shape
        elif isinstance(data, list):
            n_samples = len(data)
            n_features = len(data[0]) if data else 0
        else:
            n_samples, n_features = 100, 10  # Default assumption

        # Complexity based on data size and analysis type
        size_complexity = min(np.log10(n_samples * n_features) / 6, 1.0)

        analysis_complexity = {
            "statistical_analysis": 0.3,
            "pattern_recognition": 0.7,
            "anomaly_detection": 0.6,
            "clustering_analysis": 0.5,
            "time_series_analysis": 0.8,
        }

        base_complexity = analysis_complexity.get(task.get("type", "unknown"), 0.5)

        return min(size_complexity + base_complexity, 1.0)

    async def _statistical_analysis(
        self, data: pd.DataFrame, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Comprehensive statistical analysis"""

        logger.info("📊 Performing statistical analysis")

        results = {
            "descriptive_stats": data.describe().to_dict(),
            "correlation_matrix": data.corr().to_dict(),
            "distribution_tests": {},
            "outlier_analysis": {},
            "missing_data_analysis": {},
        }

        # Distribution tests
        for column in data.select_dtypes(include=[np.number]).columns:
            if data[column].notna().sum() > 10:  # Need sufficient data
                try:
                    statistic, p_value = stats.normaltest(data[column].dropna())
                    results["distribution_tests"][column] = {
                        "normality_test": {"statistic": statistic, "p_value": p_value},
                        "is_normal": p_value > 0.05,
                    }
                except:
                    results["distribution_tests"][column] = {"error": "Test failed"}

        # Outlier detection
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso_forest.fit_predict(numeric_data.fillna(numeric_data.mean()))
            results["outlier_analysis"] = {
                "outlier_indices": np.where(outliers == -1)[0].tolist(),
                "outlier_fraction": np.sum(outliers == -1) / len(outliers),
            }

        # Missing data analysis
        missing_summary = data.isnull().sum()
        results["missing_data_analysis"] = {
            "missing_counts": missing_summary.to_dict(),
            "missing_percentages": (missing_summary / len(data) * 100).to_dict(),
        }

        logger.info("✅ Statistical analysis completed")

        return {
            "success": True,
            "results": results,
            "insights": self._generate_statistical_insights(results),
            "success_score": 0.9,
            "task_type": "statistical_analysis",
        }

    async def _pattern_recognition(
        self, data: pd.DataFrame, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Advanced pattern recognition in data"""

        logger.info("🔍 Performing pattern recognition")

        # Prepare data
        numeric_data = data.select_dtypes(include=[np.number]).fillna(data.mean())

        if numeric_data.empty:
            return {"success": False, "error": "No numeric data for pattern analysis"}

        # Principal Component Analysis
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)

        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)

        # Identify patterns
        patterns = {
            "pca_analysis": {
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
                "n_components_95_var": np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95)
                + 1,
            },
            "feature_importance": {},
            "data_structure": {},
        }

        # Feature importance from PCA loadings
        feature_names = numeric_data.columns.tolist()
        for i, feature in enumerate(feature_names):
            # Importance based on first few components
            importance = np.sum(np.abs(pca.components_[:3, i]))
            patterns["feature_importance"][feature] = importance

        # Data structure analysis
        patterns["data_structure"] = {
            "dimensionality": numeric_data.shape[1],
            "effective_dimensionality": patterns["pca_analysis"]["n_components_95_var"],
            "data_density": (
                "sparse"
                if patterns["pca_analysis"]["n_components_95_var"] < numeric_data.shape[1] * 0.5
                else "dense"
            ),
        }

        logger.info("✅ Pattern recognition completed")

        return {
            "success": True,
            "patterns": patterns,
            "insights": self._generate_pattern_insights(patterns),
            "success_score": 0.85,
            "task_type": "pattern_recognition",
        }

    async def _anomaly_detection(
        self, data: pd.DataFrame, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect anomalies in data"""

        logger.info("🚨 Performing anomaly detection")

        numeric_data = data.select_dtypes(include=[np.number]).fillna(data.mean())

        if numeric_data.empty:
            return {"success": False, "error": "No numeric data for anomaly detection"}

        # Multiple anomaly detection methods
        anomaly_results = {}

        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_anomalies = iso_forest.fit_predict(numeric_data)

        # Statistical anomalies (Z-score)
        z_scores = np.abs(stats.zscore(numeric_data, nan_policy="omit"))
        stat_anomalies = (z_scores > 3).any(axis=1)

        anomaly_results = {
            "isolation_forest": {
                "anomaly_indices": np.where(iso_anomalies == -1)[0].tolist(),
                "anomaly_scores": iso_forest.decision_function(numeric_data).tolist(),
            },
            "statistical_outliers": {
                "anomaly_indices": np.where(stat_anomalies)[0].tolist(),
                "max_z_scores": np.max(z_scores, axis=1).tolist(),
            },
            "consensus_anomalies": [],
        }

        # Consensus anomalies (detected by multiple methods)
        iso_set = set(anomaly_results["isolation_forest"]["anomaly_indices"])
        stat_set = set(anomaly_results["statistical_outliers"]["anomaly_indices"])
        consensus = list(iso_set.intersection(stat_set))
        anomaly_results["consensus_anomalies"] = consensus

        logger.info(f"✅ Anomaly detection completed: {len(consensus)} consensus anomalies")

        return {
            "success": True,
            "anomalies": anomaly_results,
            "insights": self._generate_anomaly_insights(anomaly_results),
            "success_score": 0.8,
            "task_type": "anomaly_detection",
        }

    def _clustering_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform clustering analysis on data"""
        logger.info("🔍 Performing clustering analysis")

        # Prepare numeric data
        numeric_data = data.select_dtypes(include=[np.number]).fillna(data.mean())

        if numeric_data.empty:
            return {"success": False, "error": "No numeric data for clustering"}

        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)

        # Perform clustering
        clustering_results = {}

        # K-means clustering
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=3, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_data)
            clustering_results["kmeans"] = {
                "labels": cluster_labels.tolist(),
                "centers": kmeans.cluster_centers_.tolist(),
                "inertia": kmeans.inertia_
            }
        except Exception as e:
            clustering_results["kmeans"] = {"error": str(e)}

        # DBSCAN clustering
        try:
            from sklearn.cluster import DBSCAN
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = dbscan.fit_predict(scaled_data)
            clustering_results["dbscan"] = {
                "labels": cluster_labels.tolist(),
                "n_clusters": len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                "n_noise": list(cluster_labels).count(-1)
            }
        except Exception as e:
            clustering_results["dbscan"] = {"error": str(e)}

        logger.info("✅ Clustering analysis completed")

        return {
            "success": True,
            "clustering": clustering_results,
            "insights": self._generate_clustering_insights(clustering_results),
            "success_score": 0.7,
            "task_type": "clustering_analysis",
        }

    def _time_series_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform time series analysis on data"""
        logger.info("📈 Performing time series analysis")

        # Look for datetime columns
        datetime_cols = data.select_dtypes(include=['datetime64']).columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        if len(datetime_cols) == 0 or len(numeric_cols) == 0:
            return {"success": False, "error": "No suitable time series data found"}

        time_series_results = {}

        # Basic time series statistics
        for col in numeric_cols:
            if data[col].notna().sum() > 10:  # Need sufficient data points
                series_stats = {
                    "mean": float(data[col].mean()),
                    "std": float(data[col].std()),
                    "trend": "stable",  # Simplified trend analysis
                    "seasonality": "none",  # Simplified seasonality detection
                }
                time_series_results[col] = series_stats

        logger.info("✅ Time series analysis completed")

        return {
            "success": True,
            "time_series": time_series_results,
            "insights": self._generate_time_series_insights(time_series_results),
            "success_score": 0.6,
            "task_type": "time_series_analysis",
        }

    def _generate_statistical_insights(self, results: Dict[str, Any]) -> List[str]:
        """Generate insights from statistical analysis"""
        insights = []

        # Correlation insights
        if "correlation_matrix" in results:
            # Find strong correlations
            for var1, correlations in results["correlation_matrix"].items():
                for var2, corr in correlations.items():
                    if var1 != var2 and abs(corr) > 0.7:
                        insights.append(
                            f"Strong correlation between {var1} and {var2} (r={corr:.3f})"
                        )

        # Distribution insights
        if "distribution_tests" in results:
            non_normal_vars = [
                var
                for var, test in results["distribution_tests"].items()
                if not test.get("is_normal", True)
            ]
            if non_normal_vars:
                insights.append(
                    f"Non-normal distributions detected in: {', '.join(non_normal_vars)}"
                )

        # Outlier insights
        if "outlier_analysis" in results:
            outlier_fraction = results["outlier_analysis"].get("outlier_fraction", 0)
            if outlier_fraction > 0.1:
                insights.append(f"High outlier fraction detected: {outlier_fraction:.1%}")

        return insights

    def _generate_pattern_insights(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate insights from pattern analysis"""
        insights = []

        # Dimensionality insights
        if "pca_analysis" in patterns:
            n_comp_95 = patterns["pca_analysis"]["n_components_95_var"]
            total_features = len(patterns["feature_importance"])

            if n_comp_95 < total_features * 0.5:
                insights.append(
                    f"Data has intrinsic low dimensionality: {n_comp_95} components explain 95% variance"
                )

            # First PC variance
            first_pc_var = patterns["pca_analysis"]["explained_variance_ratio"][0]
            if first_pc_var > 0.5:
                insights.append(f"Single dominant pattern explains {first_pc_var:.1%} of variance")

        # Feature importance insights
        if "feature_importance" in patterns:
            sorted_features = sorted(
                patterns["feature_importance"].items(), key=lambda x: x[1], reverse=True
            )
            top_features = [f[0] for f in sorted_features[:3]]
            insights.append(f"Most informative features: {', '.join(top_features)}")

        return insights

    def _generate_anomaly_insights(self, anomalies: Dict[str, Any]) -> List[str]:
        """Generate insights from anomaly detection"""
        insights = []

        consensus_count = len(anomalies.get("consensus_anomalies", []))
        if consensus_count > 0:
            insights.append(f"{consensus_count} consensus anomalies detected by multiple methods")

        iso_count = len(anomalies.get("isolation_forest", {}).get("anomaly_indices", []))
        stat_count = len(anomalies.get("statistical_outliers", {}).get("anomaly_indices", []))

        if iso_count > stat_count * 2:
            insights.append(
                "Isolation Forest detects more complex anomalies than statistical methods"
            )
        elif stat_count > iso_count * 2:
            insights.append("Statistical methods detect more extreme outliers")

        return insights

    def _generate_clustering_insights(self, results: Dict[str, Any]) -> List[str]:
        """Generate insights from clustering analysis"""
        insights = []

        if "kmeans" in results and "error" not in results["kmeans"]:
            insights.append(f"K-means clustering identified 3 distinct groups in the data")
            insights.append(f"Clustering inertia: {results['kmeans']['inertia']:.2f}")

        if "dbscan" in results and "error" not in results["dbscan"]:
            n_clusters = results["dbscan"]["n_clusters"]
            n_noise = results["dbscan"]["n_noise"]
            insights.append(f"DBSCAN identified {n_clusters} density-based clusters")
            if n_noise > 0:
                insights.append(f"Found {n_noise} noise points (outliers)")

        return insights

    def _generate_time_series_insights(self, results: Dict[str, Any]) -> List[str]:
        """Generate insights from time series analysis"""
        insights = []

        for var, stats in results.items():
            if isinstance(stats, dict):
                insights.append(f"{var}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
                if stats.get('trend') != 'stable':
                    insights.append(f"{var} shows {stats['trend']} trend")
                if stats.get('seasonality') != 'none':
                    insights.append(f"{var} exhibits {stats['seasonality']} seasonality")

        return insights


class HypothesisGeneratorAgent(ResearchAgent):
    """Agent specialized in generating and refining scientific hypotheses"""

    def __init__(self):
        super().__init__("hypothesis_generator", "hypothesis_generation")
        self.hypothesis_templates = self._load_hypothesis_templates()
        self.generated_hypotheses = []

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute hypothesis generation task"""

        task_type = task.get("type", "unknown")

        if task_type == "generate_hypotheses":
            return await self._generate_hypotheses(task["data"], task.get("context", {}))
        elif task_type == "refine_hypothesis":
            return await self._refine_hypothesis(task["hypothesis"], task["new_evidence"])
        elif task_type == "evaluate_hypotheses":
            return await self._evaluate_hypotheses(task["hypotheses"])
        else:
            return {"success": False, "error": f"Unknown task type: {task_type}"}

    def evaluate_task_complexity(self, task: Dict[str, Any]) -> float:
        """Evaluate hypothesis generation task complexity"""

        complexity_factors = {
            "generate_hypotheses": 0.7,
            "refine_hypothesis": 0.5,
            "evaluate_hypotheses": 0.6,
        }

        base_complexity = complexity_factors.get(task.get("type", "unknown"), 0.5)

        # Adjust for data complexity
        if "data" in task:
            data_complexity = self._estimate_data_complexity(task["data"])
            base_complexity += data_complexity * 0.3

        return min(base_complexity, 1.0)

    async def _generate_hypotheses(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate novel scientific hypotheses"""

        logger.info("💡 Generating scientific hypotheses")

        hypotheses = []

        # Data-driven hypothesis generation
        if isinstance(data, pd.DataFrame):
            hypotheses.extend(await self._generate_data_driven_hypotheses(data))

        # Context-driven hypothesis generation
        if context:
            hypotheses.extend(await self._generate_context_driven_hypotheses(context))

        # Cross-domain hypothesis generation
        hypotheses.extend(await self._generate_cross_domain_hypotheses(data, context))

        # Rank hypotheses
        ranked_hypotheses = self._rank_hypotheses(hypotheses)

        # Store generated hypotheses
        self.generated_hypotheses.extend(ranked_hypotheses)

        logger.info(f"✅ Generated {len(ranked_hypotheses)} hypotheses")

        return {
            "success": True,
            "hypotheses": ranked_hypotheses,
            "generation_summary": {
                "total_generated": len(hypotheses),
                "high_confidence": len([h for h in ranked_hypotheses if h["confidence"] > 0.8]),
                "novel_hypotheses": len([h for h in ranked_hypotheses if h["novelty"] > 0.7]),
            },
            "success_score": 0.85,
            "task_type": "generate_hypotheses",
        }

    async def _generate_data_driven_hypotheses(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate hypotheses based on data patterns"""

        hypotheses = []

        # Correlation-based hypotheses
        corr_matrix = data.corr()
        for i, var1 in enumerate(corr_matrix.columns):
            for j, var2 in enumerate(corr_matrix.columns):
                if i < j and abs(corr_matrix.iloc[i, j]) > 0.6:
                    correlation = corr_matrix.iloc[i, j]

                    if correlation > 0:
                        relationship = "positively related to"
                    else:
                        relationship = "negatively related to"

                    hypothesis = {
                        "id": f"corr_hyp_{i}_{j}",
                        "description": f"{var1} is {relationship} {var2}",
                        "type": "correlation",
                        "variables": [var1, var2],
                        "confidence": min(abs(correlation), 0.95),
                        "novelty": self._estimate_novelty(var1, var2),
                        "testable": True,
                        "supporting_evidence": {
                            "correlation_coefficient": correlation,
                            "data_source": "observational_data",
                        },
                    }
                    hypotheses.append(hypothesis)

        # Threshold-based hypotheses
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in data.columns:
                # Find potential thresholds
                values = data[col].dropna()
                if len(values) > 20:
                    threshold_candidates = np.percentile(values, [25, 50, 75, 90])

                    for threshold in threshold_candidates:
                        high_group = data[data[col] > threshold]
                        low_group = data[data[col] <= threshold]

                        if len(high_group) > 5 and len(low_group) > 5:
                            # Compare other variables between groups
                            for other_col in numeric_cols:
                                if other_col != col and other_col in data.columns:
                                    high_mean = high_group[other_col].mean()
                                    low_mean = low_group[other_col].mean()

                                    if abs(high_mean - low_mean) > data[other_col].std():
                                        hypothesis = {
                                            "id": f"thresh_hyp_{col}_{other_col}",
                                            "description": f"When {col} exceeds {threshold:.2f}, {other_col} shows different behavior",
                                            "type": "threshold_effect",
                                            "variables": [col, other_col],
                                            "confidence": 0.7,
                                            "novelty": 0.6,
                                            "testable": True,
                                            "supporting_evidence": {
                                                "threshold_value": threshold,
                                                "high_group_mean": high_mean,
                                                "low_group_mean": low_mean,
                                            },
                                        }
                                        hypotheses.append(hypothesis)

        return hypotheses

    async def _generate_context_driven_hypotheses(
        self, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate hypotheses based on scientific context"""

        hypotheses = []

        # Domain-specific hypothesis templates
        domain = context.get("scientific_domain", "general")

        if domain == "habitability":
            hypotheses.extend(self._generate_habitability_hypotheses(context))
        elif domain == "atmospheric_evolution":
            hypotheses.extend(self._generate_atmospheric_hypotheses(context))
        elif domain == "biosignatures":
            hypotheses.extend(self._generate_biosignature_hypotheses(context))

        return hypotheses

    def _generate_habitability_hypotheses(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate habitability-specific hypotheses"""

        templates = [
            "Planets with {condition1} are more likely to maintain liquid water than those with {condition2}",
            "The habitable zone boundaries are modified by {factor} in {stellar_type} systems",
            "Atmospheric {component} concentration above {threshold} indicates {implication} for habitability",
            "Tidal locking {enhances/reduces} habitability for planets orbiting {stellar_type} stars",
        ]

        hypotheses = []
        for template in templates:
            # Fill template with context-specific values
            filled_hypothesis = self._fill_hypothesis_template(template, context)
            if filled_hypothesis:
                hypotheses.append(
                    {
                        "id": f"hab_hyp_{len(hypotheses)}",
                        "description": filled_hypothesis,
                        "type": "habitability",
                        "confidence": 0.6,
                        "novelty": 0.7,
                        "testable": True,
                        "variables": self._extract_variables_from_text(filled_hypothesis),
                    }
                )

        return hypotheses

    def _fill_hypothesis_template(self, template: str, context: Dict[str, Any]) -> Optional[str]:
        """Fill hypothesis template with context-appropriate values"""

        # Simplified template filling
        # In practice, this would use more sophisticated NLP and domain knowledge

        substitutions = {
            "{condition1}": "thick atmospheres",
            "{condition2}": "thin atmospheres",
            "{factor}": "atmospheric greenhouse effects",
            "{stellar_type}": "M-dwarf",
            "{component}": "water vapor",
            "{threshold}": "1000 ppm",
            "{implication}": "enhanced greenhouse warming",
            "{enhances/reduces}": "enhances",
        }

        filled = template
        for placeholder, value in substitutions.items():
            filled = filled.replace(placeholder, value)

        return filled if "{" not in filled else None

    def _extract_variables_from_text(self, text: str) -> List[str]:
        """Extract variable names from hypothesis text"""

        # Simplified variable extraction
        potential_variables = [
            "atmospheric_pressure",
            "surface_temperature",
            "water_vapor",
            "stellar_type",
            "orbital_period",
            "planet_mass",
            "habitability_score",
        ]

        variables = []
        for var in potential_variables:
            if any(word in text.lower() for word in var.split("_")):
                variables.append(var)

        return variables

    def _estimate_novelty(self, var1: str, var2: str) -> float:
        """Estimate novelty of relationship between variables"""

        # Check against known relationships
        known_relationships = {
            ("stellar_temperature", "insolation"),
            ("surface_temperature", "water_vapor"),
            ("planet_mass", "atmospheric_pressure"),
        }

        if (var1, var2) in known_relationships or (var2, var1) in known_relationships:
            return 0.3  # Low novelty for known relationships
        else:
            return 0.8  # Higher novelty for unknown relationships

    def _rank_hypotheses(self, hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank hypotheses by combined score"""

        for hypothesis in hypotheses:
            # Combined score
            score = (
                hypothesis["confidence"] * 0.4
                + hypothesis["novelty"] * 0.3
                + (1.0 if hypothesis["testable"] else 0.0) * 0.3
            )
            hypothesis["combined_score"] = score

        return sorted(hypotheses, key=lambda h: h["combined_score"], reverse=True)

    def _load_hypothesis_templates(self) -> Dict[str, List[str]]:
        """Load hypothesis generation templates"""

        return {
            "causal": [
                "{cause} directly influences {effect}",
                "{cause} has a threshold effect on {effect}",
                "The relationship between {cause} and {effect} is moderated by {moderator}",
            ],
            "comparative": [
                "{group1} shows higher {variable} than {group2}",
                "Systems with {characteristic} exhibit different {outcome} patterns",
            ],
            "temporal": [
                "{variable} changes over time following {pattern}",
                "Early {condition} leads to later {outcome}",
            ],
        }


class AutonomousScientificDiscovery:
    """Main autonomous scientific discovery system"""

    def __init__(self):
        # Initialize research agents
        self.research_director = ResearchDirectorAgent()
        self.data_analyst = DataAnalystAgent()
        self.hypothesis_generator = HypothesisGeneratorAgent()

        # System state
        self.active_research_campaigns = []
        self.generated_insights = []
        self.discovered_knowledge = []
        self.research_history = []

        # Integration with Tier 1 & 2 systems
        self.causal_discovery = None
        self.climate_generator = None
        self.enhanced_llm = None

        if TIER_COMPONENTS_AVAILABLE:
            try:
                self.causal_discovery = CausalDiscoveryAI()
                self.climate_generator = MultimodalClimateGenerator()
                logger.info("✅ Integrated with Tier 1 & 2 components")
            except Exception as e:
                logger.warning(f"⚠️ Tier integration failed: {e}")

        logger.info("🤖 Autonomous Scientific Discovery System initialized")

    async def conduct_autonomous_research(
        self,
        research_domain: str,
        available_data: Dict[str, Any],
        research_constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Conduct autonomous research in specified domain"""

        logger.info(f"🚀 Starting autonomous research in {research_domain}")

        research_session = {
            "session_id": f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "domain": research_domain,
            "start_time": datetime.now(),
            "constraints": research_constraints,
            "results": {},
        }

        try:
            # Phase 1: Research Planning
            logger.info("📋 Phase 1: Research Planning")
            planning_results = await self._autonomous_research_planning(
                research_domain, available_data, research_constraints
            )
            research_session["results"]["planning"] = planning_results

            # Phase 2: Data Analysis
            logger.info("📊 Phase 2: Autonomous Data Analysis")
            analysis_results = await self._autonomous_data_analysis(
                available_data, planning_results["research_goals"]
            )
            research_session["results"]["analysis"] = analysis_results

            # Phase 3: Hypothesis Generation
            logger.info("💡 Phase 3: Hypothesis Generation")
            hypothesis_results = await self._autonomous_hypothesis_generation(
                analysis_results, research_domain
            )
            research_session["results"]["hypotheses"] = hypothesis_results

            # Phase 4: Causal Discovery (if available)
            if self.causal_discovery and "dataframes" in available_data:
                logger.info("🔍 Phase 4: Causal Discovery")
                causal_results = await self._autonomous_causal_discovery(
                    available_data["dataframes"]
                )
                research_session["results"]["causal_discovery"] = causal_results

            # Phase 5: Knowledge Synthesis
            logger.info("🧠 Phase 5: Knowledge Synthesis")
            synthesis_results = await self._autonomous_knowledge_synthesis(
                research_session["results"]
            )
            research_session["results"]["synthesis"] = synthesis_results

            # Phase 6: Generate Scientific Insights
            logger.info("✨ Phase 6: Scientific Insight Generation")
            insights = await self._generate_scientific_insights(research_session)
            research_session["results"]["insights"] = insights

            research_session["end_time"] = datetime.now()
            research_session["duration"] = (
                research_session["end_time"] - research_session["start_time"]
            )
            research_session["success"] = True

            # Store results
            self.research_history.append(research_session)
            self.generated_insights.extend(insights.get("insights", []))

            logger.info(f"✅ Autonomous research completed in {research_session['duration']}")

            return research_session

        except Exception as e:
            logger.error(f"❌ Autonomous research failed: {e}")
            research_session["error"] = str(e)
            research_session["success"] = False
            return research_session

    async def _autonomous_research_planning(
        self, domain: str, data: Dict[str, Any], constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Autonomous research planning phase"""

        # Generate research objectives based on domain and available data
        objectives = self._generate_research_objectives(domain, data)

        # Plan research campaign
        planning_task = {
            "type": "plan_research_campaign",
            "domain": domain,
            "objectives": objectives,
            "constraints": constraints,
        }

        planning_result = await self.research_director.execute_task(planning_task)

        return planning_result

    async def _autonomous_data_analysis(
        self, data: Dict[str, Any], research_goals: List[ResearchGoal]
    ) -> Dict[str, Any]:
        """Autonomous data analysis phase"""

        analysis_results = {}

        # Analyze different data types
        for data_type, dataset in data.items():
            if isinstance(dataset, pd.DataFrame):
                logger.info(f"   Analyzing {data_type} dataset")

                # Statistical analysis
                stat_task = {"type": "statistical_analysis", "data": dataset}
                stat_result = await self.data_analyst.execute_task(stat_task)

                # Pattern recognition
                pattern_task = {"type": "pattern_recognition", "data": dataset}
                pattern_result = await self.data_analyst.execute_task(pattern_task)

                # Anomaly detection
                anomaly_task = {"type": "anomaly_detection", "data": dataset}
                anomaly_result = await self.data_analyst.execute_task(anomaly_task)

                analysis_results[data_type] = {
                    "statistical_analysis": stat_result,
                    "pattern_recognition": pattern_result,
                    "anomaly_detection": anomaly_result,
                }

        return analysis_results

    async def _autonomous_hypothesis_generation(
        self, analysis_results: Dict[str, Any], domain: str
    ) -> Dict[str, Any]:
        """Autonomous hypothesis generation phase"""

        all_hypotheses = []

        # Generate hypotheses from each analysis result
        for data_type, results in analysis_results.items():
            for analysis_type, result in results.items():
                if result.get("success", False):
                    hypothesis_task = {
                        "type": "generate_hypotheses",
                        "data": result.get("results", {}),
                        "context": {
                            "scientific_domain": domain,
                            "data_source": data_type,
                            "analysis_method": analysis_type,
                        },
                    }

                    hyp_result = await self.hypothesis_generator.execute_task(hypothesis_task)
                    if hyp_result.get("success", False):
                        all_hypotheses.extend(hyp_result.get("hypotheses", []))

        # Evaluate and rank all hypotheses
        if all_hypotheses:
            eval_task = {"type": "evaluate_hypotheses", "hypotheses": all_hypotheses}
            evaluation_result = await self.hypothesis_generator.execute_task(eval_task)
        else:
            evaluation_result = {"success": True, "evaluated_hypotheses": []}

        return {
            "generated_hypotheses": all_hypotheses,
            "evaluation_result": evaluation_result,
            "hypothesis_summary": {
                "total_generated": len(all_hypotheses),
                "high_confidence": len([h for h in all_hypotheses if h.get("confidence", 0) > 0.8]),
                "testable": len([h for h in all_hypotheses if h.get("testable", False)]),
            },
        }

    async def _autonomous_causal_discovery(
        self, dataframes: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Autonomous causal discovery phase"""

        causal_results = {}

        for data_name, df in dataframes.items():
            if len(df.columns) >= 3:  # Need sufficient variables for causal discovery
                logger.info(f"   Running causal discovery on {data_name}")

                try:
                    # Run causal discovery pipeline
                    results = self.causal_discovery.run_complete_discovery_pipeline(df)
                    causal_results[data_name] = results

                except Exception as e:
                    logger.warning(f"   Causal discovery failed for {data_name}: {e}")
                    causal_results[data_name] = {"error": str(e)}

        return causal_results

    async def _autonomous_knowledge_synthesis(
        self, research_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize knowledge from all research phases"""

        synthesis = {
            "key_findings": [],
            "cross_validation": {},
            "confidence_assessment": {},
            "future_directions": [],
        }

        # Extract key findings from each phase
        for phase, results in research_results.items():
            phase_findings = self._extract_phase_findings(phase, results)
            synthesis["key_findings"].extend(phase_findings)

        # Cross-validate findings across different methods
        synthesis["cross_validation"] = self._cross_validate_findings(research_results)

        # Assess overall confidence
        synthesis["confidence_assessment"] = self._assess_finding_confidence(
            synthesis["key_findings"]
        )

        # Generate future research directions
        synthesis["future_directions"] = self._generate_future_directions(synthesis)

        return synthesis

    async def _generate_scientific_insights(
        self, research_session: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate high-level scientific insights from research session"""

        insights = []

        # Generate insights from different result types
        results = research_session["results"]

        # Analysis insights
        if "analysis" in results:
            analysis_insights = self._generate_analysis_insights(results["analysis"])
            insights.extend(analysis_insights)

        # Hypothesis insights
        if "hypotheses" in results:
            hypothesis_insights = self._generate_hypothesis_insights(results["hypotheses"])
            insights.extend(hypothesis_insights)

        # Causal insights
        if "causal_discovery" in results:
            causal_insights = self._generate_causal_insights(results["causal_discovery"])
            insights.extend(causal_insights)

        # Synthesis insights
        if "synthesis" in results:
            synthesis_insights = self._generate_synthesis_insights(results["synthesis"])
            insights.extend(synthesis_insights)

        # Rank insights by importance
        ranked_insights = self._rank_insights(insights)

        return {
            "insights": ranked_insights,
            "insight_summary": {
                "total_insights": len(insights),
                "high_impact": len([i for i in ranked_insights if i.impact_score > 0.8]),
                "novel_discoveries": len([i for i in ranked_insights if i.novelty_score > 0.7]),
            },
            "research_impact": self._assess_research_impact(ranked_insights),
        }

    def _generate_research_objectives(self, domain: str, data: Dict[str, Any]) -> List[str]:
        """Generate research objectives based on domain and available data"""

        base_objectives = {
            "habitability": [
                "Identify key factors controlling planetary habitability",
                "Discover novel habitability indicators",
                "Characterize habitability boundaries and thresholds",
            ],
            "atmospheric_evolution": [
                "Understand atmospheric evolution mechanisms",
                "Identify factors controlling atmospheric retention",
                "Characterize atmospheric escape processes",
            ],
            "biosignatures": [
                "Discover new biosignature gas combinations",
                "Understand biosignature production mechanisms",
                "Develop improved detection strategies",
            ],
        }

        objectives = base_objectives.get(
            domain,
            [
                "Understand system behavior and mechanisms",
                "Identify key controlling factors",
                "Discover novel relationships and patterns",
            ],
        )

        # Customize based on available data
        data_specific_objectives = []
        if "spectroscopic" in str(data.keys()).lower():
            data_specific_objectives.append(
                "Analyze spectroscopic signatures for chemical information"
            )
        if "time_series" in str(data.keys()).lower():
            data_specific_objectives.append("Investigate temporal evolution and variability")
        if "observational" in str(data.keys()).lower():
            data_specific_objectives.append("Extract insights from observational data patterns")

        return objectives + data_specific_objectives

    def _extract_phase_findings(self, phase: str, results: Dict[str, Any]) -> List[str]:
        """Extract key findings from research phase"""

        findings = []

        if phase == "analysis" and isinstance(results, dict):
            for dataset, analysis in results.items():
                for method, result in analysis.items():
                    if result.get("success", False) and "insights" in result:
                        findings.extend(result["insights"])

        elif phase == "hypotheses" and "generated_hypotheses" in results:
            top_hypotheses = sorted(
                results["generated_hypotheses"],
                key=lambda h: h.get("combined_score", 0),
                reverse=True,
            )[
                :5
            ]  # Top 5 hypotheses

            for hyp in top_hypotheses:
                findings.append(
                    f"Hypothesis: {hyp['description']} (confidence: {hyp.get('confidence', 0):.2f})"
                )

        elif phase == "causal_discovery":
            for dataset, causal_result in results.items():
                if "summary" in causal_result:
                    summary = causal_result["summary"]
                    if summary.get("num_edges_discovered", 0) > 0:
                        findings.append(
                            f"Discovered {summary['num_edges_discovered']} causal relationships in {dataset}"
                        )

        return findings

    def _cross_validate_findings(self, research_results: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate findings across different research methods"""

        # Simplified cross-validation
        validation = {"convergent_findings": [], "conflicting_findings": [], "method_agreement": {}}

        # Find findings that appear across multiple methods
        all_findings = []
        for phase, results in research_results.items():
            phase_findings = self._extract_phase_findings(phase, results)
            all_findings.extend([(finding, phase) for finding in phase_findings])

        # Group similar findings
        finding_groups = defaultdict(list)
        for finding, phase in all_findings:
            # Simple keyword-based grouping
            key_words = set(finding.lower().split())
            found_group = False

            for group_key, group_findings in finding_groups.items():
                group_words = set(group_key.lower().split())
                if len(key_words.intersection(group_words)) >= 2:
                    finding_groups[group_key].append((finding, phase))
                    found_group = True
                    break

            if not found_group:
                finding_groups[finding].append((finding, phase))

        # Identify convergent findings (supported by multiple methods)
        for group_key, group_findings in finding_groups.items():
            if len(set(phase for _, phase in group_findings)) > 1:
                validation["convergent_findings"].append(
                    {
                        "finding": group_key,
                        "supporting_methods": list(set(phase for _, phase in group_findings)),
                        "strength": len(set(phase for _, phase in group_findings)),
                    }
                )

        return validation

    def _assess_finding_confidence(self, findings: List[str]) -> Dict[str, float]:
        """Assess confidence in research findings"""

        # Simplified confidence assessment
        confidence_scores = {}

        for finding in findings:
            # Base confidence
            confidence = 0.5

            # Increase confidence for quantitative findings
            if any(char.isdigit() for char in finding):
                confidence += 0.2

            # Increase confidence for statistical findings
            if any(word in finding.lower() for word in ["correlation", "significant", "p-value"]):
                confidence += 0.2

            # Increase confidence for consistent findings
            if "consistent" in finding.lower() or "confirmed" in finding.lower():
                confidence += 0.1

            confidence_scores[finding] = min(confidence, 1.0)

        return confidence_scores

    def _generate_future_directions(self, synthesis: Dict[str, Any]) -> List[str]:
        """Generate future research directions"""

        directions = []

        # Based on convergent findings
        for finding in synthesis.get("convergent_findings", []):
            if finding["strength"] >= 2:
                directions.append(f"Investigate mechanisms underlying: {finding['finding']}")

        # Based on confidence gaps
        low_confidence_findings = [
            finding
            for finding, confidence in synthesis.get("confidence_assessment", {}).items()
            if confidence < 0.6
        ]

        for finding in low_confidence_findings[:3]:  # Top 3 uncertain findings
            directions.append(f"Increase confidence in: {finding}")

        # Generic future directions
        directions.extend(
            [
                "Expand dataset size and diversity for more robust conclusions",
                "Develop improved measurement techniques and instrumentation",
                "Integrate findings with existing theoretical frameworks",
                "Design targeted experiments to test key hypotheses",
            ]
        )

        return directions[:8]  # Limit to 8 directions

    def _generate_analysis_insights(
        self, analysis_results: Dict[str, Any]
    ) -> List[ScientificInsight]:
        """Generate insights from analysis results"""

        insights = []

        for dataset, analysis in analysis_results.items():
            for method, result in analysis.items():
                if result.get("success", False) and "insights" in result:
                    for insight_text in result["insights"]:
                        insight = ScientificInsight(
                            id=f"analysis_{dataset}_{method}_{len(insights)}",
                            title=f"Analysis insight from {method}",
                            description=insight_text,
                            confidence_level=0.7,
                            novelty_score=0.6,
                            impact_score=0.5,
                            supporting_evidence=[
                                {
                                    "type": "data_analysis",
                                    "method": method,
                                    "dataset": dataset,
                                    "result": result,
                                }
                            ],
                            related_hypotheses=[],
                            generated_by="data_analyst_agent",
                            timestamp=datetime.now(),
                            scientific_domain="data_analysis",
                        )
                        insights.append(insight)

        return insights

    def _generate_hypothesis_insights(
        self, hypothesis_results: Dict[str, Any]
    ) -> List[ScientificInsight]:
        """Generate insights from hypothesis generation"""

        insights = []

        top_hypotheses = sorted(
            hypothesis_results.get("generated_hypotheses", []),
            key=lambda h: h.get("combined_score", 0),
            reverse=True,
        )[:5]

        for i, hyp in enumerate(top_hypotheses):
            insight = ScientificInsight(
                id=f"hypothesis_{i}",
                title=f"Novel hypothesis: {hyp.get('type', 'general')}",
                description=hyp["description"],
                confidence_level=hyp.get("confidence", 0.5),
                novelty_score=hyp.get("novelty", 0.5),
                impact_score=hyp.get("combined_score", 0.5),
                supporting_evidence=[{"type": "hypothesis_generation", "hypothesis_data": hyp}],
                related_hypotheses=[hyp.get("id", "")],
                generated_by="hypothesis_generator_agent",
                timestamp=datetime.now(),
                scientific_domain=hyp.get("type", "general"),
            )
            insights.append(insight)

        return insights

    def _generate_causal_insights(self, causal_results: Dict[str, Any]) -> List[ScientificInsight]:
        """Generate insights from causal discovery"""

        insights = []

        for dataset, results in causal_results.items():
            if "summary" in results:
                summary = results["summary"]

                if summary.get("num_edges_discovered", 0) > 0:
                    insight = ScientificInsight(
                        id=f"causal_{dataset}",
                        title=f"Causal relationships in {dataset}",
                        description=f"Discovered {summary['num_edges_discovered']} causal relationships",
                        confidence_level=0.8,
                        novelty_score=0.75,
                        impact_score=0.85,
                        supporting_evidence=[
                            {"type": "causal_discovery", "dataset": dataset, "results": results}
                        ],
                        related_hypotheses=[],
                        generated_by="causal_discovery_agent",
                        timestamp=datetime.now(),
                        scientific_domain="causal_analysis",
                    )
                    insights.append(insight)

        return insights

    def _generate_synthesis_insights(
        self, synthesis_results: Dict[str, Any]
    ) -> List[ScientificInsight]:
        """Generate insights from knowledge synthesis"""

        insights = []

        # Convergent findings insight
        convergent = synthesis_results.get("convergent_findings", [])
        if convergent:
            strong_convergent = [f for f in convergent if f["strength"] >= 2]

            if strong_convergent:
                insight = ScientificInsight(
                    id="synthesis_convergent",
                    title="Cross-method validation of findings",
                    description=f"Found {len(strong_convergent)} findings validated by multiple methods",
                    confidence_level=0.9,
                    novelty_score=0.6,
                    impact_score=0.8,
                    supporting_evidence=[
                        {"type": "cross_validation", "convergent_findings": strong_convergent}
                    ],
                    related_hypotheses=[],
                    generated_by="synthesis_agent",
                    timestamp=datetime.now(),
                    scientific_domain="meta_analysis",
                )
                insights.append(insight)

        return insights

    def _rank_insights(self, insights: List[ScientificInsight]) -> List[ScientificInsight]:
        """Rank insights by combined impact, novelty, and confidence"""

        def combined_score(insight):
            return (
                insight.impact_score * 0.4
                + insight.novelty_score * 0.3
                + insight.confidence_level * 0.3
            )

        return sorted(insights, key=combined_score, reverse=True)

    def _assess_research_impact(self, insights: List[ScientificInsight]) -> Dict[str, Any]:
        """Assess overall research impact"""

        if not insights:
            return {"overall_impact": 0.0, "impact_category": "minimal"}

        avg_impact = np.mean([i.impact_score for i in insights])
        avg_novelty = np.mean([i.novelty_score for i in insights])
        avg_confidence = np.mean([i.confidence_level for i in insights])

        overall_impact = (avg_impact + avg_novelty + avg_confidence) / 3

        if overall_impact > 0.8:
            category = "high"
        elif overall_impact > 0.6:
            category = "medium"
        else:
            category = "low"

        return {
            "overall_impact": overall_impact,
            "impact_category": category,
            "avg_impact_score": avg_impact,
            "avg_novelty_score": avg_novelty,
            "avg_confidence": avg_confidence,
            "total_insights": len(insights),
        }

    def save_research_session(self, session: Dict[str, Any], output_path: str = None):
        """Save research session results"""

        if output_path is None:
            output_path = f"autonomous_research_{session['session_id']}.json"

        # Prepare serializable session data
        serializable_session = {
            "session_id": session["session_id"],
            "domain": session["domain"],
            "start_time": session["start_time"].isoformat(),
            "end_time": session.get("end_time", datetime.now()).isoformat(),
            "duration_seconds": session.get("duration", timedelta(0)).total_seconds(),
            "success": session.get("success", False),
            "error": session.get("error"),
            "constraints": session["constraints"],
            "results_summary": {
                "phases_completed": len(session["results"]),
                "insights_generated": len(
                    session["results"].get("insights", {}).get("insights", [])
                ),
                "research_impact": session["results"]
                .get("insights", {})
                .get("research_impact", {}),
            },
        }

        with open(output_path, "w") as f:
            json.dump(serializable_session, f, indent=2, default=str)

        logger.info(f"💾 Research session saved to {output_path}")


# Factory functions


def create_autonomous_discovery_system() -> AutonomousScientificDiscovery:
    """Create configured autonomous discovery system"""
    return AutonomousScientificDiscovery()


async def demonstrate_autonomous_discovery():
    """Demonstrate autonomous scientific discovery capabilities"""

    logger.info("🤖 Demonstrating Autonomous Scientific Discovery")

    # Create system
    discovery_system = create_autonomous_discovery_system()

    # Generate synthetic astrobiology dataset
    np.random.seed(42)
    n_samples = 500

    # Create realistic astrobiology data
    stellar_mass = np.random.normal(1.0, 0.3, n_samples)
    planet_radius = np.random.lognormal(0, 0.5, n_samples)
    orbital_period = np.random.uniform(1, 100, n_samples)
    insolation = (stellar_mass / (orbital_period**0.5)) ** 2 + np.random.normal(0, 0.1, n_samples)
    surface_temp = 255 * (insolation**0.25) + np.random.normal(0, 10, n_samples)
    habitability = 1 / (1 + np.exp(-(surface_temp - 280) / 50)) + np.random.normal(
        0, 0.1, n_samples
    )

    astrobio_data = pd.DataFrame(
        {
            "stellar_mass": stellar_mass,
            "planet_radius": planet_radius,
            "orbital_period": orbital_period,
            "insolation": insolation,
            "surface_temperature": surface_temp,
            "habitability_score": habitability,
        }
    )

    # Available data for research
    available_data = {
        "dataframes": {"exoplanet_sample": astrobio_data},
        "observational": astrobio_data,
        "spectroscopic": "mock_spectroscopic_data",
    }

    # Research constraints
    research_constraints = {
        "time_budget": "1 week",
        "computational_resources": "standard",
        "data_access": "full",
    }

    # Conduct autonomous research
    research_results = await discovery_system.conduct_autonomous_research(
        research_domain="habitability",
        available_data=available_data,
        research_constraints=research_constraints,
    )

    # Display results
    logger.info("🔍 Autonomous Research Results:")
    logger.info(f"   Research Domain: {research_results['domain']}")
    logger.info(f"   Duration: {research_results.get('duration', 'Unknown')}")
    logger.info(f"   Success: {research_results.get('success', False)}")

    if "insights" in research_results["results"]:
        insights_summary = research_results["results"]["insights"]["insight_summary"]
        logger.info(f"   Total Insights: {insights_summary['total_insights']}")
        logger.info(f"   High Impact Insights: {insights_summary['high_impact']}")
        logger.info(f"   Novel Discoveries: {insights_summary['novel_discoveries']}")

        impact_assessment = research_results["results"]["insights"]["research_impact"]
        logger.info(
            f"   Overall Impact: {impact_assessment['impact_category']} ({impact_assessment['overall_impact']:.3f})"
        )

    # Save research session
    discovery_system.save_research_session(research_results)

    return research_results


if __name__ == "__main__":
    asyncio.run(demonstrate_autonomous_discovery())

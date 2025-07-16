#!/usr/bin/env python3
"""
Priority 3: Uncertainty and Emergence Modeling System
=====================================================

Final priority implementation that models fundamental unknowability and emergence 
thresholds in astrobiology. Builds on Priority 1 (evolutionary process modeling) 
and Priority 2 (narrative chat enhancement).

Key Capabilities:
1. Fundamental unknowability quantification - some aspects cannot be predicted
2. Emergence threshold detection - identify when new properties appear
3. Path dependence modeling - how history constrains future possibilities  
4. Unknowability acknowledgment - explicit recognition of prediction limits
5. Complex systems uncertainty - beyond traditional statistical uncertainty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from scipy.stats import entropy
from sklearn.mixture import GaussianMixture
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class UncertaintyType(Enum):
    """Different types of uncertainty in astrobiology"""
    STATISTICAL = "statistical"                    # Traditional measurement uncertainty
    MODEL = "model"                                # Model structure uncertainty  
    EPISTEMIC = "epistemic"                        # Knowledge-based uncertainty
    ALEATORY = "aleatory"                         # Intrinsic randomness
    EMERGENCE = "emergence"                        # Uncertainty from emergent properties
    FUNDAMENTAL = "fundamental"                    # Fundamentally unknowable aspects
    TEMPORAL = "temporal"                          # Deep time/historical uncertainty
    COMPLEXITY = "complexity"                      # Complex systems uncertainty

class EmergenceType(Enum):
    """Types of emergence in biological systems"""
    WEAK = "weak"                                  # Predictable from components
    STRONG = "strong"                              # Fundamentally unpredictable
    DIACHRONIC = "diachronic"                     # Emerges over time
    SYNCHRONIC = "synchronic"                      # Emerges from organization
    DOWNWARD_CAUSATION = "downward_causation"     # Higher levels affect lower levels

@dataclass
class UncertaintyProfile:
    """Complete uncertainty characterization for a system"""
    system_id: str
    uncertainty_types: Dict[UncertaintyType, float]  # 0-1 scale for each type
    total_uncertainty: float
    fundamental_unknowability: float                 # Aspects that cannot be known
    reducible_uncertainty: float                     # Could be reduced with more data/models
    irreducible_uncertainty: float                   # Cannot be reduced in principle
    emergence_indicators: Dict[EmergenceType, float]
    path_dependence_strength: float                  # How much history matters
    prediction_horizon: Optional[float] = None       # How far ahead we can predict
    confidence_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)

@dataclass 
class EmergenceEvent:
    """Represents an emergence event in a complex system"""
    event_id: str
    system_level: str                                # Level at which emergence occurs
    emergence_type: EmergenceType
    precursor_patterns: List[str]                    # Patterns before emergence
    emergent_properties: List[str]                   # New properties that appeared
    predictability_score: float                     # How predictable was this emergence
    downward_causation: bool                         # Does it affect lower levels
    temporal_scale: Optional[float] = None           # Timescale of emergence
    spatial_scale: Optional[float] = None            # Spatial scale
    complexity_threshold: Optional[float] = None     # Complexity level needed

class FundamentalUncertaintyEngine(nn.Module):
    """Models aspects of biological systems that are fundamentally unknowable"""
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        uncertainty_types: int = 8,
        **kwargs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.uncertainty_types = uncertainty_types
        
        # Feature extraction network
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Uncertainty type classifiers
        self.uncertainty_classifiers = nn.ModuleDict({
            'statistical': nn.Linear(hidden_dim, 1),
            'model': nn.Linear(hidden_dim, 1),
            'epistemic': nn.Linear(hidden_dim, 1), 
            'aleatory': nn.Linear(hidden_dim, 1),
            'emergence': nn.Linear(hidden_dim, 1),
            'fundamental': nn.Linear(hidden_dim, 1),
            'temporal': nn.Linear(hidden_dim, 1),
            'complexity': nn.Linear(hidden_dim, 1)
        })
        
        # Fundamental unknowability detector
        self.unknowability_detector = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(), 
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Path dependence analyzer
        self.path_dependence_analyzer = nn.Sequential(
            nn.Linear(hidden_dim + 32, 64),  # +32 for temporal context
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Prediction horizon estimator
        self.prediction_horizon = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()  # Ensure positive horizon
        )
    
    def forward(
        self, 
        system_state: torch.Tensor,
        temporal_context: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze fundamental uncertainty in a biological system
        
        Args:
            system_state: Current state representation [batch, input_dim]
            temporal_context: Historical context [batch, 32] (optional)
            
        Returns:
            Complete uncertainty analysis
        """
        # Extract features
        features = self.feature_extractor(system_state)
        
        # Classify uncertainty types
        uncertainty_scores = {}
        for uncertainty_name, classifier in self.uncertainty_classifiers.items():
            uncertainty_scores[uncertainty_name] = torch.sigmoid(classifier(features))
        
        # Detect fundamental unknowability
        fundamental_unknowability = self.unknowability_detector(features)
        
        # Analyze path dependence
        if temporal_context is not None:
            path_input = torch.cat([features, temporal_context], dim=-1)
        else:
            # Use dummy temporal context if not provided
            dummy_context = torch.zeros(features.shape[0], 32, device=features.device)
            path_input = torch.cat([features, dummy_context], dim=-1)
        
        path_dependence = self.path_dependence_analyzer(path_input)
        
        # Estimate prediction horizon
        horizon = self.prediction_horizon(features)
        
        # Calculate total uncertainty
        uncertainty_values = torch.stack(list(uncertainty_scores.values()), dim=-1)
        total_uncertainty = torch.mean(uncertainty_values, dim=-1, keepdim=True)
        
        # Separate reducible vs irreducible uncertainty
        reducible_uncertainty = torch.mean(uncertainty_values[:, :4], dim=-1, keepdim=True)  # First 4 types
        irreducible_uncertainty = torch.mean(uncertainty_values[:, 4:], dim=-1, keepdim=True)  # Last 4 types
        
        return {
            'uncertainty_scores': uncertainty_scores,
            'total_uncertainty': total_uncertainty,
            'fundamental_unknowability': fundamental_unknowability,
            'path_dependence': path_dependence,
            'prediction_horizon': horizon,
            'reducible_uncertainty': reducible_uncertainty,
            'irreducible_uncertainty': irreducible_uncertainty,
            'uncertainty_breakdown': uncertainty_values
        }
    
    def analyze_system_uncertainty(
        self, 
        system_data: Dict[str, torch.Tensor],
        system_id: str = "unknown"
    ) -> UncertaintyProfile:
        """Create complete uncertainty profile for a system"""
        
        # Extract system state representation
        system_state = self._encode_system_state(system_data)
        
        # Run uncertainty analysis
        with torch.no_grad():
            results = self.forward(system_state)
        
        # Convert to uncertainty profile
        uncertainty_types_dict = {}
        for i, uncertainty_type in enumerate(UncertaintyType):
            if uncertainty_type.value in results['uncertainty_scores']:
                uncertainty_types_dict[uncertainty_type] = results['uncertainty_scores'][uncertainty_type.value].item()
        
        profile = UncertaintyProfile(
            system_id=system_id,
            uncertainty_types=uncertainty_types_dict,
            total_uncertainty=results['total_uncertainty'].item(),
            fundamental_unknowability=results['fundamental_unknowability'].item(),
            reducible_uncertainty=results['reducible_uncertainty'].item(),
            irreducible_uncertainty=results['irreducible_uncertainty'].item(),
            emergence_indicators={},  # Will be filled by emergence detector
            path_dependence_strength=results['path_dependence'].item(),
            prediction_horizon=results['prediction_horizon'].item()
        )
        
        return profile
    
    def _encode_system_state(self, system_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode system data into state representation"""
        # Simple encoding - concatenate available features
        features = []
        
        for key, value in system_data.items():
            if isinstance(value, torch.Tensor):
                if value.dim() > 1:
                    features.append(value.flatten())
                else:
                    features.append(value)
            else:
                # Convert scalar to tensor
                features.append(torch.tensor([float(value)]))
        
        if features:
            concatenated = torch.cat(features, dim=0)
            # Pad or truncate to input_dim
            if len(concatenated) < self.input_dim:
                padding = torch.zeros(self.input_dim - len(concatenated))
                concatenated = torch.cat([concatenated, padding])
            else:
                concatenated = concatenated[:self.input_dim]
            
            return concatenated.unsqueeze(0)  # Add batch dimension
        else:
            return torch.randn(1, self.input_dim)  # Random default

class EmergenceDetector(nn.Module):
    """Detects emergence thresholds and characterizes emergent properties"""
    
    def __init__(
        self,
        input_dim: int = 128,
        temporal_window: int = 50,
        emergence_threshold: float = 0.7,
        **kwargs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.temporal_window = temporal_window
        self.emergence_threshold = emergence_threshold
        
        # Temporal pattern analyzer
        self.temporal_analyzer = nn.LSTM(
            input_size=input_dim,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Emergence type classifiers
        self.emergence_classifiers = nn.ModuleDict({
            'weak': nn.Linear(64, 1),
            'strong': nn.Linear(64, 1),
            'diachronic': nn.Linear(64, 1),
            'synchronic': nn.Linear(64, 1),
            'downward_causation': nn.Linear(64, 1)
        })
        
        # Complexity threshold detector
        self.complexity_detector = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Predictability scorer
        self.predictability_scorer = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Pattern change detector
        self.pattern_change_detector = nn.Sequential(
            nn.Linear(64 * 2, 32),  # Compare before/after patterns
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        temporal_sequence: torch.Tensor,
        system_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Detect emergence in temporal sequence
        
        Args:
            temporal_sequence: Time series data [batch, time, features]
            system_metadata: Additional system information
            
        Returns:
            Emergence analysis results
        """
        batch_size, seq_len, _ = temporal_sequence.shape
        
        # Analyze temporal patterns
        lstm_out, (hidden, cell) = self.temporal_analyzer(temporal_sequence)
        
        # Use final hidden state for emergence analysis
        final_state = lstm_out[:, -1, :]  # [batch, hidden_size]
        
        # Classify emergence types
        emergence_scores = {}
        for emergence_name, classifier in self.emergence_classifiers.items():
            emergence_scores[emergence_name] = torch.sigmoid(classifier(final_state))
        
        # Detect complexity threshold
        complexity_threshold = self.complexity_detector(final_state)
        
        # Score predictability
        predictability = self.predictability_scorer(final_state)
        
        # Detect pattern changes by comparing early vs late patterns
        early_pattern = lstm_out[:, :seq_len//3, :].mean(dim=1)  # First third
        late_pattern = lstm_out[:, -seq_len//3:, :].mean(dim=1)  # Last third
        
        pattern_change_input = torch.cat([early_pattern, late_pattern], dim=-1)
        pattern_change = self.pattern_change_detector(pattern_change_input)
        
        return {
            'emergence_scores': emergence_scores,
            'complexity_threshold': complexity_threshold,
            'predictability': predictability,
            'pattern_change': pattern_change,
            'temporal_features': lstm_out,
            'final_state': final_state
        }
    
    def detect_emergence_events(
        self,
        temporal_data: torch.Tensor,
        threshold: Optional[float] = None
    ) -> List[EmergenceEvent]:
        """Detect and characterize emergence events in temporal data"""
        
        threshold = threshold or self.emergence_threshold
        
        with torch.no_grad():
            results = self.forward(temporal_data)
        
        events = []
        
        # Check for emergence based on pattern change and complexity
        pattern_change = results['pattern_change'].item()
        complexity = results['complexity_threshold'].item()
        
        if pattern_change > threshold and complexity > 0.5:
            # Determine emergence type
            emergence_scores = results['emergence_scores']
            dominant_type = max(emergence_scores.items(), key=lambda x: x[1].item())
            
            event = EmergenceEvent(
                event_id=f"emergence_{len(events)}",
                system_level="system",  # Could be made more specific
                emergence_type=EmergenceType(dominant_type[0]),
                precursor_patterns=["pattern_complexity_increase"],
                emergent_properties=["new_system_behavior"],
                predictability_score=results['predictability'].item(),
                downward_causation=emergence_scores['downward_causation'].item() > 0.5,
                complexity_threshold=complexity
            )
            
            events.append(event)
        
        return events

class PathDependenceAnalyzer(nn.Module):
    """Analyzes how historical paths constrain future possibilities"""
    
    def __init__(
        self,
        state_dim: int = 64,
        history_length: int = 100,
        branching_points: int = 10,
        **kwargs
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.history_length = history_length
        self.branching_points = branching_points
        
        # Historical trajectory encoder
        self.trajectory_encoder = nn.LSTM(
            input_size=state_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )
        
        # Critical juncture detector
        self.juncture_detector = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Alternative path generator
        self.path_generator = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, branching_points * state_dim)
        )
        
        # Constraint strength analyzer
        self.constraint_analyzer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        historical_trajectory: torch.Tensor,
        current_state: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze path dependence in system evolution
        
        Args:
            historical_trajectory: Past states [batch, time, state_dim]
            current_state: Current state [batch, state_dim]
            
        Returns:
            Path dependence analysis
        """
        # Encode historical trajectory
        trajectory_features, (hidden, cell) = self.trajectory_encoder(historical_trajectory)
        
        # Use final hidden state for analysis
        final_encoding = trajectory_features[:, -1, :]
        
        # Detect critical junctures
        juncture_scores = self.juncture_detector(trajectory_features)
        
        # Generate alternative future paths
        alternative_paths = self.path_generator(final_encoding)
        alternative_paths = alternative_paths.view(-1, self.branching_points, self.state_dim)
        
        # Analyze constraint strength
        constraint_strength = self.constraint_analyzer(final_encoding)
        
        # Calculate path dependence metrics
        path_diversity = self._calculate_path_diversity(alternative_paths)
        historical_influence = self._calculate_historical_influence(trajectory_features, current_state)
        
        return {
            'trajectory_encoding': final_encoding,
            'critical_junctures': juncture_scores,
            'alternative_paths': alternative_paths,
            'constraint_strength': constraint_strength,
            'path_diversity': path_diversity,
            'historical_influence': historical_influence
        }
    
    def _calculate_path_diversity(self, alternative_paths: torch.Tensor) -> torch.Tensor:
        """Calculate diversity of possible future paths"""
        # Calculate pairwise distances between alternative paths
        batch_size, n_paths, state_dim = alternative_paths.shape
        
        # Reshape for pairwise distance calculation
        paths_expanded = alternative_paths.unsqueeze(2)  # [batch, n_paths, 1, state_dim]
        paths_expanded_t = alternative_paths.unsqueeze(1)  # [batch, 1, n_paths, state_dim]
        
        # Calculate L2 distances
        distances = torch.norm(paths_expanded - paths_expanded_t, dim=-1)  # [batch, n_paths, n_paths]
        
        # Average pairwise distance as diversity measure
        diversity = distances.mean(dim=(-2, -1))  # [batch]
        
        return diversity
    
    def _calculate_historical_influence(
        self, 
        trajectory_features: torch.Tensor, 
        current_state: torch.Tensor
    ) -> torch.Tensor:
        """Calculate how much history influences current state"""
        # Simple correlation-based measure
        # In practice, this could be more sophisticated
        
        # Take mean of trajectory features
        mean_history = trajectory_features.mean(dim=1)
        
        # Calculate correlation with current state
        correlation = F.cosine_similarity(mean_history, current_state, dim=-1)
        
        return torch.abs(correlation)  # Absolute correlation as influence measure

class UncertaintyEmergenceSystem(nn.Module):
    """
    Complete uncertainty and emergence modeling system
    Integrates fundamental uncertainty, emergence detection, and path dependence
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        temporal_window: int = 50,
        uncertainty_config: Dict[str, Any] = None,
        emergence_config: Dict[str, Any] = None,
        path_config: Dict[str, Any] = None,
        **kwargs
    ):
        super().__init__()
        
        # Initialize components
        uncertainty_config = uncertainty_config or {}
        self.uncertainty_engine = FundamentalUncertaintyEngine(
            input_dim=input_dim, **uncertainty_config
        )
        
        emergence_config = emergence_config or {}
        self.emergence_detector = EmergenceDetector(
            input_dim=input_dim, 
            temporal_window=temporal_window,
            **emergence_config
        )
        
        path_config = path_config or {}
        self.path_analyzer = PathDependenceAnalyzer(
            state_dim=input_dim,
            **path_config
        )
        
        # Integration network
        self.integration_network = nn.Sequential(
            nn.Linear(input_dim * 3, 256),  # Combine all analyses
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Meta-uncertainty estimator
        self.meta_uncertainty = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        system_data: Dict[str, torch.Tensor],
        temporal_sequence: Optional[torch.Tensor] = None,
        historical_trajectory: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Complete uncertainty and emergence analysis
        
        Args:
            system_data: Current system state data
            temporal_sequence: Recent temporal evolution
            historical_trajectory: Long-term historical data
            
        Returns:
            Comprehensive uncertainty and emergence analysis
        """
        # Fundamental uncertainty analysis
        uncertainty_results = self.uncertainty_engine(
            self.uncertainty_engine._encode_system_state(system_data)
        )
        
        results = {
            'uncertainty_analysis': uncertainty_results,
            'emergence_analysis': {},
            'path_analysis': {},
            'meta_analysis': {}
        }
        
        # Emergence analysis (if temporal data available)
        if temporal_sequence is not None:
            emergence_results = self.emergence_detector(temporal_sequence)
            results['emergence_analysis'] = emergence_results
            
            # Detect emergence events
            emergence_events = self.emergence_detector.detect_emergence_events(temporal_sequence)
            results['emergence_events'] = emergence_events
        
        # Path dependence analysis (if historical data available)
        if historical_trajectory is not None and temporal_sequence is not None:
            current_state = temporal_sequence[:, -1, :]  # Most recent state
            path_results = self.path_analyzer(historical_trajectory, current_state)
            results['path_analysis'] = path_results
        
        # Meta-analysis: uncertainty about uncertainty
        if temporal_sequence is not None:
            # Combine features from all analyses
            uncertainty_features = uncertainty_results['uncertainty_breakdown']
            emergence_features = results['emergence_analysis'].get('final_state', torch.zeros_like(uncertainty_features))
            path_features = results['path_analysis'].get('trajectory_encoding', torch.zeros_like(uncertainty_features))
            
            combined_features = torch.cat([
                uncertainty_features, emergence_features, path_features
            ], dim=-1)
            
            integrated = self.integration_network(combined_features)
            meta_uncertainty = self.meta_uncertainty(integrated)
            
            results['meta_analysis'] = {
                'integrated_features': integrated,
                'meta_uncertainty': meta_uncertainty,
                'analysis_confidence': 1.0 - meta_uncertainty.item()
            }
        
        return results
    
    def create_uncertainty_profile(
        self,
        system_data: Dict[str, torch.Tensor],
        temporal_sequence: Optional[torch.Tensor] = None,
        historical_trajectory: Optional[torch.Tensor] = None,
        system_id: str = "system"
    ) -> Dict[str, Any]:
        """Create comprehensive uncertainty profile for a system"""
        
        with torch.no_grad():
            analysis = self.forward(system_data, temporal_sequence, historical_trajectory)
        
        # Create uncertainty profile
        uncertainty_profile = self.uncertainty_engine.analyze_system_uncertainty(
            system_data, system_id
        )
        
        # Add emergence indicators if available
        if 'emergence_analysis' in analysis and analysis['emergence_analysis']:
            emergence_scores = analysis['emergence_analysis'].get('emergence_scores', {})
            emergence_indicators = {}
            for emergence_type_name, score in emergence_scores.items():
                emergence_type = EmergenceType(emergence_type_name)
                emergence_indicators[emergence_type] = score.item()
            uncertainty_profile.emergence_indicators = emergence_indicators
        
        # Add prediction horizon adjustments based on path dependence
        if 'path_analysis' in analysis and analysis['path_analysis']:
            constraint_strength = analysis['path_analysis'].get('constraint_strength', torch.tensor(0.5))
            # Higher constraints = shorter prediction horizon
            uncertainty_profile.prediction_horizon = uncertainty_profile.prediction_horizon * (1.0 - constraint_strength.item())
        
        return {
            'uncertainty_profile': uncertainty_profile,
            'emergence_events': analysis.get('emergence_events', []),
            'path_constraints': analysis.get('path_analysis', {}),
            'meta_analysis': analysis.get('meta_analysis', {}),
            'system_classification': self._classify_system_uncertainty(uncertainty_profile)
        }
    
    def _classify_system_uncertainty(self, profile: UncertaintyProfile) -> Dict[str, str]:
        """Classify system based on uncertainty characteristics"""
        
        fundamental_unknowability = profile.fundamental_unknowability
        emergence_strength = sum(profile.emergence_indicators.values()) if profile.emergence_indicators else 0
        path_dependence = profile.path_dependence_strength
        
        # System classification based on uncertainty characteristics
        if fundamental_unknowability > 0.7:
            uncertainty_class = "Fundamentally Unknowable"
        elif emergence_strength > 0.6:
            uncertainty_class = "Emergence-Dominated"
        elif path_dependence > 0.7:
            uncertainty_class = "History-Dependent"
        elif profile.total_uncertainty > 0.8:
            uncertainty_class = "High Uncertainty"
        else:
            uncertainty_class = "Moderately Predictable"
        
        # Predictability assessment
        if profile.prediction_horizon and profile.prediction_horizon < 0.1:
            predictability = "Extremely Limited"
        elif profile.prediction_horizon and profile.prediction_horizon < 0.3:
            predictability = "Limited"
        elif profile.prediction_horizon and profile.prediction_horizon < 0.7:
            predictability = "Moderate"
        else:
            predictability = "Good"
        
        # Research approach recommendation
        if fundamental_unknowability > 0.6:
            research_approach = "Philosophical and Narrative"
        elif emergence_strength > 0.5:
            research_approach = "Multi-Level Systems Analysis"
        elif path_dependence > 0.6:
            research_approach = "Historical and Comparative"
        else:
            research_approach = "Quantitative Modeling"
        
        return {
            'uncertainty_class': uncertainty_class,
            'predictability': predictability,
            'recommended_approach': research_approach,
            'confidence_level': 'High' if profile.total_uncertainty < 0.4 else 'Medium' if profile.total_uncertainty < 0.7 else 'Low'
        }

def create_uncertainty_emergence_system(**kwargs) -> UncertaintyEmergenceSystem:
    """Factory function to create uncertainty and emergence system"""
    return UncertaintyEmergenceSystem(**kwargs) 
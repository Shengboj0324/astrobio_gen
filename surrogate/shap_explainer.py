#!/usr/bin/env python3
"""
SHAP Explanation System for Surrogate Models
============================================

Scientific explanation system using SHAP (SHapley Additive exPlanations) for:
- Feature importance analysis
- Pathway-level explanations
- Model interpretability
- Physics-informed explanations

Inspired by the Nature paper's SAGE method for biological interpretability.
Designed for multi-domain scientific data:
- Astronomical data (stellar parameters, atmospheric features)
- Exoplanet data (planetary properties, habitability factors)
- Environmental data (climate variables, atmospheric composition)
- Biosignature data (metabolic pathways, biomarkers)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import warnings
from concurrent.futures import ThreadPoolExecutor
import time

# SHAP library
import shap
from shap import Explainer, TreeExplainer, LinearExplainer, KernelExplainer
from shap.plots import waterfall, beeswarm, bar, heatmap
from shap.utils import sample

# Scientific libraries
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import networkx as nx

# Project imports
from data_build.metadata_db import MetadataManager, DataDomain
from surrogate import SurrogateManager, SurrogateMode, BaseModelWrapper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExplanationConfig:
    """Configuration for SHAP explanations"""
    
    # SHAP parameters
    max_evals: int = 1000
    batch_size: int = 32
    background_size: int = 100
    
    # Feature grouping
    enable_pathway_grouping: bool = True
    enable_physics_grouping: bool = True
    
    # Visualization
    max_features_plot: int = 20
    save_plots: bool = True
    plot_format: str = 'png'
    
    # Performance
    parallel_computation: bool = True
    max_workers: int = 4
    
    # Caching
    enable_caching: bool = True
    cache_dir: Path = Path("explanations/cache")
    
    # Feature names and groups
    feature_names: Optional[List[str]] = None
    feature_groups: Optional[Dict[str, List[str]]] = None
    
    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)

class FeatureGroupManager:
    """Manager for feature grouping and pathway analysis"""
    
    def __init__(self, domain: DataDomain):
        self.domain = domain
        self.groups = {}
        self.feature_names = []
        self.pathway_mapping = {}
        
    def create_domain_groups(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Create domain-specific feature groups"""
        
        self.feature_names = feature_names
        
        if self.domain == DataDomain.ASTRONOMICAL:
            return self._create_astronomical_groups()
        elif self.domain == DataDomain.EXOPLANET:
            return self._create_exoplanet_groups()
        elif self.domain == DataDomain.ENVIRONMENTAL:
            return self._create_environmental_groups()
        elif self.domain == DataDomain.PHYSICS:
            return self._create_physics_groups()
        elif self.domain == DataDomain.PHYSIOLOGICAL:
            return self._create_physiological_groups()
        elif self.domain == DataDomain.BIOSIGNATURE:
            return self._create_biosignature_groups()
        else:
            return self._create_generic_groups()
    
    def _create_astronomical_groups(self) -> Dict[str, List[str]]:
        """Create astronomical feature groups"""
        
        groups = {
            'stellar_properties': [],
            'atmospheric_composition': [],
            'radiation_fields': [],
            'magnetic_fields': [],
            'convection_zones': [],
            'photosphere': [],
            'chromosphere': [],
            'corona': []
        }
        
        # Group features by keywords
        for feature in self.feature_names:
            feature_lower = feature.lower()
            
            if any(keyword in feature_lower for keyword in ['temp', 'luminosity', 'mass', 'radius', 'metallicity']):
                groups['stellar_properties'].append(feature)
            elif any(keyword in feature_lower for keyword in ['h2o', 'co2', 'ch4', 'o2', 'n2']):
                groups['atmospheric_composition'].append(feature)
            elif any(keyword in feature_lower for keyword in ['uv', 'xray', 'gamma', 'radiation']):
                groups['radiation_fields'].append(feature)
            elif any(keyword in feature_lower for keyword in ['magnetic', 'field', 'flux']):
                groups['magnetic_fields'].append(feature)
            elif any(keyword in feature_lower for keyword in ['convection', 'turbulence']):
                groups['convection_zones'].append(feature)
            elif any(keyword in feature_lower for keyword in ['photosphere', 'surface']):
                groups['photosphere'].append(feature)
            elif any(keyword in feature_lower for keyword in ['chromosphere', 'atmosphere']):
                groups['chromosphere'].append(feature)
            elif any(keyword in feature_lower for keyword in ['corona', 'wind']):
                groups['corona'].append(feature)
        
        return {k: v for k, v in groups.items() if v}  # Remove empty groups
    
    def _create_exoplanet_groups(self) -> Dict[str, List[str]]:
        """Create exoplanet feature groups"""
        
        groups = {
            'orbital_parameters': [],
            'planetary_properties': [],
            'atmospheric_properties': [],
            'surface_conditions': [],
            'habitability_factors': [],
            'biosignatures': [],
            'climate_variables': [],
            'geophysical_properties': []
        }
        
        for feature in self.feature_names:
            feature_lower = feature.lower()
            
            if any(keyword in feature_lower for keyword in ['period', 'eccentricity', 'inclination', 'semi_major']):
                groups['orbital_parameters'].append(feature)
            elif any(keyword in feature_lower for keyword in ['mass', 'radius', 'density', 'gravity']):
                groups['planetary_properties'].append(feature)
            elif any(keyword in feature_lower for keyword in ['pressure', 'temperature', 'composition', 'scale_height']):
                groups['atmospheric_properties'].append(feature)
            elif any(keyword in feature_lower for keyword in ['surface', 'albedo', 'emissivity']):
                groups['surface_conditions'].append(feature)
            elif any(keyword in feature_lower for keyword in ['habitable', 'goldilocks', 'water', 'life']):
                groups['habitability_factors'].append(feature)
            elif any(keyword in feature_lower for keyword in ['o2', 'o3', 'ch4', 'biosignature']):
                groups['biosignatures'].append(feature)
            elif any(keyword in feature_lower for keyword in ['climate', 'weather', 'circulation']):
                groups['climate_variables'].append(feature)
            elif any(keyword in feature_lower for keyword in ['magnetic', 'tidal', 'rotation']):
                groups['geophysical_properties'].append(feature)
        
        return {k: v for k, v in groups.items() if v}
    
    def _create_environmental_groups(self) -> Dict[str, List[str]]:
        """Create environmental feature groups"""
        
        groups = {
            'atmospheric_dynamics': [],
            'thermodynamics': [],
            'hydrodynamics': [],
            'radiative_transfer': [],
            'chemical_processes': [],
            'microphysics': [],
            'energy_balance': [],
            'mass_transport': []
        }
        
        for feature in self.feature_names:
            feature_lower = feature.lower()
            
            if any(keyword in feature_lower for keyword in ['wind', 'circulation', 'vorticity', 'pressure']):
                groups['atmospheric_dynamics'].append(feature)
            elif any(keyword in feature_lower for keyword in ['temperature', 'entropy', 'heat', 'thermal']):
                groups['thermodynamics'].append(feature)
            elif any(keyword in feature_lower for keyword in ['humidity', 'vapor', 'cloud', 'precipitation']):
                groups['hydrodynamics'].append(feature)
            elif any(keyword in feature_lower for keyword in ['radiation', 'flux', 'albedo', 'optical']):
                groups['radiative_transfer'].append(feature)
            elif any(keyword in feature_lower for keyword in ['chemistry', 'reaction', 'species', 'concentration']):
                groups['chemical_processes'].append(feature)
            elif any(keyword in feature_lower for keyword in ['droplet', 'particle', 'aerosol', 'nucleation']):
                groups['microphysics'].append(feature)
            elif any(keyword in feature_lower for keyword in ['energy', 'heating', 'cooling', 'balance']):
                groups['energy_balance'].append(feature)
            elif any(keyword in feature_lower for keyword in ['transport', 'diffusion', 'advection', 'mixing']):
                groups['mass_transport'].append(feature)
        
        return {k: v for k, v in groups.items() if v}
    
    def _create_physics_groups(self) -> Dict[str, List[str]]:
        """Create physics feature groups"""
        
        groups = {
            'electromagnetic': [],
            'gravitational': [],
            'thermodynamic': [],
            'quantum': [],
            'relativity': [],
            'fluid_dynamics': [],
            'plasma_physics': [],
            'particle_physics': []
        }
        
        for feature in self.feature_names:
            feature_lower = feature.lower()
            
            if any(keyword in feature_lower for keyword in ['electric', 'magnetic', 'field', 'charge']):
                groups['electromagnetic'].append(feature)
            elif any(keyword in feature_lower for keyword in ['gravity', 'mass', 'acceleration', 'force']):
                groups['gravitational'].append(feature)
            elif any(keyword in feature_lower for keyword in ['temperature', 'pressure', 'entropy', 'energy']):
                groups['thermodynamic'].append(feature)
            elif any(keyword in feature_lower for keyword in ['quantum', 'wave', 'particle', 'spin']):
                groups['quantum'].append(feature)
            elif any(keyword in feature_lower for keyword in ['relativity', 'spacetime', 'metric']):
                groups['relativity'].append(feature)
            elif any(keyword in feature_lower for keyword in ['fluid', 'flow', 'velocity', 'turbulence']):
                groups['fluid_dynamics'].append(feature)
            elif any(keyword in feature_lower for keyword in ['plasma', 'ion', 'electron', 'conductivity']):
                groups['plasma_physics'].append(feature)
            elif any(keyword in feature_lower for keyword in ['particle', 'decay', 'interaction', 'cross_section']):
                groups['particle_physics'].append(feature)
        
        return {k: v for k, v in groups.items() if v}
    
    def _create_physiological_groups(self) -> Dict[str, List[str]]:
        """Create physiological feature groups"""
        
        groups = {
            'metabolic_pathways': [],
            'enzyme_kinetics': [],
            'transport_processes': [],
            'signaling_pathways': [],
            'genetic_regulation': [],
            'stress_response': [],
            'homeostasis': [],
            'energy_production': []
        }
        
        for feature in self.feature_names:
            feature_lower = feature.lower()
            
            if any(keyword in feature_lower for keyword in ['metabolism', 'pathway', 'cycle', 'glycolysis']):
                groups['metabolic_pathways'].append(feature)
            elif any(keyword in feature_lower for keyword in ['enzyme', 'kinetic', 'catalysis', 'inhibition']):
                groups['enzyme_kinetics'].append(feature)
            elif any(keyword in feature_lower for keyword in ['transport', 'membrane', 'channel', 'pump']):
                groups['transport_processes'].append(feature)
            elif any(keyword in feature_lower for keyword in ['signal', 'receptor', 'ligand', 'cascade']):
                groups['signaling_pathways'].append(feature)
            elif any(keyword in feature_lower for keyword in ['gene', 'transcription', 'translation', 'expression']):
                groups['genetic_regulation'].append(feature)
            elif any(keyword in feature_lower for keyword in ['stress', 'shock', 'adaptation', 'survival']):
                groups['stress_response'].append(feature)
            elif any(keyword in feature_lower for keyword in ['homeostasis', 'balance', 'regulation', 'control']):
                groups['homeostasis'].append(feature)
            elif any(keyword in feature_lower for keyword in ['atp', 'energy', 'respiration', 'photosynthesis']):
                groups['energy_production'].append(feature)
        
        return {k: v for k, v in groups.items() if v}
    
    def _create_biosignature_groups(self) -> Dict[str, List[str]]:
        """Create biosignature feature groups"""
        
        groups = {
            'atmospheric_biosignatures': [],
            'surface_biosignatures': [],
            'temporal_biosignatures': [],
            'metabolic_biosignatures': [],
            'structural_biosignatures': [],
            'disequilibrium_indicators': [],
            'seasonal_variations': [],
            'anthropogenic_signatures': []
        }
        
        for feature in self.feature_names:
            feature_lower = feature.lower()
            
            if any(keyword in feature_lower for keyword in ['o2', 'o3', 'ch4', 'atmospheric', 'gas']):
                groups['atmospheric_biosignatures'].append(feature)
            elif any(keyword in feature_lower for keyword in ['surface', 'vegetation', 'chlorophyll', 'reflectance']):
                groups['surface_biosignatures'].append(feature)
            elif any(keyword in feature_lower for keyword in ['temporal', 'time', 'variation', 'periodic']):
                groups['temporal_biosignatures'].append(feature)
            elif any(keyword in feature_lower for keyword in ['metabolic', 'enzyme', 'pathway', 'biochemical']):
                groups['metabolic_biosignatures'].append(feature)
            elif any(keyword in feature_lower for keyword in ['structure', 'complexity', 'organization', 'pattern']):
                groups['structural_biosignatures'].append(feature)
            elif any(keyword in feature_lower for keyword in ['disequilibrium', 'imbalance', 'gradient', 'deviation']):
                groups['disequilibrium_indicators'].append(feature)
            elif any(keyword in feature_lower for keyword in ['seasonal', 'annual', 'cyclic', 'periodic']):
                groups['seasonal_variations'].append(feature)
            elif any(keyword in feature_lower for keyword in ['anthropogenic', 'artificial', 'technology', 'industrial']):
                groups['anthropogenic_signatures'].append(feature)
        
        return {k: v for k, v in groups.items() if v}
    
    def _create_generic_groups(self) -> Dict[str, List[str]]:
        """Create generic feature groups"""
        
        # Use clustering to create groups
        if len(self.feature_names) < 5:
            return {'all_features': self.feature_names}
        
        # Simple grouping by feature name similarity
        groups = {}
        for feature in self.feature_names:
            # Extract base name (remove numbers and common suffixes)
            base_name = feature.split('_')[0].split('.')[0]
            if base_name not in groups:
                groups[base_name] = []
            groups[base_name].append(feature)
        
        return groups

class SHAPExplainer:
    """SHAP-based explainer for surrogate models"""
    
    def __init__(self, model: BaseModelWrapper, domain: DataDomain, 
                 config: ExplanationConfig = None):
        self.model = model
        self.domain = domain
        self.config = config or ExplanationConfig()
        
        # Initialize feature group manager
        self.feature_manager = FeatureGroupManager(domain)
        
        # SHAP explainer
        self.explainer = None
        self.background_data = None
        self.feature_names = None
        self.feature_groups = None
        
        # Explanation cache
        self.explanation_cache = {}
        
        logger.info(f"Initialized SHAP explainer for {domain.value} domain")
    
    def fit(self, X: np.ndarray, feature_names: List[str] = None, 
            background_data: np.ndarray = None):
        """Fit the SHAP explainer"""
        
        logger.info(f"Fitting SHAP explainer on {X.shape[0]} samples")
        
        # Store feature names
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        # Create feature groups
        self.feature_groups = self.feature_manager.create_domain_groups(self.feature_names)
        
        # Sample background data if not provided
        if background_data is None:
            background_size = min(self.config.background_size, X.shape[0])
            background_indices = np.random.choice(X.shape[0], size=background_size, replace=False)
            self.background_data = X[background_indices]
        else:
            self.background_data = background_data
        
        # Create appropriate explainer based on model type
        try:
            # Try to create a model-specific explainer
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'predict'):
                # For sklearn-compatible models
                self.explainer = KernelExplainer(
                    self.model.predict, 
                    self.background_data
                )
            else:
                # For custom models, use KernelExplainer
                self.explainer = KernelExplainer(
                    self._model_predict_wrapper,
                    self.background_data
                )
            
            logger.info(f"Created SHAP explainer: {type(self.explainer).__name__}")
            
        except Exception as e:
            logger.error(f"Error creating SHAP explainer: {e}")
            raise
    
    def _model_predict_wrapper(self, X: np.ndarray) -> np.ndarray:
        """Wrapper for model prediction"""
        try:
            predictions = self.model.predict(X)
            
            # Ensure numpy array
            if hasattr(predictions, 'cpu'):
                predictions = predictions.cpu().numpy()
            
            # Flatten if needed
            if len(predictions.shape) > 2:
                predictions = predictions.reshape(predictions.shape[0], -1)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in model prediction wrapper: {e}")
            # Return zeros as fallback
            return np.zeros((X.shape[0], 1))
    
    def explain(self, X: np.ndarray, max_evals: int = None) -> Dict[str, Any]:
        """Generate SHAP explanations for input data"""
        
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        max_evals = max_evals or self.config.max_evals
        
        logger.info(f"Generating SHAP explanations for {X.shape[0]} samples")
        
        try:
            # Generate SHAP values
            start_time = time.time()
            shap_values = self.explainer.shap_values(X, max_evals=max_evals)
            explanation_time = time.time() - start_time
            
            logger.info(f"Generated SHAP values in {explanation_time:.2f}s")
            
            # Process results
            if isinstance(shap_values, list):
                # Multi-output case
                shap_values = shap_values[0]  # Use first output
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(shap_values)
            
            # Calculate pathway importance
            pathway_importance = self._calculate_pathway_importance(shap_values)
            
            # Generate explanations
            explanations = {
                'shap_values': shap_values,
                'feature_importance': feature_importance,
                'pathway_importance': pathway_importance,
                'feature_names': self.feature_names,
                'feature_groups': self.feature_groups,
                'explanation_time': explanation_time,
                'sample_size': X.shape[0],
                'timestamp': datetime.now().isoformat()
            }
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanations: {e}")
            raise
    
    def _calculate_feature_importance(self, shap_values: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance from SHAP values"""
        
        # Mean absolute SHAP values
        importance_scores = np.mean(np.abs(shap_values), axis=0)
        
        # Create importance dictionary
        importance_dict = {}
        for i, score in enumerate(importance_scores):
            feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            importance_dict[feature_name] = float(score)
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def _calculate_pathway_importance(self, shap_values: np.ndarray) -> Dict[str, float]:
        """Calculate pathway-level importance"""
        
        pathway_importance = {}
        
        for pathway_name, feature_indices in self.feature_groups.items():
            # Get SHAP values for features in this pathway
            pathway_shap_values = []
            
            for feature_name in feature_indices:
                if feature_name in self.feature_names:
                    feature_idx = self.feature_names.index(feature_name)
                    if feature_idx < shap_values.shape[1]:
                        pathway_shap_values.append(shap_values[:, feature_idx])
            
            if pathway_shap_values:
                # Calculate pathway importance as sum of absolute SHAP values
                pathway_shap_array = np.array(pathway_shap_values).T
                pathway_score = np.mean(np.sum(np.abs(pathway_shap_array), axis=1))
                pathway_importance[pathway_name] = float(pathway_score)
        
        # Sort by importance
        sorted_pathway_importance = dict(sorted(pathway_importance.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_pathway_importance
    
    def plot_feature_importance(self, explanations: Dict[str, Any], 
                               max_features: int = None, 
                               save_path: Path = None) -> plt.Figure:
        """Plot feature importance"""
        
        max_features = max_features or self.config.max_features_plot
        feature_importance = explanations['feature_importance']
        
        # Get top features
        top_features = list(feature_importance.items())[:max_features]
        features, importance = zip(*top_features)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, importance, alpha=0.8)
        
        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('SHAP Feature Importance')
        ax.set_title(f'Top {len(features)} Features - {self.domain.value.title()} Domain')
        ax.grid(axis='x', alpha=0.3)
        
        # Color bars
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_pathway_importance(self, explanations: Dict[str, Any], 
                               save_path: Path = None) -> plt.Figure:
        """Plot pathway importance"""
        
        pathway_importance = explanations['pathway_importance']
        
        if not pathway_importance:
            logger.warning("No pathway importance data available")
            return None
        
        # Prepare data
        pathways = list(pathway_importance.keys())
        importance = list(pathway_importance.values())
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_pos = np.arange(len(pathways))
        bars = ax.barh(y_pos, importance, alpha=0.8)
        
        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pathways)
        ax.set_xlabel('Pathway Importance (Sum of |SHAP values|)')
        ax.set_title(f'Pathway Importance - {self.domain.value.title()} Domain')
        ax.grid(axis='x', alpha=0.3)
        
        # Color bars
        colors = plt.cm.plasma(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_shap_summary(self, explanations: Dict[str, Any], 
                         X: np.ndarray, save_path: Path = None) -> plt.Figure:
        """Plot SHAP summary plot"""
        
        shap_values = explanations['shap_values']
        feature_names = explanations['feature_names']
        
        # Create summary plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Limit features for readability
        max_features = min(self.config.max_features_plot, len(feature_names))
        
        # Calculate feature importance for ordering
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        top_indices = np.argsort(feature_importance)[-max_features:][::-1]
        
        # Plot
        shap.summary_plot(
            shap_values[:, top_indices], 
            X[:, top_indices], 
            feature_names=[feature_names[i] for i in top_indices],
            show=False,
            ax=ax
        )
        
        ax.set_title(f'SHAP Summary - {self.domain.value.title()} Domain')
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_explanations(self, explanations: Dict[str, Any], 
                         save_path: Path) -> None:
        """Save explanations to file"""
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_explanations = {}
        for key, value in explanations.items():
            if isinstance(value, np.ndarray):
                serializable_explanations[key] = value.tolist()
            else:
                serializable_explanations[key] = value
        
        # Save to JSON
        with open(save_path, 'w') as f:
            json.dump(serializable_explanations, f, indent=2)
        
        logger.info(f"Saved explanations to {save_path}")

class SHAPExplainerManager:
    """Manager for SHAP explainers across different domains and models"""
    
    def __init__(self, surrogate_manager: SurrogateManager, 
                 metadata_manager: MetadataManager):
        self.surrogate_manager = surrogate_manager
        self.metadata_manager = metadata_manager
        self.explainers = {}
        
    def create_explainer(self, domain: DataDomain, model_type: str = None, 
                        config: ExplanationConfig = None) -> SHAPExplainer:
        """Create SHAP explainer for a domain"""
        
        # Get appropriate model
        if model_type and 'cube' in model_type.lower():
            surrogate_mode = SurrogateMode.DATACUBE
        else:
            surrogate_mode = SurrogateMode.SCALAR
        
        model = self.surrogate_manager.get_model(surrogate_mode)
        if not model:
            raise ValueError(f"No model available for {surrogate_mode}")
        
        # Create explainer
        explainer = SHAPExplainer(model, domain, config)
        
        # Store explainer
        key = f"{domain.value}_{model_type or 'default'}"
        self.explainers[key] = explainer
        
        return explainer
    
    def explain_prediction(self, domain: DataDomain, X: np.ndarray, 
                          model_type: str = None, feature_names: List[str] = None,
                          background_data: np.ndarray = None) -> Dict[str, Any]:
        """Generate explanations for predictions"""
        
        key = f"{domain.value}_{model_type or 'default'}"
        
        # Get or create explainer
        if key not in self.explainers:
            explainer = self.create_explainer(domain, model_type)
        else:
            explainer = self.explainers[key]
        
        # Fit explainer if not already fitted
        if explainer.explainer is None:
            explainer.fit(X, feature_names, background_data)
        
        # Generate explanations
        explanations = explainer.explain(X)
        
        return explanations
    
    def batch_explain(self, domains: List[DataDomain], X_data: Dict[str, np.ndarray],
                     feature_names: Dict[str, List[str]] = None) -> Dict[str, Any]:
        """Generate explanations for multiple domains"""
        
        all_explanations = {}
        
        for domain in domains:
            if domain.value not in X_data:
                continue
            
            try:
                X = X_data[domain.value]
                features = feature_names.get(domain.value) if feature_names else None
                
                explanations = self.explain_prediction(domain, X, feature_names=features)
                all_explanations[domain.value] = explanations
                
            except Exception as e:
                logger.error(f"Error explaining {domain.value}: {e}")
                all_explanations[domain.value] = {'error': str(e)}
        
        return all_explanations

# Utility functions
def create_shap_explainer_manager(surrogate_manager: SurrogateManager, 
                                 metadata_manager: MetadataManager) -> SHAPExplainerManager:
    """Create SHAP explainer manager"""
    return SHAPExplainerManager(surrogate_manager, metadata_manager)

def explain_surrogate_prediction(model: BaseModelWrapper, domain: DataDomain,
                                X: np.ndarray, feature_names: List[str] = None,
                                config: ExplanationConfig = None) -> Dict[str, Any]:
    """Quick function to explain a single prediction"""
    
    explainer = SHAPExplainer(model, domain, config)
    explainer.fit(X, feature_names)
    explanations = explainer.explain(X)
    
    return explanations

if __name__ == "__main__":
    # Example usage
    from surrogate import get_surrogate_manager
    from data_build.metadata_db import MetadataManager
    
    # Initialize managers
    surrogate_manager = get_surrogate_manager()
    metadata_manager = MetadataManager()
    
    # Create explainer manager
    explainer_manager = create_shap_explainer_manager(surrogate_manager, metadata_manager)
    
    # Generate sample data
    X = np.random.random((100, 50))
    feature_names = [f"feature_{i}" for i in range(50)]
    
    # Explain predictions
    explanations = explainer_manager.explain_prediction(
        DataDomain.EXOPLANET, X, feature_names=feature_names
    )
    
    print(f"Generated explanations with {len(explanations['feature_importance'])} features")
    print(f"Top 5 features: {list(explanations['feature_importance'].keys())[:5]}")
    print(f"Pathway importance: {list(explanations['pathway_importance'].keys())}") 
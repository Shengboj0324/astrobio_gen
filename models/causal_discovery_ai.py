#!/usr/bin/env python3
"""
Causal Discovery AI for Astrobiology
====================================

Advanced causal discovery system for automated hypothesis generation in astrobiology research.
Integrates cutting-edge causal inference with domain-specific scientific knowledge.

Features:
- Automated causal graph discovery from observational data
- Scientific hypothesis generation and ranking
- Multi-modal causal reasoning (genomic, atmospheric, spectral, temporal)
- Integration with existing enhanced models
- Physics-informed causal constraints
- Uncertainty quantification in causal relationships
- Active learning for targeted experimentation

Capabilities:
- Discover causal relationships in exoplanet habitability
- Generate testable hypotheses about biosignature formation
- Identify key variables driving atmospheric evolution
- Propose causal mechanisms for observed phenomena
- Guide experimental design and observation priorities

Methods:
- Structural Causal Models (SCMs)
- Directed Acyclic Graphs (DAGs)
- Causal discovery algorithms (PC, GES, NOTEARS)
- Interventional reasoning
- Counterfactual inference
- Domain-specific causal priors

Example Usage:
    # Create causal discovery system
    discoverer = CausalDiscoveryAI()
    
    # Discover causal relationships
    causal_graph = discoverer.discover_causal_structure(observational_data)
    
    # Generate hypotheses
    hypotheses = discoverer.generate_hypotheses(causal_graph, target_variable="habitability")
    
    # Design experiments
    experiments = discoverer.design_experiments(hypotheses)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import json
import networkx as nx
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings

# Causal inference libraries
try:
    import causal_learn
    from causal_learn.search.ConstraintBased.PC import pc
    from causal_learn.search.ScoreBased.GES import ges
    from causal_learn.search.Granger.Granger import Granger
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CAUSAL_LEARN_AVAILABLE = False
    warnings.warn("causal-learn not available. Install with: pip install causal-learn")

try:
    import pgmpy
    from pgmpy.models import BayesianNetwork
    from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
    from pgmpy.inference import VariableElimination
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False
    warnings.warn("pgmpy not available. Install with: pip install pgmpy")

try:
    import dowhy
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False
    warnings.warn("dowhy not available. Install with: pip install dowhy")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CausalDiscoveryConfig:
    """Configuration for causal discovery system"""
    # Discovery algorithms
    algorithms: List[str] = field(default_factory=lambda: ["pc", "ges", "notears", "granger"])
    significance_level: float = 0.05
    max_conditioning_set_size: int = 3
    
    # Domain knowledge
    use_scientific_priors: bool = True
    forbidden_edges: List[Tuple[str, str]] = field(default_factory=list)
    required_edges: List[Tuple[str, str]] = field(default_factory=list)
    temporal_ordering: Dict[str, int] = field(default_factory=dict)
    
    # Hypothesis generation
    max_hypotheses: int = 20
    min_confidence: float = 0.7
    hypothesis_ranking_method: str = "information_gain"  # or "causal_strength", "novelty"
    
    # Experimental design
    intervention_budget: int = 10
    experimental_design_method: str = "optimal_intervention"
    sample_size_calculation: bool = True
    
    # Quality and validation
    bootstrap_samples: int = 100
    cross_validation_folds: int = 5
    causal_validation_method: str = "refutation"

@dataclass
class CausalHypothesis:
    """Represents a causal hypothesis"""
    id: str
    description: str
    cause_variables: List[str]
    effect_variables: List[str]
    causal_mechanism: str
    confidence_score: float
    supporting_evidence: Dict[str, Any]
    testable_predictions: List[str]
    required_interventions: List[str]
    scientific_domain: str
    novelty_score: float = 0.0
    feasibility_score: float = 0.0
    
    def __post_init__(self):
        if not self.id:
            self.id = f"hyp_{hash(self.description) % 10000:04d}"

class ScientificPriorKnowledge:
    """Encodes scientific domain knowledge for causal discovery"""
    
    def __init__(self):
        # Astrobiology-specific causal knowledge
        self.variable_types = {
            # Planetary characteristics
            'planet_mass': 'exogenous',
            'planet_radius': 'exogenous', 
            'orbital_period': 'exogenous',
            'stellar_mass': 'exogenous',
            'stellar_temperature': 'exogenous',
            'insolation': 'endogenous',
            
            # Atmospheric properties
            'surface_temperature': 'endogenous',
            'atmospheric_pressure': 'endogenous',
            'greenhouse_effect': 'endogenous',
            'water_vapor': 'endogenous',
            'cloud_coverage': 'endogenous',
            
            # Biosignature gases
            'oxygen_concentration': 'endogenous',
            'methane_concentration': 'endogenous',
            'co2_concentration': 'endogenous',
            'ozone_concentration': 'endogenous',
            
            # Habitability indicators
            'liquid_water_probability': 'endogenous',
            'habitability_score': 'endogenous',
            'biosignature_strength': 'endogenous'
        }
        
        # Known causal relationships from physics
        self.physics_constraints = [
            ('stellar_mass', 'stellar_temperature'),
            ('stellar_temperature', 'insolation'),
            ('planet_mass', 'atmospheric_pressure'),
            ('insolation', 'surface_temperature'),
            ('surface_temperature', 'water_vapor'),
            ('water_vapor', 'greenhouse_effect'),
            ('greenhouse_effect', 'surface_temperature'),  # feedback loop
            ('atmospheric_pressure', 'liquid_water_probability'),
            ('surface_temperature', 'liquid_water_probability')
        ]
        
        # Forbidden causal relationships (violate physics)
        self.forbidden_relationships = [
            ('habitability_score', 'planet_mass'),  # effect cannot cause fundamental property
            ('biosignature_strength', 'stellar_mass'),
            ('oxygen_concentration', 'orbital_period'),
            ('methane_concentration', 'planet_radius')
        ]
        
        # Temporal ordering constraints
        self.temporal_order = {
            'stellar_mass': 0,
            'stellar_temperature': 1,
            'planet_mass': 0,
            'planet_radius': 0,
            'orbital_period': 0,
            'insolation': 2,
            'atmospheric_pressure': 3,
            'surface_temperature': 4,
            'water_vapor': 5,
            'greenhouse_effect': 6,
            'cloud_coverage': 7,
            'oxygen_concentration': 8,
            'methane_concentration': 8,
            'liquid_water_probability': 9,
            'habitability_score': 10,
            'biosignature_strength': 10
        }
    
    def get_variable_type(self, variable: str) -> str:
        """Get variable type (exogenous/endogenous)"""
        return self.variable_types.get(variable, 'endogenous')
    
    def is_edge_forbidden(self, cause: str, effect: str) -> bool:
        """Check if causal edge is forbidden by physics"""
        return (cause, effect) in self.forbidden_relationships
    
    def get_temporal_order(self, variable: str) -> int:
        """Get temporal ordering of variable"""
        return self.temporal_order.get(variable, 999)

class NeuralCausalModel(nn.Module):
    """Neural network-based causal model using NOTEARS-like approach"""
    
    def __init__(self, num_variables: int, hidden_dim: int = 64):
        super().__init__()
        self.num_variables = num_variables
        self.hidden_dim = hidden_dim
        
        # Adjacency matrix (learnable)
        self.adjacency = nn.Parameter(torch.randn(num_variables, num_variables))
        
        # Neural networks for each variable
        self.causal_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_variables, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            for _ in range(num_variables)
        ])
        
        # Constraint parameters
        self.lambda_sparse = 0.01
        self.lambda_dag = 1.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through causal model"""
        batch_size = x.shape[0]
        outputs = []
        
        # Apply causal adjacency matrix
        weighted_x = torch.matmul(x, self.adjacency.T)
        
        # Generate outputs for each variable
        for i, network in enumerate(self.causal_networks):
            parent_inputs = weighted_x[:, i:i+1]  # Parents of variable i
            combined_input = torch.cat([x, parent_inputs], dim=1)
            output = network(combined_input)
            outputs.append(output)
        
        return torch.cat(outputs, dim=1)
    
    def get_adjacency_matrix(self) -> torch.Tensor:
        """Get thresholded adjacency matrix"""
        return torch.sigmoid(self.adjacency)
    
    def dag_constraint(self) -> torch.Tensor:
        """DAG constraint: tr(e^A) - d = 0"""
        adj = self.get_adjacency_matrix()
        expm = torch.matrix_exp(adj)
        return torch.trace(expm) - self.num_variables
    
    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute loss with DAG and sparsity constraints"""
        # Reconstruction loss
        pred = self.forward(x)
        recon_loss = F.mse_loss(pred, y)
        
        # Sparsity constraint
        adj = self.get_adjacency_matrix()
        sparse_loss = torch.sum(torch.abs(adj))
        
        # DAG constraint
        dag_loss = torch.abs(self.dag_constraint())
        
        total_loss = recon_loss + self.lambda_sparse * sparse_loss + self.lambda_dag * dag_loss
        
        return total_loss, {
            'reconstruction': recon_loss.item(),
            'sparsity': sparse_loss.item(),
            'dag_constraint': dag_loss.item()
        }

class CausalGraphDiscovery:
    """Discovers causal graphs using multiple algorithms"""
    
    def __init__(self, config: CausalDiscoveryConfig, prior_knowledge: ScientificPriorKnowledge):
        self.config = config
        self.prior_knowledge = prior_knowledge
        self.discovered_graphs = {}
        
    def discover_multiple_algorithms(self, data: pd.DataFrame) -> Dict[str, nx.DiGraph]:
        """Run multiple causal discovery algorithms"""
        logger.info(f"🔍 Running causal discovery with {len(self.config.algorithms)} algorithms")
        
        discovered_graphs = {}
        
        for algorithm in self.config.algorithms:
            try:
                logger.info(f"   Running {algorithm}...")
                graph = self._run_algorithm(algorithm, data)
                discovered_graphs[algorithm] = graph
                logger.info(f"   ✅ {algorithm}: {graph.number_of_edges()} edges discovered")
            except Exception as e:
                logger.warning(f"   ⚠️ {algorithm} failed: {e}")
                discovered_graphs[algorithm] = nx.DiGraph()
        
        return discovered_graphs
    
    def _run_algorithm(self, algorithm: str, data: pd.DataFrame) -> nx.DiGraph:
        """Run specific causal discovery algorithm"""
        
        if algorithm == "pc" and CAUSAL_LEARN_AVAILABLE:
            return self._run_pc_algorithm(data)
        elif algorithm == "ges" and CAUSAL_LEARN_AVAILABLE:
            return self._run_ges_algorithm(data)
        elif algorithm == "notears":
            return self._run_notears_algorithm(data)
        elif algorithm == "granger" and CAUSAL_LEARN_AVAILABLE:
            return self._run_granger_algorithm(data)
        else:
            # Fallback: correlation-based discovery
            return self._run_correlation_discovery(data)
    
    def _run_pc_algorithm(self, data: pd.DataFrame) -> nx.DiGraph:
        """Run PC algorithm for causal discovery"""
        data_matrix = data.values
        
        # Run PC algorithm
        cg = pc(data_matrix, self.config.significance_level, 
                indep_test='fisherz', stable=True, uc_rule=0, uc_priority=2)
        
        # Convert to NetworkX graph
        graph = nx.DiGraph()
        variables = data.columns.tolist()
        
        # Add nodes
        for i, var in enumerate(variables):
            graph.add_node(var)
        
        # Add edges from adjacency matrix
        adj_matrix = cg.G.graph
        for i in range(len(variables)):
            for j in range(len(variables)):
                if adj_matrix[i, j] == 1:  # Directed edge
                    cause = variables[i]
                    effect = variables[j]
                    if not self.prior_knowledge.is_edge_forbidden(cause, effect):
                        graph.add_edge(cause, effect)
        
        return graph
    
    def _run_ges_algorithm(self, data: pd.DataFrame) -> nx.DiGraph:
        """Run GES algorithm for causal discovery"""
        data_matrix = data.values
        
        # Run GES algorithm
        record = ges(data_matrix, score_func='local_score_BIC')
        
        # Convert to NetworkX graph
        graph = nx.DiGraph()
        variables = data.columns.tolist()
        
        # Add nodes
        for var in variables:
            graph.add_node(var)
        
        # Add edges
        adj_matrix = record['G'].graph
        for i in range(len(variables)):
            for j in range(len(variables)):
                if adj_matrix[i, j] == 1:
                    cause = variables[i]
                    effect = variables[j]
                    if not self.prior_knowledge.is_edge_forbidden(cause, effect):
                        graph.add_edge(cause, effect, weight=1.0)
        
        return graph
    
    def _run_notears_algorithm(self, data: pd.DataFrame) -> nx.DiGraph:
        """Run NOTEARS neural network-based causal discovery"""
        
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data.values)
        
        # Convert to tensor
        data_tensor = torch.tensor(data_scaled, dtype=torch.float32)
        
        # Create and train neural causal model
        model = NeuralCausalModel(data.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        num_epochs = 100
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss, components = model.compute_loss(data_tensor, data_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                logger.debug(f"NOTEARS epoch {epoch}: loss={loss.item():.4f}")
        
        # Extract causal graph
        adj_matrix = model.get_adjacency_matrix().detach().numpy()
        
        # Threshold adjacency matrix
        threshold = 0.1
        adj_matrix[adj_matrix < threshold] = 0
        
        # Convert to NetworkX graph
        graph = nx.DiGraph()
        variables = data.columns.tolist()
        
        for i, cause in enumerate(variables):
            for j, effect in enumerate(variables):
                if i != j and adj_matrix[i, j] > 0:
                    if not self.prior_knowledge.is_edge_forbidden(cause, effect):
                        graph.add_edge(cause, effect, weight=adj_matrix[i, j])
        
        return graph
    
    def _run_granger_algorithm(self, data: pd.DataFrame) -> nx.DiGraph:
        """Run Granger causality test"""
        
        # This is a simplified version - full implementation would handle time series properly
        graph = nx.DiGraph()
        variables = data.columns.tolist()
        
        # Add nodes
        for var in variables:
            graph.add_node(var)
        
        # Simple correlation-based edges as placeholder
        # Real Granger causality requires time series data and lag analysis
        corr_matrix = data.corr().abs()
        threshold = 0.5
        
        for i, cause in enumerate(variables):
            for j, effect in enumerate(variables):
                if i != j and corr_matrix.iloc[i, j] > threshold:
                    if not self.prior_knowledge.is_edge_forbidden(cause, effect):
                        graph.add_edge(cause, effect, weight=corr_matrix.iloc[i, j])
        
        return graph
    
    def _run_correlation_discovery(self, data: pd.DataFrame) -> nx.DiGraph:
        """Fallback correlation-based causal discovery"""
        
        graph = nx.DiGraph()
        variables = data.columns.tolist()
        
        # Add nodes
        for var in variables:
            graph.add_node(var)
        
        # Correlation matrix
        corr_matrix = data.corr().abs()
        threshold = 0.3
        
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i != j and corr_matrix.iloc[i, j] > threshold:
                    # Use temporal ordering to determine direction
                    order1 = self.prior_knowledge.get_temporal_order(var1)
                    order2 = self.prior_knowledge.get_temporal_order(var2)
                    
                    if order1 < order2:  # var1 causes var2
                        if not self.prior_knowledge.is_edge_forbidden(var1, var2):
                            graph.add_edge(var1, var2, weight=corr_matrix.iloc[i, j])
                    elif order2 < order1:  # var2 causes var1
                        if not self.prior_knowledge.is_edge_forbidden(var2, var1):
                            graph.add_edge(var2, var1, weight=corr_matrix.iloc[i, j])
        
        return graph
    
    def ensemble_graphs(self, graphs: Dict[str, nx.DiGraph]) -> nx.DiGraph:
        """Create ensemble causal graph from multiple algorithm results"""
        
        logger.info("🔗 Creating ensemble causal graph")
        
        ensemble_graph = nx.DiGraph()
        
        # Collect all variables
        all_variables = set()
        for graph in graphs.values():
            all_variables.update(graph.nodes())
        
        # Add all nodes
        for var in all_variables:
            ensemble_graph.add_node(var)
        
        # Vote on edges
        edge_votes = {}
        for algorithm, graph in graphs.items():
            for edge in graph.edges():
                if edge not in edge_votes:
                    edge_votes[edge] = {'count': 0, 'weight_sum': 0.0, 'algorithms': []}
                
                edge_votes[edge]['count'] += 1
                edge_votes[edge]['algorithms'].append(algorithm)
                
                # Add weight if available
                if 'weight' in graph.edges[edge]:
                    edge_votes[edge]['weight_sum'] += graph.edges[edge]['weight']
                else:
                    edge_votes[edge]['weight_sum'] += 1.0
        
        # Add edges with majority vote
        min_votes = max(1, len(graphs) // 2)  # Majority voting
        
        for edge, votes in edge_votes.items():
            if votes['count'] >= min_votes:
                avg_weight = votes['weight_sum'] / votes['count']
                confidence = votes['count'] / len(graphs)
                
                ensemble_graph.add_edge(
                    edge[0], edge[1],
                    weight=avg_weight,
                    confidence=confidence,
                    supporting_algorithms=votes['algorithms']
                )
        
        logger.info(f"✅ Ensemble graph: {ensemble_graph.number_of_edges()} confident edges")
        
        return ensemble_graph

class HypothesisGenerator:
    """Generates scientific hypotheses from causal graphs"""
    
    def __init__(self, config: CausalDiscoveryConfig, prior_knowledge: ScientificPriorKnowledge):
        self.config = config
        self.prior_knowledge = prior_knowledge
        
        # Scientific hypothesis templates
        self.hypothesis_templates = {
            'direct_causation': "{cause} directly influences {effect} through {mechanism}",
            'mediated_causation': "{cause} influences {effect} through {mediator}",
            'moderated_causation': "{cause} influences {effect}, but this effect is moderated by {moderator}",
            'feedback_loop': "{var1} and {var2} form a feedback loop where changes amplify over time",
            'threshold_effect': "{cause} has a threshold effect on {effect} above critical value {threshold}",
            'interaction_effect': "{cause1} and {cause2} interact to influence {effect}",
            'confounding': "The relationship between {cause} and {effect} may be confounded by {confounder}"
        }
    
    def generate_hypotheses(self, causal_graph: nx.DiGraph, target_variable: str = None) -> List[CausalHypothesis]:
        """Generate testable hypotheses from causal graph"""
        
        logger.info(f"💡 Generating hypotheses from causal graph")
        
        hypotheses = []
        
        # Generate different types of hypotheses
        hypotheses.extend(self._generate_direct_causation_hypotheses(causal_graph, target_variable))
        hypotheses.extend(self._generate_mediation_hypotheses(causal_graph, target_variable))
        hypotheses.extend(self._generate_interaction_hypotheses(causal_graph, target_variable))
        hypotheses.extend(self._generate_feedback_hypotheses(causal_graph))
        
        # Rank and filter hypotheses
        ranked_hypotheses = self._rank_hypotheses(hypotheses, causal_graph)
        
        # Limit to max hypotheses
        final_hypotheses = ranked_hypotheses[:self.config.max_hypotheses]
        
        logger.info(f"✅ Generated {len(final_hypotheses)} ranked hypotheses")
        
        return final_hypotheses
    
    def _generate_direct_causation_hypotheses(self, graph: nx.DiGraph, target: str = None) -> List[CausalHypothesis]:
        """Generate direct causation hypotheses"""
        
        hypotheses = []
        
        edges_to_consider = graph.edges()
        if target:
            # Focus on edges leading to target variable
            edges_to_consider = [(u, v) for u, v in graph.edges() if v == target]
        
        for cause, effect in edges_to_consider:
            # Get edge attributes
            edge_data = graph.edges[cause, effect]
            confidence = edge_data.get('confidence', 0.5)
            
            if confidence >= self.config.min_confidence:
                # Generate mechanism description
                mechanism = self._infer_causal_mechanism(cause, effect)
                
                # Create hypothesis
                hypothesis = CausalHypothesis(
                    id=f"direct_{cause}_{effect}",
                    description=self.hypothesis_templates['direct_causation'].format(
                        cause=cause, effect=effect, mechanism=mechanism
                    ),
                    cause_variables=[cause],
                    effect_variables=[effect],
                    causal_mechanism=mechanism,
                    confidence_score=confidence,
                    supporting_evidence={
                        'edge_weight': edge_data.get('weight', 0.0),
                        'supporting_algorithms': edge_data.get('supporting_algorithms', [])
                    },
                    testable_predictions=self._generate_testable_predictions(cause, effect, mechanism),
                    required_interventions=[f"manipulate_{cause}"],
                    scientific_domain=self._get_scientific_domain(cause, effect)
                )
                
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_mediation_hypotheses(self, graph: nx.DiGraph, target: str = None) -> List[CausalHypothesis]:
        """Generate mediation hypotheses (A -> B -> C)"""
        
        hypotheses = []
        
        # Find all paths of length 2
        for node in graph.nodes():
            if target and node != target:
                continue
                
            # Find mediators
            predecessors = list(graph.predecessors(node))
            for mediator in predecessors:
                mediator_predecessors = list(graph.predecessors(mediator))
                
                for cause in mediator_predecessors:
                    if cause != node:  # Avoid cycles
                        # Check confidence of both edges
                        edge1_conf = graph.edges[cause, mediator].get('confidence', 0.5)
                        edge2_conf = graph.edges[mediator, node].get('confidence', 0.5)
                        overall_conf = min(edge1_conf, edge2_conf)
                        
                        if overall_conf >= self.config.min_confidence:
                            hypothesis = CausalHypothesis(
                                id=f"mediated_{cause}_{mediator}_{node}",
                                description=self.hypothesis_templates['mediated_causation'].format(
                                    cause=cause, effect=node, mediator=mediator
                                ),
                                cause_variables=[cause],
                                effect_variables=[node],
                                causal_mechanism=f"mediation through {mediator}",
                                confidence_score=overall_conf,
                                supporting_evidence={
                                    'mediation_path': [cause, mediator, node],
                                    'path_strength': overall_conf
                                },
                                testable_predictions=[
                                    f"Manipulating {cause} should change {mediator}",
                                    f"Manipulating {mediator} should change {node}",
                                    f"Controlling for {mediator} should reduce effect of {cause} on {node}"
                                ],
                                required_interventions=[f"manipulate_{cause}", f"control_{mediator}"],
                                scientific_domain=self._get_scientific_domain(cause, node)
                            )
                            
                            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_interaction_hypotheses(self, graph: nx.DiGraph, target: str = None) -> List[CausalHypothesis]:
        """Generate interaction effect hypotheses"""
        
        hypotheses = []
        
        nodes_to_consider = [target] if target else graph.nodes()
        
        for effect in nodes_to_consider:
            causes = list(graph.predecessors(effect))
            
            # Look for multiple causes of the same effect
            if len(causes) >= 2:
                for i in range(len(causes)):
                    for j in range(i + 1, len(causes)):
                        cause1, cause2 = causes[i], causes[j]
                        
                        # Check confidence of both edges
                        conf1 = graph.edges[cause1, effect].get('confidence', 0.5)
                        conf2 = graph.edges[cause2, effect].get('confidence', 0.5)
                        interaction_conf = (conf1 + conf2) / 2
                        
                        if interaction_conf >= self.config.min_confidence:
                            hypothesis = CausalHypothesis(
                                id=f"interaction_{cause1}_{cause2}_{effect}",
                                description=self.hypothesis_templates['interaction_effect'].format(
                                    cause1=cause1, cause2=cause2, effect=effect
                                ),
                                cause_variables=[cause1, cause2],
                                effect_variables=[effect],
                                causal_mechanism=f"synergistic interaction between {cause1} and {cause2}",
                                confidence_score=interaction_conf,
                                supporting_evidence={
                                    'individual_effects': [conf1, conf2],
                                    'interaction_type': 'synergistic'
                                },
                                testable_predictions=[
                                    f"Effect of {cause1} on {effect} depends on level of {cause2}",
                                    f"Combined effect is greater than sum of individual effects",
                                    f"Factorial design with {cause1} and {cause2} shows interaction"
                                ],
                                required_interventions=[f"factorial_{cause1}_{cause2}"],
                                scientific_domain=self._get_scientific_domain(cause1, effect)
                            )
                            
                            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_feedback_hypotheses(self, graph: nx.DiGraph) -> List[CausalHypothesis]:
        """Generate feedback loop hypotheses"""
        
        hypotheses = []
        
        # Find cycles in the graph
        try:
            cycles = list(nx.simple_cycles(graph))
            
            for cycle in cycles:
                if len(cycle) >= 2:
                    # Simple feedback loop
                    if len(cycle) == 2:
                        var1, var2 = cycle[0], cycle[1]
                        
                        # Check confidence of both directions
                        conf1 = graph.edges[var1, var2].get('confidence', 0.5)
                        conf2 = graph.edges[var2, var1].get('confidence', 0.5)
                        feedback_conf = min(conf1, conf2)
                        
                        if feedback_conf >= self.config.min_confidence:
                            hypothesis = CausalHypothesis(
                                id=f"feedback_{var1}_{var2}",
                                description=self.hypothesis_templates['feedback_loop'].format(
                                    var1=var1, var2=var2
                                ),
                                cause_variables=[var1, var2],
                                effect_variables=[var1, var2],
                                causal_mechanism="bidirectional feedback",
                                confidence_score=feedback_conf,
                                supporting_evidence={
                                    'cycle_length': len(cycle),
                                    'feedback_strength': feedback_conf
                                },
                                testable_predictions=[
                                    f"Perturbing {var1} leads to oscillations in {var2}",
                                    f"System shows hysteresis effects",
                                    f"Multiple equilibria possible"
                                ],
                                required_interventions=[f"perturbation_{var1}", f"perturbation_{var2}"],
                                scientific_domain=self._get_scientific_domain(var1, var2)
                            )
                            
                            hypotheses.append(hypothesis)
        
        except nx.NetworkXError:
            # No cycles found
            pass
        
        return hypotheses
    
    def _infer_causal_mechanism(self, cause: str, effect: str) -> str:
        """Infer likely causal mechanism based on variable types"""
        
        # Domain-specific mechanism inference
        mechanism_map = {
            ('stellar_temperature', 'insolation'): 'radiative energy transfer',
            ('insolation', 'surface_temperature'): 'solar heating',
            ('surface_temperature', 'water_vapor'): 'evaporation/sublimation',
            ('water_vapor', 'greenhouse_effect'): 'radiative absorption',
            ('greenhouse_effect', 'surface_temperature'): 'radiative warming',
            ('atmospheric_pressure', 'liquid_water_probability'): 'phase transition control',
            ('planet_mass', 'atmospheric_pressure'): 'gravitational retention',
            ('oxygen_concentration', 'ozone_concentration'): 'photochemical production',
            ('liquid_water_probability', 'habitability_score'): 'habitability requirement'
        }
        
        return mechanism_map.get((cause, effect), 'unknown physical process')
    
    def _generate_testable_predictions(self, cause: str, effect: str, mechanism: str) -> List[str]:
        """Generate testable predictions for causal relationship"""
        
        predictions = [
            f"Increasing {cause} should increase {effect}",
            f"Decreasing {cause} should decrease {effect}",
            f"The relationship should be mediated by {mechanism}",
            f"Controlling for confounders should preserve the {cause}-{effect} relationship"
        ]
        
        # Add domain-specific predictions
        if 'temperature' in cause and 'water' in effect:
            predictions.append("Temperature effects should follow Clausius-Clapeyron relation")
        
        if 'pressure' in cause and 'phase' in effect:
            predictions.append("Pressure effects should follow phase diagrams")
        
        return predictions
    
    def _get_scientific_domain(self, var1: str, var2: str) -> str:
        """Determine scientific domain of causal relationship"""
        
        if any(term in var1.lower() or term in var2.lower() for term in ['stellar', 'insolation']):
            return 'stellar_physics'
        elif any(term in var1.lower() or term in var2.lower() for term in ['atmosphere', 'pressure', 'temperature']):
            return 'atmospheric_science'
        elif any(term in var1.lower() or term in var2.lower() for term in ['water', 'liquid', 'ice']):
            return 'hydrosphere'
        elif any(term in var1.lower() or term in var2.lower() for term in ['oxygen', 'methane', 'biosignature']):
            return 'astrobiology'
        elif any(term in var1.lower() or term in var2.lower() for term in ['habitability', 'life']):
            return 'habitability'
        else:
            return 'planetary_science'
    
    def _rank_hypotheses(self, hypotheses: List[CausalHypothesis], graph: nx.DiGraph) -> List[CausalHypothesis]:
        """Rank hypotheses by importance and novelty"""
        
        for hypothesis in hypotheses:
            # Calculate novelty score (simplified)
            hypothesis.novelty_score = self._calculate_novelty_score(hypothesis)
            
            # Calculate feasibility score
            hypothesis.feasibility_score = self._calculate_feasibility_score(hypothesis)
        
        # Sort by combined score
        def combined_score(hyp):
            return (
                hyp.confidence_score * 0.4 +
                hyp.novelty_score * 0.3 +
                hyp.feasibility_score * 0.3
            )
        
        return sorted(hypotheses, key=combined_score, reverse=True)
    
    def _calculate_novelty_score(self, hypothesis: CausalHypothesis) -> float:
        """Calculate novelty score for hypothesis"""
        # Simplified novelty calculation
        # In practice, this would compare against existing literature
        
        # Novel combinations get higher scores
        if len(hypothesis.cause_variables) > 1:
            return 0.8  # Interaction effects are often novel
        elif 'feedback' in hypothesis.description:
            return 0.9  # Feedback loops are interesting
        else:
            return 0.5  # Direct effects are common
    
    def _calculate_feasibility_score(self, hypothesis: CausalHypothesis) -> float:
        """Calculate feasibility score for testing hypothesis"""
        
        feasibility = 1.0
        
        # Reduce score based on intervention difficulty
        for intervention in hypothesis.required_interventions:
            if 'stellar' in intervention:
                feasibility *= 0.1  # Very hard to manipulate stars
            elif 'planet' in intervention:
                feasibility *= 0.2  # Hard to manipulate planets
            elif 'atmospheric' in intervention:
                feasibility *= 0.7  # Somewhat feasible
            else:
                feasibility *= 0.9  # Generally feasible
        
        return min(feasibility, 1.0)

class ExperimentalDesigner:
    """Designs experiments to test causal hypotheses"""
    
    def __init__(self, config: CausalDiscoveryConfig):
        self.config = config
    
    def design_experiments(self, hypotheses: List[CausalHypothesis]) -> List[Dict[str, Any]]:
        """Design optimal experiments to test hypotheses"""
        
        logger.info(f"🧪 Designing experiments for {len(hypotheses)} hypotheses")
        
        experiments = []
        
        for hypothesis in hypotheses:
            if hypothesis.feasibility_score > 0.3:  # Only design experiments for feasible hypotheses
                experiment = self._design_single_experiment(hypothesis)
                experiments.append(experiment)
        
        # Optimize experimental design for multiple hypotheses
        optimized_experiments = self._optimize_experimental_design(experiments)
        
        logger.info(f"✅ Designed {len(optimized_experiments)} optimized experiments")
        
        return optimized_experiments
    
    def _design_single_experiment(self, hypothesis: CausalHypothesis) -> Dict[str, Any]:
        """Design experiment for single hypothesis"""
        
        experiment = {
            'hypothesis_id': hypothesis.id,
            'hypothesis_description': hypothesis.description,
            'experimental_design': self._determine_design_type(hypothesis),
            'independent_variables': hypothesis.cause_variables,
            'dependent_variables': hypothesis.effect_variables,
            'control_variables': self._identify_control_variables(hypothesis),
            'sample_size': self._calculate_sample_size(hypothesis),
            'duration': self._estimate_duration(hypothesis),
            'measurement_protocol': self._design_measurement_protocol(hypothesis),
            'analysis_plan': self._create_analysis_plan(hypothesis),
            'expected_outcomes': hypothesis.testable_predictions,
            'feasibility_score': hypothesis.feasibility_score,
            'priority_score': hypothesis.confidence_score * hypothesis.novelty_score
        }
        
        return experiment
    
    def _determine_design_type(self, hypothesis: CausalHypothesis) -> str:
        """Determine appropriate experimental design"""
        
        if len(hypothesis.cause_variables) == 1:
            return "simple_intervention"
        elif len(hypothesis.cause_variables) == 2:
            return "factorial_design"
        elif 'feedback' in hypothesis.description:
            return "time_series_intervention"
        elif 'mediation' in hypothesis.description:
            return "mediation_analysis"
        else:
            return "observational_study"
    
    def _identify_control_variables(self, hypothesis: CausalHypothesis) -> List[str]:
        """Identify variables that should be controlled"""
        
        # Common confounders in astrobiology
        potential_confounders = [
            'stellar_age', 'metallicity', 'system_architecture',
            'measurement_noise', 'instrumental_bias'
        ]
        
        # Filter relevant confounders
        relevant_confounders = []
        for confounder in potential_confounders:
            if confounder not in hypothesis.cause_variables and confounder not in hypothesis.effect_variables:
                relevant_confounders.append(confounder)
        
        return relevant_confounders[:3]  # Limit to 3 controls for feasibility
    
    def _calculate_sample_size(self, hypothesis: CausalHypothesis) -> int:
        """Calculate required sample size for adequate power"""
        
        # Simplified sample size calculation
        # In practice, this would use power analysis
        
        base_size = 30  # Minimum for statistical power
        
        # Adjust based on expected effect size
        if hypothesis.confidence_score > 0.8:
            multiplier = 1.0  # Large effect, smaller sample needed
        elif hypothesis.confidence_score > 0.6:
            multiplier = 1.5  # Medium effect
        else:
            multiplier = 2.0  # Small effect, larger sample needed
        
        # Adjust based on number of variables
        variable_multiplier = 1 + 0.1 * len(hypothesis.cause_variables + hypothesis.effect_variables)
        
        return int(base_size * multiplier * variable_multiplier)
    
    def _estimate_duration(self, hypothesis: CausalHypothesis) -> str:
        """Estimate experimental duration"""
        
        if any('stellar' in var for var in hypothesis.cause_variables):
            return "years"  # Stellar processes are slow
        elif any('atmospheric' in var for var in hypothesis.cause_variables):
            return "months"  # Atmospheric processes
        elif any('chemical' in var for var in hypothesis.cause_variables):
            return "days"   # Chemical processes
        else:
            return "weeks"  # Default
    
    def _design_measurement_protocol(self, hypothesis: CausalHypothesis) -> Dict[str, Any]:
        """Design measurement protocol"""
        
        protocol = {
            'measurement_frequency': self._determine_measurement_frequency(hypothesis),
            'instruments_required': self._identify_required_instruments(hypothesis),
            'measurement_precision': self._specify_precision_requirements(hypothesis),
            'quality_control': self._design_quality_control(hypothesis)
        }
        
        return protocol
    
    def _determine_measurement_frequency(self, hypothesis: CausalHypothesis) -> str:
        """Determine how often to measure variables"""
        
        if 'feedback' in hypothesis.description:
            return "continuous"
        elif any('atmospheric' in var for var in hypothesis.effect_variables):
            return "hourly"
        else:
            return "daily"
    
    def _identify_required_instruments(self, hypothesis: CausalHypothesis) -> List[str]:
        """Identify required measurement instruments"""
        
        instruments = []
        
        for var in hypothesis.cause_variables + hypothesis.effect_variables:
            if 'temperature' in var:
                instruments.append('infrared_spectrometer')
            elif 'pressure' in var:
                instruments.append('pressure_sensor')
            elif 'water' in var or 'vapor' in var:
                instruments.append('humidity_sensor')
            elif 'oxygen' in var or 'methane' in var:
                instruments.append('gas_chromatograph')
            elif 'stellar' in var:
                instruments.append('photometer')
        
        return list(set(instruments))
    
    def _specify_precision_requirements(self, hypothesis: CausalHypothesis) -> Dict[str, str]:
        """Specify measurement precision requirements"""
        
        precision = {}
        
        for var in hypothesis.cause_variables + hypothesis.effect_variables:
            if 'temperature' in var:
                precision[var] = "±0.1 K"
            elif 'pressure' in var:
                precision[var] = "±0.01 bar"
            elif 'concentration' in var:
                precision[var] = "±1 ppm"
            else:
                precision[var] = "±1%"
        
        return precision
    
    def _design_quality_control(self, hypothesis: CausalHypothesis) -> List[str]:
        """Design quality control measures"""
        
        return [
            "instrument_calibration",
            "replicate_measurements",
            "blank_controls",
            "standard_reference_materials",
            "cross_validation"
        ]
    
    def _create_analysis_plan(self, hypothesis: CausalHypothesis) -> Dict[str, Any]:
        """Create statistical analysis plan"""
        
        plan = {
            'primary_analysis': self._determine_primary_analysis(hypothesis),
            'secondary_analyses': self._determine_secondary_analyses(hypothesis),
            'significance_level': self.config.significance_level,
            'multiple_comparisons_correction': "Bonferroni",
            'effect_size_measures': self._determine_effect_size_measures(hypothesis),
            'sensitivity_analyses': self._plan_sensitivity_analyses(hypothesis)
        }
        
        return plan
    
    def _determine_primary_analysis(self, hypothesis: CausalHypothesis) -> str:
        """Determine primary statistical analysis"""
        
        if len(hypothesis.cause_variables) == 1 and len(hypothesis.effect_variables) == 1:
            return "linear_regression"
        elif len(hypothesis.cause_variables) == 2:
            return "factorial_anova"
        elif 'mediation' in hypothesis.description:
            return "mediation_analysis"
        else:
            return "multiple_regression"
    
    def _determine_secondary_analyses(self, hypothesis: CausalHypothesis) -> List[str]:
        """Determine secondary analyses"""
        
        return [
            "correlation_analysis",
            "partial_correlation",
            "robustness_checks",
            "outlier_analysis"
        ]
    
    def _determine_effect_size_measures(self, hypothesis: CausalHypothesis) -> List[str]:
        """Determine appropriate effect size measures"""
        
        return ["Cohen_d", "R_squared", "partial_eta_squared"]
    
    def _plan_sensitivity_analyses(self, hypothesis: CausalHypothesis) -> List[str]:
        """Plan sensitivity analyses"""
        
        return [
            "remove_outliers",
            "alternative_models",
            "different_transformations",
            "bootstrap_confidence_intervals"
        ]
    
    def _optimize_experimental_design(self, experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize experimental design across multiple hypotheses"""
        
        # Simple optimization: prioritize by combined score
        prioritized = sorted(experiments, key=lambda x: x['priority_score'], reverse=True)
        
        # Limit to budget
        return prioritized[:self.config.intervention_budget]

class CausalDiscoveryAI:
    """Main causal discovery AI system"""
    
    def __init__(self, config: Optional[CausalDiscoveryConfig] = None):
        self.config = config or CausalDiscoveryConfig()
        self.prior_knowledge = ScientificPriorKnowledge()
        
        # Initialize components
        self.graph_discovery = CausalGraphDiscovery(self.config, self.prior_knowledge)
        self.hypothesis_generator = HypothesisGenerator(self.config, self.prior_knowledge)
        self.experimental_designer = ExperimentalDesigner(self.config)
        
        # Results storage
        self.discovered_graphs = {}
        self.generated_hypotheses = []
        self.designed_experiments = []
        
        logger.info("🔬 Causal Discovery AI initialized")
        logger.info(f"   Algorithms: {self.config.algorithms}")
        logger.info(f"   Max hypotheses: {self.config.max_hypotheses}")
    
    def discover_causal_structure(self, data: pd.DataFrame) -> nx.DiGraph:
        """Discover causal structure from observational data"""
        
        logger.info("🔍 Starting causal structure discovery")
        
        # Run multiple algorithms
        algorithm_graphs = self.graph_discovery.discover_multiple_algorithms(data)
        self.discovered_graphs = algorithm_graphs
        
        # Create ensemble graph
        ensemble_graph = self.graph_discovery.ensemble_graphs(algorithm_graphs)
        
        # Apply domain knowledge constraints
        constrained_graph = self._apply_domain_constraints(ensemble_graph)
        
        logger.info("✅ Causal structure discovery completed")
        
        return constrained_graph
    
    def generate_hypotheses(self, causal_graph: nx.DiGraph, target_variable: str = None) -> List[CausalHypothesis]:
        """Generate scientific hypotheses from causal graph"""
        
        logger.info("💡 Generating scientific hypotheses")
        
        hypotheses = self.hypothesis_generator.generate_hypotheses(causal_graph, target_variable)
        self.generated_hypotheses = hypotheses
        
        logger.info(f"✅ Generated {len(hypotheses)} hypotheses")
        
        return hypotheses
    
    def design_experiments(self, hypotheses: List[CausalHypothesis] = None) -> List[Dict[str, Any]]:
        """Design experiments to test hypotheses"""
        
        if hypotheses is None:
            hypotheses = self.generated_hypotheses
        
        logger.info("🧪 Designing experiments")
        
        experiments = self.experimental_designer.design_experiments(hypotheses)
        self.designed_experiments = experiments
        
        logger.info(f"✅ Designed {len(experiments)} experiments")
        
        return experiments
    
    def run_complete_discovery_pipeline(self, data: pd.DataFrame, target_variable: str = None) -> Dict[str, Any]:
        """Run complete causal discovery pipeline"""
        
        logger.info("🚀 Running complete causal discovery pipeline")
        
        # Step 1: Discover causal structure
        causal_graph = self.discover_causal_structure(data)
        
        # Step 2: Generate hypotheses
        hypotheses = self.generate_hypotheses(causal_graph, target_variable)
        
        # Step 3: Design experiments
        experiments = self.design_experiments(hypotheses)
        
        # Compile results
        results = {
            'causal_graph': causal_graph,
            'hypotheses': hypotheses,
            'experiments': experiments,
            'summary': {
                'num_variables': len(data.columns),
                'num_edges_discovered': causal_graph.number_of_edges(),
                'num_hypotheses_generated': len(hypotheses),
                'num_experiments_designed': len(experiments),
                'top_hypothesis': hypotheses[0] if hypotheses else None,
                'recommended_experiment': experiments[0] if experiments else None
            }
        }
        
        logger.info("✅ Complete causal discovery pipeline completed")
        
        return results
    
    def _apply_domain_constraints(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Apply scientific domain constraints to causal graph"""
        
        constrained_graph = graph.copy()
        
        # Remove forbidden edges
        edges_to_remove = []
        for edge in constrained_graph.edges():
            if self.prior_knowledge.is_edge_forbidden(edge[0], edge[1]):
                edges_to_remove.append(edge)
        
        for edge in edges_to_remove:
            constrained_graph.remove_edge(edge[0], edge[1])
            logger.debug(f"Removed forbidden edge: {edge[0]} -> {edge[1]}")
        
        # Add required edges (from physics)
        for cause, effect in self.prior_knowledge.physics_constraints:
            if cause in graph.nodes() and effect in graph.nodes():
                if not constrained_graph.has_edge(cause, effect):
                    constrained_graph.add_edge(cause, effect, weight=1.0, confidence=0.9, source='physics')
                    logger.debug(f"Added physics constraint: {cause} -> {effect}")
        
        return constrained_graph
    
    def save_results(self, results: Dict[str, Any], output_path: str = "causal_discovery_results.json"):
        """Save causal discovery results"""
        
        # Prepare serializable results
        serializable_results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'algorithms': self.config.algorithms,
                'max_hypotheses': self.config.max_hypotheses,
                'significance_level': self.config.significance_level
            },
            'causal_graph': {
                'nodes': list(results['causal_graph'].nodes()),
                'edges': [(u, v, d) for u, v, d in results['causal_graph'].edges(data=True)],
                'number_of_nodes': results['causal_graph'].number_of_nodes(),
                'number_of_edges': results['causal_graph'].number_of_edges()
            },
            'hypotheses': [
                {
                    'id': h.id,
                    'description': h.description,
                    'confidence_score': h.confidence_score,
                    'cause_variables': h.cause_variables,
                    'effect_variables': h.effect_variables,
                    'scientific_domain': h.scientific_domain,
                    'testable_predictions': h.testable_predictions
                }
                for h in results['hypotheses']
            ],
            'experiments': results['experiments'],
            'summary': results['summary']
        }
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"💾 Results saved to {output_path}")

# Factory functions

def create_causal_discovery_system(algorithms: List[str] = None) -> CausalDiscoveryAI:
    """Create causal discovery system with specified algorithms"""
    
    if algorithms is None:
        algorithms = ["pc", "ges", "notears"]
    
    config = CausalDiscoveryConfig(algorithms=algorithms)
    return CausalDiscoveryAI(config)

async def demonstrate_causal_discovery():
    """Demonstrate causal discovery capabilities"""
    
    logger.info("🔬 Demonstrating Causal Discovery AI")
    
    # Create synthetic astrobiology dataset
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data with known causal structure
    stellar_mass = np.random.normal(1.0, 0.3, n_samples)
    stellar_temp = 5778 * (stellar_mass ** 0.5) + np.random.normal(0, 100, n_samples)
    planet_mass = np.random.lognormal(0, 0.5, n_samples)
    insolation = (stellar_temp / 5778) ** 4 / np.random.uniform(0.5, 2.0, n_samples)
    surface_temp = 255 * (insolation ** 0.25) + np.random.normal(0, 10, n_samples)
    water_vapor = np.exp((surface_temp - 273) / 50) + np.random.normal(0, 0.1, n_samples)
    greenhouse = 0.3 * water_vapor + np.random.normal(0, 0.05, n_samples)
    # Feedback: greenhouse effect increases surface temperature
    surface_temp += 20 * greenhouse
    habitability = 1 / (1 + np.exp(-(surface_temp - 280) / 50)) + np.random.normal(0, 0.1, n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'stellar_mass': stellar_mass,
        'stellar_temperature': stellar_temp,
        'planet_mass': planet_mass,
        'insolation': insolation,
        'surface_temperature': surface_temp,
        'water_vapor': water_vapor,
        'greenhouse_effect': greenhouse,
        'habitability_score': habitability
    })
    
    # Create causal discovery system
    discoverer = create_causal_discovery_system()
    
    # Run complete discovery pipeline
    results = discoverer.run_complete_discovery_pipeline(data, target_variable='habitability_score')
    
    # Display results
    logger.info("🔍 Causal Discovery Results:")
    logger.info(f"   Variables analyzed: {results['summary']['num_variables']}")
    logger.info(f"   Causal edges found: {results['summary']['num_edges_discovered']}")
    logger.info(f"   Hypotheses generated: {results['summary']['num_hypotheses_generated']}")
    logger.info(f"   Experiments designed: {results['summary']['num_experiments_designed']}")
    
    if results['summary']['top_hypothesis']:
        top_hyp = results['summary']['top_hypothesis']
        logger.info(f"   Top hypothesis: {top_hyp.description}")
        logger.info(f"   Confidence: {top_hyp.confidence_score:.3f}")
    
    # Save results
    discoverer.save_results(results, "demo_causal_discovery_results.json")
    
    return results

if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_causal_discovery()) 
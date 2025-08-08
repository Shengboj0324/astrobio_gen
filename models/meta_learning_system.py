#!/usr/bin/env python3
"""
Meta-Learning System for Astrobiology Research
==============================================

Advanced meta-learning framework for few-shot adaptation to new climate scenarios,
planetary conditions, and biological systems. Implements Model-Agnostic Meta-Learning (MAML),
Prototypical Networks, and Gradient-Based Meta-Learning.

Features:
- MAML for fast adaptation to new planetary conditions
- Prototypical networks for few-shot climate classification
- Gradient-based meta-learning for atmospheric modeling
- Task-specific meta-learning for different planet types
- Memory-augmented neural networks for experience replay
- Adaptive learning rates for different climate regimes
"""

import logging
import math
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning systems"""

    adaptation_steps: int = 5
    meta_lr: float = 1e-3
    adaptation_lr: float = 1e-2
    num_support_samples: int = 5
    num_query_samples: int = 15
    num_tasks_per_batch: int = 4
    temperature: float = 1.0
    use_second_order: bool = True
    use_adaptive_lr: bool = True
    memory_size: int = 1000


class MetaLearningBase(nn.Module, ABC):
    """Base class for meta-learning algorithms"""

    def __init__(self, config: MetaLearningConfig):
        super().__init__()
        self.config = config
        self.adaptation_steps = config.adaptation_steps
        self.meta_lr = config.meta_lr
        self.adaptation_lr = config.adaptation_lr

    @abstractmethod
    def meta_forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Meta-learning forward pass"""
        pass

    @abstractmethod
    def adapt(self, support_x: torch.Tensor, support_y: torch.Tensor) -> nn.Module:
        """Adapt model to new task"""
        pass


class MAML(MetaLearningBase):
    """Model-Agnostic Meta-Learning for fast adaptation"""

    def __init__(self, base_model: nn.Module, config: MetaLearningConfig):
        super().__init__(config)
        self.base_model = base_model
        self.meta_optimizer = torch.optim.Adam(base_model.parameters(), lr=config.meta_lr)

        # Adaptive learning rates
        if config.use_adaptive_lr:
            self.adaptation_lr_params = nn.Parameter(
                torch.ones(len(list(base_model.parameters()))) * config.adaptation_lr
            )

        logger.info("Initialized MAML meta-learning system")

    def meta_forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """MAML meta-learning forward pass"""
        # Adapt to support set
        adapted_model = self.adapt(support_x, support_y)

        # Query prediction
        query_pred = adapted_model(query_x)

        # Meta-loss
        meta_loss = F.mse_loss(query_pred, query_y)

        return {
            "meta_loss": meta_loss,
            "query_pred": query_pred,
            "adapted_params": len(list(adapted_model.parameters())),
        }

    def adapt(self, support_x: torch.Tensor, support_y: torch.Tensor) -> nn.Module:
        """Adapt model to new task using gradient descent"""
        # Clone base model
        adapted_model = deepcopy(self.base_model)

        # Gradient-based adaptation
        for step in range(self.adaptation_steps):
            # Forward pass
            support_pred = adapted_model(support_x)

            # Adaptation loss
            adaptation_loss = F.mse_loss(support_pred, support_y)

            # Compute gradients
            grads = torch.autograd.grad(
                adaptation_loss,
                adapted_model.parameters(),
                create_graph=self.config.use_second_order,
                retain_graph=True,
            )

            # Update parameters
            for param, grad, lr_param in zip(
                adapted_model.parameters(),
                grads,
                (
                    self.adaptation_lr_params
                    if self.config.use_adaptive_lr
                    else [self.adaptation_lr] * len(grads)
                ),
            ):
                if self.config.use_adaptive_lr:
                    lr = F.softplus(lr_param)
                else:
                    lr = self.adaptation_lr

                param.data = param.data - lr * grad

        return adapted_model

    def meta_update(self, meta_loss: torch.Tensor):
        """Update meta-parameters"""
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()


class PrototypicalNetworks(MetaLearningBase):
    """Prototypical Networks for few-shot classification"""

    def __init__(self, feature_extractor: nn.Module, config: MetaLearningConfig):
        super().__init__(config)
        self.feature_extractor = feature_extractor
        self.temperature = config.temperature

        logger.info("Initialized Prototypical Networks meta-learning system")

    def meta_forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Prototypical networks forward pass"""
        # Extract features
        support_features = self.feature_extractor(support_x)
        query_features = self.feature_extractor(query_x)

        # Compute prototypes
        prototypes = self._compute_prototypes(support_features, support_y)

        # Compute distances
        distances = self._compute_distances(query_features, prototypes)

        # Classification probabilities
        logits = -distances / self.temperature

        # Loss
        if query_y.dim() == 1:  # Classification
            meta_loss = F.cross_entropy(logits, query_y.long())
        else:  # Regression
            meta_loss = F.mse_loss(logits, query_y)

        return {
            "meta_loss": meta_loss,
            "logits": logits,
            "prototypes": prototypes,
            "distances": distances,
        }

    def _compute_prototypes(
        self, support_features: torch.Tensor, support_y: torch.Tensor
    ) -> torch.Tensor:
        """Compute class prototypes"""
        if support_y.dim() == 1:  # Classification
            unique_labels = torch.unique(support_y)
            prototypes = []

            for label in unique_labels:
                mask = support_y == label
                class_features = support_features[mask]
                prototype = class_features.mean(dim=0)
                prototypes.append(prototype)

            return torch.stack(prototypes)
        else:  # Regression - use k-means style clustering
            # Simple prototype computation for regression
            return support_features.mean(dim=0, keepdim=True)

    def _compute_distances(
        self, query_features: torch.Tensor, prototypes: torch.Tensor
    ) -> torch.Tensor:
        """Compute distances between query features and prototypes"""
        # Euclidean distance
        distances = torch.cdist(query_features, prototypes, p=2)
        return distances

    def adapt(self, support_x: torch.Tensor, support_y: torch.Tensor) -> nn.Module:
        """Adapt by computing prototypes"""
        # For prototypical networks, adaptation is just computing prototypes
        support_features = self.feature_extractor(support_x)
        prototypes = self._compute_prototypes(support_features, support_y)

        # Create adapted model
        adapted_model = AdaptedPrototypicalModel(
            self.feature_extractor, prototypes, self.temperature
        )
        return adapted_model


class AdaptedPrototypicalModel(nn.Module):
    """Adapted prototypical model for inference"""

    def __init__(self, feature_extractor: nn.Module, prototypes: torch.Tensor, temperature: float):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.register_buffer("prototypes", prototypes)
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adapted prototypes"""
        features = self.feature_extractor(x)
        distances = torch.cdist(features, self.prototypes, p=2)
        logits = -distances / self.temperature
        return logits


class GradientBasedMetaLearning(MetaLearningBase):
    """Gradient-based meta-learning with learned optimizers"""

    def __init__(self, base_model: nn.Module, config: MetaLearningConfig):
        super().__init__(config)
        self.base_model = base_model

        # Learned optimizer
        self.meta_optimizer = LearnedOptimizer(
            num_params=sum(p.numel() for p in base_model.parameters()), hidden_dim=64
        )

        logger.info("Initialized Gradient-Based Meta-Learning system")

    def meta_forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Gradient-based meta-learning forward pass"""
        # Adapt using learned optimizer
        adapted_model, optimization_trace = self.adapt_with_trace(support_x, support_y)

        # Query prediction
        query_pred = adapted_model(query_x)

        # Meta-loss
        meta_loss = F.mse_loss(query_pred, query_y)

        return {
            "meta_loss": meta_loss,
            "query_pred": query_pred,
            "optimization_trace": optimization_trace,
        }

    def adapt(self, support_x: torch.Tensor, support_y: torch.Tensor) -> nn.Module:
        """Adapt using learned optimizer"""
        adapted_model, _ = self.adapt_with_trace(support_x, support_y)
        return adapted_model

    def adapt_with_trace(
        self, support_x: torch.Tensor, support_y: torch.Tensor
    ) -> Tuple[nn.Module, List[float]]:
        """Adapt with optimization trace"""
        adapted_model = deepcopy(self.base_model)
        optimization_trace = []

        # Get flattened parameters
        params = torch.cat([p.view(-1) for p in adapted_model.parameters()])

        # Learned optimization
        hidden_state = self.meta_optimizer.init_hidden(params.size(0))

        for step in range(self.adaptation_steps):
            # Forward pass
            support_pred = adapted_model(support_x)

            # Adaptation loss
            adaptation_loss = F.mse_loss(support_pred, support_y)
            optimization_trace.append(adaptation_loss.item())

            # Compute gradients
            grads = torch.autograd.grad(
                adaptation_loss, adapted_model.parameters(), create_graph=True, retain_graph=True
            )

            # Flatten gradients
            flat_grads = torch.cat([g.view(-1) for g in grads])

            # Learned update
            param_update, hidden_state = self.meta_optimizer(flat_grads, hidden_state)

            # Update parameters
            param_idx = 0
            for param in adapted_model.parameters():
                param_size = param.numel()
                param.data = param.data - param_update[param_idx : param_idx + param_size].view(
                    param.shape
                )
                param_idx += param_size

        return adapted_model, optimization_trace


class LearnedOptimizer(nn.Module):
    """Learned optimizer using LSTM"""

    def __init__(self, num_params: int, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim

        # LSTM for parameter updates
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden_dim, num_layers=2, batch_first=True  # Gradient input
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)

        # Initialize for stable learning
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state"""
        device = next(self.parameters()).device
        h0 = torch.zeros(2, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(2, batch_size, self.hidden_dim, device=device)
        return h0, c0

    def forward(
        self, gradients: torch.Tensor, hidden_state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass to generate parameter updates"""
        # Reshape gradients for LSTM
        grad_input = gradients.unsqueeze(-1).unsqueeze(0)  # [1, num_params, 1]

        # LSTM forward
        lstm_out, new_hidden_state = self.lstm(grad_input, hidden_state)

        # Generate parameter updates
        param_updates = self.output_layer(lstm_out.squeeze(0))  # [num_params, 1]
        param_updates = param_updates.squeeze(-1)  # [num_params]

        return param_updates, new_hidden_state


class MemoryAugmentedMetaLearning(MetaLearningBase):
    """Memory-augmented meta-learning for experience replay"""

    def __init__(self, base_model: nn.Module, config: MetaLearningConfig):
        super().__init__(config)
        self.base_model = base_model
        self.memory_size = config.memory_size

        # External memory
        self.memory = ExternalMemory(memory_size=config.memory_size, key_size=64, value_size=128)

        # Memory controller
        self.memory_controller = MemoryController(
            input_size=(
                base_model.get_feature_size() if hasattr(base_model, "get_feature_size") else 256
            ),
            memory_size=config.memory_size,
            key_size=64,
            value_size=128,
        )

        logger.info("Initialized Memory-Augmented Meta-Learning system")

    def meta_forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Memory-augmented meta-learning forward pass"""
        # Extract features
        support_features = self.base_model(support_x)
        query_features = self.base_model(query_x)

        # Store support examples in memory
        self.memory.write(support_features, support_y)

        # Retrieve relevant memories for query
        retrieved_memories = self.memory.read(query_features)

        # Combine features with memories
        augmented_features = torch.cat([query_features, retrieved_memories], dim=-1)

        # Make predictions
        query_pred = self.memory_controller(augmented_features)

        # Meta-loss
        meta_loss = F.mse_loss(query_pred, query_y)

        return {
            "meta_loss": meta_loss,
            "query_pred": query_pred,
            "retrieved_memories": retrieved_memories,
        }

    def adapt(self, support_x: torch.Tensor, support_y: torch.Tensor) -> nn.Module:
        """Adapt by updating memory"""
        support_features = self.base_model(support_x)
        self.memory.write(support_features, support_y)

        # Return adapted model with memory
        return AdaptedMemoryModel(self.base_model, self.memory, self.memory_controller)


class ExternalMemory(nn.Module):
    """External memory for meta-learning"""

    def __init__(self, memory_size: int, key_size: int, value_size: int):
        super().__init__()
        self.memory_size = memory_size
        self.key_size = key_size
        self.value_size = value_size

        # Memory matrices
        self.register_buffer("memory_keys", torch.randn(memory_size, key_size))
        self.register_buffer("memory_values", torch.randn(memory_size, value_size))
        self.register_buffer("memory_age", torch.zeros(memory_size))

        # Key and value projections
        self.key_proj = nn.Linear(key_size, key_size)
        self.value_proj = nn.Linear(value_size, value_size)

    def write(self, keys: torch.Tensor, values: torch.Tensor):
        """Write to memory"""
        batch_size = keys.size(0)

        # Project keys and values
        projected_keys = self.key_proj(keys)
        projected_values = self.value_proj(values)

        # Find oldest memory locations
        _, oldest_indices = torch.topk(self.memory_age, k=batch_size, largest=True)

        # Write to memory
        self.memory_keys[oldest_indices] = projected_keys
        self.memory_values[oldest_indices] = projected_values
        self.memory_age[oldest_indices] = 0

        # Age all memories
        self.memory_age += 1

    def read(self, query_keys: torch.Tensor) -> torch.Tensor:
        """Read from memory"""
        # Project query keys
        projected_queries = self.key_proj(query_keys)

        # Compute similarities
        similarities = torch.mm(projected_queries, self.memory_keys.t())

        # Softmax attention
        attention_weights = F.softmax(similarities, dim=-1)

        # Read values
        retrieved_values = torch.mm(attention_weights, self.memory_values)

        return retrieved_values


class MemoryController(nn.Module):
    """Controller for memory-augmented networks"""

    def __init__(self, input_size: int, memory_size: int, key_size: int, value_size: int):
        super().__init__()
        self.input_size = input_size
        self.memory_size = memory_size

        # Controller network
        self.controller = nn.Sequential(
            nn.Linear(input_size + value_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, augmented_features: torch.Tensor) -> torch.Tensor:
        """Forward pass through controller"""
        return self.controller(augmented_features)


class AdaptedMemoryModel(nn.Module):
    """Adapted model with memory"""

    def __init__(self, base_model: nn.Module, memory: ExternalMemory, controller: MemoryController):
        super().__init__()
        self.base_model = base_model
        self.memory = memory
        self.controller = controller

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with memory"""
        features = self.base_model(x)
        retrieved_memories = self.memory.read(features)
        augmented_features = torch.cat([features, retrieved_memories], dim=-1)
        return self.controller(augmented_features)


class MetaLearningOrchestrator:
    """Orchestrator for different meta-learning algorithms"""

    def __init__(self, config: MetaLearningConfig):
        self.config = config
        self.algorithms = {}
        self.active_algorithm = None

        logger.info("Initialized Meta-Learning Orchestrator")

    def register_algorithm(self, name: str, algorithm: MetaLearningBase):
        """Register a meta-learning algorithm"""
        self.algorithms[name] = algorithm
        logger.info(f"Registered meta-learning algorithm: {name}")

    def set_active_algorithm(self, name: str):
        """Set active meta-learning algorithm"""
        if name in self.algorithms:
            self.active_algorithm = self.algorithms[name]
            logger.info(f"Set active algorithm: {name}")
        else:
            raise ValueError(f"Algorithm {name} not registered")

    def meta_train(
        self, tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Dict[str, float]:
        """Meta-training on a batch of tasks"""
        if self.active_algorithm is None:
            raise ValueError("No active algorithm set")

        total_meta_loss = 0.0
        num_tasks = len(tasks)

        for support_x, support_y, query_x, query_y in tasks:
            # Meta-learning forward pass
            outputs = self.active_algorithm.meta_forward(support_x, support_y, query_x, query_y)

            # Update meta-parameters
            if hasattr(self.active_algorithm, "meta_update"):
                self.active_algorithm.meta_update(outputs["meta_loss"])

            total_meta_loss += outputs["meta_loss"].item()

        avg_meta_loss = total_meta_loss / num_tasks

        return {"avg_meta_loss": avg_meta_loss, "num_tasks": num_tasks}

    def adapt_and_predict(
        self, support_x: torch.Tensor, support_y: torch.Tensor, query_x: torch.Tensor
    ) -> torch.Tensor:
        """Adapt to new task and make predictions"""
        if self.active_algorithm is None:
            raise ValueError("No active algorithm set")

        # Adapt to support set
        adapted_model = self.active_algorithm.adapt(support_x, support_y)

        # Make predictions
        with torch.no_grad():
            predictions = adapted_model(query_x)

        return predictions


# Global meta-learning orchestrator
_meta_orchestrator = None


def get_meta_learning_orchestrator(
    config: Optional[MetaLearningConfig] = None,
) -> MetaLearningOrchestrator:
    """Get global meta-learning orchestrator"""
    global _meta_orchestrator
    if _meta_orchestrator is None:
        if config is None:
            config = MetaLearningConfig()
        _meta_orchestrator = MetaLearningOrchestrator(config)
    return _meta_orchestrator


def create_meta_learning_system(
    base_model: nn.Module, algorithm_type: str = "maml", config: Optional[MetaLearningConfig] = None
) -> MetaLearningBase:
    """Factory function to create meta-learning systems"""
    if config is None:
        config = MetaLearningConfig()

    if algorithm_type == "maml":
        return MAML(base_model, config)
    elif algorithm_type == "prototypical":
        return PrototypicalNetworks(base_model, config)
    elif algorithm_type == "gradient_based":
        return GradientBasedMetaLearning(base_model, config)
    elif algorithm_type == "memory_augmented":
        return MemoryAugmentedMetaLearning(base_model, config)
    else:
        raise ValueError(f"Unknown algorithm type: {algorithm_type}")


if __name__ == "__main__":
    # Demonstration of meta-learning capabilities
    print("[AI] Meta-Learning System Demonstration")
    print("=" * 50)

    # Create a simple base model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, x):
            return self.layers(x)

    base_model = SimpleModel()

    # Create different meta-learning systems
    maml_system = create_meta_learning_system(base_model, "maml")
    prototypical_system = create_meta_learning_system(base_model, "prototypical")

    print(
        f"[OK] Created MAML system with {sum(p.numel() for p in maml_system.parameters())} parameters"
    )
    print(
        f"[OK] Created Prototypical system with {sum(p.numel() for p in prototypical_system.parameters())} parameters"
    )

    # Register with orchestrator
    orchestrator = get_meta_learning_orchestrator()
    orchestrator.register_algorithm("maml", maml_system)
    orchestrator.register_algorithm("prototypical", prototypical_system)

    print("[DATA] Meta-learning systems ready for few-shot adaptation!")
    print("[START] Ready for adaptation to new climate scenarios and planetary conditions!")

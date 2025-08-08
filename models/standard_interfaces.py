"""
Standard Model Interface for Astrobiology Platform
=================================================

Defines consistent interfaces for all models to prevent conflicts.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import torch


class StandardModelInterface(ABC):
    """Standard interface all models should implement"""

    @abstractmethod
    def forward(
        self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Standard forward pass"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata"""
        pass

    def validate_inputs(self, inputs: Any) -> bool:
        """Validate input format"""
        return True


class StandardDataInterface:
    """Standard data loading interface"""

    @staticmethod
    def validate_batch(batch: Any) -> bool:
        """Validate batch format"""
        return True

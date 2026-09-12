"""FirstLight-inspired local R0 components, isolated from legacy training.

This package does not launch native workers, cloud services or training on import.
"""

from .config import ModelConfig, Temperatures, TrainingConfig

__all__ = ["ModelConfig", "Temperatures", "TrainingConfig"]

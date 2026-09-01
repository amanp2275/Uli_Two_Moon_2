from dataclasses import dataclass

from .base import BaseTrainingConfig


@dataclass(frozen=True)
class RealNVPConfig(BaseTrainingConfig):
    num_layers: int = 8
    hidden_features: int = 128
    learning_rate: float = 1e-3
    epochs: int = 300


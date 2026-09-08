"""Final tuned configurations selected from the parameter sweeps.

RealNVP values correspond to EXP_024 and Transformer values correspond to
EXP_032. These configurations are intended for the final model comparison.
"""

from dataclasses import dataclass

from .base import BaseTrainingConfig


@dataclass(frozen=True)
class UpdatedRealNVPConfig(BaseTrainingConfig):
    num_layers: int = 9
    hidden_features: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 600
    early_stopping_patience: int = 6
    evaluation_frequency: int = 20
    batch_size: int = 500


@dataclass(frozen=True)
class UpdatedTransformerConfig(BaseTrainingConfig):
    in_channels: int = 2
    channels: int = 64
    num_blocks: int = 6
    layers_per_block: int = 1
    head_dim: int = 64
    expansion: int = 2
    nvp: bool = True
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 600
    early_stopping_patience: int = 6
    evaluation_frequency: int = 20
    batch_size: int = 500

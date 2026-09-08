"""Snapshot of the original model configurations before final tuning."""

from dataclasses import dataclass

from .base import BaseTrainingConfig


@dataclass(frozen=True)
class InitialRealNVPConfig(BaseTrainingConfig):
    num_layers: int = 8
    hidden_features: int = 128
    learning_rate: float = 1e-3
    epochs: int = 300


@dataclass(frozen=True)
class InitialTransformerConfig(BaseTrainingConfig):
    in_channels: int = 2
    channels: int = 64
    num_blocks: int = 4
    layers_per_block: int = 1
    head_dim: int = 64
    expansion: int = 4
    nvp: bool = True

from dataclasses import dataclass

from .base import BaseTrainingConfig


@dataclass(frozen=True)
class TransformerConfig(BaseTrainingConfig):
    in_channels: int = 2
    channels: int = 64
    num_blocks: int = 4
    layers_per_block: int = 1
    head_dim: int = 64
    expansion: int = 4
    nvp: bool = True

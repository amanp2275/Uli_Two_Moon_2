from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
	batch_size: int = 32
	points_per_batch: int = 500
	epochs: int = 300
	plot_frequency: int = 25
	noise: float = 0.05
	seed: int = 7
	plot_dir: str | Path = "plots"
	updates_per_epoch: int = 8
	plot_batches: int = 32
	device: str | None = None
	conditional: bool = True
	in_channels: int = 2
	channels: int = 128
	num_blocks: int = 8
	learning_rate: float = 1e-3


CONDITIONAL_CONFIG = TrainingConfig()

UNCONDITIONAL_CONFIG = TrainingConfig(
	epochs=600,
	plot_frequency=50,
	updates_per_epoch=16,
	device="cuda",
	conditional=False,
	plot_dir=Path(__file__).parent / "unconditional" / "plots",
)

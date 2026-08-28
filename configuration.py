from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
	batch_size: int = 32
	points_per_batch: int = 500
	epochs: int = 600
	plot_frequency: int = 50
	noise: float = 0.05
	seed: int = 7
	plot_dir: str | Path = "plots"
	updates_per_epoch: int = 16
	dataset_num_batches: int = 512
	dataset_path: str | Path = Path(__file__).parent / "two_moons_splits.pt"
	plot_batches: int = 32
	test_batches: int = 32
	early_stopping_patience: int = 3
	weight_decay: float = 1e-4
	device: str | None = None
	conditional: bool = True
	in_channels: int = 2
	channels: int = 64
	num_blocks: int = 4
	learning_rate: float = 5e-4


CONDITIONAL_CONFIG = TrainingConfig()

UNCONDITIONAL_CONFIG = replace(
	CONDITIONAL_CONFIG,
	conditional=False,
	plot_dir=Path(__file__).parent / "transformer_flows" / "plots" / "unconditional",
)

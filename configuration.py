from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class TrainingConfig:
	batch_size: int = 32 
	points_per_batch: int = 500 
	epochs: int = 600 # 600 times complete data is getting trained So, all 32 batches * 500 Points.  
	plot_frequency: int = 50 # this is how often I want to print something
	noise: float = 0.05 
	seed: int = 7 # Why the seed is just 7? What does it even mean by that? 
	plot_dir: str | Path = "plots" 
	dataset_num_batches: int = 512 # How is this one differnet then batchsizes? 
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
	checkpoint_callback: Callable[[int, float, float, float], None] | None = None
	show_loss_plot: bool = True


CONDITIONAL_CONFIG = TrainingConfig()

UNCONDITIONAL_CONFIG = replace(
	CONDITIONAL_CONFIG,
	conditional=False,
	plot_dir=Path(__file__).parent / "transformer_flows" / "plots" / "unconditional",
)

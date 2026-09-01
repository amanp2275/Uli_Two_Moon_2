from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BaseTrainingConfig:
    dataset_path: str | Path = Path(__file__).resolve().parents[1] / "two_moons_splits.pt"
    output_dir: str | Path = Path(__file__).resolve().parents[1] / "results"
    seed: int = 7
    conditional: bool = True
    batch_size: int = 32
    epochs: int = 600
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 3
    evaluation_frequency: int = 50
    points_per_batch: int = 500
    dataset_num_batches: int = 512
    noise: float = 0.05
    plot_batches: int = 32
    device: str | None = None

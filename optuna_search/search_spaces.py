"""Model-specific Optuna search spaces and model construction."""

from dataclasses import replace
from pathlib import Path

from configs import RealNVPConfig, TransformerConfig
from models import RealNVP, TransformerFlow


def build_trial(model_name, trial, *, root: Path, epochs: int, seed: int, device: str | None,
                conditional: bool, evaluation_frequency: int, early_stopping_patience: int):
    """Return the sampled config and matching model for one trial."""
    common = {
        "dataset_path": root / "data" / "two_moons_dataset.pt",
        "output_dir": root / "optuna_search" / "results",
        "seed": seed,
        "conditional": conditional,
        "epochs": epochs,
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256]),
        "evaluation_frequency": evaluation_frequency,
        "early_stopping_patience": early_stopping_patience,
        "device": device,
    }

    if model_name == "real_nvp":
        config = replace(
            RealNVPConfig(),
            num_layers=trial.suggest_int("num_layers", 2, 12, step=2),
            hidden_features=trial.suggest_categorical("hidden_features", [32, 64, 128, 256]),
            **common,
        )
        model = RealNVP(config.num_layers, config.hidden_features, config.conditional)
        return config, model

    # Every head_dim choice divides every channels choice. Keeping categorical
    # choices fixed is required when a study is resumed across trials.
    channels = trial.suggest_categorical("channels", [64, 128])
    config = replace(
        TransformerConfig(),
        channels=channels,
        num_blocks=trial.suggest_int("num_blocks", 2, 6),
        layers_per_block=trial.suggest_int("layers_per_block", 1, 3),
        head_dim=trial.suggest_categorical("head_dim", [16, 32, 64]),
        expansion=trial.suggest_categorical("expansion", [2, 4]),
        **common,
    )
    model = TransformerFlow(
        config.in_channels,
        config.points_per_batch,
        config.channels,
        config.num_blocks,
        config.layers_per_block,
        config.head_dim,
        config.expansion,
        config.nvp,
        2 if config.conditional else 0,
    )
    return config, model

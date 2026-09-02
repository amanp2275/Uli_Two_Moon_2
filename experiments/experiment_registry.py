"""Persistent CSV output registry for manual experiments."""

import csv
from pathlib import Path

RESULT_FIELDS = [
    "experiment_id", "experiment_type", "model", "learning_rate", "batch_size", "epochs", "seed",
    "conditional", "weight_decay", "early_stopping_patience", "evaluation_frequency", "points_per_batch",
    "dataset_num_batches", "noise", "plot_batches", "device", "dataset_path", "num_layers",
    "hidden_features", "in_channels", "channels", "num_blocks", "layers_per_block", "head_dim",
    "expansion", "nvp", "notes", "run", "run_id", "status", "started_at", "finished_at",
    "training_time_seconds", "epochs_completed", "best_epoch", "best_training_loss", "best_validation_loss",
    "final_train_loss", "final_validation_loss", "final_test_loss", "nll", "trainable_parameters",
    "dataset_sha256", "git_commit", "config_path", "metrics_path", "checkpoint_path", "plots_path",
    "parameters_json", "error",
]


def read_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def completed_ids(path: Path) -> set[str]:
    return {row.get("experiment_id", "") for row in read_results(path) if row.get("status") == "completed"}


def next_run_number(path: Path, experiment_id: str) -> int:
    numbers = []
    for row in read_results(path):
        if row.get("experiment_id") == experiment_id:
            try:
                numbers.append(int(str(row.get("run_id", "")).removeprefix("run_")))
            except ValueError:
                pass
    return max(numbers, default=0) + 1


def append_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if not existing:
            writer.writeheader()
        writer.writerow({field: result.get(field, "") for field in RESULT_FIELDS})

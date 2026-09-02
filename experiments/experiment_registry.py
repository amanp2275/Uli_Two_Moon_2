"""Persistent CSV registry for manually configured experiments."""

import csv
from pathlib import Path


FIELDS = [
    "experiment_id",
    "experiment_type",
    "model",
    "seed",
    "status",
    "started_at",
    "finished_at",
    "training_time_seconds",
    "epochs_requested",
    "epochs_completed",
    "best_epoch",
    "best_training_loss",
    "best_validation_loss",
    "final_train_loss",
    "final_validation_loss",
    "final_test_loss",
    "nll",
    "trainable_parameters",
    "dataset_path",
    "dataset_sha256",
    "git_commit",
    "config_path",
    "metrics_path",
    "checkpoint_path",
    "plots_path",
    "model_parameters_json",
    "training_parameters_json",
    "error",
]


def read_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def completed_ids(path: Path) -> set[str]:
    return {row.get("experiment_id", "") for row in read_results(path) if row.get("status") == "completed"}


def append_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        if not existing:
            writer.writeheader()
        writer.writerow({field: result.get(field, "") for field in FIELDS})

"""Export compact model-sweep result tables from canonical final-model artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "model_comparison"
OUTPUT = ROOT / "Analysis" / "tables" / "model_sweep"
EXPERIMENTS = ("EXP_MS_001", "EXP_MS_003", "EXP_MS_004", "EXP_MS_005")
MODELS = ("real_nvp", "transformer")

COMMON_FIELDS = [
    "experiment_id",
    "model",
    "seed",
    "status",
    "epochs",
    "epochs_completed",
    "batch_size",
    "conditional",
    "learning_rate",
    "weight_decay",
    "evaluation_frequency",
    "best_epoch",
    "best_validation_loss",
    "final_train_loss",
    "final_validation_loss",
    "final_test_loss",
    "nll",
    "trainable_parameters",
    "metrics_path",
    "config_path",
]
MODEL_FIELDS = {
    "real_nvp": ["num_layers", "hidden_features"],
    "transformer": ["in_channels", "channels", "num_blocks", "layers_per_block", "head_dim", "expansion", "nvp"],
}


def find_artifact(experiment_id: str, model: str, filename: str) -> Path:
    matches = sorted((RESULTS / experiment_id / "final_models" / model).rglob(filename))
    if not matches:
        raise FileNotFoundError(f"No canonical {filename} found for {experiment_id} ({model})")
    return matches[0]


def load_sweep_status() -> dict[str, dict[str, str]]:
    path = RESULTS / "model_sweep_results.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["experiment_id"]: row for row in csv.DictReader(handle)}


def make_rows(sweep_status: dict[str, dict[str, str]]) -> dict[str, list[dict]]:
    rows_by_model = {model: [] for model in MODELS}
    for experiment_id in EXPERIMENTS:
        for model in MODELS:
            metrics_path = find_artifact(experiment_id, model, "metrics.json")
            config_path = find_artifact(experiment_id, model, "config.json")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            config = json.loads(config_path.read_text(encoding="utf-8"))
            registry_row = sweep_status.get(experiment_id, {})
            row = {
                "experiment_id": experiment_id,
                "model": model,
                "seed": config.get("seed", registry_row.get("seed", "")),
                "status": "completed",
                "epochs": config.get("epochs", ""),
                "epochs_completed": len(metrics.get("train_losses", [])),
                "batch_size": config.get("batch_size", ""),
                "conditional": config.get("conditional", ""),
                "learning_rate": config.get("learning_rate", ""),
                "weight_decay": config.get("weight_decay", ""),
                "evaluation_frequency": config.get("evaluation_frequency", ""),
                "best_epoch": metrics.get("best_epoch", ""),
                "best_validation_loss": metrics.get("best_validation_loss", ""),
                "final_train_loss": metrics.get("final_train_loss", ""),
                "final_validation_loss": metrics.get("final_validation_loss", ""),
                "final_test_loss": metrics.get("final_test_loss", ""),
                "nll": metrics.get("final_test_loss", ""),
                "trainable_parameters": metrics.get("total_trainable_parameters", ""),
                "metrics_path": metrics_path.relative_to(ROOT).as_posix(),
                "config_path": config_path.relative_to(ROOT).as_posix(),
            }
            for field in MODEL_FIELDS[model]:
                row[field] = config.get(field, "")
            rows_by_model[model].append(row)
    return rows_by_model


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = COMMON_FIELDS + MODEL_FIELDS[rows[0]["model"]]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows_by_model = make_rows(load_sweep_status())
    for model, rows in rows_by_model.items():
        path = OUTPUT / model / f"{model}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(path, rows)
        print(f"{path.relative_to(ROOT)}: {len(rows)} rows")


if __name__ == "__main__":
    main()
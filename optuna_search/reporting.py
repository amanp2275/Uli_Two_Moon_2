"""Export Optuna results in human-readable and experiment-queue formats."""

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import optuna


QUEUE_FIELDS = [
    "experiment_id", "experiment_type", "model", "learning_rate", "batch_size", "epochs", "seed",
    "conditional", "weight_decay", "early_stopping_patience", "evaluation_frequency", "points_per_batch",
    "dataset_num_batches", "noise", "plot_batches", "device", "dataset_path", "num_layers",
    "hidden_features", "in_channels", "channels", "num_blocks", "layers_per_block", "head_dim",
    "expansion", "nvp", "notes", "run",
]


def _serializable_config(config) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}


def _completed_trials(study):
    return [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]


def export_results(study, output_dir: Path, model_name: str, best_config=None) -> None:
    """Refresh CSV, JSON, queue row, and diagnostic plots after a trial."""
    output_dir.mkdir(parents=True, exist_ok=True)
    trials = study.trials
    parameter_names = sorted({name for trial in trials for name in trial.params})
    fields = ["number", "state", "value", "started_at", "finished_at", "duration_seconds"] + parameter_names
    with (output_dir / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for trial in trials:
            row = {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "started_at": trial.datetime_start.isoformat() if trial.datetime_start else "",
                "finished_at": trial.datetime_complete.isoformat() if trial.datetime_complete else "",
                "duration_seconds": trial.duration.total_seconds() if trial.duration else "",
            }
            row.update(trial.params)
            writer.writerow(row)

    if not _completed_trials(study) or best_config is None:
        return

    best = study.best_trial
    resolved = _serializable_config(best_config)
    summary = {
        "study_name": study.study_name,
        "model": model_name,
        "objective": "minimum validation NLL per coordinate",
        "best_trial": best.number,
        "best_validation_nll": best.value,
        "best_parameters": best.params,
        "resolved_config": resolved,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (output_dir / "best_parameters.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    queue_row = {field: "" for field in QUEUE_FIELDS}
    queue_row.update({key: value for key, value in resolved.items() if key in queue_row})
    queue_row.update({
        "experiment_id": f"OPTUNA_{model_name.upper()}_CHECK",
        "experiment_type": "optuna_cross_check",
        "model": model_name,
        "conditional": "YES" if resolved["conditional"] else "NO",
        "nvp": "YES" if resolved.get("nvp") else ("NO" if model_name == "transformer" else ""),
        "notes": f"Manual cross-check of Optuna trial {best.number}; validation NLL={best.value:.8f}",
        "run": "NO",
    })
    with (output_dir / "manual_cross_check.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUEUE_FIELDS)
        writer.writeheader()
        writer.writerow(queue_row)

    plotters = {
        "optimization_history.png": optuna.visualization.matplotlib.plot_optimization_history,
        "parameter_importance.png": optuna.visualization.matplotlib.plot_param_importances,
        "parameter_slices.png": optuna.visualization.matplotlib.plot_slice,
    }
    for filename, plotter in plotters.items():
        try:
            axis = plotter(study)
            axis.figure.tight_layout()
            axis.figure.savefig(output_dir / filename, dpi=160, bbox_inches="tight")
            axis.figure.clf()
        except (ValueError, RuntimeError, ImportError):
            # Some plots require multiple completed trials or optional dependencies.
            continue


"""Create model- and experiment-type-organized training plots.

Outputs are written below ``Analysis/organized_plots``:

* separate Transformer and RealNVP loss-curve figures;
* separate parameter-sweep and model-sweep figures;
* matching parameter-summary figures for each model/sweep combination.

Run with:
    python Analysis/generate_organized_model_plots.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "Analysis" / "organized_plots"


def _sweep_groups() -> dict[str, str]:
    groups = {}
    with (ROOT / "results" / "experiment_results.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status", "").strip().lower() == "completed":
                groups[row.get("experiment_id", "").strip()] = row.get("notes", "").strip().lower()
    return groups


def _pretty_group(value: str) -> str:
    value = value.strip()
    return value[:1].upper() + value[1:] if value else "Other"


def _load(path: Path, model: str, sweep: str, group: str, run_id: str) -> dict:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": path,
        "model": model,
        "sweep": sweep,
        "group": group,
        "run_id": run_id,
        "metrics": metrics,
        "config": metrics.get("config", {}),
    }


def _collect() -> list[dict]:
    groups = _sweep_groups()
    records = []
    for path in sorted((RESULTS / "parameter_sweep").rglob("metrics.json")):
        parts = path.relative_to(RESULTS / "parameter_sweep").parts
        if len(parts) < 3:
            continue
        experiment_id = parts[0]
        model = path.parent.parent.name
        if model not in {"real_nvp", "transformer"}:
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if not metrics.get("train_losses") or experiment_id not in groups:
            continue
        records.append(_load(path, model, "parameter_sweep", _pretty_group(groups[experiment_id]), experiment_id))

    # Model sweeps write both raw and final-model copies. The final-model copy
    # is the canonical completed result and prevents duplicate curves.
    for experiment_dir in sorted((RESULTS / "model_comparison").glob("EXP_MS_*")):
        for path in sorted(experiment_dir.rglob("metrics.json")):
            if "final_models" not in path.parts:
                continue
            model = next((part for part in path.parts if part in {"real_nvp", "transformer"}), "")
            if not model:
                continue
            metrics = json.loads(path.read_text(encoding="utf-8"))
            if metrics.get("train_losses"):
                records.append(_load(path, model, "model_sweep", "Model sweep", experiment_dir.name))
    return records


def _run_title(record: dict) -> str:
    return record["run_id"]


def _plot_losses(records: list[dict], model: str, sweep: str) -> Path:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["group"]].append(record)
    names = sorted(grouped)
    columns = 2 if len(names) > 1 else 1
    rows = (len(names) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(18 if columns == 2 else 12, 5 * rows), squeeze=False)
    for axis, group in zip(axes.flat, names):
        for record in sorted(grouped[group], key=lambda item: item["run_id"]):
            metrics = record["metrics"]
            train = metrics.get("train_losses", [])
            validation = metrics.get("validation_losses", [])
            test = metrics.get("test_losses", [])
            epochs = metrics.get("evaluation_epochs", [])
            label = _run_title(record)
            axis.plot(range(1, len(train) + 1), train, label=f"{label} train", linewidth=1.2)
            if validation:
                axis.plot(epochs[: len(validation)], validation, "--", label=f"{label} val", linewidth=1)
            if test:
                axis.plot(epochs[: len(test)], test, ":", label=f"{label} test", linewidth=1)
        axis.set_title(group)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
    for axis in axes.flat[len(names) :]:
        axis.axis("off")
    figure.suptitle(f"{model.replace('_', ' ').title()} — {sweep.replace('_', ' ').title()} loss curves", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output = OUTPUT / sweep / f"{model}_loss_curves_by_experiment_type.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def _parameter_text(record: dict) -> str:
    config = record["config"]
    metrics = record["metrics"]
    keys = ["learning_rate", "weight_decay", "num_layers", "hidden_features", "channels", "num_blocks", "layers_per_block", "head_dim", "expansion"]
    values = [f"{key}={config[key]}" for key in keys if config.get(key, "") not in ("", None)]
    values.append(f"params={metrics.get('total_trainable_parameters', '')}")
    values.append(f"best val={float(metrics.get('best_validation_loss', 0)):.4f}" if metrics.get("best_validation_loss") != "" else "best val=n/a")
    return ", ".join(values)


def _plot_parameters(records: list[dict], model: str, sweep: str) -> Path:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["group"]].append(record)
    names = sorted(grouped)
    columns = 2 if len(names) > 1 else 1
    rows = (len(names) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(20 if columns == 2 else 16, max(4, rows * 3.4)), squeeze=False)
    for axis, group in zip(axes.flat, names):
        axis.axis("off")
        lines = [f"{record['run_id']}: {_parameter_text(record)}" for record in sorted(grouped[group], key=lambda item: item["run_id"])]
        axis.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=8, wrap=True)
        axis.set_title(group, loc="left", pad=12, fontweight="bold")
    for axis in axes.flat[len(names) :]:
        axis.axis("off")
    figure.suptitle(f"{model.replace('_', ' ').title()} — {sweep.replace('_', ' ').title()} final parameters", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output = OUTPUT / sweep / f"{model}_parameters_by_experiment_type.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    records = _collect()
    generated = []
    for sweep in ("parameter_sweep", "model_sweep"):
        for model in ("transformer", "real_nvp"):
            selected = [record for record in records if record["sweep"] == sweep and record["model"] == model]
            if selected:
                generated.extend([_plot_losses(selected, model, sweep), _plot_parameters(selected, model, sweep)])
    print(f"Generated {len(generated)} organized plots from {len(records)} completed runs.")
    for path in generated:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()

"""Plot RealNVP parameter sweeps as separate figures by sweep type."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "parameter_sweep"
OUTPUT = ROOT / "Analysis" / "organized_plots" / "parameter_sweep" / "real_nvp"


def sweep_name(notes: str) -> str:
    notes = notes.lower()
    if "learning rate" in notes:
        return "learning_rate"
    if "weight decay" in notes:
        return "weight_decay"
    if "flow depth" in notes:
        return "flow_depth"
    if "realnvp width" in notes:
        return "hidden_dimension"
    return "other"


def load_runs() -> dict[str, list[dict]]:
    experiment_names = {}
    with (ROOT / "results" / "experiment_results.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("model") == "real_nvp" and row.get("status") == "completed":
                experiment_names[row["experiment_id"]] = sweep_name(row.get("notes", ""))

    grouped = defaultdict(list)
    for metrics_path in sorted(RESULTS.rglob("metrics.json")):
        relative = metrics_path.relative_to(RESULTS)
        experiment_id = relative.parts[0]
        if experiment_id not in experiment_names:
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("model") != "real_nvp" or not metrics.get("train_losses"):
            continue
        grouped[experiment_names[experiment_id]].append({"id": experiment_id, "metrics": metrics})
    return grouped


def plot_group(name: str, runs: list[dict]) -> Path:
    labels = {
        "learning_rate": "Learning-rate sweep",
        "weight_decay": "Weight-decay sweep",
        "flow_depth": "Flow-depth sweep",
        "hidden_dimension": "Hidden-dimension sweep",
    }
    figure, axis = plt.subplots(figsize=(12, 7))
    for run in sorted(runs, key=lambda item: item["id"]):
        metrics = run["metrics"]
        epochs = metrics.get("evaluation_epochs", [])
        axis.plot(range(1, len(metrics["train_losses"]) + 1), metrics["train_losses"], label=f"{run['id']} train")
        axis.plot(epochs[: len(metrics.get("validation_losses", []))], metrics.get("validation_losses", []), "--", label=f"{run['id']} validation")
        axis.plot(epochs[: len(metrics.get("test_losses", []))], metrics.get("test_losses", []), ":", label=f"{run['id']} test")
    axis.set_title(f"RealNVP — {labels.get(name, name)}")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=3)
    figure.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"real_nvp_{name}_loss_curves.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


def main() -> None:
    generated = [plot_group(name, runs) for name, runs in sorted(load_runs().items()) if name != "other"]
    print(f"Generated {len(generated)} separate RealNVP parameter-sweep plots.")
    for path in generated:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()

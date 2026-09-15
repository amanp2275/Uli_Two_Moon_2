"""Plot Transformer loss curves for experiments EXP_034 through the last run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results" / "parameter_sweep"
OUTPUT = ROOT / "Analysis" / "organized_plots" / "parameter_sweep" / "transformer"
START_EXPERIMENT = 34


def find_metrics(experiment_id: str) -> Path | None:
    matches = sorted((RESULTS / experiment_id).rglob("metrics.json"))
    return matches[0] if matches else None


def format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def run_label(metrics: dict) -> str:
    config = metrics.get("config", {})
    return (
        f"lr={format_value(config.get('learning_rate', '?'))}, "
        f"wd={format_value(config.get('weight_decay', '?'))}\n"
        f"blocks={config.get('num_blocks', '?')}, "
        f"head={config.get('head_dim', '?')}, "
        f"exp={config.get('expansion', '?')}"
    )


def plot_run(axis, experiment_id: str, metrics: dict) -> int:
    train = metrics["train_losses"]
    validation = metrics.get("validation_losses", [])
    test = metrics.get("test_losses", [])
    evaluation_epochs = metrics.get("evaluation_epochs", [])

    axis.plot(range(1, len(train) + 1), train, color="#1f77b4", linewidth=1.2, label="Train")
    axis.plot(
        evaluation_epochs[: len(validation)], validation, "--",
        color="#d62728", linewidth=1.2, label="Validation"
    )
    axis.plot(
        evaluation_epochs[: len(test)], test, ":",
        color="#2ca02c", linewidth=1.5, label="Test"
    )
    axis.set_title(f"{experiment_id}\n{run_label(metrics)}", fontsize=9)
    axis.set_xlim(0, len(train))
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.grid(alpha=0.25)
    return len(train)


def main() -> None:
    experiment_numbers = sorted(
        int(path.name.split("_")[1])
        for path in RESULTS.iterdir()
        if path.is_dir() and path.name.startswith("EXP_")
        and int(path.name.split("_")[1]) >= START_EXPERIMENT
    )
    if not experiment_numbers:
        raise FileNotFoundError("No experiments found from EXP_034 onward")

    figure, axes = plt.subplots(5, 3, figsize=(18, 24), sharex=False)
    axes = axes.ravel()
    max_epochs = 0
    plotted = []
    missing = []

    for index, number in enumerate(experiment_numbers):
        experiment_id = f"EXP_{number:03d}"
        metrics_path = find_metrics(experiment_id)
        if metrics_path is None:
            axes[index].text(0.5, 0.5, "metrics.json unavailable", ha="center", va="center")
            axes[index].set_title(experiment_id, fontsize=10)
            axes[index].set_axis_off()
            missing.append(experiment_id)
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        max_epochs = max(max_epochs, plot_run(axes[index], experiment_id, metrics))
        plotted.append(experiment_id)

    for axis in axes[len(experiment_numbers):]:
        axis.set_axis_off()

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.975))
    figure.suptitle(
        "Transformer experiments EXP_034 onward: all loss curves",
        fontsize=18, fontweight="bold", y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.0, w_pad=1.5)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT / "transformer_experiments_EXP034_onward_loss_curves.png"
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    print(output_path.relative_to(ROOT))
    print(f"Plotted: {', '.join(plotted)}")
    if missing:
        print(f"Missing metrics: {', '.join(missing)}")


if __name__ == "__main__":
    main()

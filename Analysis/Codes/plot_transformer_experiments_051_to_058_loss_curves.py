"""Plot Transformer loss curves for experiments EXP_051 through EXP_058."""

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
EXPERIMENTS = tuple(f"EXP_{number:03d}" for number in range(51, 59))


def find_metrics(experiment_id: str) -> Path:
    matches = sorted((RESULTS / experiment_id).rglob("metrics.json"))
    if not matches:
        raise FileNotFoundError(f"No metrics.json found for {experiment_id}")
    return matches[0]


def format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def subtitle(metrics: dict) -> str:
    config = metrics.get("config", {})
    return (
        f"lr={format_value(config.get('learning_rate', '?'))}, "
        f"wd={format_value(config.get('weight_decay', '?'))}\n"
        f"blocks={config.get('num_blocks', '?')}, "
        f"layers/block={config.get('layers_per_block', '?')}"
    )


def plot_run(axis, experiment_id: str, metrics: dict) -> None:
    train = metrics["train_losses"]
    validation = metrics.get("validation_losses", [])
    test = metrics.get("test_losses", [])
    evaluation_epochs = metrics.get("evaluation_epochs", [])

    axis.plot(range(1, len(train) + 1), train, color="#1f77b4", linewidth=1.15, label="Train")
    axis.plot(
        evaluation_epochs[: len(validation)],
        validation,
        "--",
        color="#d62728",
        linewidth=1.15,
        label="Validation",
    )
    axis.plot(
        evaluation_epochs[: len(test)],
        test,
        ":",
        color="#2ca02c",
        linewidth=1.4,
        label="Test",
    )
    axis.set_title(f"{experiment_id}\n{subtitle(metrics)}", fontsize=10)
    axis.set_xlim(0, len(train))
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.grid(alpha=0.25)


def main() -> None:
    figure, axes = plt.subplots(2, 4, figsize=(24, 11), sharex=False, sharey=False)
    axes = axes.ravel()

    for axis, experiment_id in zip(axes, EXPERIMENTS):
        metrics = json.loads(find_metrics(experiment_id).read_text(encoding="utf-8"))
        plot_run(axis, experiment_id, metrics)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.975))
    figure.suptitle(
        "Transformer parameter sweep: EXP_051–EXP_058 loss curves",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.0, w_pad=1.4)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT / "transformer_EXP051_to_EXP058_loss_curves.png"
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    print(output_path)


if __name__ == "__main__":
    main()

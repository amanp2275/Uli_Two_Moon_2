"""Plot all RealNVP parameter-sweep loss curves in dedicated sweep columns."""

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
OUTPUT = ROOT / "Analysis" / "organized_plots" / "parameter_sweep" / "real_nvp"
SWEEPS = (
    ("Flow depth", "flow_depth", ("EXP_017", "EXP_019", "EXP_020"), "num_layers", " layers"),
    ("Hidden dimension", "hidden_dimension", ("EXP_021", "EXP_022", "EXP_023", "EXP_024"), "hidden_features", " features"),
    ("Learning rate", "learning_rate", ("EXP_001", "EXP_002", "EXP_003", "EXP_004"), "learning_rate", ""),
    ("Weight decay", "weight_decay", ("EXP_009", "EXP_010", "EXP_011", "EXP_012"), "weight_decay", ""),
)


def find_metrics(experiment_id: str) -> Path:
    matches = sorted((RESULTS / experiment_id).rglob("metrics.json"))
    if not matches:
        raise FileNotFoundError(f"No metrics.json found for {experiment_id}")
    return matches[0]


def load_metrics(experiment_id: str) -> dict:
    path = find_metrics(experiment_id)
    return json.loads(path.read_text(encoding="utf-8"))


def parameter_label(metrics: dict, parameter: str, suffix: str) -> str:
    value = metrics.get("config", {}).get(parameter, "?")
    if parameter in {"learning_rate", "weight_decay"}:
        value = f"{value:g}" if isinstance(value, float) else value
    return f"{value}{suffix}"


def plot_run(axis, experiment_id: str, metrics: dict, parameter: str, suffix: str) -> None:
    train = metrics["train_losses"]
    validation = metrics.get("validation_losses", [])
    test = metrics.get("test_losses", [])
    evaluation_epochs = metrics.get("evaluation_epochs", [])
    axis.plot(range(1, len(train) + 1), train, color="#1f77b4", linewidth=1.2, label="Train")
    axis.plot(
        evaluation_epochs[: len(validation)],
        validation,
        "--",
        color="#d62728",
        linewidth=1.2,
        label="Validation",
    )
    axis.plot(
        evaluation_epochs[: len(test)],
        test,
        ":",
        color="#2ca02c",
        linewidth=1.5,
        label="Test",
    )
    axis.set_title(
        f"{experiment_id}\n{parameter.replace('_', ' ').title()}: "
        f"{parameter_label(metrics, parameter, suffix)}",
        fontsize=10,
    )
    axis.set_xlim(0, 600)
    axis.grid(alpha=0.25)


def main() -> None:
    figure, axes = plt.subplots(4, 4, figsize=(22, 20), sharex=False)
    for column, (title, _, experiment_ids, parameter, suffix) in enumerate(SWEEPS):
        axes[0, column].set_title(title, fontsize=13, fontweight="bold", pad=14)
        for row, experiment_id in enumerate(experiment_ids):
            plot_run(axes[row, column], experiment_id, load_metrics(experiment_id), parameter, suffix)
            if row == len(experiment_ids) - 1:
                axes[row, column].set_xlabel("Epoch")
            axes[row, column].set_ylabel("Loss")
        for row in range(len(experiment_ids), 4):
            axes[row, column].axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.975))
    figure.suptitle(
        "RealNVP parameter sweep: all loss curves",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.2, w_pad=1.4)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT / "real_nvp_all_parameter_sweeps_loss_curves.png"
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    print(output_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
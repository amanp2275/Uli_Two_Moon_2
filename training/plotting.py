import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def save_training_plot(real, labels, generated, generated_labels, train_losses,
                       validation_losses, test_losses, epochs, output_path: Path,
                       parameters: dict | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    real = real.detach().cpu().reshape(-1, 2)
    labels = labels.detach().cpu().reshape(-1)
    generated = generated.detach().cpu().reshape(-1, 2)
    figure = plt.figure(figsize=(16, 9))
    layout = figure.add_gridspec(2, 3, height_ratios=(4.5, 1.5), hspace=0.35)
    axes = [figure.add_subplot(layout[0, index]) for index in range(3)]
    axes[0].scatter(real[:, 0], real[:, 1], c=labels, cmap="coolwarm", s=8)
    axes[0].set_title("Real two moons")
    if generated_labels is None:
        axes[1].scatter(generated[:, 0], generated[:, 1], color="darkgreen", s=8)
    else:
        axes[1].scatter(generated[:, 0], generated[:, 1], c=generated_labels.detach().cpu().reshape(-1), cmap="coolwarm", s=8)
    axes[1].set_title("Generated samples")
    axes[2].plot(range(1, len(train_losses) + 1), train_losses, label="Train")
    axes[2].plot(epochs, validation_losses, label="Validation")
    axes[2].plot(epochs, test_losses, label="Test")
    axes[2].set_title("Loss")
    axes[2].legend()
    for axis in axes[:2]:
        axis.set_aspect("equal")
        axis.grid(alpha=0.2)
    parameter_items = list((parameters or {}).items())
    if parameter_items:
        rows = []
        for index in range(0, len(parameter_items), 2):
            left_key, left_value = parameter_items[index]
            right_key, right_value = parameter_items[index + 1] if index + 1 < len(parameter_items) else ("", "")
            rows.append([left_key.replace("_", " "), str(left_value), right_key.replace("_", " "), str(right_value)])
        table_axis = figure.add_subplot(layout[1, :])
        table_axis.axis("off")
        table = table_axis.table(
            cellText=rows,
            colLabels=["Parameter", "Value", "Parameter", "Value"],
            colWidths=[0.18, 0.32, 0.18, 0.32],
            cellLoc="left",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.35)
    else:
        table_axis = figure.add_subplot(layout[1, :])
        table_axis.axis("off")
        table_axis.text(0.01, 0.5, "Parameters unavailable", va="center", fontsize=9)
    figure.text(
        0.99,
        0.005,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ha="right",
        va="bottom",
        fontsize=8,
        color="dimgray",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "lightgray", "alpha": 0.9},
    )
    figure.subplots_adjust(top=0.94, bottom=0.08, left=0.04, right=0.98)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def save_comparison(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")

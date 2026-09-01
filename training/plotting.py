import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def save_training_plot(real, labels, generated, generated_labels, train_losses,
                       validation_losses, test_losses, epochs, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    real = real.detach().cpu().reshape(-1, 2)
    labels = labels.detach().cpu().reshape(-1)
    generated = generated.detach().cpu().reshape(-1, 2)
    # Keep rare generated outliers from making the two point-cloud panels
    # unreadable. This changes only the displayed window, not the data.
    combined = torch.cat((real, generated), dim=0)
    lower = torch.quantile(combined, 0.005, dim=0)
    upper = torch.quantile(combined, 0.995, dim=0)
    center = (lower + upper) / 2
    half_range = (upper - lower).max() / 2
    half_range = half_range.clamp_min(1e-6) * 1.08
    x_limits = (center[0] - half_range, center[0] + half_range)
    y_limits = (center[1] - half_range, center[1] + half_range)
    figure, axes = plt.subplots(1, 3, figsize=(16, 7))
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
        axis.set_xlim(float(x_limits[0]), float(x_limits[1]))
        axis.set_ylim(float(y_limits[0]), float(y_limits[1]))
        axis.set_aspect("equal")
        axis.grid(alpha=0.2)
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
    figure.subplots_adjust(top=0.90, bottom=0.12, left=0.04, right=0.98)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_samples_plot(real, labels, generated, generated_labels, output_path: Path) -> None:
    """Save the final real/generated sample comparison without loss curves."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    real = real.detach().cpu().reshape(-1, 2)
    labels = labels.detach().cpu().reshape(-1)
    generated = generated.detach().cpu().reshape(-1, 2)
    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].scatter(real[:, 0], real[:, 1], c=labels, cmap="coolwarm", s=8)
    axes[0].set_title("Real two moons")
    if generated_labels is None:
        axes[1].scatter(generated[:, 0], generated[:, 1], color="darkgreen", s=8)
    else:
        axes[1].scatter(generated[:, 0], generated[:, 1], c=generated_labels.detach().cpu().reshape(-1), cmap="coolwarm", s=8)
    axes[1].set_title("Generated samples")
    for axis in axes:
        axis.set_aspect("equal")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_loss_curves(train_losses, validation_losses, test_losses, epochs, output_path: Path) -> None:
    """Save this run's train, validation, and test loss histories."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(range(1, len(train_losses) + 1), train_losses, label="Train")
    axis.plot(epochs, validation_losses, label="Validation")
    axis.plot(epochs, test_losses, label="Test")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("Loss curves")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_parameter_table(parameters: dict, output_path: Path) -> None:
    """Save one parameter-only reference image for a training run."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    items = list(parameters.items())
    rows = []
    for index in range(0, len(items), 2):
        left_key, left_value = items[index]
        right_key, right_value = items[index + 1] if index + 1 < len(items) else ("", "")
        rows.append([
            left_key.replace("_", " "), str(left_value),
            right_key.replace("_", " "), str(right_value),
        ])
    figure, axis = plt.subplots(figsize=(16, max(4, 1.2 + len(rows) * 0.42)))
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=["Parameter", "Value", "Parameter", "Value"],
        colWidths=[0.2, 0.3, 0.2, 0.3],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    axis.set_title("Parameters used for this run", fontsize=14, pad=18)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_summary_table(summary: dict, output_path: Path) -> None:
    """Save final training values as a single visual summary table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [[key.replace("_", " "), str(value)] for key, value in summary.items()]
    figure, axis = plt.subplots(figsize=(10, max(3, 1.5 + len(rows) * 0.55)))
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=["Final result", "Value"],
        colWidths=[0.55, 0.45],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)
    axis.set_title("Final training summary", fontsize=15, pad=18)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def save_comparison(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")

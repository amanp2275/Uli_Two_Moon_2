"""Plot selected RealNVP runs individually with complete loss histories."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "parameter_sweep"
OUTPUT = ROOT / "Analysis" / "organized_plots" / "parameter_sweep" / "real_nvp"
EXPERIMENT_GROUPS = {
    "flow_depth": ("EXP_017", "EXP_019", "EXP_020"),
    "hidden_dimension": ("EXP_021", "EXP_022", "EXP_023", "EXP_024"),
    "learning_rate": ("EXP_001", "EXP_002", "EXP_003", "EXP_004"),
    "weight_decay": ("EXP_009", "EXP_010", "EXP_011", "EXP_012"),
}


def find_metrics(experiment_id: str) -> Path:
    matches = sorted((RESULTS / experiment_id).rglob("metrics.json"))
    if not matches:
        raise FileNotFoundError(f"No metrics.json found for {experiment_id}")
    return matches[0]


def plot_run(experiment_id: str) -> Path:
    metrics = json.loads(find_metrics(experiment_id).read_text(encoding="utf-8"))
    train = metrics["train_losses"]
    validation = metrics.get("validation_losses", [])
    test = metrics.get("test_losses", [])
    epochs = metrics.get("evaluation_epochs", [])
    config = metrics.get("config", {})
    figure, axis = plt.subplots(figsize=(12, 7))
    axis.plot(range(1, len(train) + 1), train, label="Training loss", linewidth=1.5)
    axis.plot(epochs[: len(validation)], validation, "--", label="Validation loss", linewidth=1.5)
    axis.plot(epochs[: len(test)], test, ":", label="Test loss", linewidth=1.8)
    axis.set_title(
        f"RealNVP {experiment_id} — {config.get('num_layers', '?')} flow layers, "
        f"{config.get('hidden_features', '?')} hidden features"
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"{experiment_id.lower()}_full_loss_curves.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


def plot_combined(name: str, experiments: tuple[str, ...]) -> Path:
    figure, axes = plt.subplots(2, 3, figsize=(21, 11), sharey=True)
    axes = axes.flat
    loaded = []
    for axis, experiment_id in zip(axes[: len(experiments)], experiments):
        metrics = json.loads(find_metrics(experiment_id).read_text(encoding="utf-8"))
        loaded.append((experiment_id, metrics))
        train = metrics["train_losses"]
        validation = metrics.get("validation_losses", [])
        test = metrics.get("test_losses", [])
        epochs = metrics.get("evaluation_epochs", [])
        config = metrics.get("config", {})
        axis.plot(range(1, len(train) + 1), train, label="Train", linewidth=1.2)
        axis.plot(epochs[: len(validation)], validation, "--", label="Validation", linewidth=1.2)
        axis.plot(epochs[: len(test)], test, ":", label="Test", linewidth=1.5)
        axis.set_title(f"{experiment_id}\n{config.get('num_layers', '?')} flow layers")
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Loss")
    combined_axis = axes[len(experiments)]
    for experiment_id, metrics in loaded:
        combined_axis.plot(
            range(1, len(metrics["train_losses"]) + 1),
            metrics["train_losses"],
            label=f"{experiment_id} train",
            linewidth=1.2,
        )
        epochs = metrics.get("evaluation_epochs", [])
        combined_axis.plot(
            epochs[: len(metrics.get("validation_losses", []))],
            metrics.get("validation_losses", []),
            "--",
            label=f"{experiment_id} validation",
            linewidth=1.1,
        )
        combined_axis.plot(
            epochs[: len(metrics.get("test_losses", []))],
            metrics.get("test_losses", []),
            ":",
            label=f"{experiment_id} test",
            linewidth=1.3,
        )
    combined_axis.set_title("All experiments combined")
    combined_axis.set_xlabel("Epoch")
    combined_axis.set_ylabel("Loss")
    combined_axis.grid(alpha=0.25)
    combined_axis.legend(fontsize=7, ncol=2)
    for axis in axes[len(experiments) + 1 :]:
        axis.axis("off")
    figure.suptitle(f"RealNVP {name.replace('_', ' ').title()} sweep — complete loss curves", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"real_nvp_{name}_full_loss_curves.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


def main() -> None:
    for name, experiments in EXPERIMENT_GROUPS.items():
        print(plot_combined(name, experiments).relative_to(ROOT))


if __name__ == "__main__":
    main()

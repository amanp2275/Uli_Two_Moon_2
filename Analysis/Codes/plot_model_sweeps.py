"""Create organized model-sweep loss plots for Transformer and RealNVP."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "model_comparison"
OUTPUT = ROOT / "Analysis" / "organized_plots" / "model_sweep"
EXPERIMENTS = ("EXP_MS_001", "EXP_MS_003", "EXP_MS_004", "EXP_MS_005")
MODELS = ("transformer", "real_nvp")


def find_metrics(experiment_id: str, model: str) -> Path:
    matches = sorted((RESULTS / experiment_id / "final_models" / model).rglob("metrics.json"))
    if not matches:
        raise FileNotFoundError(f"No final model metrics found for {experiment_id} ({model})")
    return matches[0]


def plot_model(model: str) -> Path:
    figure, axes = plt.subplots(2, 3, figsize=(21, 11), sharey=True)
    axes = axes.flat
    loaded = []
    for axis, experiment_id in zip(axes[: len(EXPERIMENTS)], EXPERIMENTS):
        metrics = json.loads(find_metrics(experiment_id, model).read_text(encoding="utf-8"))
        loaded.append((experiment_id, metrics))
        train = metrics["train_losses"]
        validation = metrics.get("validation_losses", [])
        test = metrics.get("test_losses", [])
        epochs = metrics.get("evaluation_epochs", [])
        config = metrics.get("config", {})
        axis.plot(range(1, len(train) + 1), train, label="Train", linewidth=1.2)
        axis.plot(epochs[: len(validation)], validation, "--", label="Validation", linewidth=1.2)
        axis.plot(epochs[: len(test)], test, ":", label="Test", linewidth=1.5)
        axis.set_title(f"{experiment_id}\nseed={config.get('seed', '?')}")
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Loss")

    combined = axes[len(EXPERIMENTS)]
    for experiment_id, metrics in loaded:
        epochs = metrics.get("evaluation_epochs", [])
        combined.plot(range(1, len(metrics["train_losses"]) + 1), metrics["train_losses"], label=f"{experiment_id} train", linewidth=1.1)
        combined.plot(epochs[: len(metrics.get("validation_losses", []))], metrics.get("validation_losses", []), "--", label=f"{experiment_id} validation", linewidth=1)
        combined.plot(epochs[: len(metrics.get("test_losses", []))], metrics.get("test_losses", []), ":", label=f"{experiment_id} test", linewidth=1.2)
    combined.set_title("All model-sweep runs combined")
    combined.set_xlabel("Epoch")
    combined.set_ylabel("Loss")
    combined.grid(alpha=0.25)
    combined.legend(fontsize=7, ncol=2)
    for axis in axes[len(EXPERIMENTS) + 1 :]:
        axis.axis("off")

    title = "Transformer" if model == "transformer" else "RealNVP"
    figure.suptitle(f"{title} model sweep — complete loss curves", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    output_dir = OUTPUT / model
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{model}_model_sweep_full_loss_curves.png"
    figure.savefig(output, dpi=170)
    plt.close(figure)
    return output


def main() -> None:
    for model in MODELS:
        print(plot_model(model).relative_to(ROOT))


if __name__ == "__main__":
    main()

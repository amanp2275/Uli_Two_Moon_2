"""Create separate 2x2 Transformer parameter-sweep loss figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "parameter_sweep"
OUTPUT = ROOT / "Analysis" / "organized_plots" / "parameter_sweep" / "transformer"

# Only completed Transformer runs are included.
SWEEPS = {
    "learning_rate": ("EXP_005", "EXP_006", "EXP_007", "EXP_008"),
    "weight_decay": ("EXP_013", "EXP_014", "EXP_015", "EXP_016"),
    "block_count": ("EXP_025", "EXP_026"),
    "head_dimension": ("EXP_029", "EXP_030"),
    "expansion": ("EXP_032", "EXP_033", "EXP_034"),
}


def find_metrics(experiment_id: str) -> Path:
    matches = sorted((RESULTS / experiment_id).rglob("metrics.json"))
    if not matches:
        raise FileNotFoundError(f"No metrics.json found for {experiment_id}")
    return matches[0]


def plot_sweep(name: str, experiment_ids: tuple[str, ...]) -> Path:
    columns = 3 if len(experiment_ids) == 4 else 2
    figure, axes = plt.subplots(2, columns, figsize=(21, 11) if columns == 3 else (15, 11), sharey=True)
    axes = axes.flat
    loaded = []
    for axis, experiment_id in zip(axes[: len(experiment_ids)], experiment_ids):
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
        axis.set_title(f"{experiment_id}\n{_subtitle(name, config)}")
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Loss")

    combined = axes[len(experiment_ids)]
    for experiment_id, metrics in loaded:
        epochs = metrics.get("evaluation_epochs", [])
        combined.plot(range(1, len(metrics["train_losses"]) + 1), metrics["train_losses"], label=f"{experiment_id} train", linewidth=1.1)
        combined.plot(epochs[: len(metrics.get("validation_losses", []))], metrics.get("validation_losses", []), "--", label=f"{experiment_id} validation", linewidth=1)
        combined.plot(epochs[: len(metrics.get("test_losses", []))], metrics.get("test_losses", []), ":", label=f"{experiment_id} test", linewidth=1.2)
    combined.set_title("All experiments combined")
    combined.set_xlabel("Epoch")
    combined.set_ylabel("Loss")
    combined.grid(alpha=0.25)
    combined.legend(fontsize=7, ncol=2)
    for axis in axes[len(experiment_ids) + 1 :]:
        axis.axis("off")

    figure.suptitle(f"Transformer {name.replace('_', ' ').title()} sweep — complete loss curves", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output = OUTPUT / f"transformer_{name}_full_loss_curves.png"
    figure.savefig(output, dpi=170)
    plt.close(figure)
    return output


def _subtitle(name: str, config: dict) -> str:
    values = {
        "learning_rate": f"lr={config.get('learning_rate', '?')}",
        "weight_decay": f"wd={config.get('weight_decay', '?')}",
        "block_count": f"{config.get('num_blocks', '?')} blocks",
        "head_dimension": f"head dim={config.get('head_dim', '?')}",
        "expansion": f"expansion={config.get('expansion', '?')}",
    }
    return values.get(name, "")


def main() -> None:
    for name, experiment_ids in SWEEPS.items():
        print(plot_sweep(name, experiment_ids).relative_to(ROOT))


if __name__ == "__main__":
    main()

"""Plot grouped loss curves and samples for the canonical model sweep."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset import TwoMoonsSplits
from models.real_nvp import RealNVP
from models.transformer_flow import TransformerFlow


RESULTS = ROOT / "results" / "model_comparison"
OUTPUT = ROOT / "Analysis" / "organized_plots" / "model_sweep"
EXPERIMENTS = ("EXP_MS_001", "EXP_MS_003", "EXP_MS_004", "EXP_MS_005")
MODELS = ("real_nvp", "transformer")


def find_artifact(experiment_id: str, model: str, filename: str) -> Path:
    matches = sorted((RESULTS / experiment_id / "final_models" / model).rglob(filename))
    if not matches:
        raise FileNotFoundError(f"No {filename} found for {experiment_id} ({model})")
    return matches[0]


def load_metrics(experiment_id: str, model: str) -> dict:
    path = find_artifact(experiment_id, model, "metrics.json")
    return json.loads(path.read_text(encoding="utf-8"))


def plot_loss_curves() -> Path:
    figure, axes = plt.subplots(4, 2, figsize=(15, 20), sharex=False)
    for column, model in enumerate(MODELS):
        axes[0, column].set_title("RealNVP" if model == "real_nvp" else "Transformer", fontsize=14, fontweight="bold", pad=14)
        for row, experiment_id in enumerate(EXPERIMENTS):
            metrics = load_metrics(experiment_id, model)
            train = metrics["train_losses"]
            validation = metrics.get("validation_losses", [])
            test = metrics.get("test_losses", [])
            epochs = metrics.get("evaluation_epochs", [])
            axis = axes[row, column]
            axis.plot(range(1, len(train) + 1), train, color="#1f77b4", linewidth=1.2, label="Train")
            axis.plot(epochs[: len(validation)], validation, "--", color="#d62728", linewidth=1.2, label="Validation")
            axis.plot(epochs[: len(test)], test, ":", color="#2ca02c", linewidth=1.5, label="Test")
            axis.set_title(experiment_id, fontsize=11)
            axis.set_xlim(0, 600)
            axis.set_ylabel("Loss")
            axis.grid(alpha=0.25)
            if row == len(EXPERIMENTS) - 1:
                axis.set_xlabel("Epoch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.975))
    figure.suptitle("Model sweep: all loss curves", fontsize=18, fontweight="bold", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.0, w_pad=1.5)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "model_sweep_all_loss_curves.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


def load_generated(experiment_id: str, model_name: str, labels: torch.Tensor) -> torch.Tensor:
    checkpoint_path = find_artifact(experiment_id, model_name, "best_model.pt")
    config = json.loads((checkpoint_path.parent / "config.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if model_name == "real_nvp":
        model = RealNVP(config["num_layers"], config["hidden_features"], config["conditional"])
    else:
        model = TransformerFlow(
            in_channels=config["in_channels"],
            seq_length=config["points_per_batch"],
            channels=config["channels"],
            num_blocks=config["num_blocks"],
            layers_per_block=config["layers_per_block"],
            head_dim=config["head_dim"],
            expansion=config["expansion"],
            nvp=config["nvp"],
            num_classes=2 if config["conditional"] else 0,
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        latent = torch.randn(labels.shape[0], labels.shape[1], 2, generator=torch.Generator().manual_seed(0))
        generated = model.reverse(latent, labels)
    return generated * checkpoint["std"] + checkpoint["mean"]


def limits(reference: torch.Tensor, generated: list[torch.Tensor]) -> tuple[float, float, float, float]:
    points = torch.cat([reference.reshape(-1, 2), *[values.reshape(-1, 2) for values in generated]])
    points = points[torch.isfinite(points).all(dim=1)]
    lower = torch.quantile(points, 0.005, dim=0)
    upper = torch.quantile(points, 0.995, dim=0)
    center = (lower + upper) / 2
    half_range = (upper - lower).max() / 2 * 1.08
    return ((center[0] - half_range).item(), (center[0] + half_range).item(), (center[1] - half_range).item(), (center[1] + half_range).item())


def plot_samples() -> Path:
    splits = TwoMoonsSplits.load(ROOT / "two_moons_splits.pt")
    reference = splits.test_X[:32]
    labels = splits.test_labels[:32]
    generated = {
        model: [load_generated(experiment_id, model, labels) for experiment_id in EXPERIMENTS]
        for model in MODELS
    }
    figure, axes = plt.subplots(5, 2, figsize=(15, 25), sharex=False, sharey=False)
    for column, model in enumerate(MODELS):
        model_limits = limits(reference, generated[model])
        axes[0, column].set_title("RealNVP" if model == "real_nvp" else "Transformer", fontsize=14, fontweight="bold", pad=14)
        panels = [reference, *generated[model]]
        titles = ["Given test samples", *EXPERIMENTS]
        for row, (samples, title) in enumerate(zip(panels, titles)):
            sample_points = samples.detach().cpu().reshape(-1, 2)
            sample_labels = labels.detach().cpu().reshape(-1)
            finite = torch.isfinite(sample_points).all(dim=1)
            axis = axes[row, column]
            axis.scatter(sample_points[finite, 0], sample_points[finite, 1], c=sample_labels[finite], cmap="coolwarm", s=7)
            axis.set_title(title, fontsize=10)
            axis.set_aspect("equal")
            axis.set_xlim(model_limits[0], model_limits[1])
            axis.set_ylim(model_limits[2], model_limits[3])
            axis.grid(alpha=0.2)
    figure.suptitle("Model sweep: given and generated samples", fontsize=18, fontweight="bold", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.97), h_pad=1.5, w_pad=1.5)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "model_sweep_all_samples.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


def main() -> None:
    print(plot_loss_curves().relative_to(ROOT))
    print(plot_samples().relative_to(ROOT))


if __name__ == "__main__":
    main()
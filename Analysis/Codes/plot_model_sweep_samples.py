"""Compare given and generated samples across the canonical model sweeps."""

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


DATASET = ROOT / "two_moons_splits.pt"
OUTPUT = ROOT / "Analysis" / "organized_plots" / "model_sweep"
EXPERIMENTS = ("EXP_MS_001", "EXP_MS_003", "EXP_MS_004", "EXP_MS_005")


def find_run_file(experiment_id: str, model: str, filename: str) -> Path:
    matches = sorted((ROOT / "results" / "model_comparison" / experiment_id / "final_models" / model).rglob(filename))
    if not matches:
        raise FileNotFoundError(f"No {filename} found for {experiment_id} ({model})")
    return matches[0]


def load_generated_samples(experiment_id: str, model_name: str, labels: torch.Tensor) -> torch.Tensor:
    checkpoint_path = find_run_file(experiment_id, model_name, "best_model.pt")
    config_path = checkpoint_path.parent / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
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
        latent = torch.randn(
            labels.shape[0], labels.shape[1], 2, generator=torch.Generator().manual_seed(0)
        )
        generated = model.reverse(latent, labels)
    return generated * checkpoint["std"] + checkpoint["mean"]


def robust_limits(reference: torch.Tensor, generated: dict[str, torch.Tensor]) -> tuple[float, float, float, float]:
    points = torch.cat([reference.reshape(-1, 2), *[values.reshape(-1, 2) for values in generated.values()]])
    points = points[torch.isfinite(points).all(dim=1)]
    lower = torch.quantile(points, 0.005, dim=0)
    upper = torch.quantile(points, 0.995, dim=0)
    center = (lower + upper) / 2
    half_range = (upper - lower).max() / 2 * 1.08
    return (
        (center[0] - half_range).item(),
        (center[0] + half_range).item(),
        (center[1] - half_range).item(),
        (center[1] + half_range).item(),
    )


def plot_panel(axis, samples: torch.Tensor, labels: torch.Tensor, title: str, limits: tuple[float, float, float, float]) -> None:
    samples = samples.detach().cpu().reshape(-1, 2)
    labels = labels.detach().cpu().reshape(-1)
    finite = torch.isfinite(samples).all(dim=1)
    axis.scatter(samples[finite, 0], samples[finite, 1], c=labels[finite], cmap="coolwarm", s=8)
    axis.set_title(title)
    axis.set_aspect("equal")
    axis.set_xlim(limits[0], limits[1])
    axis.set_ylim(limits[2], limits[3])
    axis.grid(alpha=0.2)


def plot_model(model_name: str, reference: torch.Tensor, labels: torch.Tensor) -> Path:
    generated = {
        experiment_id: load_generated_samples(experiment_id, model_name, labels)
        for experiment_id in EXPERIMENTS
    }
    limits = robust_limits(reference, generated)
    figure, axes = plt.subplots(1, 5, figsize=(25, 5), sharex=True, sharey=True)
    plot_panel(axes[0], reference, labels, "Given test samples", limits)
    for axis, (experiment_id, samples) in zip(axes[1:], generated.items()):
        plot_panel(axis, samples, labels, f"Generated samples\n{experiment_id}", limits)
    title = "RealNVP" if model_name == "real_nvp" else "Transformer"
    figure.suptitle(f"{title} model sweep: given vs generated samples", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    output_dir = OUTPUT / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_model_sweep_samples_comparison.png"
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    return output_path


def main() -> None:
    splits = TwoMoonsSplits.load(DATASET)
    plot_batches = 32
    reference = splits.test_X[:plot_batches]
    labels = splits.test_labels[:plot_batches]
    for model_name in ("real_nvp", "transformer"):
        print(plot_model(model_name, reference, labels).relative_to(ROOT))


if __name__ == "__main__":
    main()
"""Compare given and generated samples across Transformer parameter sweeps."""

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
from models.transformer_flow import TransformerFlow


DATASET = ROOT / "two_moons_splits.pt"
OUTPUT = ROOT / "Analysis" / "organized_plots" / "parameter_sweep" / "transformer"
SWEEPS = {
    "learning_rate": {
        "learning rate = 1e-3": "EXP_005",
        "learning rate = 5e-4": "EXP_006",
        "learning rate = 1e-4": "EXP_007",
        "learning rate = 5e-5": "EXP_008",
    },
    "weight_decay": {
        "weight decay = 0": "EXP_013",
        "weight decay = 1e-5": "EXP_014",
        "weight decay = 1e-4": "EXP_015",
        "weight decay = 1e-3": "EXP_016",
    },
    "block_count": {
        "2 blocks": "EXP_025",
        "6 blocks": "EXP_026",
    },
    "head_dimension": {
        "16 head dimension": "EXP_029",
        "32 head dimension": "EXP_030",
    },
    "expansion": {
        "2x expansion": "EXP_032",
        "6x expansion": "EXP_033",
        "4x expansion": "EXP_034",
    },
}


def find_run_file(experiment_id: str, filename: str) -> Path:
    matches = sorted((ROOT / "results" / "parameter_sweep" / experiment_id).rglob(filename))
    if not matches:
        raise FileNotFoundError(f"No {filename} found for {experiment_id}")
    return matches[0]


def load_generated_samples(experiment_id: str, labels: torch.Tensor) -> torch.Tensor:
    checkpoint_path = find_run_file(experiment_id, "best_model.pt")
    config_path = checkpoint_path.parent / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
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
    mean = checkpoint["mean"]
    std = checkpoint["std"]
    with torch.no_grad():
        latent = torch.randn(
            labels.shape[0], labels.shape[1], 2, generator=torch.Generator().manual_seed(0)
        )
        generated = model.reverse(latent, labels)
    return generated * std + mean


def robust_limits(reference: torch.Tensor, generated: dict[str, torch.Tensor]) -> tuple[float, float, float, float]:
    samples = [reference.reshape(-1, 2), *[values.reshape(-1, 2) for values in generated.values()]]
    points = torch.cat(samples, dim=0)
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


def plot_panel(
    axis,
    samples: torch.Tensor,
    labels: torch.Tensor,
    title: str,
    limits: tuple[float, float, float, float],
) -> None:
    samples = samples.detach().cpu().reshape(-1, 2)
    labels = labels.detach().cpu().reshape(-1)
    finite = torch.isfinite(samples).all(dim=1)
    samples = samples[finite]
    labels = labels[finite]
    axis.scatter(samples[:, 0], samples[:, 1], c=labels, cmap="coolwarm", s=8)
    axis.set_title(title)
    axis.set_aspect("equal")
    axis.set_xlim(limits[0], limits[1])
    axis.set_ylim(limits[2], limits[3])
    axis.grid(alpha=0.2)


def plot_sweep(name: str, runs: dict[str, str], reference: torch.Tensor, labels: torch.Tensor) -> Path:
    generated = {
        title: load_generated_samples(experiment_id, labels)
        for title, experiment_id in runs.items()
    }
    limits = robust_limits(reference, generated)
    figure, axes = plt.subplots(
        1, len(generated) + 1, figsize=(5 * (len(generated) + 1), 5), sharex=True, sharey=True
    )
    plot_panel(axes[0], reference, labels, "Given test samples", limits)
    for axis, (title, samples) in zip(axes[1:], generated.items()):
        plot_panel(axis, samples, labels, f"Generated samples\n{title}", limits)
    figure.suptitle(f"Transformer {name.replace('_', ' ')} sweep: given vs generated samples", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT / f"transformer_{name}_samples_comparison.png"
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    return output_path


def main() -> None:
    splits = TwoMoonsSplits.load(DATASET)
    plot_batches = 32
    reference = splits.test_X[:plot_batches]
    labels = splits.test_labels[:plot_batches]
    for name, runs in SWEEPS.items():
        print(plot_sweep(name, runs, reference, labels).relative_to(ROOT))


if __name__ == "__main__":
    main()
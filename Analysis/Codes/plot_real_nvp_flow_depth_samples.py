"""Compare given and generated samples across the RealNVP flow-depth sweep."""

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


DATASET = ROOT / "two_moons_splits.pt"
OUTPUT = ROOT / "Analysis" / "organized_plots" / "parameter_sweep" / "real_nvp"
SWEEPS = {
    "flow_depth": {
        "3 flow layers": "EXP_017",
        "7 flow layers": "EXP_019",
        "9 flow layers": "EXP_020",
    },
    "hidden_dimension": {
        "32 hidden features": "EXP_021",
        "64 hidden features": "EXP_022",
        "128 hidden features": "EXP_023",
        "256 hidden features": "EXP_024",
    },
    "learning_rate": {
        "learning rate = 1e-3": "EXP_001",
        "learning rate = 5e-4": "EXP_002",
        "learning rate = 1e-4": "EXP_003",
        "learning rate = 5e-5": "EXP_004",
    },
    "weight_decay": {
        "weight decay = 0": "EXP_009",
        "weight decay = 1e-5": "EXP_010",
        "weight decay = 1e-4": "EXP_011",
        "weight decay = 1e-3": "EXP_012",
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
    model = RealNVP(
        num_layers=config["num_layers"],
        hidden_features=config["hidden_features"],
        conditional=config["conditional"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    mean = checkpoint["mean"]
    std = checkpoint["std"]
    with torch.no_grad():
        latent = torch.randn(labels.shape[0], labels.shape[1], 2, generator=torch.Generator().manual_seed(0))
        generated = model.reverse(latent, labels)
    return generated * std + mean


def plot_panel(axis, samples: torch.Tensor, labels: torch.Tensor, title: str) -> None:
    samples = samples.detach().cpu().reshape(-1, 2)
    labels = labels.detach().cpu().reshape(-1)
    finite = torch.isfinite(samples).all(dim=1)
    samples = samples[finite]
    labels = labels[finite]
    axis.scatter(samples[:, 0], samples[:, 1], c=labels, cmap="coolwarm", s=8)
    axis.set_title(title)
    axis.set_aspect("equal")
    axis.grid(alpha=0.2)


def plot_sweep(name: str, runs: dict[str, str], reference: torch.Tensor, labels: torch.Tensor) -> Path:
    generated = {
        title: load_generated_samples(experiment_id, labels)
        for title, experiment_id in runs.items()
    }
    figure, axes = plt.subplots(1, len(generated) + 1, figsize=(5 * (len(generated) + 1), 5), sharex=True, sharey=True)
    plot_panel(axes[0], reference, labels, "Given test samples")
    for axis, (title, samples) in zip(axes[1:], generated.items()):
        plot_panel(axis, samples, labels, f"Generated samples\n{title}")
    figure.suptitle(f"RealNVP {name.replace('_', ' ')} sweep: given vs generated samples", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT / f"real_nvp_{name}_samples_comparison.png"
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
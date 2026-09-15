"""Plot EXP_034 onward samples in the established reference/generated layout."""

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
RESULTS = ROOT / "results" / "parameter_sweep"
OUTPUT = ROOT / "Analysis" / "organized_plots" / "parameter_sweep" / "transformer"


def find_file(experiment_id: str, filename: str) -> Path | None:
    matches = sorted((RESULTS / experiment_id).rglob(filename))
    return matches[0] if matches else None


def load_generated(experiment_id: str, labels: torch.Tensor) -> torch.Tensor:
    checkpoint_path = find_file(experiment_id, "best_model.pt")
    if checkpoint_path is None:
        raise FileNotFoundError(f"No best_model.pt found for {experiment_id}")
    config_file = checkpoint_path.parent / "config.json"
    raw_config = json.loads(config_file.read_text(encoding="utf-8"))
    config = raw_config.get("resolved_config", raw_config.get("config", raw_config))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = TransformerFlow(
        in_channels=int(config["in_channels"]),
        seq_length=int(config["points_per_batch"]),
        channels=int(config["channels"]),
        num_blocks=int(config["num_blocks"]),
        layers_per_block=int(config["layers_per_block"]),
        head_dim=int(config["head_dim"]),
        expansion=int(config["expansion"]),
        nvp=config["nvp"] in (True, "YES", "yes"),
        num_classes=2 if config["conditional"] in (True, "YES", "yes") else 0,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        latent = torch.randn(
            labels.shape[0], labels.shape[1], 2,
            generator=torch.Generator().manual_seed(0),
        )
        generated = model.reverse(latent, labels)
    return generated * checkpoint["std"] + checkpoint["mean"]


def limits(reference: torch.Tensor, generated: list[torch.Tensor]) -> tuple[float, float, float, float]:
    points = torch.cat([reference.reshape(-1, 2), *[values.reshape(-1, 2) for values in generated]])
    points = points[torch.isfinite(points).all(dim=1)]
    lower = torch.quantile(points, 0.005, dim=0)
    upper = torch.quantile(points, 0.995, dim=0)
    center = (lower + upper) / 2
    half_range = (upper - lower).max() / 2 * 1.08
    return tuple((value.item() for value in (
        center[0] - half_range, center[0] + half_range,
        center[1] - half_range, center[1] + half_range,
    )))


def draw(axis, samples: torch.Tensor, labels: torch.Tensor, title: str, bounds: tuple[float, float, float, float]) -> None:
    samples = samples.detach().cpu().reshape(-1, 2)
    labels = labels.detach().cpu().reshape(-1)
    finite = torch.isfinite(samples).all(dim=1)
    axis.scatter(samples[finite, 0], samples[finite, 1], c=labels[finite], cmap="coolwarm", s=7)
    axis.set_title(title, fontsize=9)
    axis.set_aspect("equal")
    axis.set_xlim(bounds[0], bounds[1])
    axis.set_ylim(bounds[2], bounds[3])
    axis.grid(alpha=0.2)


def experiment_title(experiment_id: str) -> str:
    config_file = find_file(experiment_id, "config.json")
    raw = json.loads(config_file.read_text(encoding="utf-8"))
    config = raw.get("resolved_config", raw.get("config", raw))
    return (
        f"{experiment_id}\n"
        f"blocks={config['num_blocks']}, head={config['head_dim']}, exp={config['expansion']}"
    )


def main() -> None:
    experiment_ids = [
        f"EXP_{number:03d}"
        for number in sorted(
            int(path.name.split("_")[1]) for path in RESULTS.iterdir()
            if path.is_dir() and path.name.startswith("EXP_")
            and int(path.name.split("_")[1]) >= 34
        )
    ]
    splits = TwoMoonsSplits.load(DATASET)
    reference = splits.test_X[:32]
    labels = splits.test_labels[:32]
    generated = {}
    missing = []
    for experiment_id in experiment_ids:
        try:
            generated[experiment_id] = load_generated(experiment_id, labels)
        except FileNotFoundError:
            missing.append(experiment_id)

    bounds = limits(reference, list(generated.values()))
    figure, axes = plt.subplots(4, 5, figsize=(25, 20), sharex=True, sharey=True)
    for column in range(5):
        draw(axes[0, column], reference, labels, "Given test samples", bounds)
    for index, experiment_id in enumerate(experiment_ids):
        axis = axes[1 + index // 5, index % 5]
        if experiment_id in generated:
            draw(axis, generated[experiment_id], labels, experiment_title(experiment_id), bounds)
        else:
            axis.text(0.5, 0.5, "samples unavailable", ha="center", va="center")
            axis.set_title(experiment_id, fontsize=9)
            axis.set_axis_off()
    for axis in axes.ravel()[len(experiment_ids) + 5:]:
        axis.set_axis_off()

    figure.suptitle("Transformer experiments EXP_034 onward: given and generated samples", fontsize=18, fontweight="bold", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.97), h_pad=1.5, w_pad=1.0)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT / "transformer_experiments_EXP034_onward_samples_reference_format.png"
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    print(output_path.relative_to(ROOT))
    print(f"Generated: {', '.join(generated)}")
    if missing:
        print(f"Missing: {', '.join(missing)}")


if __name__ == "__main__":
    main()

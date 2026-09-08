"""Plot all parameter-sweep given and generated samples in dedicated columns."""

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
OUTPUTS = {
    "real_nvp": ROOT / "Analysis" / "organized_plots" / "parameter_sweep" / "real_nvp",
    "transformer": ROOT / "Analysis" / "organized_plots" / "parameter_sweep" / "transformer",
}
SWEEPS = {
    "real_nvp": (
        ("Flow depth", ("EXP_017", "EXP_019", "EXP_020")),
        ("Hidden dimension", ("EXP_021", "EXP_022", "EXP_023", "EXP_024")),
        ("Learning rate", ("EXP_001", "EXP_002", "EXP_003", "EXP_004")),
        ("Weight decay", ("EXP_009", "EXP_010", "EXP_011", "EXP_012")),
    ),
    "transformer": (
        ("Learning rate", ("EXP_005", "EXP_006", "EXP_007", "EXP_008")),
        ("Weight decay", ("EXP_013", "EXP_014", "EXP_015", "EXP_016")),
        ("Block count", ("EXP_025", "EXP_026")),
        ("Head dimension", ("EXP_029", "EXP_030")),
        ("Expansion", ("EXP_032", "EXP_033", "EXP_034")),
    ),
}


def find_run_file(model: str, experiment_id: str, filename: str) -> Path:
    matches = sorted((ROOT / "results" / "parameter_sweep" / experiment_id).rglob(filename))
    matches = [path for path in matches if model in path.parts]
    if not matches:
        raise FileNotFoundError(f"No {filename} found for {model} {experiment_id}")
    return matches[0]


def load_generated_samples(model_name: str, experiment_id: str, labels: torch.Tensor) -> torch.Tensor:
    checkpoint_path = find_run_file(model_name, experiment_id, "best_model.pt")
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


def robust_limits(reference: torch.Tensor, generated: list[torch.Tensor]) -> tuple[float, float, float, float]:
    points = torch.cat([reference.reshape(-1, 2), *[values.reshape(-1, 2) for values in generated]])
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
    axis.scatter(samples[finite, 0], samples[finite, 1], c=labels[finite], cmap="coolwarm", s=7)
    axis.set_title(title, fontsize=9)
    axis.set_aspect("equal")
    axis.set_xlim(limits[0], limits[1])
    axis.set_ylim(limits[2], limits[3])
    axis.grid(alpha=0.2)


def format_experiment_title(model_name: str, experiment_id: str) -> str:
    path = find_run_file(model_name, experiment_id, "config.json")
    config = json.loads(path.read_text(encoding="utf-8"))
    if model_name == "real_nvp":
        values = [("layers", config["num_layers"]), ("hidden", config["hidden_features"])]
    else:
        values = [("blocks", config["num_blocks"]), ("head", config["head_dim"]), ("exp", config["expansion"])]
    return f"{experiment_id}\n" + ", ".join(f"{name}={value}" for name, value in values)


def plot_model(model_name: str, reference: torch.Tensor, labels: torch.Tensor) -> Path:
    sweep_groups = SWEEPS[model_name]
    generated = {
        experiment_id: load_generated_samples(model_name, experiment_id, labels)
        for _, experiment_ids in sweep_groups
        for experiment_id in experiment_ids
    }
    limits = robust_limits(reference, list(generated.values()))
    rows = 5
    figure, axes = plt.subplots(rows, len(sweep_groups), figsize=(5 * len(sweep_groups), 5 * rows), sharex=True, sharey=True)
    if len(sweep_groups) == 1:
        axes = axes.reshape(rows, 1)
    for column, (sweep_name, experiment_ids) in enumerate(sweep_groups):
        axes[0, column].set_title(sweep_name, fontsize=13, fontweight="bold", pad=12)
        plot_panel(axes[0, column], reference, labels, "Given test samples", limits)
        for row, experiment_id in enumerate(experiment_ids, start=1):
            plot_panel(axes[row, column], generated[experiment_id], labels, format_experiment_title(model_name, experiment_id), limits)
        for row in range(len(experiment_ids) + 1, rows):
            axes[row, column].axis("off")
    title = "RealNVP" if model_name == "real_nvp" else "Transformer"
    figure.suptitle(f"{title} parameter sweeps: given and generated samples", fontsize=18, fontweight="bold", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.97), h_pad=1.5, w_pad=1.0)
    output_path = OUTPUTS[model_name] / f"{model_name}_all_parameter_sweeps_samples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    return output_path


def main() -> None:
    splits = TwoMoonsSplits.load(DATASET)
    reference = splits.test_X[:32]
    labels = splits.test_labels[:32]
    for model_name in ("real_nvp", "transformer"):
        print(plot_model(model_name, reference, labels).relative_to(ROOT))


if __name__ == "__main__":
    main()
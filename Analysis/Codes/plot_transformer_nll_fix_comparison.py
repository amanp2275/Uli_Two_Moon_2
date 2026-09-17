"""Compare Transformer experiments before and after the NLL fixes.

The figure is deliberately built from the canonical experiment artifacts:
metrics.json for loss curves, samples_final.png for the saved generated
samples, and config/registry data for parameters and implementation notes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "results" / "experiment_results.csv"
RESULTS = ROOT / "results" / "parameter_sweep"
OUTPUT = ROOT / "Analysis" / "nll_fix_comparison"
EXPERIMENTS = ("EXP_041", "EXP_043", "EXP_047", "EXP_048", "EXP_049", "EXP_050")

IMPLEMENTATION_NOTES = {
    "EXP_041": (
        "Old NLL: Gaussian constant was added once per point, not once per coordinate. "
        "Loss ignored variance in the prior; generation used self.var."
    ),
    "EXP_043": (
        "Old NLL: Gaussian constant was added once per point, not once per coordinate. "
        "Loss ignored variance in the prior; generation used self.var."
    ),
    "EXP_047": (
        "Coordinate-wise NLL fix: sums [sequence, coordinate] and normalizes by T*C. "
        "Loss uses unit variance; generation still uses self.var."
    ),
    "EXP_048": (
        "Coordinate-wise NLL fix: sums [sequence, coordinate] and normalizes by T*C. "
        "Loss uses unit variance; generation still uses self.var."
    ),
    "EXP_049": (
        "Coordinate-wise NLL fix plus learned self.var in the loss and generation "
        "(per-position, per-coordinate prior variance)."
    ),
    "EXP_050": (
        "Coordinate-wise NLL fix plus learned self.var in the loss and generation "
        "(per-position, per-coordinate prior variance)."
    ),
}


def first_match(experiment_id: str, filename: str) -> Path:
    matches = sorted((RESULTS / experiment_id).rglob(filename))
    if not matches:
        raise FileNotFoundError(f"No {filename} found for {experiment_id}")
    return matches[0]


def load_registry() -> dict[str, dict[str, str]]:
    with REGISTRY.open(encoding="utf-8-sig", newline="") as handle:
        return {row["experiment_id"]: row for row in csv.DictReader(handle)}


def load_experiment(experiment_id: str, registry: dict[str, dict[str, str]]) -> dict:
    metrics_path = first_match(experiment_id, "metrics.json")
    config_path = first_match(experiment_id, "config.json")
    config_raw = json.loads(config_path.read_text(encoding="utf-8"))
    config = config_raw.get("resolved_config", config_raw.get("config", config_raw))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "id": experiment_id,
        "metrics": metrics,
        "config": config,
        "registry": registry[experiment_id],
        "sample_path": first_match(experiment_id, "samples_final.png"),
        "metrics_path": metrics_path,
        "config_path": config_path,
    }


def fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def write_summary(rows: list[dict]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT / "experiment_summary.csv"
    fields = [
        "experiment", "git_commit", "epochs", "best_epoch", "final_test_nll",
        "best_validation_nll", "trainable_parameters", "learning_rate", "weight_decay",
        "channels", "num_blocks", "layers_per_block", "head_dim", "expansion",
        "nll_interpretation",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            metrics = item["metrics"]
            config = item["config"]
            registry = item["registry"]
            writer.writerow({
                "experiment": item["id"],
                "git_commit": registry.get("git_commit", ""),
                "epochs": len(metrics.get("train_losses", [])),
                "best_epoch": metrics.get("best_epoch", ""),
                "final_test_nll": metrics.get("final_test_loss", ""),
                "best_validation_nll": metrics.get("best_validation_loss", ""),
                "trainable_parameters": metrics.get("total_trainable_parameters", ""),
                "learning_rate": config.get("learning_rate", ""),
                "weight_decay": config.get("weight_decay", ""),
                "channels": config.get("channels", ""),
                "num_blocks": config.get("num_blocks", ""),
                "layers_per_block": config.get("layers_per_block", ""),
                "head_dim": config.get("head_dim", ""),
                "expansion": config.get("expansion", ""),
                "nll_interpretation": IMPLEMENTATION_NOTES[item["id"]],
            })


def write_report(rows: list[dict]) -> None:
    lines = [
        "# Transformer NLL-fix comparison",
        "",
        "Selected experiments: EXP_041, EXP_043, EXP_047, EXP_048, EXP_049, and EXP_050.",
        "The figure uses the saved loss history and saved final sample panels from each run.",
        "",
        "## Cross-check",
        "",
        "The original Transformer loss formed `0.5 * z.pow(2).sum(dim=-1)` and then added "
        "the Gaussian constant once per point. Because the latent has two coordinates, "
        "the constant should be applied to both coordinates. The corrected implementation "
        "works elementwise, sums over dimensions `[1, 2]`, and divides by `z.size(1) * "
        "z.size(2)`.",
        "",
        "The repository history shows that EXP_047/048 used the corrected coordinate-wise "
        "NLL with a unit-variance loss, but their trainer still called `model.reverse(...)` "
        "without the `latent_var` override; therefore generation used the learned `self.var`. "
        "EXP_049/050 changed the loss to use the learned per-position, per-coordinate "
        "`self.var` as well. So the claim that EXP_047/048 generated with fixed variance "
        "0.1 or 1 is not supported by the recorded code commit.",
        "",
        "## Results",
        "",
        "| Experiment | Final test NLL | Best epoch | Parameters | NLL implementation |",
        "|---|---:|---:|---:|---|",
    ]
    for item in rows:
        metrics = item["metrics"]
        lines.append(
            f"| {item['id']} | {metrics['final_test_loss']:.6f} | "
            f"{metrics['best_epoch']} | {metrics['total_trainable_parameters']:,} | "
            f"{IMPLEMENTATION_NOTES[item['id']]} |"
        )
    lines.extend([
        "",
        "Raw NLL values across these groups should be interpreted with care: EXP_041/043 "
        "use the under-normalized old objective, EXP_047/048 use the corrected unit-variance "
        "objective, and EXP_049/050 use the corrected learned-variance objective.",
        "",
        "Source artifacts are linked in `experiment_summary.csv`; the registry also records "
        "the Git commit used for each run.",
    ])
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(rows: list[dict]) -> Path:
    figure, axes = plt.subplots(
        len(rows), 3, figsize=(22, 5.4 * len(rows)),
        gridspec_kw={"width_ratios": [1.15, 1.0, 1.25]},
    )
    if len(rows) == 1:
        axes = [axes]

    for row_index, item in enumerate(rows):
        loss_axis, sample_axis, text_axis = axes[row_index]
        metrics = item["metrics"]
        config = item["config"]
        experiment_id = item["id"]
        epochs = list(range(1, len(metrics["train_losses"]) + 1))
        eval_epochs = metrics.get("evaluation_epochs", [])

        loss_axis.plot(epochs, metrics["train_losses"], label="Train", linewidth=1.2)
        loss_axis.plot(
            eval_epochs[:len(metrics.get("validation_losses", []))],
            metrics.get("validation_losses", []), "--", label="Validation", linewidth=1.2,
        )
        loss_axis.plot(
            eval_epochs[:len(metrics.get("test_losses", []))],
            metrics.get("test_losses", []), ":", label="Test", linewidth=1.6,
        )
        loss_axis.set_title(f"{experiment_id}: loss / NLL curves")
        loss_axis.set_xlabel("Epoch")
        loss_axis.set_ylabel("NLL per coordinate")
        loss_axis.grid(alpha=0.25)
        loss_axis.legend(fontsize=8)

        sample_axis.imshow(mpimg.imread(item["sample_path"]))
        sample_axis.set_title(f"{experiment_id}: saved final samples")
        sample_axis.axis("off")

        registry = item["registry"]
        parameter_text = (
            f"Final test NLL: {metrics['final_test_loss']:.6f}\n"
            f"Best validation NLL: {metrics['best_validation_loss']:.6f}\n"
            f"Best epoch: {metrics['best_epoch']} / {len(metrics['train_losses'])}\n"
            f"Trainable parameters: {metrics['total_trainable_parameters']:,}\n\n"
            f"lr={fmt(config.get('learning_rate'))}; wd={fmt(config.get('weight_decay'))}\n"
            f"batch={config.get('batch_size')}; seed={config.get('seed')}\n"
            f"channels={config.get('channels')}; blocks={config.get('num_blocks')}\n"
            f"layers/block={config.get('layers_per_block')}; head={config.get('head_dim')}\n"
            f"expansion={config.get('expansion')}; conditional={config.get('conditional')}\n\n"
            f"NLL implementation\n{IMPLEMENTATION_NOTES[experiment_id]}\n\n"
            f"Run commit: {registry.get('git_commit', '')[:8]}"
        )
        text_axis.text(0.02, 0.98, parameter_text, va="top", ha="left", fontsize=10, wrap=True)
        text_axis.set_title(f"{experiment_id}: parameters and interpretation")
        text_axis.axis("off")

    figure.suptitle(
        "Transformer NLL correction comparison: EXP_041, EXP_043, EXP_047–050",
        fontsize=18, fontweight="bold", y=0.997,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.992), h_pad=2.0, w_pad=2.0)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT / "transformer_nll_fix_comparison.png"
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    registry = load_registry()
    rows = [load_experiment(experiment_id, registry) for experiment_id in EXPERIMENTS]
    write_summary(rows)
    write_report(rows)
    output = plot(rows)
    print(output.relative_to(ROOT))
    print((OUTPUT / "experiment_summary.csv").relative_to(ROOT))
    print((OUTPUT / "README.md").relative_to(ROOT))


if __name__ == "__main__":
    main()

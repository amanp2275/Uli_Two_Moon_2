"""Create a matching multi-panel figure of final generated samples."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "parameter_sweep"
OUTPUT = ROOT / "Analysis" / "organized_plots" / "parameter_sweep" / "transformer"
START_EXPERIMENT = 34


def find_file(experiment_id: str, filename: str) -> Path | None:
    matches = sorted((RESULTS / experiment_id).rglob(filename))
    return matches[0] if matches else None


def format_value(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value:g}"
    return str(value)


def title(experiment_id: str) -> str:
    config_path = find_file(experiment_id, "config.json")
    if config_path is None:
        return experiment_id
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config = config.get("resolved_config", config.get("config", config))
    return (
        f"{experiment_id}\n"
        f"lr={format_value(config.get('learning_rate', '?'))}, "
        f"wd={format_value(config.get('weight_decay', '?'))}; "
        f"blocks={config.get('num_blocks', '?')}, head={config.get('head_dim', '?')}, "
        f"exp={config.get('expansion', '?')}"
    )


def main() -> None:
    experiment_numbers = sorted(
        int(path.name.split("_")[1])
        for path in RESULTS.iterdir()
        if path.is_dir() and path.name.startswith("EXP_")
        and int(path.name.split("_")[1]) >= START_EXPERIMENT
    )
    figure, axes = plt.subplots(5, 3, figsize=(18, 25))
    axes = axes.ravel()
    plotted = []
    missing = []

    for index, number in enumerate(experiment_numbers):
        experiment_id = f"EXP_{number:03d}"
        axis = axes[index]
        sample_path = find_file(experiment_id, "samples_final.png")
        if sample_path is None:
            axis.text(0.5, 0.5, "samples_final.png unavailable", ha="center", va="center")
            axis.set_title(experiment_id, fontsize=10)
            axis.set_axis_off()
            missing.append(experiment_id)
            continue
        axis.imshow(mpimg.imread(sample_path))
        axis.set_title(title(experiment_id), fontsize=9)
        axis.axis("off")
        plotted.append(experiment_id)

    for axis in axes[len(experiment_numbers):]:
        axis.set_axis_off()

    figure.suptitle(
        "Transformer experiments EXP_034 onward: final generated samples",
        fontsize=18, fontweight="bold", y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97), h_pad=1.5, w_pad=1.0)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT / "transformer_experiments_EXP034_onward_samples.png"
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    print(output_path.relative_to(ROOT))
    print(f"Plotted: {', '.join(plotted)}")
    if missing:
        print(f"Missing samples: {', '.join(missing)}")


if __name__ == "__main__":
    main()

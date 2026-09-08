"""Plot model-sweep test NLL for RealNVP and Transformer."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TABLES = ROOT / "Analysis" / "tables" / "model_sweep"
OUTPUT = ROOT / "Analysis" / "organized_plots" / "model_sweep"
EXPERIMENTS = ("EXP_MS_001", "EXP_MS_003", "EXP_MS_004", "EXP_MS_005")


def load_nll(model: str) -> dict[str, float]:
    path = TABLES / model / f"{model}.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return {row["experiment_id"]: float(row["nll"]) for row in rows}


def main() -> None:
    real_nvp = load_nll("real_nvp")
    transformer = load_nll("transformer")
    real_values = [real_nvp[experiment_id] for experiment_id in EXPERIMENTS]
    transformer_values = [transformer[experiment_id] for experiment_id in EXPERIMENTS]

    positions = np.arange(len(EXPERIMENTS))
    width = 0.36
    figure, axis = plt.subplots(figsize=(12, 7))
    real_bars = axis.bar(positions - width / 2, real_values, width, label="RealNVP", color="#3f51b5")
    transformer_bars = axis.bar(
        positions + width / 2,
        transformer_values,
        width,
        label="Transformer",
        color="#c2185b",
    )
    axis.set_title("Model sweep: test NLL comparison")
    axis.set_xlabel("Model-sweep experiment")
    axis.set_ylabel("Test NLL")
    axis.set_xticks(positions, EXPERIMENTS)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()

    for bars in (real_bars, transformer_bars):
        axis.bar_label(bars, labels=[f"{bar.get_height():.4f}" for bar in bars], padding=3, fontsize=9)
    axis.set_ylim(0, max(real_values) * 1.22)
    figure.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT / "model_sweep_nll_comparison.png"
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    print(output_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
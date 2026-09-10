"""Build one browsable overview of every completed training run.

Usage:
    python Analysis/generate_training_overview.py

The script reads results/**/metrics.json and writes a self-contained index
table plus links to each run's existing loss and parameter plots.
"""

from __future__ import annotations

import csv
import html
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "Analysis"
ORGANIZED_PLOTS = OUTPUT / "organized_plots"


def _label(metrics_path: Path) -> str:
    relative = metrics_path.relative_to(RESULTS)
    return " / ".join(relative.parts[:-1])


def _plot_path(metrics_path: Path, filename: str) -> str:
    path = metrics_path.parent / filename
    return Path(os.path.relpath(path, OUTPUT)).as_posix() if path.exists() else ""


def _read_runs() -> list[dict]:
    runs = []
    for metrics_path in sorted(RESULTS.rglob("metrics.json")):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        train = metrics.get("train_losses", [])
        validation = metrics.get("validation_losses", [])
        test = metrics.get("test_losses", [])
        epochs = metrics.get("evaluation_epochs", [])
        if not train:
            continue
        config = metrics.get("config", {})
        runs.append(
            {
                "label": _label(metrics_path),
                "metrics_path": metrics_path,
                "model": metrics.get("model", config.get("model", "")),
                "config": config,
                "metrics": metrics,
                "train": train,
                "validation": validation,
                "test": test,
                "epochs": epochs,
                "loss_plot": _plot_path(metrics_path, "loss_curves.png"),
                "parameter_plot": _plot_path(metrics_path, "parameters.png"),
            }
        )
    return runs


def _write_parameters_csv(runs: list[dict]) -> None:
    fields = [
        "run", "model", "best_epoch", "best_validation_loss", "final_train_loss",
        "final_validation_loss", "final_test_loss", "total_trainable_parameters",
        "learning_rate", "weight_decay", "epochs", "batch_size", "seed",
        "num_layers", "hidden_features", "channels", "num_blocks",
        "layers_per_block", "head_dim", "expansion", "conditional",
    ]
    with (OUTPUT / "all_training_parameters.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            metrics = run["metrics"]
            config = run["config"]
            row = {field: metrics.get(field, config.get(field, "")) for field in fields}
            row["run"] = run["label"]
            row["model"] = run["model"]
            writer.writerow(row)


def _write_loss_overview(runs: list[dict]) -> None:
    columns = 4
    rows = (len(runs) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(18, max(4, rows * 3.1)), squeeze=False)
    for axis, run in zip(axes.flat, runs):
        axis.plot(range(1, len(run["train"]) + 1), run["train"], label="train", linewidth=1)
        if run["validation"]:
            axis.plot(run["epochs"][: len(run["validation"])], run["validation"], label="validation", linewidth=1)
        if run["test"]:
            axis.plot(run["epochs"][: len(run["test"])], run["test"], label="test", linewidth=1)
        axis.set_title(run["label"].replace(" / ", "\n"), fontsize=7)
        axis.grid(alpha=0.2)
        axis.tick_params(labelsize=7)
    for axis in axes.flat[len(runs) :]:
        axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3)
    figure.suptitle("Loss curves for all completed training runs", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(OUTPUT / "all_training_loss_curves.png", dpi=160)
    plt.close(figure)


def _organized_plot_gallery() -> str:
    groups: dict[str, list[Path]] = {}
    for plot_path in sorted(ORGANIZED_PLOTS.rglob("*.png")):
        relative = plot_path.relative_to(ORGANIZED_PLOTS)
        group = relative.parts[0] if len(relative.parts) > 1 else "other"
        groups.setdefault(group, []).append(plot_path)

    sections = []
    for group, plot_paths in groups.items():
        images = []
        for plot_path in plot_paths:
            relative = Path(os.path.relpath(plot_path, OUTPUT)).as_posix()
            title = plot_path.stem.replace("_", " ")
            images.append(
                f'<figure><img src="{html.escape(relative)}" alt="{html.escape(title)}">'
                f"<figcaption>{html.escape(title)}</figcaption></figure>"
            )
        sections.append(
            f"<h3>{html.escape(group.replace('_', ' ').title())}</h3>"
            f'<div class="gallery">{"".join(images)}</div>'
        )
    return "".join(sections)


def _write_html(runs: list[dict]) -> None:
    cards = []
    for run in runs:
        metrics = run["metrics"]
        config_rows = []
        for key, value in run["config"].items():
            config_rows.append(f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>")
        plots = []
        if run["loss_plot"]:
            plots.append(f'<img src="{html.escape(run["loss_plot"])}" alt="Loss curves">')
        if run["parameter_plot"]:
            plots.append(f'<img src="{html.escape(run["parameter_plot"])}" alt="Final parameters">')
        cards.append(
            f"""<details class="run" open>
<summary><strong>{html.escape(run['label'])}</strong> — {html.escape(str(run['model']))}, 
best validation loss: {html.escape(str(metrics.get('best_validation_loss', '')))}</summary>
<div class="plots">{''.join(plots)}</div>
<table><tr><th>Metric</th><th>Value</th></tr>
<tr><td>Best epoch</td><td>{html.escape(str(metrics.get('best_epoch', '')))}</td></tr>
<tr><td>Final train loss</td><td>{html.escape(str(metrics.get('final_train_loss', '')))}</td></tr>
<tr><td>Final validation loss</td><td>{html.escape(str(metrics.get('final_validation_loss', '')))}</td></tr>
<tr><td>Final test loss</td><td>{html.escape(str(metrics.get('final_test_loss', '')))}</td></tr>
<tr><td>Trainable parameters</td><td>{html.escape(str(metrics.get('total_trainable_parameters', '')))}</td></tr>
</table>
<table><tr><th>Final configuration</th><th>Value</th></tr>{''.join(config_rows)}</table>
</details>"""
        )
    gallery = _organized_plot_gallery()
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>All training overview</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 2rem; color: #222; }}
h1 {{ margin-bottom: .25rem; }}
.links {{ margin: 1rem 0 2rem; }}
.run {{ border: 1px solid #ccc; border-radius: 8px; margin: 1rem 0; padding: .8rem; }}
summary {{ cursor: pointer; font-size: 1.05rem; }}
.plots {{ display: flex; flex-wrap: wrap; gap: 1rem; margin: 1rem 0; }}
.plots img {{ max-width: 48%; min-width: 360px; height: auto; border: 1px solid #ddd; }}
.gallery {{ display: flex; flex-wrap: wrap; gap: 1rem; margin: 1rem 0 2rem; }}
.gallery figure {{ margin: 0; width: min(48%, 720px); }}
.gallery img {{ display: block; width: 100%; height: auto; border: 1px solid #ddd; }}
figcaption {{ margin-top: .35rem; font-size: .9rem; color: #555; }}
table {{ border-collapse: collapse; margin: .75rem 0; }}
th, td {{ border: 1px solid #ddd; padding: .3rem .55rem; text-align: left; }}
th {{ background: #f3f3f3; }}
</style></head><body>
<h1>All training runs</h1>
<p>{len(runs)} completed runs discovered from <code>results/**/metrics.json</code>.</p>
<p class="links"><a href="all_training_loss_curves.png">Combined loss curves</a> ·
<a href="all_training_parameters.csv">Parameters and final metrics CSV</a></p>
<h2>Analysis plots</h2>
{gallery or '<p>No organized analysis plots found.</p>'}
{''.join(cards)}
</body></html>"""
    (OUTPUT / "all_training_overview.html").write_text(page, encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    runs = _read_runs()
    if not runs:
        raise SystemExit("No completed metrics.json files with training histories were found.")
    _write_parameters_csv(runs)
    _write_loss_overview(runs)
    _write_html(runs)
    print(f"Wrote overview for {len(runs)} runs to Analysis/all_training_overview.html")


if __name__ == "__main__":
    main()

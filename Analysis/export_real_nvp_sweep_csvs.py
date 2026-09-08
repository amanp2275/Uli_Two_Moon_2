"""Export one scalar result row per RealNVP parameter-sweep experiment."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "results" / "experiment_results.csv"
OUTPUT = ROOT / "Analysis" / "tables" / "parameter_sweep" / "real_nvp"

SWEEPS = {
    "flow_depth": ("num_layers", ("EXP_017", "EXP_019", "EXP_020")),
    "hidden_dimension": ("hidden_features", ("EXP_021", "EXP_022", "EXP_023", "EXP_024")),
    "learning_rate": ("learning_rate", ("EXP_001", "EXP_002", "EXP_003", "EXP_004")),
    "weight_decay": ("weight_decay", ("EXP_009", "EXP_010", "EXP_011", "EXP_012")),
}

def load_registry() -> dict[str, dict[str, str]]:
    with REGISTRY.open(encoding="utf-8", newline="") as handle:
        return {
            row["experiment_id"]: row
            for row in csv.DictReader(handle)
            if row.get("model") == "real_nvp" and row.get("status") == "completed"
        }


def make_rows(sweep: str, parameter: str, experiment_ids: tuple[str, ...], registry: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for experiment_id in experiment_ids:
        row = dict(registry[experiment_id])
        row["sweep"] = sweep
        row["sweep_parameter"] = parameter
        row["sweep_value"] = row.get(parameter, "")
        rows.append(row)
    return rows


def write_csv(sweep: str, rows: list[dict]) -> Path:
    path = OUTPUT / f"realnvp_{sweep}_loss_curves.csv"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["sweep", "sweep_parameter", "sweep_value"] + [
            field for field in rows[0] if field not in {"sweep", "sweep_parameter", "sweep_value"}
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_combined_csv(rows: list[dict]) -> Path:
    path = OUTPUT / "real_nvp.csv"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sweep", "sweep_parameter", "sweep_value"] + [
        field for field in rows[0] if field not in {"sweep", "sweep_parameter", "sweep_value"}
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    registry = load_registry()
    combined_rows = []
    for sweep, (parameter, experiment_ids) in SWEEPS.items():
        rows = make_rows(sweep, parameter, experiment_ids, registry)
        combined_rows.extend(rows)
        path = write_csv(sweep, rows)
        print(f"{path.relative_to(ROOT)}: {len(rows)} rows")
    path = write_combined_csv(combined_rows)
    print(f"{path.relative_to(ROOT)}: {len(combined_rows)} rows")


if __name__ == "__main__":
    main()
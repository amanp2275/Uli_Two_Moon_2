"""Export one scalar result row per Transformer parameter-sweep experiment."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "results" / "experiment_results.csv"
OUTPUT = ROOT / "Analysis" / "tables" / "parameter_sweep" / "transformer"

SWEEPS = {
    "learning_rate": ("learning_rate", ("EXP_005", "EXP_006", "EXP_007", "EXP_008")),
    "weight_decay": ("weight_decay", ("EXP_013", "EXP_014", "EXP_015", "EXP_016")),
    "block_count": ("num_blocks", ("EXP_025", "EXP_026")),
    "head_dimension": ("head_dim", ("EXP_029", "EXP_030")),
    "expansion": ("expansion", ("EXP_032", "EXP_033", "EXP_034")),
}


def load_registry() -> dict[str, dict[str, str]]:
    with REGISTRY.open(encoding="utf-8", newline="") as handle:
        return {
            row["experiment_id"]: row
            for row in csv.DictReader(handle)
            if row.get("model") == "transformer" and row.get("status") == "completed"
        }


def make_rows(
    sweep: str,
    parameter: str,
    experiment_ids: tuple[str, ...],
    registry: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    rows = []
    for experiment_id in experiment_ids:
        row = dict(registry[experiment_id])
        row["sweep"] = sweep
        row["sweep_parameter"] = parameter
        row["sweep_value"] = row.get(parameter, "")
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["sweep", "sweep_parameter", "sweep_value"] + [
        field for field in rows[0] if field not in {"sweep", "sweep_parameter", "sweep_value"}
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    registry = load_registry()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    combined_rows = []
    for sweep, (parameter, experiment_ids) in SWEEPS.items():
        rows = make_rows(sweep, parameter, experiment_ids, registry)
        combined_rows.extend(rows)
        path = OUTPUT / f"transformer_{sweep}.csv"
        write_csv(path, rows)
        print(f"{path.relative_to(ROOT)}: {len(rows)} rows")

    path = OUTPUT / "transformer.csv"
    write_csv(path, combined_rows)
    print(f"{path.relative_to(ROOT)}: {len(combined_rows)} rows")


if __name__ == "__main__":
    main()
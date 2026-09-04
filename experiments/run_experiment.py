"""Run manually selected experiments from experiments/experiment_queue.csv."""

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import RealNVPConfig, TransformerConfig
from models import RealNVP, TransformerFlow
from training import train_model
from training.storage import config_dict
from experiments.experiment_registry import append_result, completed_ids, next_run_number

ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = Path(__file__).with_name("experiment_queue.csv")
RESULTS_ROOT = ROOT / "results"
RESULTS_PATH = RESULTS_ROOT / "experiment_results.csv"

QUEUE_FIELDS = [
    "experiment_id", "experiment_type", "model", "learning_rate", "batch_size", "epochs", "seed",
    "conditional", "weight_decay", "early_stopping_patience", "evaluation_frequency", "points_per_batch",
    "dataset_num_batches", "noise", "plot_batches", "device", "dataset_path", "num_layers",
    "hidden_features", "in_channels", "channels", "num_blocks", "layers_per_block", "head_dim",
    "expansion", "nvp", "notes", "run",
]
INT_FIELDS = {
    "batch_size", "epochs", "seed", "early_stopping_patience", "evaluation_frequency", "points_per_batch",
    "dataset_num_batches", "plot_batches", "num_layers", "hidden_features", "in_channels", "channels",
    "num_blocks", "layers_per_block", "head_dim", "expansion",
}
FLOAT_FIELDS = {"learning_rate", "weight_decay", "noise"}
BOOL_FIELDS = {"conditional", "nvp"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _notify_telegram(message: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    payload = json.dumps({"chat_id": chat_id, "text": message}).encode("utf-8")
    request = Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=10):
            pass
    except Exception as error:
        print(f"Telegram notification failed: {error}", file=sys.stderr)


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _sha256(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_queue() -> list[dict[str, str]]:
    with QUEUE_PATH.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("experiment_queue.csv must have a header row")
        missing = {"experiment_id", "experiment_type", "model", "run"} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"experiment_queue.csv is missing required columns: {', '.join(sorted(missing))}")
        return [dict(row) for row in reader]


def _parse_value(name: str, value: str):
    value = value.strip()
    if not value:
        return None
    if name in BOOL_FIELDS:
        if value.lower() in {"yes", "true", "1"}:
            return True
        if value.lower() in {"no", "false", "0"}:
            return False
        raise ValueError(f"{name} must be YES/NO or true/false, got {value!r}")
    if name in INT_FIELDS:
        return int(value)
    if name in FLOAT_FIELDS:
        return float(value)
    return value


def _build_config(row: dict[str, str], experiment_dir: Path):
    model = row.get("model", "").strip().lower()
    config_type = RealNVPConfig if model == "real_nvp" else TransformerConfig if model == "transformer" else None
    if config_type is None:
        raise ValueError(f"unsupported model: {model!r}")
    allowed = {field.name for field in fields(config_type)}
    values = {}
    for name, raw in row.items():
        if name in allowed and raw is not None:
            parsed = _parse_value(name, raw)
            if parsed is not None:
                values[name] = parsed
    values["output_dir"] = experiment_dir
    return config_type(**values), model


def _make_model(config, model_name):
    if model_name == "real_nvp":
        return RealNVP(config.num_layers, config.hidden_features, config.conditional)
    return TransformerFlow(
        config.in_channels, config.points_per_batch, config.channels, config.num_blocks,
        config.layers_per_block, config.head_dim, config.expansion, config.nvp,
        2 if config.conditional else 0,
    )


def _base_row(row: dict[str, str], run_id: str) -> dict:
    return {
        **{field: row.get(field, "") for field in QUEUE_FIELDS},
        "experiment_id": row.get("experiment_id", "").strip(),
        "experiment_type": row.get("experiment_type", "").strip(),
        "model": row.get("model", "").strip().lower(),
        "run_id": run_id, "status": "failed", "started_at": "", "finished_at": "",
        "training_time_seconds": "", "epochs_completed": "", "best_epoch": "", "best_training_loss": "",
        "best_validation_loss": "", "final_train_loss": "", "final_validation_loss": "",
        "final_test_loss": "", "nll": "", "trainable_parameters": "", "dataset_sha256": "",
        "git_commit": _git_commit(), "config_path": "", "metrics_path": "", "checkpoint_path": "",
        "plots_path": "", "parameters_json": "", "error": "",
    }


def _run(row: dict[str, str], run_number: int) -> bool:
    experiment_id = row.get("experiment_id", "").strip()
    run_id = f"run_{run_number:03d}"
    experiment_type = row.get("experiment_type", "").strip()
    experiment_dir = RESULTS_ROOT / experiment_type / experiment_id / run_id
    result_row = _base_row(row, run_id)
    started = _now()
    start_time = time.perf_counter()
    try:
        if not experiment_id:
            raise ValueError("experiment_id cannot be empty")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", experiment_type):
            raise ValueError(
                "experiment_type must be a non-empty directory-safe label "
                "containing only letters, numbers, underscores, or hyphens"
            )
        experiment_dir.mkdir(parents=True, exist_ok=True)
        config, model_name = _build_config(row, experiment_dir)
        config_path = experiment_dir / "config.json"
        config_path.write_text(json.dumps({"queue_row": row, "resolved_config": config_dict(config)}, indent=2), encoding="utf-8")
        dataset_path = Path(config.dataset_path)
        result = train_model(_make_model(config, model_name), config, model_name)
        elapsed = time.perf_counter() - start_time
        candidates = list((experiment_dir / "raw" / model_name).rglob("metrics.json"))
        run_dir = candidates[0].parent if candidates else experiment_dir / "raw" / model_name
        metrics = dict(result)
        best_epoch = int(metrics.get("best_epoch", 0))
        train_losses = metrics.get("train_losses", [])
        result_row.update({
            "status": "completed", "started_at": started, "finished_at": _now(),
            "training_time_seconds": f"{elapsed:.6f}", "epochs_completed": len(train_losses),
            "best_epoch": best_epoch,
            "best_training_loss": train_losses[best_epoch - 1] if 0 < best_epoch <= len(train_losses) else "",
            "best_validation_loss": metrics.get("best_validation_loss", ""),
            "final_train_loss": metrics.get("final_train_loss", ""),
            "final_validation_loss": metrics.get("final_validation_loss", ""),
            "final_test_loss": metrics.get("final_test_loss", ""), "nll": metrics.get("final_test_loss", ""),
            "trainable_parameters": metrics.get("total_trainable_parameters", ""),
            "dataset_sha256": _sha256(dataset_path), "git_commit": _git_commit(),
            "config_path": str(config_path), "metrics_path": str(run_dir / "metrics.json"),
            "checkpoint_path": str(run_dir / "best_model.pt"), "plots_path": str(run_dir),
            "parameters_json": json.dumps(config_dict(config), sort_keys=True),
        })
        append_result(RESULTS_PATH, result_row)
        print(f"{experiment_id} completed — results saved to {run_dir}")
        return True
    except Exception as exc:
        result_row.update({
            "started_at": started, "finished_at": _now(),
            "training_time_seconds": f"{time.perf_counter() - start_time:.6f}",
            "error": f"{type(exc).__name__}: {exc}",
        })
        append_result(RESULTS_PATH, result_row)
        print(f"{experiment_id or '<blank id>'} failed — {exc}", file=sys.stderr)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", dest="experiment_id", help="run one queue row by experiment ID")
    parser.add_argument("--force", action="store_true", help="rerun completed IDs and preserve the previous record")
    args = parser.parse_args()
    rows = _read_queue()
    selected = [row for row in rows if args.experiment_id is None or row.get("experiment_id", "").strip() == args.experiment_id]
    if args.experiment_id is not None and not selected:
        raise SystemExit(f"experiment not found in queue: {args.experiment_id}")
    completed = completed_ids(RESULTS_PATH)
    attempted = successful = failed = skipped = 0
    for row in selected:
        experiment_id = row.get("experiment_id", "").strip()
        if args.experiment_id is None and row.get("run", "").strip().upper() != "YES":
            skipped += 1
            continue
        if experiment_id in completed and not args.force:
            print(f"{experiment_id} already completed — skipping.")
            skipped += 1
            continue
        attempted += 1
        if _run(row, next_run_number(RESULTS_PATH, experiment_id)):
            successful += 1
        else:
            failed += 1

    status = "completed successfully" if failed == 0 else "finished with failures"
    _notify_telegram(
        f"Experiment queue {status}.\n"
        f"Trainings attempted: {attempted}\n"
        f"Successful: {successful}\n"
        f"Failed: {failed}\n"
        f"Skipped: {skipped}"
    )


if __name__ == "__main__":
    main()

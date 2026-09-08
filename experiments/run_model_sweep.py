"""Run paired model comparisons listed in model_sweep_queue.csv."""

import argparse
import csv
import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import UpdatedRealNVPConfig, UpdatedTransformerConfig
from models import RealNVP, TransformerFlow
from training import train_model
from training.storage import publish_model_comparison, raw_run_path


ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = Path(__file__).with_name("model_sweep_queue.csv")
RESULTS_ROOT = ROOT / "results" / "model_comparison"
RESULTS_PATH = RESULTS_ROOT / "model_sweep_results.csv"


def notify_telegram(message: str) -> None:
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


def _bool(value: str) -> bool:
    return value.strip().upper() in {"YES", "TRUE", "1"}


def _int(value: str) -> int:
    return int(value.strip())


def _float(value: str) -> float:
    return float(value.strip())


def _read_queue() -> list[dict[str, str]]:
    with QUEUE_PATH.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("model_sweep_queue.csv is empty")
    return rows


def _completed_ids() -> set[str]:
    if not RESULTS_PATH.exists():
        return set()
    with RESULTS_PATH.open(newline="", encoding="utf-8") as handle:
        return {row["experiment_id"] for row in csv.DictReader(handle) if row.get("status") == "completed"}


def _append_result(row: dict[str, object]) -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment_id", "seed", "status", "real_nvp_test_nll", "transformer_test_nll",
        "real_nvp_validation_loss", "transformer_validation_loss", "real_nvp_parameters",
        "transformer_parameters", "output_dir", "finished_at", "error",
    ]
    exists = RESULTS_PATH.exists()
    with RESULTS_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def _run(row: dict[str, str]) -> None:
    experiment_id = row["experiment_id"].strip()
    seed = _int(row["seed"])
    output_dir = RESULTS_ROOT / experiment_id
    shared = {
        "seed": seed,
        "conditional": _bool(row["conditional"]),
        "batch_size": _int(row["batch_size"]),
        "epochs": _int(row["epochs"]),
        "early_stopping_patience": _int(row["early_stopping_patience"]),
        "evaluation_frequency": _int(row["evaluation_frequency"]),
        "points_per_batch": _int(row["points_per_batch"]),
        "dataset_num_batches": _int(row["dataset_num_batches"]),
        "noise": _float(row["noise"]),
        "plot_batches": _int(row["plot_batches"]),
        "output_dir": output_dir,
    }
    if row.get("device", "").strip():
        shared["device"] = row["device"].strip()
    if row.get("dataset_path", "").strip():
        dataset_path = Path(row["dataset_path"].strip())
        shared["dataset_path"] = dataset_path if dataset_path.is_absolute() else ROOT / dataset_path

    real_config = replace(
        UpdatedRealNVPConfig(),
        **shared,
        learning_rate=_float(row["real_nvp_learning_rate"]),
        weight_decay=_float(row["real_nvp_weight_decay"]),
        num_layers=_int(row["real_nvp_num_layers"]),
        hidden_features=_int(row["real_nvp_hidden_features"]),
    )
    transformer_config = replace(
        UpdatedTransformerConfig(),
        **shared,
        learning_rate=_float(row["transformer_learning_rate"]),
        weight_decay=_float(row["transformer_weight_decay"]),
        in_channels=_int(row["transformer_in_channels"]),
        channels=_int(row["transformer_channels"]),
        num_blocks=_int(row["transformer_num_blocks"]),
        layers_per_block=_int(row["transformer_layers_per_block"]),
        head_dim=_int(row["transformer_head_dim"]),
        expansion=_int(row["transformer_expansion"]),
        nvp=_bool(row["transformer_nvp"]),
    )

    notify_telegram(f"Model sweep {experiment_id} started for seed {seed}.")
    try:
        real_result = train_model(
            RealNVP(real_config.num_layers, real_config.hidden_features, real_config.conditional),
            real_config,
            "real_nvp",
        )
        transformer_result = train_model(
            TransformerFlow(
                transformer_config.in_channels,
                transformer_config.points_per_batch,
                transformer_config.channels,
                transformer_config.num_blocks,
                transformer_config.layers_per_block,
                transformer_config.head_dim,
                transformer_config.expansion,
                transformer_config.nvp,
                2 if transformer_config.conditional else 0,
            ),
            transformer_config,
            "transformer",
        )
        publish_model_comparison(output_dir, {
            "real_nvp": raw_run_path(output_dir, "real_nvp", real_config),
            "transformer": raw_run_path(output_dir, "transformer", transformer_config),
        })
        result = {
            "experiment_id": experiment_id,
            "seed": seed,
            "status": "completed",
            "real_nvp_test_nll": real_result["final_test_loss"],
            "transformer_test_nll": transformer_result["final_test_loss"],
            "real_nvp_validation_loss": real_result["best_validation_loss"],
            "transformer_validation_loss": transformer_result["best_validation_loss"],
            "real_nvp_parameters": real_result["total_trainable_parameters"],
            "transformer_parameters": transformer_result["total_trainable_parameters"],
            "output_dir": output_dir,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        _append_result(result)
        notify_telegram(
            f"Model sweep {experiment_id} completed.\n"
            f"RealNVP NLL: {real_result['final_test_loss']:.6f}\n"
            f"Transformer NLL: {transformer_result['final_test_loss']:.6f}"
        )
    except Exception as error:
        _append_result({
            "experiment_id": experiment_id,
            "seed": seed,
            "status": "failed",
            "output_dir": output_dir,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "error": str(error),
        })
        notify_telegram(f"Model sweep {experiment_id} failed: {error}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", dest="experiment_id", help="run one queue row by experiment ID")
    parser.add_argument("--force", action="store_true", help="rerun completed IDs")
    args = parser.parse_args()
    rows = _read_queue()
    if args.experiment_id:
        rows = [row for row in rows if row["experiment_id"].strip() == args.experiment_id]
        if not rows:
            raise SystemExit(f"experiment not found in queue: {args.experiment_id}")
    completed = _completed_ids()
    selected = [row for row in rows if row.get("run", "").strip().upper() == "YES"]
    if args.experiment_id:
        selected = rows
    for row in selected:
        experiment_id = row["experiment_id"].strip()
        if experiment_id in completed and not args.force:
            print(f"{experiment_id} already completed — skipping.")
            continue
        _run(row)


if __name__ == "__main__":
    main()

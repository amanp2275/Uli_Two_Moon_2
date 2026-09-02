"""Run one explicitly configured experiment and register its result."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import RealNVPConfig, TransformerConfig
from models import RealNVP, TransformerFlow
from training import train_model
from training.storage import config_dict
from experiments.experiment_registry import append_result, completed_ids


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).with_name("experiment_config.json")
RESULTS_ROOT = ROOT / "results"
REGISTRY_PATH = RESULTS_ROOT / "experiments.csv"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
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


def _load_experiments() -> list[dict]:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    experiments = payload.get("experiments")
    if not isinstance(experiments, list):
        raise ValueError("experiment_config.json must contain an 'experiments' list")
    return experiments


def _config_fields(config_type) -> set[str]:
    return {field.name for field in fields(config_type)}


def _build_config(spec: dict, experiment_dir: Path):
    model = spec["model"].lower()
    config_type = RealNVPConfig if model == "real_nvp" else TransformerConfig if model == "transformer" else None
    if config_type is None:
        raise ValueError(f"unsupported model: {model!r}")
    values = dict(spec.get("parameters", {}))
    if "seed" in spec:
        values["seed"] = spec["seed"]
    unknown = sorted(set(values) - _config_fields(config_type) - {"output_dir"})
    if unknown:
        raise ValueError(f"unknown {model} parameter(s): {', '.join(unknown)}")
    values["output_dir"] = experiment_dir
    return config_type(**values), model


def _model(config, model_name):
    if model_name == "real_nvp":
        return RealNVP(config.num_layers, config.hidden_features, config.conditional)
    return TransformerFlow(
        config.in_channels, config.points_per_batch, config.channels, config.num_blocks,
        config.layers_per_block, config.head_dim, config.expansion, config.nvp,
        2 if config.conditional else 0,
    )


def _run(spec: dict) -> None:
    experiment_id = str(spec.get("experiment_id", "")).strip()
    experiment_type = spec.get("experiment_type")
    if not experiment_id or experiment_type not in {"parameter_sweep", "model_comparison"}:
        raise ValueError("each experiment needs experiment_id and experiment_type (parameter_sweep/model_comparison)")
    experiment_dir = RESULTS_ROOT / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    config, model_name = _build_config(spec, experiment_dir)
    config_path = experiment_dir / "config.json"
    config_path.write_text(json.dumps({"experiment": spec, "resolved_config": config_dict(config)}, indent=2), encoding="utf-8")
    dataset_path = Path(config.dataset_path)
    started = _now()
    start_time = time.perf_counter()
    try:
        result = train_model(_model(config, model_name), config, model_name)
        elapsed = time.perf_counter() - start_time
        metrics_path = experiment_dir / "raw" / model_name
        candidates = list(metrics_path.rglob("metrics.json"))
        run_dir = candidates[0].parent if candidates else metrics_path
        metrics = dict(result)
        best_epoch = int(metrics.get("best_epoch", 0))
        train_losses = metrics.get("train_losses", [])
        best_training_loss = train_losses[best_epoch - 1] if 0 < best_epoch <= len(train_losses) else ""
        resolved = config_dict(config)
        model_keys = {"num_layers", "hidden_features", "in_channels", "channels", "num_blocks", "layers_per_block", "head_dim", "expansion", "nvp"}
        model_parameters = {key: resolved[key] for key in resolved if key in model_keys}
        training_parameters = {key: resolved[key] for key in resolved if key not in model_keys}
        row = {
            "experiment_id": experiment_id, "experiment_type": experiment_type, "model": model_name,
            "seed": config.seed, "status": "completed", "started_at": started, "finished_at": _now(),
            "training_time_seconds": f"{elapsed:.6f}", "epochs_requested": config.epochs,
            "epochs_completed": len(train_losses), "best_epoch": best_epoch, "best_training_loss": best_training_loss,
            "best_validation_loss": metrics.get("best_validation_loss", ""), "final_train_loss": metrics.get("final_train_loss", ""),
            "final_validation_loss": metrics.get("final_validation_loss", ""), "final_test_loss": metrics.get("final_test_loss", ""),
            "nll": metrics.get("final_test_loss", ""), "trainable_parameters": metrics.get("total_trainable_parameters", ""),
            "dataset_path": str(dataset_path), "dataset_sha256": _sha256(dataset_path), "git_commit": _git_commit(),
            "config_path": str(config_path), "metrics_path": str(run_dir / "metrics.json"),
            "checkpoint_path": str(run_dir / "best_model.pt"), "plots_path": str(run_dir),
            "model_parameters_json": json.dumps(model_parameters, sort_keys=True),
            "training_parameters_json": json.dumps(training_parameters, sort_keys=True),
        }
        append_result(REGISTRY_PATH, row)
        print(f"Completed {experiment_id}; results saved to {run_dir}")
    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        append_result(REGISTRY_PATH, {
            "experiment_id": experiment_id, "experiment_type": experiment_type, "model": model_name,
            "seed": config.seed, "status": "failed", "started_at": started, "finished_at": _now(),
            "training_time_seconds": f"{elapsed:.6f}", "epochs_requested": config.epochs,
            "dataset_path": str(dataset_path), "dataset_sha256": _sha256(dataset_path), "git_commit": _git_commit(),
            "config_path": str(config_path), "error": f"{type(exc).__name__}: {exc}",
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--id", dest="experiment_id")
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()
    specs = _load_experiments()
    selected = specs if args.all else [spec for spec in specs if spec.get("experiment_id") == args.experiment_id]
    if not selected:
        raise SystemExit(f"experiment not found: {args.experiment_id}")
    done = completed_ids(REGISTRY_PATH)
    for spec in selected:
        experiment_id = spec.get("experiment_id")
        if experiment_id in done:
            print(f"{experiment_id} has already been completed. Skipping it.")
            continue
        _run(spec)


if __name__ == "__main__":
    main()

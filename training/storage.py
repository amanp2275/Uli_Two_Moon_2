"""Deterministic paths and filesystem operations for training results."""

import json
import math
import re
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path


NLL_MATCH_TOLERANCE = 0.05


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def config_dict(config) -> dict:
    """Return the complete config as JSON-safe values."""
    values = asdict(config) if is_dataclass(config) else dict(vars(config))
    return _json_value(values)


def _number(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"signature values must be finite, got {value!r}")
    mantissa, exponent = format(number, ".12e").split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exponent = str(int(exponent))
    return f"{mantissa}e{exponent}"


def _safe(value) -> str:
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value)
    return value.strip(".-") or "value"


def parameter_signature(config, model_name: str) -> str:
    """Build the stable model-specific signature used by raw run folders."""
    fields = [
        ("lr", _number(config.learning_rate)),
        ("ep", _number(config.epochs)),
        ("bs", _number(config.batch_size)),
        ("wd", _number(config.weight_decay)),
        ("cond", _number(config.conditional)),
        ("seed", _number(config.seed)),
    ]
    if model_name == "real_nvp":
        fields.extend((
            ("layers", _number(config.num_layers)),
            ("h", _number(config.hidden_features)),
        ))
    elif model_name == "transformer":
        fields.extend((
            ("ch", _number(config.channels)),
            ("blocks", _number(config.num_blocks)),
            ("lpb", _number(config.layers_per_block)),
        ))
    else:
        raise ValueError(f"unsupported model name: {model_name}")
    return "params__" + "_".join(f"{key}{_safe(value)}" for key, value in fields)


def shared_parameter_signature(config) -> str:
    """Build the Experiment 1 signature from protocol fields only."""
    fields = [
        ("lr", _number(config.learning_rate)),
        ("ep", _number(config.epochs)),
        ("bs", _number(config.batch_size)),
        ("wd", _number(config.weight_decay)),
        ("cond", _number(config.conditional)),
        ("seed", _number(config.seed)),
    ]
    return "params__" + "_".join(f"{key}{_safe(value)}" for key, value in fields)


def raw_run_path(results_root, model_name: str, config) -> Path:
    return Path(results_root) / "raw" / model_name / parameter_signature(config, model_name)


def prepare_run_directory(path: Path) -> Path:
    """Clear artifacts from a repeated run while retaining the run directory."""
    path.mkdir(parents=True, exist_ok=True)
    names = {
        "config.json", "metrics.json", "best_model.pt", "samples_final.png",
        "loss_curves.png", "parameters.png", "final_summary.png",
    }
    for file_path in path.iterdir():
        if file_path.name in names or (file_path.name.startswith("epoch_") and file_path.suffix == ".png"):
            if file_path.is_file() or file_path.is_symlink():
                file_path.unlink()
    return path


def _replace_directory(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)


def publish_same_params(results_root, config, runs: dict[str, Path]) -> Path:
    destination = Path(results_root) / "01_same_params_diff_results" / shared_parameter_signature(config)
    for model_name, source in runs.items():
        _replace_directory(Path(source), destination / model_name)
    return destination


def nll_band(final_test_loss: float, width: float = NLL_MATCH_TOLERANCE) -> str:
    """Round measured Test NLL for display/grouping; never predict it pre-training."""
    if not math.isfinite(float(final_test_loss)):
        raise ValueError("final_test_loss must be finite")
    if width <= 0:
        raise ValueError("NLL band width must be positive")
    band = round(float(final_test_loss) / width) * width
    decimals = max(2, len(str(width).split(".")[-1].rstrip("0"))) if "." in str(width) else 0
    return f"{band:.{decimals}f}"


def publish_nll_group(results_root, model_name: str, config, final_test_loss: float) -> Path:
    """Copy a completed raw run into its measured NLL band view.

    Two runs are considered the same result when their final Test NLL values
    differ by at most NLL_MATCH_TOLERANCE (0.05 by default).
    """
    source = raw_run_path(results_root, model_name, config)
    destination = (
        Path(results_root) / "02_diff_params_same_results" / f"nll__{nll_band(final_test_loss)}"
        / f"{model_name}__{parameter_signature(config, model_name)[len('params__'):]}"
    )
    _replace_directory(source, destination)
    return destination


def save_config(config, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_dict(config), indent=2), encoding="utf-8")

"""Real NVP implementation and experiment entry points."""

import importlib.util
from pathlib import Path


_implementation_path = Path(__file__).resolve().parents[1] / "real_nvp.py"
_spec = importlib.util.spec_from_file_location("_real_nvp_implementation", _implementation_path)
if _spec is None or _spec.loader is None:
	raise ImportError(f"could not load Real NVP implementation from {_implementation_path}")
_implementation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_implementation)

RealNVP = _implementation.RealNVP
RealNVPConfig = _implementation.RealNVPConfig
train_two_moons = _implementation.train_two_moons

__all__ = ["RealNVP", "RealNVPConfig", "train_two_moons"]
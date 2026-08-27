from dataclasses import replace
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from real_nvp import RealNVPConfig, train_two_moons


if __name__ == "__main__":
	config = replace(
		RealNVPConfig(),
		conditional=True,
		plot_dir=project_root / "real_nvp" / "plots" / "conditional",
	)
	train_two_moons(config)

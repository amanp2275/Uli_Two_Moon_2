from dataclasses import replace
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from configuration import UNCONDITIONAL_CONFIG
from train import train_two_moons


if __name__ == "__main__":
	config = replace(UNCONDITIONAL_CONFIG, plot_dir=project_root / "transformer_flows" / "plots" / "unconditional")
	train_two_moons(config)

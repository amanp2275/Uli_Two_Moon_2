from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import RealNVPConfig, TransformerConfig
from models import RealNVP, TransformerFlow
from training import train_model
from training.plotting import save_comparison


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    real_config = replace(RealNVPConfig(), output_dir=root / "results" / "tuned_params")
    transformer_config = replace(TransformerConfig(), output_dir=root / "results" / "tuned_params")
    real_result = train_model(RealNVP(real_config.num_layers, real_config.hidden_features, real_config.conditional), real_config, "real_nvp")
    transformer_result = train_model(TransformerFlow(transformer_config.in_channels, transformer_config.points_per_batch, transformer_config.channels, transformer_config.num_blocks, transformer_config.layers_per_block, transformer_config.head_dim, transformer_config.expansion, transformer_config.nvp, 2 if transformer_config.conditional else 0), transformer_config, "transformer")
    save_comparison({"comparison": "tuned_params", "real_nvp": real_result, "transformer": transformer_result}, root / "results" / "tuned_params" / "comparison.json")


if __name__ == "__main__":
    main()

from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import RealNVPConfig, TransformerConfig
from models import RealNVP, TransformerFlow
from training import train_model
from training.storage import publish_nll_group


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    results_root = root / "results"
    real_config = replace(RealNVPConfig(), output_dir=results_root)
    transformer_config = replace(TransformerConfig(), output_dir=results_root)
    real_result = train_model(RealNVP(real_config.num_layers, real_config.hidden_features, real_config.conditional), real_config, "real_nvp")
    transformer_result = train_model(TransformerFlow(transformer_config.in_channels, transformer_config.points_per_batch, transformer_config.channels, transformer_config.num_blocks, transformer_config.layers_per_block, transformer_config.head_dim, transformer_config.expansion, transformer_config.nvp, 2 if transformer_config.conditional else 0), transformer_config, "transformer")
    publish_nll_group(results_root, "real_nvp", real_config, real_result["final_test_loss"])
    publish_nll_group(results_root, "transformer", transformer_config, transformer_result["final_test_loss"])


if __name__ == "__main__":
    main()

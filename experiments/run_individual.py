import argparse
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import RealNVPConfig, TransformerConfig
from models import RealNVP, TransformerFlow
from training import train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("real_nvp", "transformer"), required=True)
    parser.add_argument("--unconditional", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.model == "real_nvp":
        config = replace(RealNVPConfig(output_dir=root / "results" / "individual"), conditional=not args.unconditional)
        model = RealNVP(config.num_layers, config.hidden_features, config.conditional)
    else:
        config = replace(TransformerConfig(output_dir=root / "results" / "individual"), conditional=not args.unconditional)
        model = TransformerFlow(config.in_channels, config.points_per_batch, config.channels, config.num_blocks, config.layers_per_block, config.head_dim, config.expansion, config.nvp, 2 if config.conditional else 0)
    train_model(model, config, args.model)


if __name__ == "__main__":
    main()

from dataclasses import replace
import argparse
import json
import os
from pathlib import Path
import sys
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import UpdatedRealNVPConfig, UpdatedTransformerConfig
from models import RealNVP, TransformerFlow
from training import train_model
from training.storage import publish_model_comparison, raw_run_path


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


def main() -> None:
    try:
        parser = argparse.ArgumentParser(description="Run the finalized model sweep for one random seed.")
        parser.add_argument("--seed", type=int, default=7, help="random seed used by both finalized models")
        args = parser.parse_args()
        root = Path(__file__).resolve().parents[1]
        results_root = root / "results"
        comparison_root = results_root / "model_comparison" / f"manual_seed_{args.seed}"
        real_config = replace(UpdatedRealNVPConfig(), seed=args.seed, output_dir=comparison_root)
        transformer_config = replace(UpdatedTransformerConfig(), seed=args.seed, output_dir=comparison_root)
        notify_telegram(f"Model comparison started for seed {args.seed}: RealNVP EXP_024 and Transformer EXP_032.")
        real_result = train_model(RealNVP(real_config.num_layers, real_config.hidden_features, real_config.conditional), real_config, "real_nvp")
        transformer_result = train_model(TransformerFlow(transformer_config.in_channels, transformer_config.points_per_batch, transformer_config.channels, transformer_config.num_blocks, transformer_config.layers_per_block, transformer_config.head_dim, transformer_config.expansion, transformer_config.nvp, 2 if transformer_config.conditional else 0), transformer_config, "transformer")
        publish_model_comparison(comparison_root, {
            "real_nvp": raw_run_path(comparison_root, "real_nvp", real_config),
            "transformer": raw_run_path(comparison_root, "transformer", transformer_config),
        })
    except Exception as error:
        notify_telegram(f"Model comparison failed for seed {args.seed}: {error}")
        raise
    else:
        notify_telegram(
            f"Model comparison completed successfully for seed {args.seed}.\n"
            f"RealNVP test NLL: {real_result['final_test_loss']:.6f}\n"
            f"Transformer test NLL: {transformer_result['final_test_loss']:.6f}"
        )


if __name__ == "__main__":
    main()

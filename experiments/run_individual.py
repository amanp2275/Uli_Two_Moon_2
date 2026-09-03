import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import RealNVPConfig, TransformerConfig
from models import RealNVP, TransformerFlow
from training import train_model


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("real_nvp", "transformer"), required=True)
    parser.add_argument("--unconditional", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.model == "real_nvp":
        config = replace(RealNVPConfig(output_dir=root / "results"), conditional=not args.unconditional)
        model = RealNVP(config.num_layers, config.hidden_features, config.conditional)
    else:
        config = replace(TransformerConfig(output_dir=root / "results"), conditional=not args.unconditional)
        model = TransformerFlow(config.in_channels, config.points_per_batch, config.channels, config.num_blocks, config.layers_per_block, config.head_dim, config.expansion, config.nvp, 2 if config.conditional else 0)
    try:
        train_model(model, config, args.model)
    except Exception as error:
        notify_telegram(f"❌ {args.model} experiment failed: {error}")
        raise
    else:
        notify_telegram(f"✅ {args.model} experiment completed successfully.")


if __name__ == "__main__":
    main()

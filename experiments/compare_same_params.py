from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import BaseTrainingConfig, RealNVPConfig, TransformerConfig
from models import RealNVP, TransformerFlow
from training import train_model
from training.storage import publish_same_params, raw_run_path


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
        root = Path(__file__).resolve().parents[1]
        results_root = root / "results" / "model_comparison"
        base = BaseTrainingConfig(output_dir=results_root, epochs=300, learning_rate=5e-4)
        real_config = replace(RealNVPConfig(), **base.__dict__)
        transformer_config = replace(TransformerConfig(), **base.__dict__)
        notify_telegram("Same-parameter model comparison started: RealNVP and Transformer.")
        real_result = train_model(RealNVP(real_config.num_layers, real_config.hidden_features, real_config.conditional), real_config, "real_nvp")
        transformer_result = train_model(TransformerFlow(transformer_config.in_channels, transformer_config.points_per_batch, transformer_config.channels, transformer_config.num_blocks, transformer_config.layers_per_block, transformer_config.head_dim, transformer_config.expansion, transformer_config.nvp, 2 if transformer_config.conditional else 0), transformer_config, "transformer")
        publish_same_params(results_root, base, {
            "real_nvp": raw_run_path(results_root, "real_nvp", real_config),
            "transformer": raw_run_path(results_root, "transformer", transformer_config),
        })
    except Exception as error:
        notify_telegram(f"Same-parameter model comparison failed: {error}")
        raise
    else:
        notify_telegram(
            "Same-parameter model comparison completed successfully.\n"
            f"RealNVP test loss: {real_result['final_test_loss']:.6f}\n"
            f"Transformer test loss: {transformer_result['final_test_loss']:.6f}"
        )


if __name__ == "__main__":
    main()

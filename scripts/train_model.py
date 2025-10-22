"""CLI helper to train the AI trading model."""

from __future__ import annotations

import argparse
import json

from ai_trading_bot.pipeline import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AI trading model.")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file.")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download fresh data instead of using the cached dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train(args.config, force_download=args.force_download)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

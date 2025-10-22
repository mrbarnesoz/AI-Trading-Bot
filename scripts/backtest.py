"""CLI helper to run the backtest using the trained model."""

from __future__ import annotations

import argparse
import json

from ai_trading_bot.pipeline import backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the backtest for the AI trading strategy.")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file.")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download fresh data instead of using the cached dataset.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Probability threshold for entering a long position.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, result = backtest(args.config, force_download=args.force_download, probability_threshold=args.threshold)
    print(json.dumps(result.summary, indent=2))


if __name__ == "__main__":
    main()

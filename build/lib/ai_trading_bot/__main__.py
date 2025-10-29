"""Command-line entry point for the AI Trading Bot."""

from __future__ import annotations

import argparse
import json

from ai_trading_bot.config import load_config
from ai_trading_bot.pipeline import backtest, prepare_dataset, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Trading Bot command-line interface.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_download = subparsers.add_parser("download", help="Download OHLCV market data.")
    parser_download.add_argument("--config", default="config.yaml", help="Path to configuration file.")
    parser_download.add_argument("--force", action="store_true", help="Force download even if cached data exists.")

    parser_train = subparsers.add_parser("train", help="Train the machine learning model.")
    parser_train.add_argument("--config", default="config.yaml", help="Path to configuration file.")
    parser_train.add_argument("--force-download", action="store_true", help="Refresh downloaded data before training.")

    parser_backtest = subparsers.add_parser("backtest", help="Run the backtest using the trained model.")
    parser_backtest.add_argument("--config", default="config.yaml", help="Path to configuration file.")
    parser_backtest.add_argument("--force-download", action="store_true", help="Refresh data before backtesting.")
    parser_backtest.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="(Deprecated) Alias for --long-threshold when only long trades are desired.",
    )
    parser_backtest.add_argument(
        "--long-threshold",
        type=float,
        default=None,
        help="Probability required to enter a long position (defaults to config).",
    )
    parser_backtest.add_argument(
        "--short-threshold",
        type=float,
        default=None,
        help="Probability at or below which to enter a short position (defaults to config).",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "download":
        config = load_config(args.config)
        prepare_dataset(config, force_download=args.force)
    elif args.command == "train":
        metrics = train(args.config, force_download=args.force_download)
        print(json.dumps(metrics, indent=2))
    elif args.command == "backtest":
        long_threshold = args.long_threshold if args.long_threshold is not None else args.threshold
        strategy_output, result, metadata = backtest(
            args.config,
            force_download=args.force_download,
            long_threshold=long_threshold,
            short_threshold=args.short_threshold,
        )
        payload = {
            "metadata": metadata,
            "summary": result.summary,
            "signal_preview": strategy_output.signals.tail(5).to_dict(),
        }
        print(json.dumps(payload, indent=2, default=str))
    else:  # pragma: no cover - argparse prevents this
        parser.print_help()


if __name__ == "__main__":
    main()


"""CLI helper for walk-forward validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ai_trading_bot.backtesting.walk_forward import walk_forward_backtest
from ai_trading_bot.config import load_config
from ai_trading_bot.monitoring import MetricsCollector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run walk-forward validation for the AI trading strategy.")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file.")
    parser.add_argument("--train-days", type=int, required=True, help="Size of the training window in days.")
    parser.add_argument("--test-days", type=int, required=True, help="Size of the test window in days.")
    parser.add_argument(
        "--step-days",
        type=int,
        help="Step size in days between windows (defaults to test window length).",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download fresh data instead of using any cached dataset.",
    )
    parser.add_argument("--output", type=Path, help="Optional path to persist JSON results.")
    parser.add_argument(
        "--metrics-port",
        type=int,
        help="Expose Prometheus metrics for the walk-forward aggregate on this port.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    train_period = pd.Timedelta(days=args.train_days)
    test_period = pd.Timedelta(days=args.test_days)
    step_period = pd.Timedelta(days=args.step_days) if args.step_days else None

    report = walk_forward_backtest(
        config,
        train_period=train_period,
        test_period=test_period,
        step_period=step_period,
        force_download=args.force_download,
    )

    payload = {
        "aggregate": report.aggregate,
        "segments": [
            {
                "index": segment.index,
                "train_start": segment.train_start.isoformat(),
                "train_end": segment.train_end.isoformat(),
                "test_start": segment.test_start.isoformat(),
                "test_end": segment.test_end.isoformat(),
                "summary": segment.summary,
            }
            for segment in report.segments
        ],
    }

    if args.metrics_port is not None:
        collector = MetricsCollector()
        collector.start_server(args.metrics_port)
        collector.record_walk_forward(report.aggregate)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

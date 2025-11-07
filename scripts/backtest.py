"""CLI helper to run the backtest using the trained model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_trading_bot.pipeline import backtest
from ai_trading_bot.monitoring import MetricsCollector


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
        default=None,
        help="(Deprecated) Alias for --long-threshold.",
    )
    parser.add_argument(
        "--long-threshold",
        type=float,
        default=None,
        help="Probability required to enter a long position (defaults to config).",
    )
    parser.add_argument(
        "--short-threshold",
        type=float,
        default=None,
        help="Probability at or below which to enter a short position (defaults to config).",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        help="Expose Prometheus metrics on this port.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    long_threshold = args.long_threshold if args.long_threshold is not None else args.threshold
    short_threshold = args.short_threshold
    strategy_output, result, metadata = backtest(
        args.config,
        force_download=args.force_download,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
    )
    payload = {
        "symbol": metadata.get("symbol"),
        "interval": metadata.get("interval"),
        "rows": metadata.get("rows"),
        "meta_score": getattr(strategy_output, "meta_score", None),
        "meta_metrics": getattr(strategy_output, "meta_metrics", {}),
        "summary": result.summary,
    }
    decisions = getattr(strategy_output, "decisions", None)
    if hasattr(decisions, "iloc") and len(decisions) > 0:  # pragma: no branch - simple duck typing check
        latest = decisions.iloc[-1].to_dict()
        payload["latest_decision"] = latest
    if args.metrics_port is not None:
        collector = MetricsCollector()
        collector.start_server(args.metrics_port)
        collector.record_backtest(result.summary, metadata)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

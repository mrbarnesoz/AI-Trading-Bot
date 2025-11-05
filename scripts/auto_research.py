"""CLI entrypoint for automated strategy research."""

from __future__ import annotations

import argparse
from pathlib import Path

from orchestration.auto_research import run_auto_research


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run automated strategy backtests and produce selection summary.")
    parser.add_argument("--symbols", nargs="*", help="Symbols to include (defaults to all discovered).")
    parser.add_argument("--timeframes", nargs="*", help="Timeframes to include (defaults to all discovered).")
    parser.add_argument("--strategies", nargs="*", help="Strategy names to include (defaults to all discovered).")
    parser.add_argument("--min-trades", type=int, default=100, help="Minimum trades required to mark a candidate as viable.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top candidates to retain per symbol/timeframe.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_auto_research(
        symbols=args.symbols,
        timeframes=args.timeframes,
        strategies=args.strategies,
        min_trades=args.min_trades,
        top_k=args.top_k,
    )
    print(f"Auto research summary written to {Path(result)}")


if __name__ == "__main__":
    main()

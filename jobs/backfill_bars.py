"""Backfill job to build OHLCV bars from normalized trades."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from etl.resample_bars import BarSpec, resample_trades

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill OHLCV bars from normalized trades.")
    parser.add_argument("--trades-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timeframes", nargs="+", default=["1m", "5m", "15m", "1h", "4h", "1d"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = [BarSpec(timeframe=tf, partition=f"timeframe={tf}") for tf in args.timeframes]
    trade_files = sorted(args.trades_root.rglob("*.parquet"))
    for trade_file in trade_files:
        logger.info("Resampling %s", trade_file)
        resample_trades(trade_file, specs, args.output_root)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

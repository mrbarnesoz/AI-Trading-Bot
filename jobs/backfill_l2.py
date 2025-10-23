"""Backfill job to download and normalize BitMEX L2 updates."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

from etl.bitmex_rest_archives import BitmexArchiveDownloader
from etl.normalize_l2 import normalize_l2_updates

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill BitMEX L2 updates.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbols", nargs="+", required=True)
    return parser.parse_args()


def daterange(start: datetime, end: datetime) -> list[str]:
    return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end - start).days + 1)]


def main() -> None:
    args = parse_args()
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    dates = daterange(start, end)
    downloader = BitmexArchiveDownloader(args.base_url, args.output_root)
    downloader.download_daily_archives(dates, args.symbols)
    for date in dates:
        for symbol in args.symbols:
            source = args.output_root / "orderBookL2" / f"dt={date}" / f"{symbol}.gz"
            if source.exists():
                normalize_l2_updates(source, args.output_root / "bronze" / "l2_updates" / f"dt={date}" / f"symbol={symbol}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

"""Download BitMEX trade and order book archives for a date range."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

from etl.bitmex_rest_archives import BitmexArchiveDownloader
from utils.config import load_yaml

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download BitMEX public archives.")
    parser.add_argument("--storage-config", type=Path, default=Path("conf/storage.yaml"))
    parser.add_argument("--sources-config", type=Path, default=Path("conf/sources.yaml"))
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbols", nargs="+", required=True)
    return parser.parse_args()


def daterange(start: datetime, end: datetime):
    days = (end - start).days
    for i in range(days + 1):
        yield (start + timedelta(days=i)).strftime("%Y-%m-%d")


def main() -> None:
    args = parse_args()
    storage_cfg = load_yaml(args.storage_config)
    sources_cfg = load_yaml(args.sources_config)
    raw_root = Path(storage_cfg["raw_root"])
    base_url = sources_cfg.get("archives_base_url", sources_cfg["rest"]["base_url"] + "/data")
    l2_prefix = storage_cfg.get("l2_archive_prefix", "orderBookL2")
    downloader = BitmexArchiveDownloader(base_url=base_url, output_root=raw_root, l2_prefix=l2_prefix)
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    dates = list(daterange(start, end))
    downloader.download_daily_archives(dates, args.symbols, include_l2=storage_cfg.get("auto_ingest_l2", False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

"""Daily incremental orchestration for BitMEX datasets."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from etl.normalize_trades import normalize_trade_file
from etl.normalize_l2 import normalize_l2_updates
from etl.qc_validations import check_trade_minute_completeness
from utils import io as io_utils

logger = logging.getLogger(__name__)


def current_utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def roll_previous_day(raw_root: Path, bronze_root: Path, date_str: str) -> None:
    trade_archives = sorted((raw_root / "trade" / f"dt={date_str}").glob("*.gz"))
    for archive in trade_archives:
        symbol = archive.stem
        normalize_trade_file(archive, bronze_root / "trades" / f"dt={date_str}" / f"symbol={symbol}")

    l2_archives = sorted((raw_root / "orderBookL2" / f"dt={date_str}").glob("*.gz"))
    for archive in l2_archives:
        symbol = archive.stem
        normalize_l2_updates(archive, bronze_root / "l2_updates" / f"dt={date_str}" / f"symbol={symbol}")


def run_daily_incremental(raw_root: Path, bronze_root: Path) -> None:
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    logger.info("Rolling daily data for %s", yesterday)
    roll_previous_day(raw_root, bronze_root, yesterday)
    trade_files = list((bronze_root / "trades" / f"dt={yesterday}").rglob("*.parquet"))
    for trade_file in trade_files:
        df = io_utils.read_trade_file(trade_file)
        completeness = check_trade_minute_completeness(df)
        if not completeness.passed():
            logger.warning("Trade completeness below threshold for %s: %.2f", trade_file, completeness.coverage)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_daily_incremental(Path("raw"), Path("bronze"))

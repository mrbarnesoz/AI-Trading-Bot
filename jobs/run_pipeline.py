"""End-to-end ETL pipeline runner."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from glob import glob

import polars as pl

from etl.bitmex_rest_archives import BitmexArchiveDownloader
from etl.normalize_trades import normalize_trade_file
from etl.normalize_l2 import normalize_l2_updates
from etl.orderbook_snapshots import SnapshotConfig, build_snapshots, snapshots_to_parquet
from etl.resample_bars import BarSpec, resample_trades
from utils.config import load_yaml
from utils.duckdb_utils import ensure_views

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BitMEX ETL pipeline.")
    parser.add_argument("--storage-config", type=Path, default=Path("conf/storage.yaml"))
    parser.add_argument("--cadence-config", type=Path, default=Path("conf/cadence.yaml"))
    parser.add_argument("--sources-config", type=Path, default=Path("conf/sources.yaml"))
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--symbols", nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    storage_cfg = load_yaml(args.storage_config)
    cadence_cfg = load_yaml(args.cadence_config)
    sources_cfg = load_yaml(args.sources_config)
    raw_root = Path(storage_cfg["raw_root"])
    bronze_root = Path(storage_cfg["bronze_root"])
    silver_root = Path(storage_cfg["silver_root"])
    duckdb_path = Path(storage_cfg["duckdb_path"])

    auto_ingest_l2 = bool(storage_cfg.get("auto_ingest_l2", False))
    l2_prefix = storage_cfg.get("l2_archive_prefix", "orderBookL2")
    selected_dates = None

    if args.start and args.end:
        base_url = sources_cfg.get("archives_base_url", sources_cfg["rest"]["base_url"] + "/data")
        downloader = BitmexArchiveDownloader(base_url=base_url, output_root=raw_root, l2_prefix=l2_prefix)
        symbols = args.symbols or sources_cfg.get("default_symbols", ["XBTUSD"])
        start = datetime.fromisoformat(args.start)
        end = datetime.fromisoformat(args.end)
        selected_dates = list(daterange(start, end))
        downloader.download_daily_archives(selected_dates, symbols, include_l2=auto_ingest_l2)

    normalize_trades_step(raw_root, bronze_root, dates=selected_dates)
    normalize_l2_step(raw_root, bronze_root, dates=selected_dates, include=auto_ingest_l2, l2_prefix=l2_prefix)
    snapshot_step(bronze_root, silver_root, cadence_cfg, dates=selected_dates)
    bars_step(bronze_root, silver_root, cadence_cfg, dates=selected_dates)
    build_duckdb_views(duckdb_path, bronze_root, silver_root, storage_cfg)


def normalize_trades_step(raw_root: Path, bronze_root: Path, dates: list[str] | None) -> None:
    if dates:
        trade_archives = []
        for dt in dates:
            trade_archives.extend(sorted((raw_root / "trade" / f"dt={dt}").glob("*.gz")))
    else:
        trade_archives = sorted((raw_root / "trade").rglob("*.gz"))
    for archive in trade_archives:
        dt = archive.parent.name.split("=")[-1]
        symbol = "ALL"
        destination = bronze_root / "trades" / f"dt={dt}" / f"symbol={symbol}"
        normalize_trade_file(archive, destination)


def normalize_l2_step(raw_root: Path, bronze_root: Path, dates: list[str] | None, include: bool, l2_prefix: str) -> None:
    if not include:
        logger.info("Skipping L2 normalization (auto_ingest disabled)")
        return
    if dates:
        book_archives = []
        for dt in dates:
            book_archives.extend(sorted((raw_root / l2_prefix / f"dt={dt}").glob("*.gz")))
    else:
        book_archives = sorted((raw_root / l2_prefix).rglob("*.gz"))
    for archive in book_archives:
        dt = archive.parent.name.split("=")[-1]
        symbol = "ALL"
        destination = bronze_root / "l2_updates" / f"dt={dt}" / f"symbol={symbol}"
        normalize_l2_updates(archive, destination)


def snapshot_step(bronze_root: Path, silver_root: Path, cadence_cfg: dict, dates: list[str] | None) -> None:
    updates_root = bronze_root / "l2_updates"
    output_root = silver_root / "l2_snapshots"
    config = SnapshotConfig(
        cadence_ms=int(cadence_cfg.get("l2_snapshot_interval", "250ms").replace("ms", "")),
        burst_on_best_change=True,
    )
    if dates:
        update_files = []
        for dt in dates:
            update_files.extend(sorted((updates_root / f"dt={dt}").rglob("*.parquet")))
    else:
        update_files = sorted(updates_root.rglob("*.parquet"))
    for update_file in update_files:
        updates = pl.read_parquet(update_file)
        snapshots = build_snapshots(updates, config)
        if snapshots.height == 0:
            continue
        relative = update_file.relative_to(updates_root)
        output_path = output_root / relative
        snapshots_to_parquet(snapshots, output_path)


def bars_step(bronze_root: Path, silver_root: Path, cadence_cfg: dict, dates: list[str] | None) -> None:
    trades_root = bronze_root / "trades"
    output_root = silver_root / "bars"
    base_timeframes = cadence_cfg.get("aggregated_bars", ["1m", "5m", "15m", "1h", "4h", "1d"])
    specs = [BarSpec(timeframe=tf, partition=f"timeframe={tf}") for tf in base_timeframes]
    if dates:
        trade_files = []
        for dt in dates:
            trade_files.extend(sorted((trades_root / f"dt={dt}").rglob("*.parquet")))
    else:
        trade_files = sorted(trades_root.rglob("*.parquet"))
    for trade_file in trade_files:
        resample_trades(trade_file, specs, output_root)


def build_duckdb_views(duckdb_path: Path, bronze_root: Path, silver_root: Path, storage_cfg: dict) -> None:
    trades_path = (bronze_root / "trades").as_posix() + "/**/*.parquet"
    updates_path = (bronze_root / "l2_updates").as_posix() + "/**/*.parquet"
    bars_path = (silver_root / "bars").as_posix() + "/**/*.parquet"
    snapshots_path = (silver_root / "l2_snapshots").as_posix() + "/**/*.parquet"
    statements = []
    if glob(trades_path, recursive=True):
        statements.append(f"CREATE OR REPLACE VIEW bronze_trades AS SELECT * FROM read_parquet('{trades_path}');")
    if glob(updates_path, recursive=True):
        statements.append(f"CREATE OR REPLACE VIEW bronze_l2_updates AS SELECT * FROM read_parquet('{updates_path}');")
    if glob(bars_path, recursive=True):
        statements.append(f"CREATE OR REPLACE VIEW silver_bars AS SELECT * FROM read_parquet('{bars_path}');")
    if glob(snapshots_path, recursive=True):
        statements.append(
            f"CREATE OR REPLACE VIEW silver_l2_snapshots AS SELECT * FROM read_parquet('{snapshots_path}');"
        )
    ensure_views(duckdb_path, statements)


def daterange(start: datetime, end: datetime):
    days = (end - start).days
    for i in range(days + 1):
        yield (start + timedelta(days=i)).strftime("%Y-%m-%d")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

"""Normalize raw BitMEX trade archive rows into the bronze Parquet schema."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from utils import io as io_utils
from utils.time import ensure_utc_ns

logger = logging.getLogger(__name__)

TRADE_SCHEMA = {
    "timestamp": pl.Datetime(time_unit="ns", time_zone="UTC"),
    "symbol": pl.String,
    "side": pl.Categorical,
    "price": pl.Float64,
    "size": pl.Int64,
    "tick_id": pl.UInt64,
}


def normalize_trade_file(source_path: Path, destination_dir: Path) -> Path:
    """Read a BitMEX trade archive (gzipped NDJSON) and emit normalized Parquet."""
    logger.info("Normalizing trades from %s", source_path)
    df = io_utils.read_trade_file(source_path)
    df = df.rename({"timestamp": "ts"})
    df = df.with_columns(
        [
            (
                pl.col("ts")
                .str.replace("D", "T")
                .str.strptime(pl.Datetime(time_unit="ns", time_zone="UTC"), strict=False)
                .alias("ts")
            ),
            pl.col("side").cast(pl.Categorical),
            pl.arange(0, pl.count()).cast(pl.UInt64).alias("tick_id"),
            pl.lit(datetime.now(timezone.utc)).alias("ingest_ts"),
        ]
    )
    df = df.select(
        [
            pl.col("ts"),
            pl.col("symbol"),
            pl.col("side"),
            pl.col("price").cast(pl.Float64),
            pl.col("size").cast(pl.Int64),
            pl.col("tick_id").cast(pl.UInt64),
            pl.col("ingest_ts").cast(pl.Datetime(time_unit="ns", time_zone="UTC")),
        ]
    )
    df = df.sort(["ts", "symbol", "tick_id"]).unique(subset=["ts", "symbol", "tick_id"], maintain_order=True)
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / f"{source_path.stem}.parquet"
    df.write_parquet(target)
    io_utils.write_manifest(
        target,
        rows=df.height,
        extra={"source": str(source_path)},
    )
    logger.info("Wrote normalized trades to %s", target)
    return target

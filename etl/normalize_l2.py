"""Normalize BitMEX order book L2 updates into bronze schema."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from utils import io as io_utils
from utils.time import ensure_utc_ns

logger = logging.getLogger(__name__)


def normalize_l2_updates(source_path: Path, destination_dir: Path) -> Path:
    """Convert raw L2 JSON to normalized Parquet with enforced schema."""
    logger.info("Normalizing order book updates from %s", source_path)
    df = io_utils.read_l2_file(source_path)
    df = df.rename({"timestamp": "ts", "side": "side"})
    df = df.with_columns(
        [
            ensure_utc_ns(pl.col("ts")),
            pl.col("action").cast(pl.Categorical),
            pl.col("side").cast(pl.Categorical),
            pl.col("price").cast(pl.Float64),
            pl.col("size").cast(pl.Float64),
            pl.col("id").alias("level_id"),
            pl.lit(datetime.now(timezone.utc)).alias("ingest_ts"),
        ]
    )
    df = df.select(
        ["ts", "symbol", "action", "side", "level_id", "price", "size", "ingest_ts"]
    ).sort(["ts", "symbol", "side", "level_id"])
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / f"{source_path.stem}.parquet"
    df.write_parquet(target)
    io_utils.write_manifest(
        target,
        rows=df.height,
        extra={"source": str(source_path)},
    )
    logger.info("Wrote normalized L2 updates to %s", target)
    return target

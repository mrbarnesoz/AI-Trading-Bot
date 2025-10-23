"""Resample normalized trade data into OHLCV bar aggregates."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class BarSpec:
    timeframe: str  # e.g. "1m", "5m"
    partition: str  # relative output partition path


def resample_trades(
    trade_parquet: Path,
    bar_specs: Sequence[BarSpec],
    output_root: Path,
) -> None:
    """Resample trades to multiple bar intervals and persist to Parquet."""
    df = pl.read_parquet(trade_parquet)
    df = df.rename({"ts": "timestamp"})
    df = df.sort(["symbol", "timestamp"])
    for spec in bar_specs:
        logger.info("Resampling %s to %s bars", trade_parquet.name, spec.timeframe)
        grouped = (
            df.group_by_dynamic(
                "timestamp",
                every=spec.timeframe,
                period=spec.timeframe,
                closed="right",
                by="symbol",
                label="right",
            )
            .agg(
                [
                    pl.first("price").alias("open"),
                    pl.max("price").alias("high"),
                    pl.min("price").alias("low"),
                    pl.last("price").alias("close"),
                    pl.sum("size").alias("volume"),
                    (pl.col("price") * pl.col("size")).sum().alias("notional"),
                    pl.count().alias("trade_count"),
                ]
            )
            .with_columns(
                [
                    (pl.col("notional") / pl.col("volume")).alias("vwap"),
                    pl.col("timestamp").alias("ts"),
                    pl.lit(spec.timeframe).alias("timeframe"),
                ]
            )
            .select(
                [
                    "ts",
                    "symbol",
                    "timeframe",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "vwap",
                    "trade_count",
                ]
            )
        )
        if "notional" in grouped.columns:
            grouped = grouped.drop("notional")
        output_dir = output_root / spec.partition
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / f"{trade_parquet.stem}_{spec.timeframe}.parquet"
        grouped.write_parquet(target)
        logger.info("Wrote bars to %s", target)

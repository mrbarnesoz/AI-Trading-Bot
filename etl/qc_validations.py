"""Quality control checks for BitMEX bronze/silver datasets."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import polars as pl

from utils.time import is_sorted

logger = logging.getLogger(__name__)


@dataclass
class CompletenessResult:
    coverage: float
    threshold: float

    def passed(self) -> bool:
        return self.coverage >= self.threshold


def validate_spread_non_negative(orderbook_snapshot_df: pl.DataFrame) -> None:
    if (orderbook_snapshot_df["spread"] < 0).any():
        raise ValueError("Detected negative spreads in order book snapshots.")
    logger.debug("Spread validation passed.")


def validate_monotonic_timestamps(df: pl.DataFrame, column: str = "ts") -> None:
    if not is_sorted(df[column]):
        raise ValueError(f"Timestamps in column '{column}' are not monotonic.")
    logger.debug("Timestamp monotonicity validation passed.")


def check_trade_minute_completeness(trade_df: pl.DataFrame, threshold: float = 0.95) -> CompletenessResult:
    """Check minute-level coverage for a full day."""
    minutes_present = trade_df["ts"].dt.truncate("1m").unique().len()
    coverage = minutes_present / (24 * 60)
    logger.debug("Trade minute coverage: %.3f", coverage)
    return CompletenessResult(coverage=coverage, threshold=threshold)

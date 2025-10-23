"""Time utility helpers."""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc_ns(column: pl.Expr) -> pl.Expr:
    """Ensure a Polars datetime expression is UTC with nanosecond resolution."""
    return column.cast(pl.Datetime(time_unit="ns", time_zone="UTC"))


def is_sorted(column: pl.Series) -> bool:
    return column.is_sorted()

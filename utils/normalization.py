"""Normalization pipeline for feature engineering."""

from __future__ import annotations

import polars as pl


def rolling_zscore(data: pl.LazyFrame, column: str, window: int) -> pl.Expr:
    mean = pl.col(column).rolling_mean(window_size=window)
    std = pl.col(column).rolling_std(window_size=window)
    return (pl.col(column) - mean) / (std + 1e-9)


def apply_normalization(frame: pl.LazyFrame, window: int) -> pl.LazyFrame:
    cols = [col for col in frame.columns if col not in {"ts", "symbol"}]
    normalized = frame
    for col in cols:
        normalized = normalized.with_columns(
            [
                rolling_zscore(normalized, col, window).clip(-5, 5).alias(f"{col}_z"),
            ]
        )
    return normalized

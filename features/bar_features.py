"""Bar-level feature engineering utilities."""

from __future__ import annotations

import polars as pl


def add_basic_returns(bars: pl.LazyFrame) -> pl.LazyFrame:
    return bars.with_columns(
        [
            pl.col("close")
            .pct_change()
            .alias("ret_1"),
            pl.col("close")
            .pct_change(3)
            .alias("ret_3"),
            pl.col("close")
            .pct_change(12)
            .alias("ret_12"),
        ]
    )


def add_volatility_features(bars: pl.LazyFrame) -> pl.LazyFrame:
    return bars.with_columns(
        [
            pl.col("ret_1").rolling_std(window_size=60).alias("vol_60"),
            pl.col("ret_1").rolling_std(window_size=120).alias("vol_120"),
            (pl.col("ret_1") / (pl.col("ret_1").rolling_std(window_size=60) + 1e-9)).alias("zret_1"),
        ]
    )

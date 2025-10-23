"""Triple-barrier labeling utilities."""

from __future__ import annotations

import polars as pl


def triple_barrier_labels(
    bars: pl.LazyFrame,
    horizon: int,
    sigma_col: str = "vol_60",
    k_sigma: float = 0.5,
) -> pl.LazyFrame:
    """Apply triple-barrier scheme and return label columns."""
    return bars.with_columns(
        [
            (pl.col("close").shift(-horizon) / pl.col("close") - 1.0).alias("fwd_ret"),
            (k_sigma * pl.col(sigma_col) * (horizon**0.5)).alias("barrier"),
        ]
    ).with_columns(
        [
            pl.when(pl.col("fwd_ret") > pl.col("barrier"))
            .then(1)
            .when(pl.col("fwd_ret") < -pl.col("barrier"))
            .then(-1)
            .otherwise(0)
            .alias("label_sign"),
        ]
    )

"""Feature transforms for microstructure and order-book data."""

from __future__ import annotations

import polars as pl


def build_depth_imbalance(snapshot_df: pl.DataFrame) -> pl.DataFrame:
    """Compute depth imbalances for top-of-book levels."""
    return snapshot_df.with_columns(
        [
            (pl.col("bid_sz_1") - pl.col("ask_sz_1")) / (pl.col("bid_sz_1") + pl.col("ask_sz_1") + 1e-9).alias(
                "imbalance_l1"
            ),
            (pl.col("total_bid_sz_1_5") - pl.col("total_ask_sz_1_5"))
            / (pl.col("total_bid_sz_1_5") + pl.col("total_ask_sz_1_5") + 1e-9).alias("imbalance_l1_5"),
        ]
    )


def rolling_microstructure_features(snapshot_df: pl.LazyFrame, window: int = 40) -> pl.LazyFrame:
    """Generate rolling features over the specified window of snapshots."""
    return snapshot_df.with_columns(
        [
            pl.col("mid").diff().alias("mid_change"),
            pl.col("spread").rolling_mean(window, by="symbol").alias("spread_mean"),
            pl.col("spread").rolling_std(window, by="symbol").alias("spread_std"),
        ]
    )

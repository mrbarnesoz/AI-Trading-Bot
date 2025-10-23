"""Utilities to join labels with feature matrices."""

from __future__ import annotations

import polars as pl


def join_features_labels(features: pl.DataFrame, labels: pl.DataFrame, on: tuple[str, str] = ("symbol", "ts")) -> pl.DataFrame:
    merged = features.join(labels, on=list(on), how="inner")
    merged = merged.drop_nulls()
    return merged

"""Utilities for loading model-ready datasets from the gold feature store."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import pandas as pd
import polars as pl


@dataclass(slots=True)
class DatasetSpec:
    """Specification describing where to load a regime dataset."""

    root: Path
    regime: str
    label_column: str = "label_sign"
    weight_column: Optional[str] = None
    drop_columns: Sequence[str] = field(default_factory=lambda: ("symbol",))
    glob: str = "*.parquet"

    @property
    def feature_root(self) -> Path:
        return self.root / "features" / self.regime

    @property
    def label_root(self) -> Path:
        return self.root / "labels" / self.regime


def _collect_parquet(paths: Iterable[Path], columns: Optional[Sequence[str]] = None) -> pl.DataFrame:
    scans = [pl.scan_parquet(str(path)) for path in paths]
    if not scans:
        raise FileNotFoundError("No parquet files located for dataset spec.")
    lazy = pl.concat(scans)
    if columns:
        lazy = lazy.select(list(columns))
    return lazy.collect()


def _list_files(base: Path, pattern: str) -> list[Path]:
    if not base.exists():
        raise FileNotFoundError(f"Dataset location {base} does not exist.")
    files = sorted(base.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files matched {pattern} under {base}.")
    return files


def load_dataset(
    spec: DatasetSpec,
    symbols: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    columns: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    """Load a dataset for a given regime as pandas objects suitable for modeling.

    Parameters
    ----------
    spec:
        Dataset specification describing the regime and gold root.
    symbols:
        Optional subset of symbols to include. When omitted, all symbols are loaded.
    limit:
        Optional row cap (applied after filtering) useful for smoke tests.
    columns:
        Optional explicit column list to project before collecting from Polars.
    """

    feature_files = _list_files(spec.feature_root, spec.glob)
    df = _collect_parquet(feature_files, columns=columns)

    if symbols:
        df = df.filter(pl.col("symbol").is_in(symbols))
    if limit is not None:
        df = df.limit(limit)

    pandas_df = df.to_pandas(use_pyarrow_extension=True)
    if "ts" in pandas_df.columns:
        pandas_df["ts"] = pd.to_datetime(pandas_df["ts"], utc=True, errors="coerce")
        pandas_df = pandas_df.sort_values(["symbol", "ts"])
        pandas_df = pandas_df.set_index("ts")

    label_series = pandas_df.pop(spec.label_column)
    weight_series: Optional[pd.Series] = None
    if spec.weight_column and spec.weight_column in pandas_df.columns:
        weight_series = pandas_df.pop(spec.weight_column)

    for col in spec.drop_columns:
        if col in pandas_df.columns:
            pandas_df = pandas_df.drop(columns=[col])

    feature_columns = pandas_df.columns.tolist()
    if not feature_columns:
        raise ValueError("No feature columns remain after dropping labels and exclusions.")

    return pandas_df.astype("float64"), label_series.astype("int32"), weight_series


__all__ = ["DatasetSpec", "load_dataset"]

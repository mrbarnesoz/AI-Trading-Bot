"""Feature generation orchestrator for HFT, intraday, and swing regimes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

import polars as pl

from features import bar_features, microstructure
from labels import triple_barrier
from utils import normalization


@dataclass
class RegimeConfig:
    name: str
    sampling: str
    lookback: int
    normalization_window: int
    feature_funcs: tuple[Callable[[pl.LazyFrame], pl.LazyFrame], ...]
    label_horizon: int
    sigma_col: str
    k_sigma: float
    mutual_information_top_k: int


def build_intraday_pipeline(bars: pl.LazyFrame, cfg: RegimeConfig) -> pl.DataFrame:
    features = bars
    for func in cfg.feature_funcs:
        features = func(features)
    labels_df = triple_barrier.triple_barrier_labels(features, cfg.label_horizon, cfg.sigma_col, cfg.k_sigma)
    normalized = normalization.apply_normalization(labels_df, cfg.normalization_window)
    return normalized.collect()


REGIME_PRESETS: Dict[str, RegimeConfig] = {
    "intraday": RegimeConfig(
        name="intraday",
        sampling="1m",
        lookback=1440,
        normalization_window=500,
        feature_funcs=(bar_features.add_basic_returns, bar_features.add_volatility_features),
        label_horizon=15,
        sigma_col="vol_60",
        k_sigma=0.5,
        mutual_information_top_k=80,
    )
}

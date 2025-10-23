"""Forward testing utilities for shadow trading."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict

import polars as pl

from eval.metrics import compute_backtest_metrics


@dataclass
class ForwardTestConfig:
    regime: str
    duration_days: int
    latency_ms: int


def run_shadow_test(trades: pl.DataFrame, config: ForwardTestConfig) -> Dict[str, float]:
    window_start = trades["ts"].min()
    window_end = window_start + timedelta(days=config.duration_days)
    window = trades.filter((pl.col("ts") >= window_start) & (pl.col("ts") < window_end))
    metrics = compute_backtest_metrics(window)
    return metrics

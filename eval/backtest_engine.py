"""Vectorized backtest engine with BitMEX cost/latency model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import polars as pl

from eval.costs import CostModel, compute_trade_cost
from eval.metrics import compute_backtest_metrics


@dataclass(slots=True)
class BacktestConfig:
    regime: str
    long_threshold: float
    short_threshold: float
    cross_threshold: float
    latency_bars: int = 0
    notional: float = 1.0
    funding: Optional[pl.DataFrame] = None


def _ensure_dataframe(frame) -> pl.DataFrame:
    if isinstance(frame, pl.DataFrame):
        return frame.clone()
    if hasattr(frame, "to_dict"):
        return pl.from_pandas(frame)
    raise TypeError("Unsupported frame type for backtest input.")


def run_backtest(
    features,
    probs,
    labels,
    cost_model: CostModel,
    config: BacktestConfig,
) -> Tuple[Dict[str, float], pl.DataFrame, pl.DataFrame]:
    """Run a vectorized maker-first backtest and return metrics and trade logs."""
    df = _ensure_dataframe(features)
    if "ts" not in df.columns:
        raise ValueError("features must contain 'ts' column for chronological ordering.")
    df = df.with_columns(
        [
            pl.Series("prob", probs).cast(pl.Float64),
            pl.Series("label", labels).cast(pl.Float64),
        ]
    ).sort("ts")

    df = df.with_columns((1.0 - pl.col("prob")).alias("short_prob"))
    df = df.with_columns(
        [
            pl.when(pl.col("prob") >= config.long_threshold)
            .then(1)
            .when(pl.col("short_prob") >= config.short_threshold)
            .then(-1)
            .otherwise(0)
            .alias("raw_signal"),
            (
                (pl.col("prob") >= config.cross_threshold)
                | (pl.col("short_prob") >= config.cross_threshold)
            ).alias("cross"),
        ]
    )

    position_expr = pl.col("raw_signal")
    if config.latency_bars > 0:
        position_expr = position_expr.shift(config.latency_bars).fill_null(0)
    df = df.with_columns(position_expr.alias("position"))

    df = df.with_columns(
        [
            pl.col("position").shift(1, fill_value=0).alias("prev_position"),
        ]
    ).with_columns(
        [
            (pl.col("position") - pl.col("prev_position")).abs().alias("turnover"),
        ]
    )

    df = df.with_columns(
        [
            (pl.col("position") * pl.col("label") * config.notional).alias("gross_pnl"),
            (compute_trade_cost(pl.col("turnover"), pl.col("cross"), cost_model) * config.notional).alias("cost"),
        ]
    )
    df = df.with_columns((pl.col("gross_pnl") - pl.col("cost")).alias("net_pnl"))
    df = df.with_columns(pl.col("net_pnl").cum_sum().alias("equity"))

    metrics = compute_backtest_metrics(df)
    symbol_cols = ["symbol"] if "symbol" in df.columns else []
    portfolio_cols = ["ts", *symbol_cols, "prob", "short_prob", "label", "position", "gross_pnl", "cost", "net_pnl", "equity"]
    trade_cols = ["ts", *symbol_cols, "position", "turnover", "cross", "prob", "short_prob", "gross_pnl", "cost"]
    portfolio = df.select(portfolio_cols)
    trades = df.filter(pl.col("turnover") > 0).select(trade_cols)
    return metrics, portfolio, trades


def save_results(metrics: Dict[str, float], trades: pl.DataFrame, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "metrics.json"
    trades_path = results_dir / "trades.parquet"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    trades.write_parquet(trades_path)

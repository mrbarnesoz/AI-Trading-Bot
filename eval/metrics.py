"""Backtest evaluation metrics."""

from __future__ import annotations

from typing import Dict

import numpy as np
import polars as pl


def compute_backtest_metrics(df: pl.DataFrame) -> Dict[str, float]:
    column = "net_pnl" if "net_pnl" in df.columns else "pnl"
    pnl = df[column].to_numpy()
    cumulative = np.cumsum(pnl)
    sharpe = _sharpe(pnl)
    sortino = _sortino(pnl)
    calmar = _calmar(cumulative)
    max_dd = _max_drawdown(cumulative)
    hit_rate = float((pnl > 0).mean())
    total_positive = pnl[pnl > 0].sum()
    total_negative = -pnl[pnl < 0].sum()
    profit_factor = float(total_positive / total_negative) if total_negative > 0 else float("inf")
    avg_return = float(pnl.mean())
    turnover = float(df["turnover"].sum()) if "turnover" in df.columns else float("nan")
    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "hit_rate": hit_rate,
        "profit_factor": profit_factor,
        "avg_return": avg_return,
        "turnover": turnover,
    }


def _sharpe(pnl: np.ndarray, eps: float = 1e-9) -> float:
    mean = pnl.mean()
    std = pnl.std()
    if std < eps:
        return 0.0
    return float(mean / std * np.sqrt(252))


def _sortino(pnl: np.ndarray, eps: float = 1e-9) -> float:
    downside = pnl[pnl < 0]
    if downside.size == 0:
        return float(np.inf)
    denominator = downside.std()
    if denominator < eps:
        return 0.0
    return float(pnl.mean() / denominator * np.sqrt(252))


def _max_drawdown(equity: np.ndarray) -> float:
    running_max = np.maximum.accumulate(equity)
    drawdown = equity - running_max
    return float(drawdown.min())


def _calmar(equity: np.ndarray) -> float:
    max_dd = abs(_max_drawdown(equity))
    total_return = equity[-1]
    if max_dd == 0:
        return float(total_return)
    return float(total_return / max_dd)

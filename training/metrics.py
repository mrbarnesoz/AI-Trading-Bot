"""Training and evaluation metrics."""

from __future__ import annotations

import numpy as np


def compute_sharpe(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-9) -> float:
    pnl = probs * labels
    mean = pnl.mean()
    std = pnl.std()
    if std < eps:
        return 0.0
    return float(mean / std * np.sqrt(252))


def compute_calmar(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-9) -> float:
    pnl = probs * labels
    cumulative = np.cumsum(pnl)
    drawdown = cumulative - np.maximum.accumulate(cumulative)
    max_dd = drawdown.min()
    total_return = cumulative[-1]
    if abs(max_dd) < eps:
        return float(total_return)
    return float(total_return / abs(max_dd))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from eval.backtest_engine import BacktestConfig, run_backtest
from eval.costs import CostModel


def _make_features(rows: int = 100) -> pl.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ts = [start + timedelta(minutes=i) for i in range(rows)]
    symbols = ["XBTUSD"] * rows
    return pl.DataFrame({"ts": ts, "symbol": symbols})


def test_run_backtest_produces_metrics_and_trades() -> None:
    features = _make_features(120)
    probs = pl.Series(np.linspace(0.4, 0.6, 120))
    labels = pl.Series(np.sin(np.linspace(0, np.pi, 120)))  # synthetic returns
    cost_model = CostModel(taker_fee_bps=2.5, slippage_bps=1.0, maker_fee_bps=0.0)
    config = BacktestConfig(
        regime="intraday",
        long_threshold=0.55,
        short_threshold=0.55,
        cross_threshold=0.75,
        latency_bars=1,
        notional=1.0,
    )
    metrics, portfolio, trades = run_backtest(features, probs, labels, cost_model, config)
    assert "sharpe" in metrics
    assert portfolio.height == 120
    assert "net_pnl" in portfolio.columns
    assert trades["turnover"].sum() >= 0

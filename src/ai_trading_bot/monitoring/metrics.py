"""Prometheus metrics helpers for the trading bot."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional

from prometheus_client import CollectorRegistry, Gauge, start_http_server

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Lightweight helper that maintains Prometheus gauges."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self._server_started = False
        self._server_port: Optional[int] = None

        self._equity = Gauge(
            "trading_equity",
            "Account equity or benchmark benchmarked equity level.",
            labelnames=("mode",),
            registry=self.registry,
        )
        self._drawdown = Gauge(
            "trading_drawdown",
            "Drawdown magnitude (absolute value).",
            labelnames=("mode",),
            registry=self.registry,
        )
        self._sharpe = Gauge(
            "trading_sharpe",
            "Sharpe ratio observed for the strategy.",
            labelnames=("mode",),
            registry=self.registry,
        )
        self._symbol_pnl = Gauge(
            "trading_symbol_pnl",
            "Per-symbol profit or loss (mark-to-market).",
            labelnames=("mode", "symbol"),
            registry=self.registry,
        )

    # ------------------------------------------------------------------#
    # Server management
    # ------------------------------------------------------------------#
    def start_server(self, port: int = 8001) -> None:
        """Expose the metrics endpoint if not already running."""
        if self._server_started:
            if self._server_port != port:
                logger.info("Metrics server already running on port %s", self._server_port)
            return
        start_http_server(port, registry=self.registry)
        self._server_started = True
        self._server_port = port
        logger.info("Prometheus metrics server listening on %s", port)

    # ------------------------------------------------------------------#
    # Recorders
    # ------------------------------------------------------------------#
    def record_backtest(self, summary: Dict[str, float], metadata: Dict[str, object]) -> None:
        """Record backtest summary metrics."""
        mode = "backtest"
        final_equity = float(summary.get("final_equity", 0.0))
        initial_capital = float(metadata.get("initial_capital", summary.get("initial_capital", final_equity)))
        total_return = float(summary.get("total_return", 0.0))
        sharpe = float(summary.get("sharpe_ratio", 0.0))
        max_drawdown = float(abs(summary.get("max_drawdown", 0.0)))
        symbol = str(metadata.get("symbol", "unknown"))

        self._equity.labels(mode=mode).set(final_equity)
        self._drawdown.labels(mode=mode).set(max_drawdown)
        self._sharpe.labels(mode=mode).set(sharpe)
        pnl = final_equity - initial_capital
        self._symbol_pnl.labels(mode=mode, symbol=symbol).set(pnl)
        logger.debug("Recorded backtest metrics for %s", symbol)

    def record_walk_forward(self, aggregate: Dict[str, float]) -> None:
        """Record walk-forward aggregate metrics."""
        mode = "walk_forward"
        cumulative_return = float(aggregate.get("cumulative_return", 0.0))
        worst_drawdown = float(abs(aggregate.get("worst_drawdown", 0.0)))
        average_sharpe = float(aggregate.get("average_sharpe", 0.0))
        self._equity.labels(mode=mode).set(1.0 + cumulative_return)
        self._drawdown.labels(mode=mode).set(worst_drawdown)
        self._sharpe.labels(mode=mode).set(average_sharpe)

    def record_live_state(
        self,
        *,
        equity: float,
        drawdown: float,
        sharpe: Optional[float] = None,
        symbol_pnl: Optional[Dict[str, float]] = None,
    ) -> None:
        """Update live gauges with the latest controller/risk metrics."""
        mode = "live"
        self._equity.labels(mode=mode).set(float(equity))
        self._drawdown.labels(mode=mode).set(float(abs(drawdown)))
        if sharpe is not None:
            self._sharpe.labels(mode=mode).set(float(sharpe))
        if symbol_pnl:
            for symbol, pnl in symbol_pnl.items():
                self._symbol_pnl.labels(mode=mode, symbol=symbol).set(float(pnl))


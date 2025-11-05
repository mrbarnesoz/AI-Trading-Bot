"""High-level live trading controller that ties data, strategy, risk, and execution."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, TYPE_CHECKING

from .data import BarEvent, DataFeed
from .execution import ExecutionRouter, OrderRequest
from .risk import RiskManager
from .strategy import StrategyDecision, StrategyEngine

if TYPE_CHECKING:
    from ai_trading_bot.monitoring.metrics import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class ControllerSettings:
    default_contract_size: float = 1.0
    market_order_type: str = "Market"


class LiveTradingController:
    """Coordinates live trading workflow."""

    def __init__(
        self,
        *,
        data_feed: DataFeed,
        strategy_engine: StrategyEngine,
        risk_manager: RiskManager,
        execution_router: ExecutionRouter,
        settings: ControllerSettings | None = None,
        metrics_collector: Optional["MetricsCollector"] = None,
    ) -> None:
        self.data_feed = data_feed
        self.strategy_engine = strategy_engine
        self.risk_manager = risk_manager
        self.execution_router = execution_router
        self.settings = settings or ControllerSettings()
        self._last_decision: Dict[str, StrategyDecision] = {}
        self.metrics = metrics_collector
        self._symbol_initial_price: Dict[str, float] = {}
        self._symbol_pnl: Dict[str, float] = {}

        self.data_feed.register_bar_callback(self._on_bar)

    def start(self) -> None:
        logger.info("Starting live trading controller.")
        self.data_feed.start()

    def stop(self) -> None:
        logger.info("Stopping live trading controller.")
        self.data_feed.stop()

    # ------------------------------------------------------------------#
    # Event handlers
    # ------------------------------------------------------------------#
    def _on_bar(self, bar: BarEvent) -> None:
        try:
            self.risk_manager.update_price(bar.symbol, bar.close)
            logger.debug("Received bar for %s close=%.4f", bar.symbol, bar.close)
            if bar.symbol not in self._symbol_initial_price:
                self._symbol_initial_price[bar.symbol] = bar.close
            qty = self.risk_manager.positions.get(bar.symbol, 0.0)
            initial_price = self._symbol_initial_price.get(bar.symbol, bar.close)
            self._symbol_pnl[bar.symbol] = qty * (bar.close - initial_price)
            if self.metrics:
                self.metrics.record_live_state(
                    equity=self.risk_manager.account_equity,
                    drawdown=self.risk_manager.cumulative_drawdown,
                    symbol_pnl=self._symbol_pnl,
                )
            decision = self.strategy_engine.on_bar(bar)
            if decision is None:
                return
            self._last_decision[bar.symbol] = decision
            logger.debug(
                "Strategy decision for %s: strategy=%s regime=%s signal=%.3f prob=%.3f",
                bar.symbol,
                getattr(decision, "strategy", "unknown"),
                getattr(decision, "regime", ""),
                decision.signal,
                decision.probability,
            )
            self._execute_decision(decision, bar)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Error handling bar event for %s", bar.symbol)

    # ------------------------------------------------------------------#
    def _execute_decision(self, decision: StrategyDecision, bar: BarEvent) -> None:
        target_qty = decision.signal * self.settings.default_contract_size
        current_qty = self.risk_manager.positions.get(decision.symbol, 0.0)
        delta = target_qty - current_qty
        if abs(delta) < 1e-6:
            logger.debug("No position change required for %s (target=%s current=%s).", decision.symbol, target_qty, current_qty)
            return

        side = "Buy" if delta > 0 else "Sell"
        quantity = abs(delta)
        order = OrderRequest(
            symbol=decision.symbol,
            side=side,
            quantity=quantity,
            order_type=self.settings.market_order_type,
            price=bar.close,
        )
        assessed = self.risk_manager.assess_order(order)
        if assessed is None:
            logger.info("Risk manager blocked order for %s", decision.symbol)
            return

        result = self.execution_router.submit_order(assessed)
        if result.status.lower() in {"rejected", "cancelled"}:
            logger.warning("Order for %s rejected: %s", decision.symbol, result.status)
            return

        filled = result.filled_qty if result.filled_qty else assessed.quantity
        new_qty = current_qty + filled if side == "Buy" else current_qty - filled
        self.risk_manager.update_position(decision.symbol, new_qty)
        logger.info(
            "Executed %s %s contracts for %s (new position=%s).",
            side,
            filled,
            decision.symbol,
            new_qty,
        )
        if self.metrics:
            self.metrics.record_live_state(
                equity=self.risk_manager.account_equity,
                drawdown=self.risk_manager.cumulative_drawdown,
                symbol_pnl=self._symbol_pnl,
            )

    # ------------------------------------------------------------------#
    # Accessors for external monitoring
    # ------------------------------------------------------------------#
    @property
    def last_decisions(self) -> Dict[str, StrategyDecision]:
        return dict(self._last_decision)

    def update_account_state(
        self,
        *,
        equity: float,
        daily_pnl: float,
        drawdown: float,
        sharpe: Optional[float] = None,
    ) -> None:
        self.risk_manager.update_account(equity=equity, daily_pnl=daily_pnl, drawdown=drawdown)
        if self.metrics:
            self.metrics.record_live_state(
                equity=equity,
                drawdown=drawdown,
                sharpe=sharpe,
                symbol_pnl=self._symbol_pnl,
            )

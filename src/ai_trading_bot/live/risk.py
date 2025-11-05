"""Risk management primitives for live trading."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Dict, Optional

from .execution import OrderRequest

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Configuration for live risk limits."""

    max_position_notional: float = 50_000.0
    max_portfolio_leverage: float = 3.0
    max_daily_loss: float = 1_000.0
    max_drawdown: float = 5_000.0


class RiskManager:
    """Applies risk guardrails to proposed orders."""

    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits
        self.account_equity: float = 0.0
        self.daily_pnl: float = 0.0
        self.cumulative_drawdown: float = 0.0
        self.positions: Dict[str, float] = {}
        self.prices: Dict[str, float] = {}
        self.daily_start_equity: float = 0.0
        self.peak_equity: float = 0.0

    # ------------------------------------------------------------------#
    # State synchronisation
    # ------------------------------------------------------------------#
    def update_account(self, *, equity: float, daily_pnl: float, drawdown: float) -> None:
        self.account_equity = max(equity, 0.0)
        self.daily_pnl = daily_pnl
        self.cumulative_drawdown = drawdown
        if self.daily_start_equity <= 0.0:
            self.daily_start_equity = self.account_equity
        self.peak_equity = max(self.peak_equity, self.account_equity)

    def update_position(self, symbol: str, quantity: float) -> None:
        if abs(quantity) < 1e-9:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = quantity

    def update_price(self, symbol: str, price: float) -> None:
        if price > 0:
            self.prices[symbol] = price

    # ------------------------------------------------------------------#
    # Assessment
    # ------------------------------------------------------------------#
    def assess_order(self, order: OrderRequest) -> Optional[OrderRequest]:
        """Return a possibly adjusted order or ``None`` if blocked."""
        if self._halts_active():
            logger.warning("RiskManager halting trading due to risk limits.")
            return None

        mark_price = order.price or self.prices.get(order.symbol)
        if not mark_price:
            logger.debug("Missing price for %s; cannot evaluate order.", order.symbol)
            return None

        order_notional = mark_price * abs(order.quantity)
        if order_notional <= 0:
            return None

        # Cap per-symbol notional
        if order_notional > self.limits.max_position_notional:
            scale = self.limits.max_position_notional / order_notional
            if scale <= 0:
                logger.info("Blocking %s order: exceeds per-symbol notional limit.", order.symbol)
                return None
            adjusted_qty = order.quantity * scale
            logger.debug("Scaling order %s quantity from %s to %s due to symbol limit.", order.symbol, order.quantity, adjusted_qty)
            order = replace(order, quantity=adjusted_qty)
            order_notional = mark_price * abs(order.quantity)

        # Cap aggregate leverage
        prospective_total = self._portfolio_notional() + order_notional
        allowed_total = self.limits.max_portfolio_leverage * self.account_equity if self.account_equity > 0 else float("inf")
        if prospective_total > allowed_total:
            allowed_delta = max(allowed_total - self._portfolio_notional(), 0.0)
            if allowed_delta <= 0:
                logger.info("Blocking %s order: portfolio leverage limit reached.", order.symbol)
                return None
            scale = allowed_delta / order_notional
            adjusted_qty = order.quantity * scale
            if abs(adjusted_qty) < 1e-6:
                logger.info("Blocking %s order: leverage limit leaves no capacity.", order.symbol)
                return None
            logger.debug("Scaling order %s quantity from %s to %s due to leverage limit.", order.symbol, order.quantity, adjusted_qty)
            order = replace(order, quantity=adjusted_qty)

        return order

    def _portfolio_notional(self) -> float:
        notional = 0.0
        for symbol, qty in self.positions.items():
            price = self.prices.get(symbol)
            if price:
                notional += abs(qty) * price
        return notional

    def _halts_active(self) -> bool:
        if self.limits.max_daily_loss:
            if self.daily_pnl <= -abs(self.limits.max_daily_loss):
                logger.warning("Daily loss limit breached (%.2f <= -%.2f).", self.daily_pnl, self.limits.max_daily_loss)
                return True
            daily_equity_change = self.account_equity - self.daily_start_equity
            if daily_equity_change <= -abs(self.limits.max_daily_loss):
                logger.warning(
                    "Daily equity change breach (%.2f <= -%.2f).",
                    daily_equity_change,
                    self.limits.max_daily_loss,
                )
                return True
        if self.limits.max_drawdown:
            peak = self.peak_equity or self.account_equity
            if peak > 0:
                drawdown_amount = peak - self.account_equity
                if drawdown_amount >= abs(self.limits.max_drawdown):
                    logger.warning("Drawdown limit breached (%.2f >= %.2f).", drawdown_amount, self.limits.max_drawdown)
                    return True
        return False

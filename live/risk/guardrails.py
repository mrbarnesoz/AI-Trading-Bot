"""Risk guardrails for leverage, exposure, and loss stops."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from live.state.positions import PositionManager


@dataclass
class RiskLimits:
    leverage: float
    daily_loss: float
    atr_stop: float
    max_position_units: float = 1.0


class RiskGuardrails:
    """Evaluate whether proposed actions respect leverage and loss constraints."""

    def __init__(
        self,
        limits: Dict[str, RiskLimits],
        position_manager: Optional[PositionManager] = None,
        checkpoint_interval_seconds: int = 5,
    ) -> None:
        self.limits = limits
        self.positions = position_manager or PositionManager()
        now = datetime.now(timezone.utc)
        self._daily_loss = 0.0
        self._day = now.date()
        self._checkpoint_interval = timedelta(seconds=checkpoint_interval_seconds)
        self._last_checkpoint = now
        self._default_limits = limits.get("default", RiskLimits(leverage=1.0, daily_loss=0.1, atr_stop=5.0))

    def _resolve_limits(self, symbol: str, regime: Optional[str]) -> RiskLimits:
        if symbol in self.limits:
            return self.limits[symbol]
        if regime and regime in self.limits:
            return self.limits[regime]
        return self._default_limits

    def _roll_day(self, timestamp: datetime) -> None:
        ts_day = timestamp.date()
        if ts_day != self._day:
            self._day = ts_day
            self._daily_loss = 0.0

    def record_pnl(self, pnl: float) -> None:
        self._daily_loss += pnl

    def is_allowed(self, symbol: str, decision, market_state) -> bool:
        timestamp = getattr(market_state, "timestamp", datetime.now(timezone.utc))
        self._roll_day(timestamp)

        side = getattr(decision, "side", None)
        if side is None and isinstance(decision, dict):
            side = decision.get("side")
        if side is None:
            return False

        size = getattr(decision, "size", 0.0)
        if size == 0.0 and isinstance(decision, dict):
            size = float(decision.get("size", 0.0))
        target_position = getattr(decision, "target_position", None)
        regime = getattr(market_state, "regime", None)
        limits = self._resolve_limits(symbol, regime)

        if self._daily_loss <= -abs(limits.daily_loss):
            return False

        if side in {"hold", "flat"} or size == 0.0:
            return True

        current_position = self.positions.net_position(symbol)
        if target_position is None:
            delta = size if side in {"buy", "long"} else -size
            target_position = current_position + delta

        if abs(target_position) > limits.max_position_units:
            return False

        account = getattr(market_state, "account", None)
        if account is not None:
            leverage = getattr(account, "leverage", 0.0)
            if leverage and leverage > limits.leverage:
                return False

        atr = getattr(market_state, "atr", None)
        price = getattr(market_state, "price", None)
        if atr and price and atr * limits.atr_stop >= price * 0.5:
            return False

        self.positions.set_target(symbol, target_position)
        return True

    def should_checkpoint(self) -> bool:
        return datetime.now(timezone.utc) - self._last_checkpoint >= self._checkpoint_interval

    def mark_checkpoint(self) -> None:
        self._last_checkpoint = datetime.now(timezone.utc)

    def snapshot(self) -> dict:
        return {
            "daily_loss": self._daily_loss,
            "day": self._day.isoformat(),
        }

    def restore(self, payload: dict | None) -> None:
        if not payload:
            return
        self._daily_loss = float(payload.get("daily_loss", 0.0))
        day = payload.get("day")
        if day:
            self._day = datetime.fromisoformat(day).date()


__all__ = ["RiskLimits", "RiskGuardrails"]

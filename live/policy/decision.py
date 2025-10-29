"""Decision policy linking model probabilities to actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional


@dataclass
class Thresholds:
    entry: float
    exit: float
    cross: float
    unit_size: float = 1.0


@dataclass
class Probabilities:
    long: float
    short: float
    flat: float = 0.0

    def normalized(self) -> "Probabilities":
        total = max(self.long + self.short + self.flat, 1e-9)
        return Probabilities(self.long / total, self.short / total, self.flat / total)


@dataclass
class AccountState:
    equity: float
    leverage: float = 0.0
    daily_pnl: float = 0.0


@dataclass
class MarketState:
    symbol: str
    regime: str
    probabilities: Probabilities
    position: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    atr: float = 0.0
    price: float = 0.0
    lot_size: float = 1.0
    account: Optional[AccountState] = None
    high: float = 0.0
    low: float = 0.0
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    spread_bps: float = 0.0
    latency_ok: bool = True
    bar_closed: bool = True
    tick_size: float = 1.0
    mid_price: Optional[float] = None


@dataclass
class Decision:
    side: str
    size: float
    confidence: float
    cross: bool
    long_probability: float
    short_probability: float
    target_position: float


class DecisionPolicy:
    """Apply per-regime thresholds to model outputs."""

    _sampling_seconds = {"hft": 0.25, "intraday": 60.0, "swing": 3600.0}

    def __init__(self, thresholds: Dict[str, Thresholds]) -> None:
        self.thresholds = thresholds

    def decide(self, symbol: str, market_state: MarketState) -> Decision:
        thresholds = self.thresholds.get(market_state.regime)
        if thresholds is None:
            raise KeyError(f"No thresholds configured for regime '{market_state.regime}'.")

        probs = market_state.probabilities.normalized()
        long_prob = max(0.0, min(probs.long, 1.0))
        short_prob = max(0.0, min(probs.short, 1.0))

        current_position = market_state.position
        desired_position = current_position
        if long_prob >= thresholds.entry and long_prob >= short_prob:
            desired_position = thresholds.unit_size
        elif short_prob >= thresholds.entry and short_prob > long_prob:
            desired_position = -thresholds.unit_size
        elif abs(current_position) > 0 and max(long_prob, short_prob) <= thresholds.exit:
            desired_position = 0.0

        if desired_position > current_position:
            side = "buy"
            size = desired_position - current_position
        elif desired_position < current_position:
            side = "sell"
            size = current_position - desired_position
        else:
            side = "hold"
            size = 0.0

        confidence = max(long_prob, short_prob)
        cross = long_prob >= thresholds.cross or short_prob >= thresholds.cross

        return Decision(
            side=side,
            size=float(size),
            confidence=float(confidence),
            cross=cross,
            long_probability=long_prob,
            short_probability=short_prob,
            target_position=float(desired_position),
        )

    @staticmethod
    def sampling_interval(regime: str) -> float:
        return DecisionPolicy._sampling_seconds.get(regime.lower(), 60.0)


__all__ = [
    "Thresholds",
    "Probabilities",
    "AccountState",
    "MarketState",
    "Decision",
    "DecisionPolicy",
]

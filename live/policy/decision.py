"""Decision policy linking model probabilities to actions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

import pandas as pd

from ai_trading_bot.meta.select import MetaStrategySelector, Decision as MetaDecisionRecord

logger = logging.getLogger(__name__)


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
    features: Dict[str, float] = field(default_factory=dict)
    risk_metrics: Dict[str, float] = field(default_factory=dict)


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

    def __init__(self, thresholds: Dict[str, Thresholds], meta_selector: MetaStrategySelector | None = None) -> None:
        self.thresholds = thresholds
        self.meta_selector = meta_selector

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

        decision = Decision(
            side=side,
            size=float(size),
            confidence=float(confidence),
            cross=cross,
            long_probability=long_prob,
            short_probability=short_prob,
            target_position=float(desired_position),
        )
        return self._apply_meta(decision, market_state, thresholds)

    def _apply_meta(self, decision: Decision, market_state: MarketState, thresholds: Thresholds) -> Decision:
        if not self.meta_selector:
            return decision
        try:
            meta_decision = self._select_meta_decision(self.meta_selector, market_state)
        except Exception as exc:  # pragma: no cover - guardrail for unexpected selector failures
            logger.debug("Meta selector evaluation failed: %s", exc, exc_info=True)
            return decision
        if meta_decision is None:
            return decision
        try:
            return self._merge_meta(decision, meta_decision, thresholds, market_state)
        except Exception as exc:  # pragma: no cover - fallback to base behaviour
            logger.debug("Meta merge failed: %s", exc, exc_info=True)
            return decision

    def _select_meta_decision(
        self,
        selector: MetaStrategySelector,
        market_state: MarketState,
    ) -> MetaDecisionRecord | None:
        ts_value = getattr(market_state, "timestamp", datetime.now(timezone.utc))
        try:
            ts = pd.Timestamp(ts_value)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
        except Exception:
            ts = pd.Timestamp(datetime.now(timezone.utc))

        row_dict: Dict[str, float] = {}
        row_dict.update(getattr(market_state, "features", {}) or {})
        row_dict.update(getattr(market_state, "risk_metrics", {}) or {})
        row_dict.setdefault("atr", float(getattr(market_state, "atr", 0.0)))
        row_dict.setdefault("latency_ok", 1.0 if getattr(market_state, "latency_ok", True) else 0.0)
        row_dict.setdefault("spread_z", float(getattr(market_state, "spread_bps", 0.0)))

        account = getattr(market_state, "account", None)
        if account is not None:
            row_dict.setdefault("current_leverage", float(getattr(account, "leverage", 0.0)))
            daily_pnl = getattr(account, "daily_pnl", None)
            if daily_pnl is not None:
                row_dict.setdefault("daily_pnl_pct", float(daily_pnl))

        if "funding_z" not in row_dict:
            funding_metric = row_dict.get("fundingRate") or row_dict.get("funding_rate") or 0.0
            row_dict.setdefault("funding_z", float(funding_metric))

        row = pd.Series(row_dict, dtype="float64") if row_dict else pd.Series(dtype="float64")
        probability = float(market_state.probabilities.long)
        context = selector.create_context_from_row(
            ts,
            market_state.symbol,
            market_state.regime,
            row,
            probability,
        )
        return selector.select_strategy(context)

    def _merge_meta(
        self,
        decision: Decision,
        meta_decision: MetaDecisionRecord,
        thresholds: Thresholds,
        market_state: MarketState,
    ) -> Decision:
        current_position = float(getattr(market_state, "position", 0.0))
        meta_direction = meta_decision.direction.lower()

        if meta_direction == "flat":
            if abs(current_position) <= 1e-9:
                return Decision(
                    side="hold",
                    size=0.0,
                    confidence=max(decision.confidence, meta_decision.confidence),
                    cross=False,
                    long_probability=decision.long_probability,
                    short_probability=decision.short_probability,
                    target_position=0.0,
                )
            exit_side = "sell" if current_position > 0 else "buy"
            exit_size = abs(current_position)
            return Decision(
                side=exit_side,
                size=float(exit_size),
                confidence=max(decision.confidence, meta_decision.confidence),
                cross=meta_decision.execution == "cross",
                long_probability=decision.long_probability,
                short_probability=decision.short_probability,
                target_position=0.0,
            )

        if decision.side == "hold" and abs(current_position) < 1e-9:
            return decision

        delta = decision.target_position - current_position
        if delta > 1e-9:
            base_direction = "long"
        elif delta < -1e-9:
            base_direction = "short"
        else:
            base_direction = "flat"

        if base_direction in {"long", "short"} and meta_direction != base_direction:
            if abs(current_position) > 1e-9:
                exit_side = "sell" if current_position > 0 else "buy"
                exit_size = abs(current_position)
                return Decision(
                    side=exit_side,
                    size=float(exit_size),
                    confidence=max(decision.confidence, meta_decision.confidence),
                    cross=meta_decision.execution == "cross",
                    long_probability=decision.long_probability,
                    short_probability=decision.short_probability,
                    target_position=0.0,
                )
            return Decision(
                side="hold",
                size=0.0,
                confidence=decision.confidence,
                cross=False,
                long_probability=decision.long_probability,
                short_probability=decision.short_probability,
                target_position=float(current_position),
            )

        if base_direction == "flat" and abs(current_position) > 1e-9:
            current_direction = "long" if current_position > 0 else "short"
            if meta_direction != current_direction:
                exit_side = "sell" if current_position > 0 else "buy"
                exit_size = abs(current_position)
                return Decision(
                    side=exit_side,
                    size=float(exit_size),
                    confidence=max(decision.confidence, meta_decision.confidence),
                    cross=meta_decision.execution == "cross",
                    long_probability=decision.long_probability,
                    short_probability=decision.short_probability,
                    target_position=0.0,
                )

        selector = self.meta_selector
        sizing_cfg = {}
        if selector is not None:
            sizing_cfg = (
                selector.sizing.get(meta_decision.regime)
                or selector.sizing.get(meta_decision.regime.lower(), {})
                or {}
            )
        base_frac = float(sizing_cfg.get("base_size_frac", thresholds.unit_size or 1.0))
        max_frac = float(sizing_cfg.get("max_size_frac", base_frac))
        if base_frac <= 0:
            base_frac = 1.0
        max_scale = max_frac / base_frac if base_frac else 1.0
        scale = meta_decision.size_frac / base_frac if base_frac else meta_decision.size_frac
        scale = max(0.0, min(scale, max_scale))

        base_target = abs(decision.target_position)
        if base_target <= 1e-9:
            base_target = thresholds.unit_size
        target_units = base_target * scale
        target_units = min(target_units, thresholds.unit_size * max_scale)
        desired_sign = 1.0 if meta_direction == "long" else -1.0
        target_units *= desired_sign

        delta_target = target_units - current_position
        if abs(delta_target) <= 1e-9:
            return Decision(
                side="hold",
                size=0.0,
                confidence=max(decision.confidence, meta_decision.confidence),
                cross=decision.cross or meta_decision.execution == "cross",
                long_probability=decision.long_probability,
                short_probability=decision.short_probability,
                target_position=float(target_units),
            )

        side = "buy" if delta_target > 0 else "sell"
        size = abs(delta_target)
        return Decision(
            side=side,
            size=float(size),
            confidence=max(decision.confidence, meta_decision.confidence),
            cross=decision.cross or meta_decision.execution == "cross",
            long_probability=decision.long_probability,
            short_probability=decision.short_probability,
            target_position=float(target_units),
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

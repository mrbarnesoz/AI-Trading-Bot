from __future__ import annotations

import pytest
from ai_trading_bot.meta.select import Context, Decision as MetaDecision
from live.policy.decision import (
    DecisionPolicy,
    MarketState,
    Probabilities,
    Thresholds,
)


class DummyMetaSelector:
    def __init__(self, *, direction: str = "long", size_frac: float = 0.1, execution: str = "maker") -> None:
        self.direction = direction
        self.size_frac = size_frac
        self.execution = execution
        self.sizing = {"intraday": {"base_size_frac": 0.1, "max_size_frac": 0.2}}
        self.feature_keys: list[str] = []
        self._last_context: Context | None = None

    def create_context_from_row(self, timestamp, symbol: str, regime: str, row, probability: float) -> Context:
        self._last_context = Context(
            timestamp=timestamp,
            symbol=symbol,
            regime=regime,
            features={},
            model_outputs={"p_long": probability, "p_short": max(0.0, 1.0 - probability), "p_hold": 0.0},
            market_state={},
            risk_state={},
        )
        return self._last_context

    def select_strategy(self, context: Context) -> MetaDecision:
        return MetaDecision(
            timestamp=context.timestamp,
            symbol=context.symbol,
            regime=context.regime,
            strategy_id="S1",
            weight=1.0,
            direction=self.direction,
            size_frac=self.size_frac,
            confidence=0.9,
            execution=self.execution,
            rationale="dummy",
        )

    def flush(self) -> None:  # pragma: no cover - compatibility hook
        pass


def _policy(selector: DummyMetaSelector) -> DecisionPolicy:
    thresholds = {"intraday": Thresholds(entry=0.6, exit=0.4, cross=0.75, unit_size=1.0)}
    return DecisionPolicy(thresholds, meta_selector=selector)


def test_meta_policy_scales_long_position():
    selector = DummyMetaSelector(direction="long", size_frac=0.2)
    policy = _policy(selector)
    market_state = MarketState(
        symbol="XBTUSD",
        regime="intraday",
        probabilities=Probabilities(long=0.8, short=0.1),
        position=0.0,
    )

    decision = policy.decide("XBTUSD", market_state)

    assert decision.side == "buy"
    assert pytest.approx(decision.size, rel=1e-6) == 2.0
    assert pytest.approx(decision.target_position, rel=1e-6) == 2.0


def test_meta_policy_requires_threshold_before_opening():
    selector = DummyMetaSelector(direction="long", size_frac=0.2)
    policy = _policy(selector)
    market_state = MarketState(
        symbol="XBTUSD",
        regime="intraday",
        probabilities=Probabilities(long=0.2, short=0.2),
        position=0.0,
    )

    decision = policy.decide("XBTUSD", market_state)

    assert decision.side == "hold"
    assert decision.size == 0.0
    assert decision.target_position == 0.0


def test_meta_policy_flattens_when_direction_conflicts():
    selector = DummyMetaSelector(direction="short", size_frac=0.2)
    policy = _policy(selector)
    market_state = MarketState(
        symbol="XBTUSD",
        regime="intraday",
        probabilities=Probabilities(long=0.55, short=0.3),
        position=1.0,
    )

    decision = policy.decide("XBTUSD", market_state)

    assert decision.side == "sell"
    assert pytest.approx(decision.size, rel=1e-6) == 1.0
    assert decision.target_position == 0.0


def test_meta_policy_flat_decision_closes_position():
    selector = DummyMetaSelector(direction="flat", size_frac=0.0)
    policy = _policy(selector)
    market_state = MarketState(
        symbol="XBTUSD",
        regime="intraday",
        probabilities=Probabilities(long=0.5, short=0.4),
        position=-2.0,
    )

    decision = policy.decide("XBTUSD", market_state)

    assert decision.side == "buy"
    assert pytest.approx(decision.size, rel=1e-6) == 2.0
    assert decision.target_position == 0.0

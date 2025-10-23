from __future__ import annotations

from datetime import datetime, timedelta, timezone

from live.policy.decision import AccountState, Decision, DecisionPolicy, MarketState, Probabilities, Thresholds
from live.risk.guardrails import RiskGuardrails, RiskLimits


def _market_state(long_prob: float, short_prob: float, position: float = 0.0, flat_prob: float = 0.0) -> MarketState:
    return MarketState(
        symbol="XBTUSD",
        regime="intraday",
        probabilities=Probabilities(long_prob, short_prob, flat_prob),
        position=position,
        timestamp=datetime.now(timezone.utc),
        price=30000.0,
        atr=100.0,
        lot_size=1.0,
        account=AccountState(equity=100_000.0, leverage=1.5),
    )


def test_decision_policy_prefers_long_when_probability_high() -> None:
    policy = DecisionPolicy({"intraday": Thresholds(entry=0.6, exit=0.45, cross=0.75, unit_size=1.0)})
    decision = policy.decide("XBTUSD", _market_state(0.7, 0.2))
    assert decision.side == "buy"
    assert decision.cross is True
    assert decision.target_position == 1.0


def test_decision_policy_flattens_when_probability_drops() -> None:
    policy = DecisionPolicy({"intraday": Thresholds(entry=0.6, exit=0.4, cross=0.75, unit_size=1.0)})
    state = _market_state(0.2, 0.1, position=1.0, flat_prob=0.7)
    decision = policy.decide("XBTUSD", state)
    assert decision.side == "sell"
    assert decision.target_position == 0.0


def test_risk_guardrails_blocks_excess_position() -> None:
    limits = {"intraday": RiskLimits(leverage=3.0, daily_loss=10_000.0, atr_stop=5.0, max_position_units=1.0)}
    risk = RiskGuardrails(limits)
    state = _market_state(0.65, 0.2)
    policy = DecisionPolicy({"intraday": Thresholds(entry=0.6, exit=0.4, cross=0.8)})
    action = policy.decide("XBTUSD", state)
    allowed = risk.is_allowed("XBTUSD", action, state)
    assert allowed is True

    heavy_action = Decision(
        side="buy",
        size=1.5,
        confidence=0.9,
        cross=True,
        long_probability=0.9,
        short_probability=0.1,
        target_position=1.5,
    )
    assert risk.is_allowed("XBTUSD", heavy_action, state) is False


def test_should_checkpoint_triggers_after_interval() -> None:
    limits = {"intraday": RiskLimits(leverage=3.0, daily_loss=10_000.0, atr_stop=5.0)}
    risk = RiskGuardrails(limits, checkpoint_interval_seconds=1)
    # force last checkpoint in past
    risk._last_checkpoint -= timedelta(seconds=2)
    assert risk.should_checkpoint() is True

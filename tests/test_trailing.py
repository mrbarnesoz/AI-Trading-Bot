
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ai_trading_bot.config import TrailingConfig
from live.policy.decision import MarketState, Probabilities
from live.risk.trailing import TrailingManager
from live.state.positions import PositionManager


def _cfg(enabled: bool = True) -> TrailingConfig:
    return TrailingConfig(
        enabled=enabled,
        update={
            "hft": {"snapshots": 4},
            "intraday": {"on_bar_close": True},
            "swing": {"on_bar_close": True},
        },
        k_atr={
            "stop": {"intraday": 3.0, "swing": 4.0, "hft": 2.0},
            "take": {"intraday": 1.5, "swing": 2.0, "hft": 1.0},
        },
        min_lock={"R_multiple": 1.0},
        floor_ceiling={
            "long": {"min_px_distance_ticks": 1},
            "short": {"min_px_distance_ticks": 1},
        },
        slippage_guard_bps=10,
        max_updates_per_min=60,
    )


def _state(
    price: float,
    atr: float,
    *,
    regime: str = "intraday",
    position: float = 1.0,
    long_prob: float = 0.7,
    short_prob: float = 0.1,
    high: float | None = None,
    low: float | None = None,
    spread_bps: float = 0.0,
    latency_ok: bool = True,
    bar_closed: bool = True,
    minute_offset: int = 0,
) -> MarketState:
    base_ts = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=minute_offset)
    return MarketState(
        symbol="XBTUSD",
        regime=regime,
        probabilities=Probabilities(long_prob, short_prob, 1.0 - long_prob - short_prob),
        position=position,
        timestamp=base_ts,
        atr=atr,
        price=price,
        lot_size=1.0,
        high=high if high is not None else price,
        low=low if low is not None else price,
        spread_bps=spread_bps,
        latency_ok=latency_ok,
        bar_closed=bar_closed,
        tick_size=1.0,
    )


def _create_manager(tmp_path, cfg: TrailingConfig | None = None) -> TrailingManager:
    positions = PositionManager()
    config = cfg or _cfg()
    return TrailingManager(config, positions, export_dir=tmp_path / "trail_logs")


def test_trailing_long_ratchets_stop_and_exits(tmp_path) -> None:
    manager = _create_manager(tmp_path)

    entry_state = _state(price=100.0, atr=2.0, high=101.0, low=99.0, position=0.0)
    manager.record_execution("XBTUSD", {"side": "buy", "size": 1.0}, entry_state)

    sub = manager._get_position("XBTUSD", "long")
    assert sub is not None
    assert pytest.approx(94.0, rel=1e-6) == sub.stop_price

    # Price rallies strongly (best_favorable up to 110)
    rally_state = _state(price=110.0, atr=2.0, high=110.0, low=108.0, position=1.0, minute_offset=1)
    events = manager.update("XBTUSD", rally_state)
    assert events == []  # no exit yet
    assert sub.stop_price is not None and sub.stop_price >= 104.0  # floored by bar low

    # First drop triggers partial take profit
    reversal_state = _state(price=sub.stop_price, atr=2.0, high=sub.stop_price + 0.5, low=sub.stop_price - 0.5, position=1.0, minute_offset=2)
    events = manager.update("XBTUSD", reversal_state)
    assert len(events) == 1
    event = events[0]
    assert event["exit_reason"] == "TTP_hit"
    manager.record_execution("XBTUSD", event, reversal_state)
    sub = manager._get_position("XBTUSD", "long")
    assert sub is not None
    assert sub.take_profit_price is None

    # Further decline should trigger trailing stop on the remainder
    final_state = _state(price=sub.stop_price, atr=2.0, high=sub.stop_price + 0.5, low=sub.stop_price - 0.5, position=sub.size * 1.0, minute_offset=3)
    events = manager.update("XBTUSD", final_state)
    assert len(events) == 1
    tsl_event = events[0]
    assert tsl_event["exit_reason"] == "TSL_hit"
    manager.record_execution("XBTUSD", tsl_event, final_state)
    assert manager.positions.net_position("XBTUSD") == 0.0


def test_trailing_requires_min_r_multiple(tmp_path) -> None:
    cfg = _cfg()
    cfg.min_lock = {"R_multiple": 1.5}
    manager = _create_manager(tmp_path, cfg)

    entry_state = _state(price=100.0, atr=4.0, high=101.0, low=99.0, position=0.0)
    manager.record_execution("XBTUSD", {"side": "buy", "size": 1.0}, entry_state)
    sub = manager._get_position("XBTUSD", "long")
    assert sub is not None
    initial_stop = sub.stop_price

    modest_state = _state(price=104.0, atr=4.0, high=104.0, low=102.0, position=1.0, minute_offset=1)
    manager.update("XBTUSD", modest_state)
    assert sub.stop_price == initial_stop  # R multiple < 1.5 so no update


def test_trailing_respects_spread_guard(tmp_path) -> None:
    cfg = _cfg()
    cfg.slippage_guard_bps = 1.0
    manager = _create_manager(tmp_path, cfg)
    entry_state = _state(price=100.0, atr=3.0, high=101.0, low=99.0, position=0.0)
    manager.record_execution("XBTUSD", {"side": "buy", "size": 1.0}, entry_state)
    sub = manager._get_position("XBTUSD", "long")
    stop_before = sub.stop_price

    spread_state = _state(
        price=110.0,
        atr=3.0,
        high=110.0,
        low=108.0,
        position=1.0,
        spread_bps=5.0,
        minute_offset=1,
    )
    manager.update("XBTUSD", spread_state)
    assert sub.stop_price == stop_before  # update skipped due to spread guard


def test_trailing_partial_take_profit_then_trail_only(tmp_path) -> None:
    manager = _create_manager(tmp_path)
    entry_state = _state(price=100.0, atr=2.0, high=101.0, low=99.0, position=0.0)
    manager.record_execution("XBTUSD", {"side": "buy", "size": 1.0}, entry_state)
    sub = manager._get_position("XBTUSD", "long")

    rally_state = _state(price=112.0, atr=2.0, high=112.0, low=110.0, position=1.0, minute_offset=1)
    manager.update("XBTUSD", rally_state)
    assert sub.take_profit_price is not None

    # Move to take profit level (below top but above stop)
    tp_state = _state(price=sub.take_profit_price, atr=2.0, high=sub.take_profit_price + 0.5, low=sub.take_profit_price - 0.5, position=1.0, minute_offset=2)
    events = manager.update("XBTUSD", tp_state)
    assert len(events) == 1
    event = events[0]
    assert event["exit_reason"] == "TTP_hit"
    assert pytest.approx(0.5, rel=1e-6) == event["size"]
    manager.record_execution("XBTUSD", event, tp_state)

    # Remaining position should be managed by stop only
    sub = manager._get_position("XBTUSD", "long")
    assert sub is not None
    assert sub.size == pytest.approx(0.5)
    assert sub.take_profit_price is None


def test_trailing_short_mirror_behaviour(tmp_path) -> None:
    manager = _create_manager(tmp_path)
    entry_state = _state(price=100.0, atr=2.0, high=101.0, low=99.0, position=0.0)
    manager.record_execution("XBTUSD", {"side": "sell", "size": 1.0}, entry_state)
    sub = manager._get_position("XBTUSD", "short")
    assert sub is not None
    initial_stop = sub.stop_price
    assert initial_stop and initial_stop > sub.entry_price

    favorable_state = _state(price=90.0, atr=2.0, high=92.0, low=90.0, position=-1.0, minute_offset=1)
    manager.update("XBTUSD", favorable_state)
    assert sub.stop_price < initial_stop  # stop moves lower for shorts

    reversal_state = _state(price=sub.stop_price, atr=2.0, high=sub.stop_price + 0.5, low=sub.stop_price - 0.5, position=-1.0, minute_offset=2)
    events = manager.update("XBTUSD", reversal_state)
    assert len(events) == 1
    event = events[0]
    assert event["side"] == "buy"
    assert event["exit_reason"] == "TTP_hit"
    manager.record_execution("XBTUSD", event, reversal_state)
    sub = manager._get_position("XBTUSD", "short")
    assert sub is not None
    assert sub.take_profit_price is None

    final_state = _state(price=sub.stop_price, atr=2.0, high=sub.stop_price + 0.5, low=sub.stop_price - 0.5, position=-0.5, minute_offset=3)
    events = manager.update("XBTUSD", final_state)
    assert len(events) == 1
    tsl_event = events[0]
    assert tsl_event["exit_reason"] == "TSL_hit"
    manager.record_execution("XBTUSD", tsl_event, final_state)
    assert manager.positions.net_position("XBTUSD") == 0.0

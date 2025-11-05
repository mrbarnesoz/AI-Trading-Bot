"""Volatility-aware trailing stop and take-profit management."""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ai_trading_bot.config import TrailingConfig
from live.state.positions import PositionManager, SubPosition


logger = logging.getLogger(__name__)

_TP_SL_SENTINEL_LOGGED = False


def sided_levels(best: float, atr: float, direction: int, k_tp: float, k_sl: float) -> Tuple[float, float]:
    global _TP_SL_SENTINEL_LOGGED
    if not _TP_SL_SENTINEL_LOGGED:
        logger.warning("TP_SL_FIX_ACTIVE k_tp=%s k_sl=%s", k_tp, k_sl)
        _TP_SL_SENTINEL_LOGGED = True
    tp = best + direction * k_tp * atr
    sl = best - direction * k_sl * atr
    if direction == 1:
        if not (tp > best and sl < best):
            raise ValueError(f"SideError dir={direction} best={best} atr={atr} tp={tp} sl={sl}")
    else:
        if not (tp < best and sl > best):
            raise ValueError(f"SideError dir={direction} best={best} atr={atr} tp={tp} sl={sl}")
    return tp, sl


@dataclass
class TrailingEvent:
    """Instruction emitted by the trailing manager."""

    side: str
    size: float
    cross: bool
    confidence: float
    target_position: float
    reason: str
    exit_reason: Optional[str] = None
    price: Optional[float] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "side": self.side,
            "size": float(self.size),
            "cross": bool(self.cross),
            "confidence": float(self.confidence),
            "target_position": float(self.target_position),
            "reason": self.reason,
        }
        if self.exit_reason:
            payload["exit_reason"] = self.exit_reason
        if self.price is not None:
            payload["price"] = float(self.price)
        return payload


class TrailingManager:
    """Applies ATR-based trailing logic across regimes."""

    def __init__(
        self,
        config: TrailingConfig,
        positions: PositionManager,
        export_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.positions = positions
        self._export_dir = Path(export_dir or Path("results") / "trailing_decisions")
        self._buffer: Dict[str, List[Dict[str, Any]]] = {}
        self._flush_threshold = 100
        self._export_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: Counter[str] = Counter()
        logger.warning("TRAIL_INSTANCE id=%s", hex(id(self)))

    @staticmethod
    def _direction(side: str) -> int:
        return 1 if side == "long" else -1

    def _validate_levels(
        self,
        symbol: str,
        side: str,
        best: float,
        stop: Optional[float],
        take: Optional[float],
        min_ticks: int,
        tick_size: float,
    ) -> bool:
        min_gap = tick_size * max(1, min_ticks)
        if stop is None:
            return True
        if side == "long":
            if stop >= best:
                logger.error("Invalid long stop for %s: stop %.4f >= best %.4f", symbol, stop, best)
                self.metrics["side_errors"] += 1
                return False
            if take is not None:
                if take <= best:
                    logger.error("Invalid long take for %s: take %.4f <= best %.4f", symbol, take, best)
                    self.metrics["side_errors"] += 1
                    return False
                if take - stop <= min_gap:
                    logger.error(
                        "Invalid long levels for %s: take %.4f - stop %.4f <= min_gap %.4f",
                        symbol,
                        take,
                        stop,
                        min_gap,
                    )
                    self.metrics["side_errors"] += 1
                    return False
        else:
            if stop <= best:
                logger.error("Invalid short stop for %s: stop %.4f <= best %.4f", symbol, stop, best)
                self.metrics["side_errors"] += 1
                return False
            if take is not None:
                if take >= best:
                    logger.error("Invalid short take for %s: take %.4f >= best %.4f", symbol, take, best)
                    self.metrics["side_errors"] += 1
                    return False
                if stop - take <= min_gap:
                    logger.error(
                        "Invalid short levels for %s: stop %.4f - take %.4f <= min_gap %.4f",
                        symbol,
                        stop,
                        take,
                        min_gap,
                    )
                    self.metrics["side_errors"] += 1
                    return False
        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def update(self, symbol: str, market_state) -> List[Dict[str, Any]]:
        """Evaluate all active sub-positions and emit trailing adjustments."""
        if not self.enabled():
            return []
        now = getattr(market_state, "timestamp", datetime.now(timezone.utc))
        regime = getattr(market_state, "regime", "intraday")

        events: List[Dict[str, Any]] = []
        for sub in list(self.positions.iter_symbol(symbol)):
            emitted = self._process_subposition(symbol, regime, sub, market_state, now)
            for event in emitted:
                events.append(event.to_payload())
        return events

    def record_execution(self, symbol: str, payload: Dict[str, Any], market_state) -> None:
        """Update position state after an execution (assumes immediate fills)."""
        side = payload.get("side")
        size = float(payload.get("size", 0.0))
        if side not in {"buy", "sell"} or size <= 0:
            # hold / flat decisions do not affect exposure
            if side in {"flat", "hold"}:
                for sub in self.positions.iter_symbol(symbol):
                    sub.pending_exit = None
            return

        price = self._resolve_price(payload, market_state)
        regime = getattr(market_state, "regime", "intraday")
        now = getattr(market_state, "timestamp", datetime.now(timezone.utc))
        atr = float(getattr(market_state, "atr", 0.0) or 0.0)
        tick_size = self._tick_size(market_state)

        net_before = self.positions.net_position(symbol)
        direction = 1.0 if side in {"buy", "long"} else -1.0
        net_after = net_before + direction * size

        # Remove opposing exposure first
        if net_before > 0 and net_after <= 0:
            self._close_position(symbol, "long")
        elif net_before < 0 and net_after >= 0:
            self._close_position(symbol, "short")

        if abs(net_after) < 1e-9:
            self._close_position(symbol, "long")
            self._close_position(symbol, "short")
            return

        target_side = "long" if net_after > 0 else "short"
        existing = self._get_position(symbol, target_side)
        new_size = abs(net_after)

        if existing is None:
            sub_id = f"{symbol}-{target_side}"
            entry_price = price
            sub = SubPosition(
                id=sub_id,
                side=target_side,
                size=new_size,
                entry_price=entry_price,
                stop_price=None,
                take_profit_price=None,
                take_profit_active=True,
                regime=regime,
                atr_at_entry=atr if atr > 0 else None,
                initial_stop_price=None,
                best_favorable_price=entry_price,
                best_adverse_price=entry_price,
                r_multiple=0.0,
                last_update=now,
                updates_this_minute=0,
                last_update_minute=None,
                snapshots_since_update=0,
                pending_exit=None,
            )
            self._seed_levels(sub, atr, tick_size)
            self.positions.open(symbol, sub)
        else:
            # Adjust entry price as weighted average when adding to position
            prev_size = existing.size
            if prev_size <= 0:
                existing.entry_price = price
            else:
                existing.entry_price = self._weighted_avg(
                    existing.entry_price,
                    prev_size,
                    price,
                    abs(direction * size),
                )
            existing.size = new_size
            existing.regime = regime
            existing.atr_at_entry = existing.atr_at_entry or (atr if atr > 0 else None)
            existing.best_adverse_price = (
                min(existing.best_adverse_price or price, price)
                if target_side == "long"
                else max(existing.best_adverse_price or price, price)
            )
            existing.best_favorable_price = (
                max(existing.best_favorable_price or price, price)
                if target_side == "long"
                else min(existing.best_favorable_price or price, price)
            )
            if direction * net_after > 0:
                existing.take_profit_active = True
            existing.pending_exit = None
            existing.snapshots_since_update = 0
            existing.last_update = now
            existing.updates_this_minute = 0
            existing.last_update_minute = None
            if existing.initial_stop_price is None:
                self._seed_levels(existing, atr, tick_size)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _process_subposition(
        self,
        symbol: str,
        regime: str,
        sub: SubPosition,
        market_state,
        now: datetime,
    ) -> List[TrailingEvent]:
        # Update cadence counters up-front
        sub.snapshots_since_update += 1

        price = self._resolve_price({}, market_state)
        if price is None:
            return []

        pre_events = self._check_triggers(symbol, sub, price)
        if pre_events:
            return pre_events

        high = self._resolve_high(market_state, price)
        low = self._resolve_low(market_state, price)
        atr = float(getattr(market_state, "atr", 0.0) or (sub.atr_at_entry or 0.0))
        if atr <= 0:
            return []

        tick_size = self._tick_size(market_state)
        stop_mult = self.config.stop_multiplier(regime, default=3.0)
        take_mult = self.config.take_multiplier(regime, default=1.5)
        min_lock = float(self.config.min_lock.get("R_multiple", 1.0))

        prev_best = sub.best_favorable_price or sub.entry_price
        best_changed = False
        if sub.side == "long":
            if high > prev_best:
                sub.best_favorable_price = high
                best_changed = True
            sub.best_adverse_price = min(sub.best_adverse_price or low, low)
        else:
            if low < (sub.best_favorable_price or sub.entry_price):
                sub.best_favorable_price = low
                best_changed = True
            sub.best_adverse_price = max(sub.best_adverse_price or high, high)

        best_favourable = sub.best_favorable_price or prev_best
        initial_stop = self._initial_stop(sub, stop_mult, atr)
        risk_per_unit = self._risk_per_unit(sub, initial_stop)
        if risk_per_unit <= 0:
            return []

        gain = (
            (best_favourable - sub.entry_price)
            if sub.side == "long"
            else (sub.entry_price - best_favourable)
        )
        sub.r_multiple = gain / risk_per_unit

        if sub.r_multiple < min_lock:
            return self._check_triggers(symbol, sub, price)

        spread_guard = self.config.slippage_guard_bps
        spread = float(getattr(market_state, "spread_bps", 0.0) or 0.0)
        latency_ok = bool(getattr(market_state, "latency_ok", True))
        if spread_guard and spread_guard > 0 and spread > spread_guard:
            return self._check_triggers(symbol, sub, price)
        if not latency_ok:
            return self._check_triggers(symbol, sub, price)

        if not sub.take_profit_active:
            stop_before = sub.stop_price
            take_before = sub.take_profit_price
            sub.take_profit_active = True
            changed = self._reprice_levels(sub, best_favourable, atr, stop_mult, take_mult, low, high, tick_size)
            if changed and (stop_before != sub.stop_price or take_before != sub.take_profit_price):
                self._record_update(
                    symbol=symbol,
                    sub=sub,
                    best=best_favourable,
                    atr=atr,
                    stop_before=stop_before,
                    stop_after=sub.stop_price,
                    take_before=take_before,
                    take_after=sub.take_profit_price,
                    now=now,
                    reason="trail_activate",
                )
            self._touch_update_counters(sub, now)
            return []

        should_update = self._should_update(
            sub=sub,
            regime=regime,
            market_state=market_state,
            now=now,
            atr=atr,
            best_changed=best_changed,
            best_favourable=best_favourable,
            prev_best=prev_best,
        )
        if should_update:
            stop_before = sub.stop_price
            take_before = sub.take_profit_price
            changed = self._reprice_levels(sub, best_favourable, atr, stop_mult, take_mult, low, high, tick_size)
            if changed and (stop_before != sub.stop_price or take_before != sub.take_profit_price):
                self._record_update(
                    symbol=symbol,
                    sub=sub,
                    best=best_favourable,
                    atr=atr,
                    stop_before=stop_before,
                    stop_after=sub.stop_price,
                    take_before=take_before,
                    take_after=sub.take_profit_price,
                    now=now,
                    reason="trail_update",
                )
            self._touch_update_counters(sub, now)
        return self._check_triggers(symbol, sub, price)

    def _check_triggers(self, symbol: str, sub: SubPosition, price: float) -> List[TrailingEvent]:
        """Return exit events when price breaches trailing levels."""
        if sub.pending_exit is not None:
            return []

        events: List[TrailingEvent] = []
        if sub.side == "long":
            if sub.take_profit_active and sub.take_profit_price and price >= sub.take_profit_price:
                exit_size = sub.size * 0.5
                remaining = max(sub.size - exit_size, 0.0)
                events.append(
                    TrailingEvent(
                        side="sell",
                        size=exit_size,
                        cross=False,
                        confidence=0.9,
                        target_position=remaining,
                        reason="trailing_ttp",
                        exit_reason="TTP_hit",
                        price=price,
                    )
                )
                sub.pending_exit = "TTP_hit"
                sub.take_profit_price = None
                sub.take_profit_active = False
                self.metrics["exit_trailing_ttp"] += 1
            elif sub.stop_price and price <= sub.stop_price:
                events.append(
                    TrailingEvent(
                        side="sell",
                        size=sub.size,
                        cross=True,
                        confidence=1.0,
                        target_position=0.0,
                        reason="trailing_tsl",
                        exit_reason="TSL_hit",
                        price=price,
                    )
                )
                sub.pending_exit = "TSL_hit"
                self.metrics["exit_trailing_tsl"] += 1
        else:
            if sub.take_profit_active and sub.take_profit_price and price <= sub.take_profit_price:
                exit_size = sub.size * 0.5
                remaining = max(sub.size - exit_size, 0.0)
                events.append(
                    TrailingEvent(
                        side="buy",
                        size=exit_size,
                        cross=False,
                        confidence=0.9,
                        target_position=-remaining,
                        reason="trailing_ttp",
                        exit_reason="TTP_hit",
                        price=price,
                    )
                )
                sub.pending_exit = "TTP_hit"
                sub.take_profit_price = None
                sub.take_profit_active = False
                self.metrics["exit_trailing_ttp"] += 1
            elif sub.stop_price and price >= sub.stop_price:
                events.append(
                    TrailingEvent(
                        side="buy",
                        size=sub.size,
                        cross=True,
                        confidence=1.0,
                        target_position=0.0,
                        reason="trailing_tsl",
                        exit_reason="TSL_hit",
                        price=price,
                    )
                )
                sub.pending_exit = "TSL_hit"
                self.metrics["exit_trailing_tsl"] += 1
        return events

    def _seed_levels(self, sub: SubPosition, atr: float, tick_size: float) -> None:
        self.metrics["seed_attempts"] += 1
        if atr <= 0:
            atr = sub.atr_at_entry or 0.0
        if not atr or atr <= 0:
            logger.error("Cannot seed trailing for %s due to non-positive ATR.", sub.id)
            self.metrics["seed_failed_zero_atr"] += 1
            raise ValueError(f"Cannot seed trailing for {sub.id}: ATR={atr}")

        regime = sub.regime
        stop_mult = self.config.stop_multiplier(regime, default=3.0)
        take_mult = self.config.take_multiplier(regime, default=1.5)
        best = sub.entry_price
        direction = self._direction(sub.side)
        min_ticks = max(1, self.config.min_ticks("long" if sub.side == "long" else "short"))
        min_gap = tick_size * min_ticks

        try:
            raw_tp, raw_sl = sided_levels(best, atr, direction, take_mult, stop_mult)
        except ValueError as exc:
            self.metrics["seed_side_errors"] += 1
            logger.error("Failed to compute sided levels for %s: %s", sub.id, exc)
            raise

        if sub.side == "long":
            stop_price = min(raw_sl, best - min_gap)
            take_price = max(raw_tp, best + min_gap, stop_price + min_gap)
        else:
            stop_price = max(raw_sl, best + min_gap)
            take_price = min(raw_tp, best - min_gap, stop_price - min_gap)

        if not self._validate_levels(sub.id, sub.side, best, stop_price, take_price, min_ticks, tick_size):
            self.metrics["seed_invalid_levels"] += 1
            logger.error(
                "Invalid seed levels for %s (best=%s stop=%s take=%s).",
                sub.id,
                best,
                stop_price,
                take_price,
            )
            raise ValueError(f"Invalid seed levels for {sub.id}: stop={stop_price} take={take_price}")

        sub.stop_price = stop_price
        sub.initial_stop_price = stop_price
        sub.take_profit_price = take_price
        sub.take_profit_active = False
        sub.best_favorable_price = best
        sub.best_adverse_price = best
        sub.pending_exit = None
        self.metrics["seed_attached"] += 1
        now = sub.last_update if isinstance(sub.last_update, datetime) else datetime.now(timezone.utc)
        self._record_update(
            symbol=sub.id,
            sub=sub,
            best=best,
            atr=atr,
            stop_before=None,
            stop_after=sub.stop_price,
            take_before=None,
            take_after=sub.take_profit_price,
            now=sub.last_update,
            reason="seed",
        )

    def _reprice_levels(
        self,
        sub: SubPosition,
        best: float,
        atr: float,
        stop_mult: float,
        take_mult: float,
        low: float,
        high: float,
        tick_size: float,
    ) -> bool:
        direction = self._direction(sub.side)
        min_ticks = max(1, self.config.min_ticks("long" if sub.side == "long" else "short"))
        min_gap = min_ticks * tick_size

        current_stop = sub.stop_price if sub.stop_price is not None else self._initial_stop(sub, stop_mult, atr)

        try:
            target_tp, target_sl = sided_levels(best, atr, direction, take_mult, stop_mult)
        except ValueError as exc:
            self.metrics["modify_side_errors"] += 1
            logger.error("Sided level failure during reprice for %s: %s", sub.id, exc)
            return False

        if sub.side == "long":
            stop_target = min(target_sl, low)
            new_stop = stop_target if current_stop is None else max(current_stop, stop_target)
            new_stop = min(new_stop, low)
            if current_stop is not None and self._ticks_between(new_stop, current_stop, tick_size) < min_ticks:
                new_stop = current_stop
        else:
            stop_target = max(target_sl, high)
            new_stop = stop_target if current_stop is None else min(current_stop, stop_target)
            new_stop = max(new_stop, high)
            if current_stop is not None and self._ticks_between(new_stop, current_stop, tick_size) < min_ticks:
                new_stop = current_stop

        current_take = sub.take_profit_price
        take_target = target_tp
        if sub.side == "long":
            take_target = max(take_target, new_stop + min_gap, best + min_gap)
            new_take = take_target if current_take is None else max(current_take, take_target)
        else:
            take_target = min(take_target, new_stop - min_gap, best - min_gap)
            new_take = take_target if current_take is None else min(current_take, take_target)

        if not self._validate_levels(
            sub.id,
            sub.side,
            best,
            new_stop,
            new_take if sub.take_profit_active else None,
            min_ticks,
            tick_size,
        ):
            logger.error(
                "Skipping trailing update for %s due to invalid levels (best=%s, stop=%s, take=%s)",
                sub.id,
                best,
                new_stop,
                new_take,
            )
            self.metrics["modify_invalid_levels"] += 1
            return False

        if sub.side == "long":
            if sub.stop_price is None or new_stop > sub.stop_price:
                sub.stop_price = new_stop
        else:
            if sub.stop_price is None or new_stop < sub.stop_price:
                sub.stop_price = new_stop

        if sub.take_profit_active:
            sub.take_profit_price = new_take
        else:
            sub.take_profit_price = None
        self.metrics["modify_success"] += 1
        return True

    def _should_update(
        self,
        sub: SubPosition,
        regime: str,
        market_state,
        now: datetime,
        atr: float,
        best_changed: bool,
        best_favourable: float,
        prev_best: float,
    ) -> bool:
        minute_key = int(now.timestamp() // 60)
        if sub.last_update_minute == minute_key and sub.updates_this_minute >= self.config.max_updates_per_min:
            return False

        schedule = self.config.cadence_for(regime)
        regime_lower = regime.lower()
        if regime_lower == "hft":
            snapshots = int(schedule.get("snapshots", 0) or 0)
            if best_changed:
                return True
            if snapshots > 0 and sub.snapshots_since_update < snapshots:
                return False
            return True

        on_close = bool(schedule.get("on_bar_close", False))
        bar_closed = bool(getattr(market_state, "bar_closed", not on_close))
        if on_close and not bar_closed:
            threshold = 0.5 * atr
            improvement = abs(best_favourable - prev_best)
            if improvement >= threshold:
                return True
            return False
        return True

    def _record_update(
        self,
        symbol: str,
        sub: SubPosition,
        best: float,
        atr: float,
        stop_before: Optional[float],
        stop_after: Optional[float],
        take_before: Optional[float],
        take_after: Optional[float],
        now: datetime,
        reason: str,
    ) -> None:
        record = {
            "ts": now.isoformat(),
            "symbol": symbol,
            "sub_id": sub.id,
            "direction": sub.side,
            "regime": sub.regime,
            "best_price": best,
            "atr": atr,
            "stop_before": stop_before,
            "stop_after": stop_after,
            "take_before": take_before,
            "take_after": take_after,
            "r_multiple": sub.r_multiple,
            "reason": reason,
        }
        self.metrics[f"updates_{reason}"] += 1
        logger.debug("Trailing update: %s", json.dumps(record))
        date_key = now.strftime("%Y-%m-%d")
        self._buffer.setdefault(date_key, []).append(record)
        if len(self._buffer[date_key]) >= self._flush_threshold:
            self.flush(date_key)

    def flush(self, date_key: Optional[str] = None) -> None:
        keys: Sequence[str]
        if date_key:
            keys = [date_key]
        else:
            keys = list(self._buffer.keys())
        for key in keys:
            records = self._buffer.get(key)
            if not records:
                continue
            path = self._export_dir / f"{key}.jsonl"
            with path.open("a", encoding="utf-8") as fh:
                for record in records:
                    fh.write(json.dumps(record) + "\n")
            self._buffer[key] = []

    def _touch_update_counters(self, sub: SubPosition, now: datetime) -> None:
        minute_key = int(now.timestamp() // 60)
        if sub.last_update_minute == minute_key:
            sub.updates_this_minute += 1
        else:
            sub.updates_this_minute = 1
            sub.last_update_minute = minute_key
        sub.last_update = now
        sub.snapshots_since_update = 0

    def snapshot_metrics(self, reset: bool = False) -> Dict[str, int]:
        metrics = dict(self.metrics)
        if reset:
            self.metrics = Counter()
        return metrics

    def _close_position(self, symbol: str, side: str) -> None:
        sub = self._get_position(symbol, side)
        if sub:
            self.positions.close(symbol, sub.id)

    def _get_position(self, symbol: str, side: str) -> Optional[SubPosition]:
        for sub in self.positions.iter_symbol(symbol):
            if sub.side == side:
                return sub
        return None

    @staticmethod
    def _weighted_avg(p1: float, q1: float, p2: float, q2: float) -> float:
        if q1 + q2 <= 0:
            return p2
        return (p1 * q1 + p2 * q2) / (q1 + q2)

    def _initial_stop(self, sub: SubPosition, stop_mult: float, atr: float) -> Optional[float]:
        if atr <= 0:
            return sub.stop_price
        if sub.side == "long":
            return sub.entry_price - stop_mult * atr
        return sub.entry_price + stop_mult * atr

    def _initial_take(
        self,
        sub: SubPosition,
        take_mult: float,
        atr: float,
        stop_price: Optional[float],
        tick_size: float,
        side: str,
    ) -> Optional[float]:
        if atr <= 0:
            return None
        direction = 1 if side == "long" else -1
        min_ticks = max(1, self.config.min_ticks("long" if side == "long" else "short"))
        min_gap = min_ticks * tick_size
        base = sub.entry_price
        candidate = base + direction * take_mult * atr

        if side == "long":
            candidate = max(candidate, base + min_gap)
            if stop_price is not None:
                candidate = max(candidate, stop_price + min_gap)
        else:
            candidate = min(candidate, base - min_gap)
            if stop_price is not None:
                candidate = min(candidate, stop_price - min_gap)
        return candidate

    def _risk_per_unit(self, sub: SubPosition, initial_stop: Optional[float]) -> float:
        if initial_stop is None:
            return 0.0
        if sub.side == "long":
            return sub.entry_price - initial_stop
        return initial_stop - sub.entry_price

    @staticmethod
    def _ticks_between(a: float, b: float, tick_size: float) -> float:
        if tick_size <= 0:
            return abs(a - b)
        return abs(a - b) / tick_size

    @staticmethod
    def _tick_size(market_state) -> float:
        return float(getattr(market_state, "tick_size", 1.0) or 1.0)

    @staticmethod
    def _resolve_price(payload: Dict[str, Any], market_state) -> Optional[float]:
        if "price" in payload and payload["price"] is not None:
            return float(payload["price"])
        if hasattr(market_state, "price") and market_state.price is not None:
            return float(market_state.price)
        if hasattr(market_state, "mid_price") and market_state.mid_price is not None:
            return float(market_state.mid_price)
        bid = getattr(market_state, "best_bid", None)
        ask = getattr(market_state, "best_ask", None)
        if bid is not None and ask is not None:
            return (float(bid) + float(ask)) / 2.0
        return None

    @staticmethod
    def _resolve_high(market_state, default: float) -> float:
        return float(getattr(market_state, "high", default) or default)

    @staticmethod
    def _resolve_low(market_state, default: float) -> float:
        return float(getattr(market_state, "low", default) or default)

    @staticmethod
    def _is_fx(symbol: str) -> bool:
        return "/" in symbol or symbol.endswith("USD") or symbol.endswith("USDT")


__all__ = ["TrailingEvent", "TrailingManager"]


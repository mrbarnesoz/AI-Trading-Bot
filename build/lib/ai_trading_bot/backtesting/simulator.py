"""Simple vectorised backtesting engine."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ai_trading_bot.config import BacktestConfig, TrailingConfig
try:  # pragma: no cover - fallback for local usage
    from ai_trading_bot.live.policy.decision import MarketState, Probabilities
    from ai_trading_bot.live.risk.trailing import TrailingManager
    from ai_trading_bot.live.state.positions import PositionManager
except ImportError:  # pragma: no cover - compatibility with legacy layout
    from live.policy.decision import MarketState, Probabilities
    from live.risk.trailing import TrailingManager
    from live.state.positions import PositionManager

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    performance: pd.DataFrame
    summary: Dict[str, float]


def _compute_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    high = data["High"]
    low = data["Low"]
    close = data["Close"]
    prev_close = close.shift(1)
    tr_components = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1)
    true_range = tr_components.max(axis=1)
    atr = true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return atr.bfill()


def run_backtest(
    price_data: pd.DataFrame,
    signals: pd.Series,
    backtest_cfg: BacktestConfig,
    trailing_cfg: Optional[TrailingConfig] = None,
    symbol: str = "SYMBOL",
    regime: str = "intraday",
) -> BacktestResult:
    """Run a backtest using generated signals and risk controls."""
    signals = signals.astype(float)
    signals = _enforce_signal_persistence(
        signals,
        backtest_cfg.signal_persistence_bars,
        backtest_cfg.flip_cooldown_bars,
        backtest_cfg.min_hold_bars,
        backtest_cfg.cooldown_bars_after_exit,
        invert=backtest_cfg.invert_side,
    )
    data = price_data.loc[signals.index].copy()
    data["signal"] = signals

    dataset_token = _dataset_hash(data)
    cfg_token = _config_fingerprint(backtest_cfg)
    start_label = _fmt_ts(data.index[0]) if len(data) else "NA"
    end_label = _fmt_ts(data.index[-1]) if len(data) else "NA"
    logger.info(
        "Backtest telemetry dataset=%s cfg=%s rows=%s start=%s end=%s maker_bps=%.4f taker_bps=%.4f slip_model=half_spread transaction_cost=%.6f",
        dataset_token,
        cfg_token,
        len(data),
        start_label,
        end_label,
        backtest_cfg.maker_fee_bps,
        backtest_cfg.taker_fee_bps,
        backtest_cfg.transaction_cost,
    )

    data["_spread_bps"] = _derive_spread_bps(data, backtest_cfg.fallback_spread_bps)

    use_trailing = trailing_cfg is not None
    trailing_metrics: Dict[str, int] = {}
    signal_entries = 0
    signal_flips = 0
    signal_flats = 0
    maker_event_flags = np.zeros(len(data), dtype=bool)
    taker_event_flags = np.zeros(len(data), dtype=bool)

    if trailing_cfg is None:
        unit_fraction = backtest_cfg.position_capital_fraction
        max_units = backtest_cfg.max_position_units
        max_total_fraction = backtest_cfg.max_total_capital_fraction

        allocation_units = np.clip(np.abs(data["signal"]), 0, max_units)
        requested_fraction = allocation_units * unit_fraction
        effective_fraction = np.minimum(requested_fraction, max_total_fraction)
        allocation_series = np.sign(data["signal"]) * effective_fraction
        allocation_series = pd.Series(allocation_series, index=data.index, name="allocation")

        data["allocation"] = allocation_series
        data["position"] = allocation_series.shift(1).fillna(0.0)
    else:
        positions = PositionManager()
        trailing_cfg_local = copy.deepcopy(trailing_cfg)
        if backtest_cfg.backtest_min_lock is not None:
            trailing_cfg_local.min_lock = dict(trailing_cfg_local.min_lock)
            trailing_cfg_local.min_lock["R_multiple"] = backtest_cfg.backtest_min_lock
        trailing_manager = TrailingManager(trailing_cfg_local, positions)
        logger.warning("POSITION_MANAGER id=%s", hex(id(positions)))
        atr_series = _compute_atr(data)
        tick_size = float(data["Close"].diff().abs().replace(0, np.nan).min() or 1.0)

        index_list = list(data.index)
        maker_event_flags = np.zeros(len(index_list), dtype=bool)
        taker_event_flags = np.zeros(len(index_list), dtype=bool)
        position_units: List[float] = []
        pending_order: Optional[Dict[str, Any]] = None
        max_wait = backtest_cfg.max_entry_wait_bars
        reprice_offset = backtest_cfg.reprice_tick_offset
        allow_cross_after_timeout = backtest_cfg.allow_cross_after_timeout
        partial_min_frac = max(0.0, min(1.0, backtest_cfg.partial_fill_min_qty_frac))

        for bar_pos, idx in enumerate(index_list):
            row = data.loc[idx]
            current_position = positions.net_position(symbol)
            atr_value = float(atr_series.loc[idx] if idx in atr_series.index and not pd.isna(atr_series.loc[idx]) else 0.0)
            if atr_value <= 0:
                atr_value = tick_size
            spread_bps = float(data.at[idx, "_spread_bps"])
            mid_price = float(row["Close"])
            bid_px, ask_px = _best_prices(mid_price, spread_bps)

            # Update existing maker order
            if pending_order is not None:
                side = pending_order["side"]
                if spread_bps > backtest_cfg.cancel_spread_bps:
                    pending_order = None
                else:
                    limit_px = _maker_limit_price(side, bid_px, ask_px, tick_size, reprice_offset)
                    pending_order["limit_price"] = limit_px
                    filled = _price_hit_limit(
                        side,
                        limit_px,
                        float(row["High"]),
                        float(row["Low"]),
                    )
                    if filled:
                        payload = {
                            "side": side,
                            "size": pending_order["size"],
                            "confidence": 0.9,
                            "cross": False,
                            "target_position": pending_order["target_units"],
                        }
                        market_state_fill = MarketState(
                            symbol=symbol,
                            regime=regime,
                            probabilities=Probabilities(0.5, 0.5, 0.0),
                            position=current_position,
                            timestamp=idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx,
                            atr=atr_value,
                            price=limit_px,
                            lot_size=1.0,
                            high=float(row["High"]),
                            low=float(row["Low"]),
                            spread_bps=spread_bps,
                            latency_ok=True,
                            bar_closed=True,
                            tick_size=tick_size,
                        )
                        trailing_manager.record_execution(symbol, payload, market_state_fill)
                        maker_event_flags[bar_pos] = True
                        pending_order = None
                        current_position = positions.net_position(symbol)
                    else:
                        pending_order["bars_waited"] += 1
                        if pending_order["bars_waited"] >= max_wait:
                            if allow_cross_after_timeout:
                                residual_target = pending_order["target_units"]
                                residual_delta = residual_target - current_position
                                if abs(residual_delta) > 1e-9:
                                    cross_side = "buy" if residual_delta > 0 else "sell"
                                    payload = {
                                        "side": cross_side,
                                        "size": abs(residual_delta),
                                        "confidence": 0.9,
                                        "cross": True,
                                        "target_position": residual_target,
                                    }
                                    trailing_manager.record_execution(symbol, payload, market_state)
                                    taker_event_flags[bar_pos] = True
                            pending_order = None

            market_state = MarketState(
                symbol=symbol,
                regime=regime,
                probabilities=Probabilities(0.5, 0.5, 0.0),
                position=current_position,
                timestamp=idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx,
                atr=atr_value,
                price=float(row["Close"]),
                lot_size=1.0,
                high=float(row["High"]),
                low=float(row["Low"]),
                spread_bps=spread_bps,
                latency_ok=True,
                bar_closed=True,
                tick_size=tick_size,
            )

            events = trailing_manager.update(symbol, market_state)
            for event in events:
                taker_event_flags[bar_pos] = True
                trailing_manager.record_execution(symbol, event, market_state)
            current_position = positions.net_position(symbol)

            desired_units = float(data.at[idx, "signal"])
            desired_units = max(
                min(desired_units, backtest_cfg.max_position_units),
                -backtest_cfg.max_position_units,
            )

            if pending_order is not None:
                desired_sign = np.sign(desired_units)
                pending_sign = 1 if pending_order["side"] == "buy" else -1
                if desired_sign == pending_sign:
                    pending_order["target_units"] = desired_units
                    pending_order["size"] = abs(desired_units - current_position)
                    if pending_order["size"] <= 1e-9:
                        pending_order = None
                else:
                    pending_order = None

            delta = desired_units - current_position

            if abs(delta) > 1e-9:
                prev_sign = np.sign(current_position)
                desired_sign = np.sign(desired_units)
                is_flat = abs(desired_units) <= 1e-9
                is_entry = prev_sign == 0 and desired_sign != 0
                is_flip = prev_sign != 0 and desired_sign != 0 and prev_sign != desired_sign
                is_add_same_dir = (
                    desired_sign != 0 and desired_sign == prev_sign and abs(desired_units) > abs(current_position)
                )
                is_new_position = prev_sign == 0 and desired_sign != 0
                is_reduce = desired_sign == prev_sign and abs(desired_units) < abs(current_position)

                if is_flat:
                    signal_flats += 1
                elif is_entry:
                    signal_entries += 1
                elif is_flip:
                    signal_flips += 1

                delta_sign = np.sign(delta)
                side = "buy" if delta_sign > 0 else "sell"

                maker_candidate = backtest_cfg.entry_mode == "maker_post_only" and abs(delta) > 0

                if maker_candidate:
                    if pending_order is None:
                        pending_order = {
                            "side": side,
                            "size": abs(delta),
                            "target_units": desired_units,
                            "bars_waited": 0,
                            "limit_price": None,
                        }
                        limit_px = _maker_limit_price(side, bid_px, ask_px, tick_size, reprice_offset)
                        fill_condition = (
                            spread_bps <= backtest_cfg.cancel_spread_bps
                            and _price_hit_limit(
                                side,
                                limit_px,
                                float(row["High"]),
                                float(row["Low"]),
                            )
                        )
                        if fill_condition:
                            payload = {
                                "side": side,
                                "size": abs(delta),
                                "confidence": 0.9,
                                "cross": False,
                                "target_position": desired_units,
                            }
                            trailing_manager.record_execution(symbol, payload, market_state)
                            maker_event_flags[bar_pos] = True
                            pending_order = None
                        elif spread_bps > backtest_cfg.cancel_spread_bps:
                            pending_order = None
                        else:
                            pending_order["limit_price"] = limit_px
                else:
                    payload = {
                        "side": "buy" if delta > 0 else "sell",
                        "size": abs(delta),
                        "confidence": 0.9,
                        "cross": True,
                        "target_position": desired_units,
                    }
                    trailing_manager.record_execution(symbol, payload, market_state)
                    taker_event_flags[bar_pos] = True

            position_units.append(positions.net_position(symbol))

        trailing_manager.flush()
        trailing_metrics = trailing_manager.snapshot_metrics(reset=True)
        if trailing_metrics:
            logger.info("Trailing metrics snapshot: %s", trailing_metrics)

        unit_fraction = backtest_cfg.position_capital_fraction
        max_units = backtest_cfg.max_position_units
        max_total_fraction = backtest_cfg.max_total_capital_fraction

        position_series = pd.Series(position_units, index=data.index)
        allocation_units = np.clip(position_series.abs(), 0, max_units)
        requested_fraction = allocation_units * unit_fraction
        effective_fraction = np.minimum(requested_fraction, max_total_fraction)
        allocation_series = np.sign(position_series) * effective_fraction
        allocation_series = pd.Series(allocation_series, index=data.index, name="allocation")

        data["signal"] = position_series
        data["allocation"] = allocation_series
        data["position"] = allocation_series.shift(1).fillna(0.0)

    data["returns"] = data["Close"].pct_change().fillna(0.0)
    allocation = data["allocation"].astype(float)
    trade_change = allocation.diff().fillna(allocation)
    prev_allocation = allocation.shift(1).fillna(0.0)
    trade_mask = trade_change.abs() > 1e-9
    trade_mask_np = trade_mask.to_numpy(dtype=bool)
    maker_mask = trade_mask_np & maker_event_flags[: len(trade_mask_np)] & ~taker_event_flags[: len(trade_mask_np)]
    taker_mask = trade_mask_np & (~maker_mask | taker_event_flags[: len(trade_mask_np)])
    trade_abs = trade_change.abs().to_numpy()

    base_spread = data["_spread_bps"].to_numpy()
    configured_slip = max(0.0, backtest_cfg.slippage_bps)
    taker_slippage = np.maximum(base_spread * 0.5, 0.0)
    slippage_bps = np.zeros_like(base_spread, dtype=float)
    slippage_bps[taker_mask] = configured_slip + taker_slippage[taker_mask]
    slippage_bps[maker_mask] = configured_slip
    min_slip = max(backtest_cfg.min_slippage_bps, configured_slip)
    slippage_bps = np.where(trade_mask_np, np.maximum(slippage_bps, min_slip), 0.0)

    extra_cost = backtest_cfg.transaction_cost * trade_abs
    slippage_cost = (slippage_bps / 10000.0) * trade_abs + extra_cost

    maker_fee_rate = backtest_cfg.maker_fee_bps / 10000.0
    taker_fee_rate = backtest_cfg.taker_fee_bps / 10000.0
    per_side_rate = backtest_cfg.per_side_fee_bps / 10000.0
    fee_rates = np.where(taker_mask, taker_fee_rate, maker_fee_rate) + per_side_rate
    fee_cost = fee_rates * trade_abs

    data["slippage_cost"] = pd.Series(slippage_cost, index=data.index)
    data["execution_fee"] = pd.Series(fee_cost, index=data.index)
    data["trade_slippage_bps"] = pd.Series(
        np.where(trade_mask_np, slippage_bps, 0.0),
        index=data.index,
    )
    trading_costs = data["slippage_cost"] + data["execution_fee"]

    taker_count = int(taker_mask.sum())
    maker_count = int(maker_mask.sum())
    weighted_idx = trade_mask_np & (trade_abs > 0)
    if np.any(weighted_idx):
        avg_slippage_bps = float(np.average(slippage_bps[weighted_idx], weights=trade_abs[weighted_idx]))
    else:
        avg_slippage_bps = 0.0

    data["strategy_return"] = data["position"] * data["returns"] - trading_costs

    funding_col = backtest_cfg.funding_rate_column
    if backtest_cfg.simulate_funding and funding_col and funding_col in data.columns:
        funding_series = data[funding_col].fillna(0.0)
        funding_cost = funding_series * data["position"].shift(1).fillna(0.0)
        data["strategy_return"] -= funding_cost
        data["funding_cost"] = funding_cost
    else:
        data["funding_cost"] = 0.0

    data["equity_curve"] = (1 + data["strategy_return"]).cumprod() * backtest_cfg.initial_capital
    data["capital_in_position"] = data["position"].abs() * backtest_cfg.initial_capital

    equity_curve = data["equity_curve"]
    total_return = (equity_curve.iloc[-1] / backtest_cfg.initial_capital) - 1
    ann_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / len(data)) - 1 if len(data) > 0 else 0.0
    ann_volatility = data["strategy_return"].std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (
        (ann_return - backtest_cfg.risk_free_rate)
        / ann_volatility
        if ann_volatility and ann_volatility > 0
        else 0.0
    )
    win_rate = (data["strategy_return"] > 0).sum() / max(len(data), 1)

    trade_events = trade_change.abs() > 1e-9
    trades_count = int(trade_events.sum())
    position_sequence = data["position"]
    non_zero = position_sequence != 0
    if non_zero.any():
        run_keys = non_zero.ne(non_zero.shift()).cumsum()
        holding_lengths = position_sequence[non_zero].groupby(run_keys[non_zero]).size()
        avg_holding_bars = float(holding_lengths.mean())
    else:
        avg_holding_bars = 0.0

    gross_exposure_avg = float(data["allocation"].abs().mean())
    bars_per_day = _estimate_bars_per_day(data.index)
    trades_per_day = trades_count / max(1.0, len(data) / max(bars_per_day, 1e-6))

    trade_gross_pnls: List[float] = []
    trade_net_pnls: List[float] = []
    gross_returns_series = data["position"] * data["returns"]
    if non_zero.any():
        groups: Dict[int, List[pd.Timestamp]] = {}
        for ts, key in zip(position_sequence.index[non_zero], run_keys[non_zero]):
            groups.setdefault(int(key), []).append(ts)
        for idxs in groups.values():
            window = data.loc[idxs]
            gross_pnl = float(gross_returns_series.loc[idxs].sum())
            net_pnl = float(window["strategy_return"].sum())
            trade_gross_pnls.append(gross_pnl)
            trade_net_pnls.append(net_pnl)

    expectancy_no_costs = float(np.mean(trade_gross_pnls)) if trade_gross_pnls else 0.0
    expectancy_after_costs = float(np.mean(trade_net_pnls)) if trade_net_pnls else 0.0
    wins = [p for p in trade_net_pnls if p > 0]
    losses = [-p for p in trade_net_pnls if p < 0]
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    payoff = avg_win / avg_loss if avg_win and avg_loss else 0.0

    rolling_window = max(1, min(len(data), 30))
    rolling_returns = data["strategy_return"].rolling(rolling_window).sum()
    rolling_vol = data["strategy_return"].rolling(rolling_window).std()
    rolling_sharpe = (rolling_returns / (rolling_vol + 1e-12)) * np.sqrt(TRADING_DAYS_PER_YEAR)
    data["rolling_sharpe"] = rolling_sharpe.fillna(0.0)

    summary = {
        "total_return": float(total_return),
        "annualised_return": float(ann_return),
        "annualised_volatility": float(ann_volatility),
        "sharpe_ratio": float(sharpe),
        "win_rate": float(win_rate),
        "max_drawdown": float(_max_drawdown(equity_curve)),
        "final_equity": float(equity_curve.iloc[-1]),
        "average_capital_fraction": float(data["position"].abs().mean()),
        "total_fees_paid": float(data["execution_fee"].sum()),
        "total_slippage_cost": float(data["slippage_cost"].sum()),
        "total_funding_paid": float(data["funding_cost"].sum()),
        "trades_count": trades_count,
        "avg_holding_bars": float(avg_holding_bars),
        "gross_exposure_avg": gross_exposure_avg,
        "avg_slippage_bps": float(avg_slippage_bps),
        "maker_trades": maker_count,
        "taker_trades": taker_count,
        "maker_fill_ratio": float(maker_count / trades_count) if trades_count else 0.0,
        "signal_entries": int(signal_entries),
        "signal_flips": int(signal_flips),
        "signal_flats": int(signal_flats),
        "bars_per_day_est": float(bars_per_day),
        "trades_per_day": float(trades_per_day),
        "expectancy_no_costs": expectancy_no_costs,
        "expectancy_after_costs": expectancy_after_costs,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff,
        "rolling_sharpe_last": float(data["rolling_sharpe"].iloc[-1]) if len(data) else 0.0,
    }
    if trailing_metrics:
        for key, value in trailing_metrics.items():
            summary[f"trail_{key}"] = float(value) if isinstance(value, (int, float)) else value
    if trades_count < 50:
        logger.warning("Backtest generated only %s trades. Results may be statistically fragile.", trades_count)
    logger.info("Backtest summary: %s", summary)
    if "_spread_bps" in data.columns:
        data = data.drop(columns=["_spread_bps"])
    return BacktestResult(
        equity_curve=data["equity_curve"],
        performance=data[
            [
                "signal",
                "allocation",
                "position",
                "returns",
                "strategy_return",
                "execution_fee",
                "slippage_cost",
                "trade_slippage_bps",
                "funding_cost",
                "equity_curve",
                "capital_in_position",
            ]
        ],
        summary=summary,
    )


def _max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return drawdown.min()


def _dataset_hash(df: pd.DataFrame) -> str:
    if df.empty:
        return "empty"
    cols = [col for col in ["Open", "High", "Low", "Close", "Volume"] if col in df.columns]
    if not cols:
        cols = list(df.columns)
    hashed = pd.util.hash_pandas_object(df[cols], index=True).values
    return hashlib.sha1(hashed.tobytes()).hexdigest()[:12]


def _config_fingerprint(cfg: BacktestConfig) -> str:
    payload = json.dumps(asdict(cfg), sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _fmt_ts(value) -> str:
    if value is None:
        return "NA"
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _derive_spread_bps(df: pd.DataFrame, fallback_bps: float) -> pd.Series:
    fallback_bps = max(0.0, float(fallback_bps))
    if "spread_bps" in df.columns:
        return df["spread_bps"].astype(float).abs().replace(0.0, fallback_bps).fillna(fallback_bps)

    approx: pd.Series
    if {"best_bid", "best_ask"}.issubset(df.columns):
        spread = (df["best_ask"] - df["best_bid"]).abs()
        mid = ((df["best_ask"] + df["best_bid"]) / 2).replace(0.0, np.nan)
        approx = (spread / mid).replace([np.inf, -np.inf], np.nan) * 10000.0
    elif {"High", "Low", "Close"}.issubset(df.columns):
        close = df["Close"].replace(0.0, np.nan)
        hi_lo = (df["High"] - df["Low"]).abs()
        approx = (hi_lo / close).replace([np.inf, -np.inf], np.nan) * 10000.0
    else:
        approx = pd.Series(float(fallback_bps), index=df.index, dtype=float)

    approx = approx.fillna(fallback_bps)
    lower = max(fallback_bps * 0.5, 0.0)
    upper = max(fallback_bps * 3.0, fallback_bps + 15.0)
    return approx.clip(lower=lower, upper=upper)


def _best_prices(mid_price: float, spread_bps: float) -> tuple[float, float]:
    return mid_price, mid_price


def _maker_limit_price(side: str, bid: float, ask: float, tick_size: float, offset_ticks: float) -> float:
    if side == "buy":
        return max(bid, 0.0)
    return ask


def _price_hit_limit(side: str, limit_price: float, bar_high: float, bar_low: float) -> bool:
    if side == "buy":
        return bar_low <= limit_price
    return bar_high >= limit_price
def _enforce_signal_persistence(
    signals: pd.Series,
    min_bars: int,
    cooldown_bars: int,
    min_hold_bars: int,
    cooldown_after_exit: int,
    *,
    invert: bool = False,
) -> pd.Series:
    """Require signals to persist before flips and enforce hold/cooldown windows."""
    min_bars = max(1, int(min_bars))
    cooldown_bars = max(0, int(cooldown_bars))
    min_hold_bars = max(0, int(min_hold_bars))
    cooldown_after_exit = max(0, int(cooldown_after_exit))

    values = signals.to_numpy(dtype=float)
    result = np.zeros_like(values, dtype=float)

    current_value = 0.0
    pending_sign = 0.0
    pending_value = 0.0
    persistence = 0
    cooldown = 0
    hold_counter = 0

    for idx, requested in enumerate(values):
        req_sign = np.sign(requested) if abs(requested) > 1e-9 else 0.0

        if req_sign == 0.0:
            if current_value != 0.0 and hold_counter < min_hold_bars:
                hold_counter += 1
                result[idx] = current_value
                continue
            current_value = 0.0
            pending_sign = 0.0
            pending_value = 0.0
            persistence = 0
            if cooldown <= 0:
                cooldown = cooldown_after_exit
            else:
                cooldown -= 1
            hold_counter = 0
            result[idx] = 0.0
            continue

        current_sign = np.sign(current_value)
        if req_sign == current_sign and current_sign != 0.0:
            current_value = requested
            pending_sign = 0.0
            pending_value = 0.0
            persistence = 0
            if hold_counter < min_hold_bars:
                hold_counter += 1
            if cooldown > 0:
                cooldown -= 1
            result[idx] = current_value
            continue

        if hold_counter < min_hold_bars and current_value != 0.0:
            hold_counter += 1
            result[idx] = current_value
            continue

        if pending_sign != req_sign:
            pending_sign = req_sign
            pending_value = requested
            persistence = 1
        else:
            persistence += 1
            pending_value = requested

        if persistence >= min_bars and cooldown <= 0:
            current_value = pending_value
            pending_sign = 0.0
            pending_value = 0.0
            persistence = 0
            cooldown = cooldown_bars
            hold_counter = 1
        else:
            if cooldown > 0:
                cooldown -= 1

        result[idx] = current_value

    if invert:
        result = -result

    return pd.Series(result, index=signals.index, dtype=float)


def _estimate_bars_per_day(index: pd.Index) -> float:
    if len(index) < 2:
        return 1.0
    try:
        deltas = pd.Series(index).diff().dropna()
    except Exception:
        return 1.0
    if deltas.empty:
        return 1.0
    seconds = deltas.dt.total_seconds().median()
    if not seconds or seconds <= 0:
        return 1.0
    return max(1.0, 86400.0 / seconds)

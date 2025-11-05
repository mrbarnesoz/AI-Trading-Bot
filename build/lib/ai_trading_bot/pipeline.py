"""High-level orchestration for the AI trading bot pipeline."""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ai_trading_bot.backtesting.simulator import BacktestResult, run_backtest as simulate_backtest
from ai_trading_bot.config import AppConfig, FilterConfig, load_config
from ai_trading_bot.data_pipeline import prepare_dataset as build_dataset
from ai_trading_bot.decision.mode_selector import ModeDecision, select_mode
from ai_trading_bot.experiments.atr_defaults import load_top_candidate
from ai_trading_bot.meta.select import MetaStrategySelector
from ai_trading_bot.models.predictor import train_model
from ai_trading_bot.risk.guardrails import validate_app_config
from ai_trading_bot.orchestrator.multi_strategy import orchestrate_multi_strategy
from ai_trading_bot.regime.detector import summarise_regime
from ai_trading_bot.strategies.ml_strategy import StrategyOutput, generate_signals
from ai_trading_bot.signals.post import apply_hysteresis
from ai_trading_bot.utils.logging import configure_logging

logger = logging.getLogger(__name__)
_META_SELECTOR = MetaStrategySelector()


def _apply_filters(strategy_output: StrategyOutput, engineered: pd.DataFrame, config: AppConfig) -> StrategyOutput:
    filters: FilterConfig = getattr(config, "filters", FilterConfig())
    signals = strategy_output.signals.astype(float).copy()
    raw_components = getattr(strategy_output, "components", {}) or {}
    component_signals: Dict[str, pd.Series] = {}
    for name, series in raw_components.items():
        component_signals[name] = pd.Series(series, index=signals.index, dtype=float).fillna(0.0)

    component_probabilities = getattr(strategy_output, "component_probabilities", {}) or {}
    aligned_component_probs: Dict[str, pd.Series] = {}
    for name, series in component_probabilities.items():
        aligned_component_probs[name] = pd.Series(series, index=signals.index, dtype=float).fillna(0.5)

    if not filters.allow_long:
        signals[signals > 0] = 0.0
    if not filters.allow_short:
        signals[signals < 0] = 0.0

    mask = pd.Series(True, index=signals.index)

    if filters.min_atr_frac:
        atr_col = None
        for candidate in ("atr_14", "ATR", "atr"):
            if candidate in engineered.columns:
                atr_col = engineered[candidate]
                break
        if atr_col is not None and "Close" in engineered.columns:
            atr_frac = (atr_col / engineered["Close"]).reindex(signals.index).astype(float)
            mask &= atr_frac >= float(filters.min_atr_frac)

    if filters.min_trend_slope is not None:
        if "trend_slope" in engineered.columns:
            slope_series = engineered["trend_slope"]
            lookback = max(1, int(filters.trend_slope_lookback or 1))
            if lookback > 1:
                slope_series = slope_series.rolling(window=lookback, min_periods=1).mean()
            slope_series = slope_series.reindex(signals.index)
            mask &= slope_series.abs() >= float(filters.min_trend_slope)

    if filters.min_confidence:
        prob_series = getattr(strategy_output, "probabilities", None)
        if prob_series is not None:
            probs = prob_series.reindex(signals.index).astype(float).fillna(0.5)
            conf_mask = (probs - 0.5).abs() >= float(filters.min_confidence)
            mask &= conf_mask
        else:
            logger.debug("Skipping min_confidence filter: probability series missing.")

    if filters.min_adx is not None:
        adx_series = None
        for candidate in ("adx", "ADX", "trend_strength"):
            if candidate in engineered.columns:
                adx_series = engineered[candidate].reindex(signals.index)
                break
        if adx_series is not None:
            mask &= adx_series >= float(filters.min_adx)
        else:
            logger.debug("Skipping min_adx filter: ADX/trend column missing.")

    signals[~mask.fillna(False)] = 0.0
    masked = mask.reindex(signals.index).fillna(False)
    if component_signals:
        for name, series in component_signals.items():
            series.loc[~masked] = 0.0

    hysteresis_k = getattr(filters, "hysteresis_k_atr", None)
    if hysteresis_k and hysteresis_k > 0:
        close_series = None
        for price_col in ("Close", "close", "close_price", "last_price"):
            if price_col in engineered.columns:
                close_series = engineered[price_col]
                break
        atr_series = None
        for candidate in ("atr_14", "ATR", "atr"):
            if candidate in engineered.columns:
                atr_series = engineered[candidate]
                break
        if close_series is not None and atr_series is not None:
            signals = apply_hysteresis(
                signals,
                close_series.reindex(signals.index),
                atr_series.reindex(signals.index),
                k=float(hysteresis_k),
            )
            if "trend" in component_signals:
                component_signals["trend"] = pd.Series(
                    np.sign(signals.values), index=signals.index, dtype=float
                )
        else:
            logger.debug(
                "Skipping hysteresis application: required Close/ATR columns missing (%s/%s).",
                "Close" if close_series is not None else "missing",
                "ATR" if atr_series is not None else "missing",
            )

    strategy_output.signals = signals
    if component_signals:
        strategy_output.components = component_signals
    if aligned_component_probs:
        strategy_output.component_probabilities = aligned_component_probs
    return strategy_output


def _select_mode_config(config: AppConfig, force_download: bool = False) -> tuple[AppConfig, ModeDecision | None]:
    if getattr(config, "sweep_mode", False):
        return config, None
    modes = getattr(config, "modes", None)
    if not modes:
        return config, None
    try:
        decision = select_mode(config, force_download=force_download)
    except Exception as exc:  # pragma: no cover - defensive guard for selection errors
        logger.warning("Mode selection failed; falling back to base configuration: %s", exc)
        return config, None

    active = deepcopy(config)
    active.data = decision.data_config
    active.pipeline = decision.pipeline_config
    active.backtest = decision.backtest_config
    return active, decision


def _apply_risk_overrides(config: AppConfig) -> Dict[str, float]:
    if getattr(config, "sweep_mode", False):
        return {}
    candidate = load_top_candidate()
    if not candidate:
        return {}

    applied: Dict[str, float] = {}

    atr_mult = candidate.get("atr_mult")
    if atr_mult is not None:
        config.filters.min_atr_frac = float(atr_mult)
        applied["min_atr_frac"] = float(atr_mult)

    hyst_k = candidate.get("hyst_k")
    if hyst_k is not None:
        config.filters.hysteresis_k_atr = float(hyst_k)
        applied["hysteresis_k_atr"] = float(hyst_k)

    min_hold = candidate.get("min_hold")
    if min_hold is not None:
        config.backtest.min_hold_bars = int(min_hold)
        applied["min_hold_bars"] = float(config.backtest.min_hold_bars)

    cap_fraction = candidate.get("cap_fraction")
    if cap_fraction is not None:
        config.backtest.position_capital_fraction = float(cap_fraction)
        config.backtest.max_total_capital_fraction = min(0.5, float(cap_fraction) * 3.0)
        applied["position_capital_fraction"] = float(cap_fraction)
        applied["max_total_capital_fraction"] = float(config.backtest.max_total_capital_fraction)

    cancel_spread = candidate.get("cancel_spread_bps")
    if cancel_spread is not None:
        config.backtest.cancel_spread_bps = float(cancel_spread)
        applied["cancel_spread_bps"] = float(cancel_spread)

    trail_k = candidate.get("trail_k")
    ttp_k = candidate.get("ttp_k")
    if trail_k and ttp_k:
        config.risk.trailing.enabled = True
        config.risk.trailing.k_atr.setdefault("stop", {})["swing"] = float(trail_k)
        config.risk.trailing.k_atr.setdefault("take", {})["swing"] = float(ttp_k)
        applied["trail_k"] = float(trail_k)
        applied["ttp_k"] = float(ttp_k)

    return applied


def _enforce_backtest_gates(summary: dict, config: AppConfig) -> None:
    if not config.backtest.enforce_gates:
        return
    trades_count = summary.get("trades_count", 0)
    if trades_count < config.backtest.min_gate_trades:
        raise AssertionError(
            f"sample gate fail: trades_count={trades_count} < {config.backtest.min_gate_trades}"
        )
    maker_ratio = summary.get("maker_fill_ratio", 0.0)
    if maker_ratio < config.backtest.maker_ratio_target:
        raise AssertionError(
            f"maker gate fail: ratio={maker_ratio:.2f} < {config.backtest.maker_ratio_target:.2f}"
        )
    avg_slip = summary.get("avg_slippage_bps", 0.0)
    if avg_slip > config.backtest.slippage_bps_target:
        raise AssertionError(
            f"slippage gate fail: avg_slippage_bps={avg_slip:.2f} > {config.backtest.slippage_bps_target:.2f}"
        )
    trades_per_day = summary.get("trades_per_day", 0.0)
    max_trades_allowed = config.backtest.trades_per_day_target
    if trades_per_day > max_trades_allowed:
        raise AssertionError(
            f"frequency gate fail: trades_per_day={trades_per_day:.2f} > {max_trades_allowed:.2f}"
        )
    max_dd = summary.get("max_drawdown", 0.0)
    if max_dd < config.backtest.max_drawdown_floor:
        raise AssertionError(
            f"risk gate fail: max_drawdown={max_dd:.3f} < {config.backtest.max_drawdown_floor:.2f}"
        )
    cap_fraction = summary.get("average_capital_fraction", 0.0)
    if cap_fraction > config.backtest.max_total_capital_fraction + 1e-6:
        raise AssertionError(
            f"capital gate fail: average_capital_fraction={cap_fraction:.4f} > "
            f"{config.backtest.max_total_capital_fraction}"
        )
    calc_sharpe = summary.get("calc_sharpe", 0.0)
    if calc_sharpe <= 0.10:
        raise AssertionError(f"sharpe gate fail: calc_sharpe={calc_sharpe:.6f} <= 0.10")


def prepare_dataset(config: AppConfig, force_download: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch raw data, engineer indicators, and build training/backtest dataset."""
    config_dict = config.to_nested_dict()
    return build_dataset(config_dict, force_download=force_download)


def train(config_path: Path | str = "config.yaml", force_download: bool = False, verbose: bool = False) -> dict:
    """Train the machine learning model."""
    log_level = logging.DEBUG if verbose else logging.INFO
    configure_logging(log_level)
    config = load_config(config_path)
    active_config, mode_decision = _select_mode_config(config, force_download=force_download)
    validate_app_config(active_config, stage="backtest")
    validate_app_config(active_config, stage="train")
    risk_overrides = _apply_risk_overrides(active_config)
    _, engineered = prepare_dataset(active_config, force_download=force_download)
    metrics = train_model(engineered, active_config.model, active_config.pipeline)
    metrics.update(
        {
            "symbol": active_config.data.symbol,
            "interval": active_config.data.interval,
            "rows": len(engineered),
        }
    )
    if mode_decision is not None:
        metrics.update(
            {
                "mode": mode_decision.mode.name,
                "mode_interval": mode_decision.data_config.interval,
                "mode_score": float(mode_decision.score),
            }
        )
    if risk_overrides:
        metrics["risk_overrides"] = risk_overrides
    logger.info("Training metrics: %s", metrics)
    return metrics


def execute_backtest(
    config: AppConfig,
    force_download: bool = False,
    long_threshold: Optional[float] = None,
    short_threshold: Optional[float] = None,
    verbose: bool = False,
    enforce_gates: bool = True,
) -> Tuple[StrategyOutput, BacktestResult, dict]:
    active_config, mode_decision = _select_mode_config(config, force_download=force_download)
    risk_overrides = _apply_risk_overrides(active_config)

    price_frame, engineered = prepare_dataset(active_config, force_download=force_download)
    long_thr = long_threshold if long_threshold is not None else active_config.pipeline.long_threshold
    short_thr = short_threshold if short_threshold is not None else active_config.pipeline.short_threshold

    post_cfg = getattr(active_config, "signals", None)
    if post_cfg and getattr(post_cfg, "post", None):
        if getattr(post_cfg.post, "min_hold_bars", 0):
            active_config.backtest.min_hold_bars = int(post_cfg.post.min_hold_bars)
        if getattr(post_cfg.post, "hysteresis_k_atr", 0):
            active_config.filters.hysteresis_k_atr = float(post_cfg.post.hysteresis_k_atr)

    strategy_output = generate_signals(
        engineered,
        active_config.model,
        active_config.pipeline,
        probability_long_threshold=long_thr,
        probability_short_threshold=short_thr,
    )
    strategy_output, regime_series, strategy_slices = orchestrate_multi_strategy(engineered, strategy_output)
    strategy_output = _apply_filters(strategy_output, engineered, active_config)
    strategy_output = _apply_meta_selector(strategy_output, engineered, active_config)

    regime_name = _infer_regime(active_config)
    result = simulate_backtest(
        price_frame,
        strategy_output.signals,
        active_config.backtest,
        trailing_cfg=active_config.risk.trailing,
        symbol=active_config.data.symbol,
        regime=regime_name,
    )
    metadata = {
        "symbol": active_config.data.symbol,
        "interval": active_config.data.interval,
        "rows": int(len(engineered)),
    }
    trend_count, breakout_count, range_count = summarise_regime(regime_series)
    metadata.update({
        "multi_strategy_regimes": {
            "trend": trend_count,
            "breakout": breakout_count,
            "range": range_count,
        },
        "strategy_modules": {name: slice_.diagnostics for name, slice_ in strategy_slices.items()},
    })
    metadata["initial_capital"] = float(active_config.backtest.initial_capital)
    if mode_decision is not None:
        metadata["mode"] = mode_decision.mode.name
        metadata["mode_interval"] = mode_decision.data_config.interval
        metadata["mode_score"] = float(mode_decision.score)
        metadata["mode_metrics"] = {k: float(v) for k, v in mode_decision.metrics.items()}
    if risk_overrides:
        metadata["risk_overrides"] = risk_overrides

    selector_stats = getattr(strategy_output, "selector_stats", {})
    if selector_stats:
        metadata["selector_stats"] = selector_stats
        for key, value in selector_stats.items():
            try:
                result.summary[f"selector_{key}"] = float(value)
            except (TypeError, ValueError):
                result.summary[f"selector_{key}"] = value

    ann_ret = result.summary.get("annualised_return", 0.0)
    ann_vol = result.summary.get("annualised_volatility", 0.0) or 0.0
    vol_denom = max(abs(ann_vol), 1e-12)
    calc_sharpe = (ann_ret - 0.0) / vol_denom
    result.summary.setdefault("regime_trend_count", trend_count)
    result.summary.setdefault("regime_breakout_count", breakout_count)
    result.summary.setdefault("regime_range_count", range_count)

    result.summary["calc_sharpe"] = calc_sharpe

    if verbose:
        logger.debug("Backtest metadata: %s", metadata)
        logger.debug("Signal preview:\n%s", strategy_output.signals.tail())
        logger.debug(
            "Diagnostics: trades=%s maker_ratio=%.3f avg_slip_bps=%.3f calc_sharpe=%.6f",
            result.summary.get("trades_count"),
            result.summary.get("maker_fill_ratio"),
            result.summary.get("avg_slippage_bps"),
            calc_sharpe,
        )

    if enforce_gates:
        _enforce_backtest_gates(result.summary, active_config)

    if mode_decision is not None:
        result.summary["mode"] = mode_decision.mode.name
        result.summary["mode_score"] = float(mode_decision.score)
        for key, value in mode_decision.metrics.items():
            result.summary[f"mode_{key}"] = float(value)
    if risk_overrides:
        for key, value in risk_overrides.items():
            result.summary[f"risk_{key}"] = float(value)

    return strategy_output, result, metadata


def backtest(
    config_path: Path | str = "config.yaml",
    force_download: bool = False,
    long_threshold: Optional[float] = None,
    short_threshold: Optional[float] = None,
    verbose: bool = False,
) -> tuple[StrategyOutput, BacktestResult, dict]:
    """Run the full pipeline through to backtesting."""
    configure_logging(logging.DEBUG if verbose else logging.INFO)
    config = load_config(config_path)
    return execute_backtest(
        config,
        force_download=force_download,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        verbose=verbose,
        enforce_gates=True,
    )


def run_backtest(
    config_path: Path | str = "config.yaml",
    force_download: bool = False,
    long_threshold: Optional[float] = None,
    short_threshold: Optional[float] = None,
    verbose: bool = False,
) -> tuple[StrategyOutput, BacktestResult, dict]:
    """Convenience wrapper to align CLI expectations."""
    return backtest(
        config_path=config_path,
        force_download=force_download,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _infer_regime(config: AppConfig) -> str:
    interval = (config.data.interval or "").lower()
    for mode in getattr(config, 'modes', []):
        if getattr(mode, 'interval', '').lower() == interval:
            return mode.name
    mapping = {"1m": "scalping", "3m": "scalping", "5m": "intraday", "10m": "intraday", "15m": "intraday", "30m": "intraday", "1h": "swing", "4h": "swing", "1d": "swing"}
    return mapping.get(interval, config.pipeline.__dict__.get('mode', 'intraday'))



def _apply_meta_selector(
    strategy_output: StrategyOutput,
    engineered: pd.DataFrame,
    config: AppConfig,
) -> StrategyOutput:
    selector = _META_SELECTOR
    selector.set_cache_enabled(not getattr(config, "sweep_mode", False))
    previous_mode = getattr(selector, "mode", "live")
    selector.set_mode("backtest")
    try:
        regime_name = _infer_regime(config)
        regime = selector._normalise_regime(regime_name)  # type: ignore[attr-defined]
        base_fraction = config.backtest.position_capital_fraction
        max_units = config.backtest.max_position_units

        sized: list[float] = []
        records: list[dict] = []
        index = strategy_output.signals.index

        base_prob_series = strategy_output.probabilities.reindex(index, fill_value=0.5).astype(float)
        component_signals = {
            name: pd.Series(series, index=index, dtype=float).fillna(0.0)
            for name, series in getattr(strategy_output, "components", {}).items()
        }
        component_probabilities = {
            name: pd.Series(series, index=index, dtype=float).fillna(0.5)
            for name, series in getattr(strategy_output, "component_probabilities", {}).items()
        }
        trend_prob_series = component_probabilities.get("trend", base_prob_series)

        for pos, timestamp in enumerate(index):
            row = engineered.loc[timestamp] if timestamp in engineered.index else pd.Series(dtype=float)
            if not isinstance(row, pd.Series):
                row = pd.Series(dtype=float)
            else:
                row = row.copy()

            for name, series in component_signals.items():
                row[f"signal_{name}"] = float(series.iloc[pos])
            for name, series in component_probabilities.items():
                row[f"prob_{name}"] = float(series.iloc[pos])

            ts = pd.Timestamp(timestamp) if not isinstance(timestamp, pd.Timestamp) else timestamp
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            context = selector.create_context_from_row(
                timestamp=ts,
                symbol=config.data.symbol,
                regime=regime,
                row=row,
                probability=float(trend_prob_series.iloc[pos]),
                health={"simulated": 1.0, "latency_ok": 1.0, "liq_buffer_atr": 999.0},
            )
            decision = selector.select_strategy(context)
            records.append(decision.to_record())
            if decision.direction == "long":
                units = min(decision.size_frac / base_fraction, max_units)
                sized.append(float(units))
            elif decision.direction == "short":
                units = min(decision.size_frac / base_fraction, max_units)
                sized.append(float(-units))
            else:
                sized.append(0.0)

        selector.flush()
        strategy_output.signals = pd.Series(sized, index=index, dtype=float)
        strategy_output.decisions = pd.DataFrame(records)
        stats = selector.snapshot_stats(reset=True)
        setattr(strategy_output, "selector_stats", stats)
        if stats:
            logger.debug("Meta selector blockers: %s", stats)
        return strategy_output
    finally:
        selector.set_mode(previous_mode)


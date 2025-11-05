"""Validation utilities enforcing risk guardrails on configuration."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Tuple

from ai_trading_bot.config import (
    AppConfig,
    BacktestConfig,
    FilterConfig,
    LatencyArbConfig,
)


class GuardrailViolation(ValueError):
    """Raised when guardrails detect unsafe configuration values."""

    def __init__(self, violations: List[str]) -> None:
        message = "Guardrail validation failed:\n- " + "\n- ".join(violations)
        super().__init__(message)
        self.violations = violations


def _range_check(name: str, value: float, lower: float, upper: float, inclusive: Tuple[bool, bool] = (True, True)) -> str | None:
    if math.isnan(value):
        return f"{name} must be a number."
    lower_bad = value < lower if inclusive[0] else value <= lower
    upper_bad = value > upper if inclusive[1] else value >= upper
    if lower_bad or upper_bad:
        bounds = f"[{lower}, {upper}]" if all(inclusive) else f"{'(' if not inclusive[0] else '['}{lower}, {upper}{')' if not inclusive[1] else ']'}"
        return f"{name}={value:.6g} outside safe range {bounds}."
    return None


def _validate_backtest(backtest: BacktestConfig) -> List[str]:
    violations: List[str] = []
    ranges = [
        ("position_capital_fraction", backtest.position_capital_fraction, 0.001, 0.2),
        ("max_total_capital_fraction", backtest.max_total_capital_fraction, 0.01, 0.9),
        ("max_position_units", float(backtest.max_position_units), 1, 20, (True, True)),
        ("maker_fee_bps", backtest.maker_fee_bps, -5.0, 5.0),
        ("taker_fee_bps", backtest.taker_fee_bps, -5.0, 10.0),
        ("slippage_bps", backtest.slippage_bps, 0.0, 50.0),
        ("slippage_bps_target", backtest.slippage_bps_target, 0.0, 50.0),
        ("max_daily_loss_pct", backtest.max_daily_loss_pct, 0.0, 0.20),
        ("max_drawdown_pct", backtest.max_drawdown_pct, 0.0, 0.60),
        ("trades_per_day_target", backtest.trades_per_day_target, 0.05, 20.0),
        ("min_gate_trades", float(backtest.min_gate_trades), 10, 5000, (True, True)),
    ]
    for entry in ranges:
        if len(entry) == 4:
            name, value, lower, upper = entry
            problem = _range_check(name, float(value), lower, upper)
        else:
            name, value, lower, upper, inc = entry
            problem = _range_check(name, float(value), lower, upper, inc)
        if problem:
            violations.append(problem)

    if backtest.max_total_capital_fraction < backtest.position_capital_fraction:
        violations.append(
            f"max_total_capital_fraction ({backtest.max_total_capital_fraction:.3f}) must be >= position_capital_fraction ({backtest.position_capital_fraction:.3f})."
        )
    if backtest.max_daily_loss_pct and backtest.max_drawdown_pct and backtest.max_daily_loss_pct > backtest.max_drawdown_pct:
        violations.append(
            "max_daily_loss_pct must not exceed max_drawdown_pct."
        )
    return violations


def _validate_filters(filters: FilterConfig) -> List[str]:
    violations: List[str] = []
    if filters.min_atr_frac is not None and filters.min_atr_frac < 0.0:
        violations.append("filters.min_atr_frac must be >= 0.")
    if filters.min_confidence is not None:
        problem = _range_check("filters.min_confidence", filters.min_confidence, 0.0, 0.49)
        if problem:
            violations.append(problem)
    if filters.min_adx is not None:
        problem = _range_check("filters.min_adx", filters.min_adx, 0.0, 60.0)
        if problem:
            violations.append(problem)
    return violations


def _validate_latency_arbitrage(cfg: LatencyArbConfig) -> List[str]:
    violations: List[str] = []
    if not cfg.enabled:
        return violations

    summary_path: Path = cfg.summary_path
    if not summary_path.exists():
        violations.append(
            f"Latency arbitrage summary missing at {summary_path}. Run analysis/latency_arbitrage_sim.py before enabling."
        )
        return violations

    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        violations.append(f"Latency arbitrage summary invalid JSON: {exc}")
        return violations

    summary = payload.get("summary") or {}
    trades = int(summary.get("trades", 0) or 0)
    avg_bps = float(summary.get("average_realized_bps", 0.0) or 0.0)
    net_pnl = float(summary.get("net_pnl", 0.0) or 0.0)

    if trades < cfg.min_trades:
        violations.append(
            f"Latency arbitrage requires at least {cfg.min_trades} trades in simulation; found {trades}."
        )
    if avg_bps < cfg.min_average_bps:
        violations.append(
            f"Latency arbitrage average realised edge {avg_bps:.2f}bps below guardrail minimum {cfg.min_average_bps:.2f}bps."
        )
    if net_pnl <= 0:
        violations.append("Latency arbitrage net PnL must be positive before enabling live execution.")

    return violations


def validate_app_config(config: AppConfig, stage: str = "runtime") -> None:
    """Validate guardrails for the provided configuration."""
    violations: List[str] = []
    violations.extend(_validate_backtest(config.backtest))
    violations.extend(_validate_filters(config.filters))
    violations.extend(_validate_latency_arbitrage(config.risk.latency_arb))
    if config.model.model_type == "random_forest":
        if config.model.n_estimators > 2000:
            violations.append("model.n_estimators too high for realtime usage.")
        if config.model.max_depth and config.model.max_depth > 20:
            violations.append("model.max_depth > 20 may overfit.")
    if violations:
        raise GuardrailViolation(violations)

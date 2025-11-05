"""ATR / trailing parameter sweep utilities."""

from __future__ import annotations

import csv
import logging
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, Tuple

from ai_trading_bot.config import AppConfig
from ai_trading_bot.pipeline import execute_backtest

logger = logging.getLogger(__name__)

GRID = {
    "atr_mult": [0.0015, 0.0025, 0.0035],
    "trail_k": [0.0, 1.5, 2.0, 3.0],
    "ttp_k": [0.0, 2.5, 3.5, 4.5],
    "min_hold": [24, 48, 72],
    "hyst_k": [0.80, 1.00, 1.20],
    "cap_frac": [0.025, 0.03, 0.035],
    "cancel_spread_bps": [6.0, 8.0, 10.0],
}

ACCEPT = {
    "min_trades": 400,
    "max_dd": -0.12,
    "min_expect": 0.0,
    "max_slip_bps": 0.6,
    "min_maker": 0.90,
    "min_sharpe": 0.0,
}

CSV_FIELDS = [
    "atr_mult",
    "trail_k",
    "ttp_k",
    "min_hold",
    "hyst_k",
    "cap_fraction",
    "cancel_spread_bps",
    "total_return",
    "annualised_return",
    "annualised_volatility",
    "calc_sharpe",
    "expectancy_after_costs",
    "max_drawdown",
    "trades_count",
    "avg_slippage_bps",
    "maker_fill_ratio",
    "pass",
]


def _passes(summary: Dict[str, float]) -> bool:
    return (
        summary.get("trades_count", 0) >= ACCEPT["min_trades"]
        and summary.get("max_drawdown", -1.0) >= ACCEPT["max_dd"]
        and summary.get("expectancy_after_costs", -1.0) > ACCEPT["min_expect"]
        and summary.get("avg_slippage_bps", float("inf")) <= ACCEPT["max_slip_bps"]
        and summary.get("maker_fill_ratio", 0.0) >= ACCEPT["min_maker"]
        and summary.get("calc_sharpe", -1.0) > ACCEPT["min_sharpe"]
    )


def _configure_iteration(
    base_config: AppConfig,
    atr_mult: float,
    trail_k: float,
    ttp_k: float,
    min_hold: int,
    hyst_k: float,
    cap_frac: float,
    cancel_spread_bps: float,
) -> AppConfig:
    cfg = deepcopy(base_config)
    cfg.sweep_mode = True
    cfg.filters.min_atr_frac = float(atr_mult)
    cfg.filters.hysteresis_k_atr = float(hyst_k)
    cfg.backtest.min_hold_bars = int(min_hold)
    cfg.backtest.position_capital_fraction = float(cap_frac)
    cfg.backtest.max_total_capital_fraction = min(0.5, float(cap_frac) * 3.0)
    cfg.backtest.cancel_spread_bps = float(cancel_spread_bps)

    trailing = cfg.risk.trailing
    trail_active = trail_k > 0 and ttp_k > 0
    trailing.enabled = trail_active
    if trail_active:
        trailing.k_atr.setdefault("stop", {})["swing"] = float(trail_k)
        trailing.k_atr.setdefault("take", {})["swing"] = float(ttp_k)
    else:
        # Ensure swing trailing is effectively disabled
        trailing.k_atr.setdefault("stop", {}).pop("swing", None)
        trailing.k_atr.setdefault("take", {}).pop("swing", None)

    return cfg


def run_iteration(cfg: AppConfig) -> Tuple[Dict[str, float], Dict[str, object]]:
    _, result, metadata = execute_backtest(cfg, enforce_gates=False)
    return result.summary, metadata


def main(base_config: AppConfig | Dict[str, object], output_dir: Path | str = "sweeps", grid_override: Dict[str, Iterable[object]] | None = None) -> Path:
    """Execute the ATR sweep using the provided base configuration."""
    if not isinstance(base_config, AppConfig):
        base_config = AppConfig.from_dict(dict(base_config))

    grid = {**GRID}
    if grid_override:
        for key, values in grid_override.items():
            grid[key] = list(values)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_csv = out_dir / f"atr_sweep_{timestamp}.csv"

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()

        keys = [
            "atr_mult",
            "trail_k",
            "ttp_k",
            "min_hold",
            "hyst_k",
            "cap_frac",
            "cancel_spread_bps",
        ]
        grid_values: Iterable[Tuple[object, ...]] = product(*(grid[k] for k in keys))
        for values in grid_values:
            atr_mult, trail_k, ttp_k, min_hold, hyst_k, cap_frac, cancel_spread = values
            cfg = _configure_iteration(
                base_config,
                atr_mult,
                trail_k,
                ttp_k,
                min_hold,
                hyst_k,
                cap_frac,
                cancel_spread,
            )
            summary, _ = run_iteration(cfg)

            calc_sharpe = summary.get("calc_sharpe")
            if calc_sharpe is None:
                ann_ret = summary.get("annualised_return", 0.0)
                ann_vol = summary.get("annualised_volatility", 0.0)
                denom = max(abs(ann_vol), 1e-12)
                calc_sharpe = (ann_ret - 0.0) / denom

            row = {
                "atr_mult": atr_mult,
                "trail_k": trail_k,
                "ttp_k": ttp_k,
                "min_hold": min_hold,
                "hyst_k": hyst_k,
                "cap_fraction": cap_frac,
                "cancel_spread_bps": cancel_spread,
                "total_return": summary.get("total_return"),
                "annualised_return": summary.get("annualised_return"),
                "annualised_volatility": summary.get("annualised_volatility"),
                "calc_sharpe": calc_sharpe,
                "expectancy_after_costs": summary.get("expectancy_after_costs"),
                "max_drawdown": summary.get("max_drawdown"),
                "trades_count": summary.get("trades_count"),
                "avg_slippage_bps": summary.get("avg_slippage_bps"),
                "maker_fill_ratio": summary.get("maker_fill_ratio"),
                "pass": _passes(summary),
            }
            writer.writerow(row)
            logger.debug(
                "ATR sweep iteration atr=%.4f trail=%.2f ttp=%.2f hold=%s hyst=%.2f cap=%.3f spread=%.1f -> Sharpe=%.4f DD=%.4f trades=%s",
                atr_mult,
                trail_k,
                ttp_k,
                min_hold,
                hyst_k,
                cap_frac,
                cancel_spread,
                row["calc_sharpe"],
                row["max_drawdown"],
                row["trades_count"],
            )

    logger.info("ATR sweep results written to %s", out_csv)
    return out_csv


__all__ = ["main", "GRID", "ACCEPT"]



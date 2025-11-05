"""Parameter sweep for the maker trend pullback strategy."""

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

DEFAULT_GRID = {
    "timeframe": ["1h"],
    "trend_ma_len": [50, 100],
    "trend_strength_min": [0.30, 0.45],
    "pullback_k_atr": [0.8, 1.2, 1.6],
    "reentry_block_bars": [8, 16],
    "signals.post.min_hold_bars": [24, 48],
    "signals.post.hysteresis_k_atr": [0.4, 0.7],
    "trailing.trail_k": [2.0, 3.0],
    "trailing.ttp_k": [2.0, 3.0],
    "risk.cap_fraction": [0.02, 0.03],
    "execution.cancel_spread_bps": [1.0, 2.0, 3.0],
}

ACCEPT = {
    "min_trades": 250,
    "max_dd": -0.12,
    "min_expect": 0.0,
    "min_maker": 0.92,
    "max_slip_bps": 0.6,
    "min_sharpe": 0.0,
}

CSV_FIELDS = [
    "timeframe",
    "trend_ma_len",
    "trend_strength_min",
    "pullback_k_atr",
    "reentry_block_bars",
    "signals.post.min_hold_bars",
    "signals.post.hysteresis_k_atr",
    "trailing.trail_k",
    "trailing.ttp_k",
    "risk.cap_fraction",
    "execution.cancel_spread_bps",
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
        and summary.get("maker_fill_ratio", 0.0) >= ACCEPT["min_maker"]
        and summary.get("avg_slippage_bps", float("inf")) <= ACCEPT["max_slip_bps"]
        and summary.get("calc_sharpe", -1.0) > ACCEPT["min_sharpe"]
    )


def _configure_iteration(base_cfg: AppConfig, combo: Dict[str, object]) -> AppConfig:
    cfg = deepcopy(base_cfg)
    cfg.strategy.name = "maker_trend_pullback"
    cfg.strategy.params = dict(cfg.strategy.params)

    timeframe = str(combo["timeframe"])
    cfg.data.interval = timeframe

    cfg.strategy.params.update(
        {
            "trend_ma_len": int(combo["trend_ma_len"]),
            "trend_strength_min": float(combo["trend_strength_min"]),
            "pullback_k_atr": float(combo["pullback_k_atr"]),
            "reentry_block_bars": int(combo["reentry_block_bars"]),
        }
    )

    min_hold = int(combo["signals.post.min_hold_bars"])
    hyst_k = float(combo["signals.post.hysteresis_k_atr"])
    cfg.signals.post.min_hold_bars = min_hold
    cfg.signals.post.hysteresis_k_atr = hyst_k
    cfg.backtest.min_hold_bars = min_hold
    cfg.filters.hysteresis_k_atr = hyst_k

    cap_frac = float(combo["risk.cap_fraction"])
    cfg.backtest.position_capital_fraction = cap_frac
    cfg.backtest.max_total_capital_fraction = min(0.3, cap_frac * 3.0)

    cancel_spread = float(combo["execution.cancel_spread_bps"])
    cfg.backtest.cancel_spread_bps = cancel_spread

    trail_k = float(combo["trailing.trail_k"])
    ttp_k = float(combo["trailing.ttp_k"])
    trailing = cfg.risk.trailing
    trailing.enabled = True
    trailing.k_atr.setdefault("stop", {})["swing"] = trail_k
    trailing.k_atr.setdefault("take", {})["swing"] = ttp_k

    return cfg


def main(
    base_config: AppConfig | Dict[str, object],
    output_dir: Path | str = "sweeps",
    grid_override: Dict[str, Iterable[object]] | None = None,
) -> Path:
    if not isinstance(base_config, AppConfig):
        base_cfg = AppConfig.from_dict(dict(base_config))
    else:
        base_cfg = deepcopy(base_config)

    grid = {**DEFAULT_GRID}
    if grid_override:
        for key, values in grid_override.items():
            grid[key] = list(values)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_csv = out_dir / f"maker_trend_pullback_{timestamp}.csv"

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()

        keys = list(grid.keys())
        grid_values: Iterable[Tuple[object, ...]] = product(*(grid[k] for k in keys))
        for values in grid_values:
            combo = dict(zip(keys, values))
            cfg = _configure_iteration(base_cfg, combo)
            _, result, _ = execute_backtest(cfg, enforce_gates=False)
            summary = result.summary

            calc_sharpe = summary.get("calc_sharpe")
            if calc_sharpe is None:
                ann_ret = summary.get("annualised_return", 0.0)
                ann_vol = summary.get("annualised_volatility", 0.0)
                denom = max(abs(ann_vol), 1e-12)
                calc_sharpe = (ann_ret - 0.0) / denom
                summary["calc_sharpe"] = calc_sharpe

            row = {
                **combo,
                "total_return": summary.get("total_return"),
                "annualised_return": summary.get("annualised_return"),
                "annualised_volatility": summary.get("annualised_volatility"),
                "calc_sharpe": calc_sharpe,
                "expectancy_after_costs": summary.get("expectancy_after_costs"),
                "max_drawdown": summary.get("max_drawdown"),
                "trades_count": summary.get("trades_count"),
                "avg_slippage_bps": summary.get("avg_slippage_bps"),
                "maker_fill_ratio": summary.get("maker_fill_ratio"),
            }
            row["pass"] = _passes(summary)
            writer.writerow(row)

            logger.debug(
                "Trend pullback sweep timeframe=%s trend_ma=%s trend_strength=%.2f pullback=%.2f reentry=%s hold=%s hyst=%.2f trail=%.2f/%.2f cap=%.3f spread=%.1f -> Sharpe=%.4f trades=%s",
                combo["timeframe"],
                combo["trend_ma_len"],
                combo["trend_strength_min"],
                combo["pullback_k_atr"],
                combo["reentry_block_bars"],
                combo["signals.post.min_hold_bars"],
                combo["signals.post.hysteresis_k_atr"],
                combo["trailing.trail_k"],
                combo["trailing.ttp_k"],
                combo["risk.cap_fraction"],
                combo["execution.cancel_spread_bps"],
                row["calc_sharpe"],
                row["trades_count"],
            )

    logger.info("Maker trend pullback sweep results written to %s", out_csv)
    return out_csv


__all__ = ["main", "DEFAULT_GRID", "ACCEPT"]


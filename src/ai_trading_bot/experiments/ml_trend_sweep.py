"""Parameter sweep for the ml_trend swing strategy."""

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
    "timeframe": ["1h", "4h"],
    "pipeline.long_threshold": [0.52, 0.55, 0.58],
    "pipeline.short_threshold": [0.48, 0.45, 0.42],
    "filters.min_confidence": [0.04, 0.06],
    "filters.min_atr_frac": [0.0015, 0.0020, 0.0025],
    "filters.min_trend_slope": [0.0, 0.0005, 0.001],
    "filters.min_adx": [10, 15, 20],
    "signals.post.min_hold_bars": [4, 8, 12],
    "signals.post.hysteresis_k_atr": [0.15, 0.25, 0.35],
    "backtest.position_capital_fraction": [0.02, 0.03, 0.04],
    "backtest.max_total_capital_fraction": [0.06, 0.09, 0.12],
    "risk.trailing.enabled": [False, True],
    "risk.trailing.stop_k": [3.0, 4.0, 5.0],
    "risk.trailing.take_k": [1.5, 2.0, 2.5],
}

ACCEPT = {
    "min_trades_1h": 150,
    "min_trades_4h": 80,
    "max_dd": -0.15,
    "min_expect": 0.0,
    "max_slip_bps": 2.5,
    "min_sharpe": 0.2,
}

CSV_FIELDS = [
    "timeframe",
    "pipeline.long_threshold",
    "pipeline.short_threshold",
    "filters.min_confidence",
    "filters.min_atr_frac",
    "filters.min_trend_slope",
    "filters.min_adx",
    "signals.post.min_hold_bars",
    "signals.post.hysteresis_k_atr",
    "backtest.position_capital_fraction",
    "backtest.max_total_capital_fraction",
    "risk.trailing.enabled",
    "risk.trailing.stop_k",
    "risk.trailing.take_k",
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


def _passes(summary: Dict[str, float], timeframe: str) -> bool:
    min_trades = ACCEPT["min_trades_4h"] if timeframe.endswith("4h") else ACCEPT["min_trades_1h"]
    return (
        summary.get("trades_count", 0) >= min_trades
        and summary.get("max_drawdown", -1.0) >= ACCEPT["max_dd"]
        and summary.get("expectancy_after_costs", -1.0) > ACCEPT["min_expect"]
        and summary.get("avg_slippage_bps", float("inf")) <= ACCEPT["max_slip_bps"]
        and summary.get("calc_sharpe", -1.0) > ACCEPT["min_sharpe"]
    )


def _configure_iteration(base_cfg: AppConfig, combo: Dict[str, object]) -> AppConfig:
    cfg = deepcopy(base_cfg)
    cfg.strategy.name = "ml_trend"
    cfg.strategy.params = dict(cfg.strategy.params)

    timeframe = str(combo["timeframe"])
    cfg.data.interval = timeframe

    cfg.pipeline.long_threshold = float(combo["pipeline.long_threshold"])
    cfg.pipeline.short_threshold = float(combo["pipeline.short_threshold"])

    cfg.filters.min_confidence = float(combo["filters.min_confidence"])
    cfg.filters.min_atr_frac = float(combo["filters.min_atr_frac"])
    cfg.filters.min_trend_slope = float(combo["filters.min_trend_slope"])
    cfg.filters.min_adx = float(combo["filters.min_adx"])

    min_hold = int(combo["signals.post.min_hold_bars"])
    hyst_k = float(combo["signals.post.hysteresis_k_atr"])
    cfg.signals.post.min_hold_bars = min_hold
    cfg.signals.post.hysteresis_k_atr = hyst_k
    cfg.backtest.min_hold_bars = min_hold

    position_frac = float(combo["backtest.position_capital_fraction"])
    cfg.backtest.position_capital_fraction = position_frac
    cfg.backtest.max_total_capital_fraction = float(combo["backtest.max_total_capital_fraction"])

    trailing_enabled = bool(combo["risk.trailing.enabled"])
    stop_k = float(combo["risk.trailing.stop_k"])
    take_k = float(combo["risk.trailing.take_k"])
    trailing = cfg.risk.trailing
    trailing.enabled = trailing_enabled
    if trailing_enabled:
        trailing.k_atr.setdefault("stop", {})["swing"] = stop_k
        trailing.k_atr.setdefault("take", {})["swing"] = take_k
    else:
        trailing.k_atr.setdefault("stop", {}).pop("swing", None)
        trailing.k_atr.setdefault("take", {}).pop("swing", None)

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
    out_csv = out_dir / f"ml_trend_{timestamp}.csv"

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
            row["pass"] = _passes(summary, timeframe=str(combo["timeframe"]))
            writer.writerow(row)

            logger.debug(
                "ml_trend sweep timeframe=%s p_long=%.2f p_short=%.2f conf=%.3f atr=%.5f slope=%.5f adx=%.1f hold=%s hyst=%.2f cap=%.3f totcap=%.2f trail=%s stop=%.1f take=%.1f -> Sharpe=%.4f trades=%s",
                combo["timeframe"],
                combo["pipeline.long_threshold"],
                combo["pipeline.short_threshold"],
                combo["filters.min_confidence"],
                combo["filters.min_atr_frac"],
                combo["filters.min_trend_slope"],
                combo["filters.min_adx"],
                combo["signals.post.min_hold_bars"],
                combo["signals.post.hysteresis_k_atr"],
                combo["backtest.position_capital_fraction"],
                combo["backtest.max_total_capital_fraction"],
                combo["risk.trailing.enabled"],
                combo["risk.trailing.stop_k"],
                combo["risk.trailing.take_k"],
                row["calc_sharpe"],
                row["trades_count"],
            )

    logger.info("ml_trend sweep results written to %s", out_csv)
    return out_csv


__all__ = ["main", "DEFAULT_GRID", "ACCEPT"]


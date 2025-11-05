"""Parameter sweep for the maker VWAP band reversion strategy."""

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
    "timeframe": ["15m"],
    "vwap_band_sigma": [1.0, 1.5, 2.0],
    "session": ["24x7", "US_only"],
    "tp_back_to_vwap": [0.5, 0.8],
    "sl_k_atr": [1.0, 1.3, 1.6],
    "signals.post.min_hold_bars": [8, 16, 24],
    "signals.post.hysteresis_k_atr": [0.6, 0.9],
    "trailing.trail_k": ["off", 3.0],
    "trailing.ttp_k": ["off", 3.0],
    "risk.cap_fraction": [0.01, 0.02, 0.03],
    "execution.cancel_spread_bps": [1.0, 2.0],
}

ACCEPT = {
    "min_trades_15m": 400,
    "min_trades_1h": 250,
    "max_dd": -0.10,
    "min_expect": 0.0,
    "min_maker": 0.93,
    "max_slip_bps": 0.5,
    "min_sharpe": 0.0,
}

CSV_FIELDS = [
    "timeframe",
    "vwap_band_sigma",
    "session",
    "tp_back_to_vwap",
    "sl_k_atr",
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


def _passes(summary: Dict[str, float], timeframe: str) -> bool:
    min_trades = ACCEPT["min_trades_1h"] if timeframe.endswith("1h") else ACCEPT["min_trades_15m"]
    return (
        summary.get("trades_count", 0) >= min_trades
        and summary.get("max_drawdown", -1.0) >= ACCEPT["max_dd"]
        and summary.get("expectancy_after_costs", -1.0) > ACCEPT["min_expect"]
        and summary.get("maker_fill_ratio", 0.0) >= ACCEPT["min_maker"]
        and summary.get("avg_slippage_bps", float("inf")) <= ACCEPT["max_slip_bps"]
        and summary.get("calc_sharpe", -1.0) > ACCEPT["min_sharpe"]
    )


def _configure_iteration(base_cfg: AppConfig, combo: Dict[str, object]) -> AppConfig:
    cfg = deepcopy(base_cfg)
    cfg.strategy.name = "maker_vwap_reversion"
    cfg.strategy.params = dict(cfg.strategy.params)

    timeframe = str(combo["timeframe"])
    cfg.data.interval = timeframe

    cfg.strategy.params.update(
        {
            "vwap_band_sigma": float(combo["vwap_band_sigma"]),
            "session": str(combo["session"]),
            "tp_back_to_vwap": float(combo["tp_back_to_vwap"]),
            "sl_k_atr": float(combo["sl_k_atr"]),
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
    cfg.backtest.max_total_capital_fraction = min(0.25, cap_frac * 3.0)

    cancel_spread = float(combo["execution.cancel_spread_bps"])
    cfg.backtest.cancel_spread_bps = cancel_spread

    trail_k = combo["trailing.trail_k"]
    ttp_k = combo["trailing.ttp_k"]
    trailing = cfg.risk.trailing
    if isinstance(trail_k, str) and trail_k.lower() == "off":
        trailing.enabled = False
        trailing.k_atr.setdefault("stop", {}).pop("swing", None)
    else:
        trailing.enabled = True
        trailing.k_atr.setdefault("stop", {})["swing"] = float(trail_k)

    if isinstance(ttp_k, str) and ttp_k.lower() == "off":
        trailing.k_atr.setdefault("take", {}).pop("swing", None)
        if not trailing.enabled:
            trailing.k_atr.setdefault("take", {})
    else:
        trailing.enabled = True
        trailing.k_atr.setdefault("take", {})["swing"] = float(ttp_k)

    if not trailing.enabled:
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
    out_csv = out_dir / f"maker_vwap_reversion_{timestamp}.csv"

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
            row["pass"] = _passes(summary, str(combo["timeframe"]))
            writer.writerow(row)

            logger.debug(
                "VWAP reversion sweep timeframe=%s sigma=%.1f session=%s tp=%.2f sl=%.2f hold=%s hyst=%.2f trail=%s/%s cap=%.3f spread=%.1f -> Sharpe=%.4f trades=%s",
                combo["timeframe"],
                combo["vwap_band_sigma"],
                combo["session"],
                combo["tp_back_to_vwap"],
                combo["sl_k_atr"],
                combo["signals.post.min_hold_bars"],
                combo["signals.post.hysteresis_k_atr"],
                combo["trailing.trail_k"],
                combo["trailing.ttp_k"],
                combo["risk.cap_fraction"],
                combo["execution.cancel_spread_bps"],
                row["calc_sharpe"],
                row["trades_count"],
            )

    logger.info("Maker VWAP reversion sweep results written to %s", out_csv)
    return out_csv


__all__ = ["main", "DEFAULT_GRID", "ACCEPT"]


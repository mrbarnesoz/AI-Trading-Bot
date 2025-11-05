"""Parameter sweep for the taker momentum burst strategy."""

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
    "timeframe": ["1m"],
    "momo_len": [8, 14, 21],
    "momo_threshold": [1.5, 2.0, 2.5],
    "cooldown_bars": [6, 12],
    "tp_k_atr": [0.5, 0.8, 1.0],
    "sl_k_atr": [0.6, 0.9, 1.2],
    "risk.cap_fraction": [0.0025, 0.005],
    "execution.max_slippage_bps": [2.0, 3.0, 4.0],
}

ACCEPT = {
    "min_trades_1m": 800,
    "min_trades_5m": 500,
    "max_dd": -0.06,
    "min_expect": 0.0,
    "max_slip_bps": 3.0,
    "min_sharpe": 0.5,
}

CSV_FIELDS = [
    "timeframe",
    "momo_len",
    "momo_threshold",
    "cooldown_bars",
    "tp_k_atr",
    "sl_k_atr",
    "risk.cap_fraction",
    "execution.max_slippage_bps",
    "total_return",
    "annualised_return",
    "annualised_volatility",
    "calc_sharpe",
    "expectancy_after_costs",
    "max_drawdown",
    "trades_count",
    "avg_slippage_bps",
    "pass",
]


def _passes(summary: Dict[str, float], timeframe: str) -> bool:
    min_trades = ACCEPT["min_trades_5m"] if timeframe.endswith("5m") else ACCEPT["min_trades_1m"]
    return (
        summary.get("trades_count", 0) >= min_trades
        and summary.get("max_drawdown", -1.0) >= ACCEPT["max_dd"]
        and summary.get("expectancy_after_costs", -1.0) > ACCEPT["min_expect"]
        and summary.get("avg_slippage_bps", float("inf")) <= ACCEPT["max_slip_bps"]
        and summary.get("calc_sharpe", -1.0) > ACCEPT["min_sharpe"]
    )


def _configure_iteration(base_cfg: AppConfig, combo: Dict[str, object]) -> AppConfig:
    cfg = deepcopy(base_cfg)
    cfg.strategy.name = "taker_momo_burst"
    cfg.strategy.params = dict(cfg.strategy.params)

    timeframe = str(combo["timeframe"])
    cfg.data.interval = timeframe

    cfg.backtest.entry_mode = "taker"

    cfg.strategy.params.update(
        {
            "momo_len": int(combo["momo_len"]),
            "momo_threshold": float(combo["momo_threshold"]),
            "cooldown_bars": int(combo["cooldown_bars"]),
            "tp_k_atr": float(combo["tp_k_atr"]),
            "sl_k_atr": float(combo["sl_k_atr"]),
            "execution_max_slippage_bps": float(combo["execution.max_slippage_bps"]),
        }
    )

    cap_frac = float(combo["risk.cap_fraction"])
    cfg.backtest.position_capital_fraction = cap_frac
    cfg.backtest.max_total_capital_fraction = min(0.02, cap_frac * 3.0)

    # Taker slippage cap stored in strategy params; backtester enforces taker mode.
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
    out_csv = out_dir / f"taker_momo_burst_{timestamp}.csv"

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
            }
            row["pass"] = _passes(summary, timeframe=str(combo["timeframe"]))
            writer.writerow(row)

            logger.debug(
                "Taker momo sweep timeframe=%s len=%s threshold=%.2f cooldown=%s tp=%.2f sl=%.2f cap=%.4f slip=%.1f -> Sharpe=%.4f trades=%s",
                combo["timeframe"],
                combo["momo_len"],
                combo["momo_threshold"],
                combo["cooldown_bars"],
                combo["tp_k_atr"],
                combo["sl_k_atr"],
                combo["risk.cap_fraction"],
                combo["execution.max_slippage_bps"],
                row["calc_sharpe"],
                row["trades_count"],
            )

    logger.info("Taker momentum burst sweep results written to %s", out_csv)
    return out_csv


__all__ = ["main", "DEFAULT_GRID", "ACCEPT"]


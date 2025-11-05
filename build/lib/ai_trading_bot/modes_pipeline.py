"""Mode-specific dataset preparation helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from ai_trading_bot.config import AppConfig
from ai_trading_bot.data.pipeline import prepare_dataset


def _resolve_window(end_iso: str | None, lookback_days: int) -> Tuple[str, str]:
    if end_iso:
        end = pd.to_datetime(end_iso, utc=True)
    else:
        end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=lookback_days)
    return start.strftime("%Y-%m-%dT%H:%M:%SZ"), end.strftime("%Y-%m-%dT%H:%M:%SZ")


def _label_from_iso(value: str) -> str:
    ts = pd.to_datetime(value, utc=True)
    return ts.strftime("%Y%m%dT%H%M%SZ")


def prepare_datasets_for_modes(config: Dict, force_download: bool = False) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Build engineered datasets for each mode defined in the configuration."""
    base_config = AppConfig.from_dict(config)
    symbol = base_config.data.symbol
    end_iso = base_config.data.end_date

    engineered_dir = Path("data/engineered")
    engineered_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}

    for mode_cfg in base_config.modes:
        start_iso, mode_end_iso = _resolve_window(end_iso, mode_cfg.lookback_days)

        mode_config = deepcopy(config)
        mode_config["data"]["interval"] = mode_cfg.interval
        mode_config["data"]["start_date"] = start_iso
        mode_config["data"]["end_date"] = mode_end_iso

        mode_config["pipeline"]["long_threshold"] = mode_cfg.long_threshold or mode_config["pipeline"]["long_threshold"]
        mode_config["pipeline"]["short_threshold"] = mode_cfg.short_threshold or mode_config["pipeline"]["short_threshold"]
        if mode_cfg.long_bands:
            mode_config["pipeline"]["long_bands"] = mode_cfg.long_bands
        if mode_cfg.short_bands:
            mode_config["pipeline"]["short_bands"] = mode_cfg.short_bands

        decision, engineered = prepare_dataset(mode_config, force_download=force_download)

        if not {"target_return", "target"}.issubset(engineered.columns):
            raise ValueError(f"Missing target columns for mode '{mode_cfg.name}'.")

        start_label = _label_from_iso(start_iso)
        end_label = _label_from_iso(mode_end_iso)
        suffix = f"{symbol}_{mode_cfg.name}_{mode_cfg.interval}_{start_label}_{end_label}"
        engineered.to_parquet(engineered_dir / f"{suffix}.parquet")
        engineered.to_csv(engineered_dir / f"{suffix}.csv")

        results[mode_cfg.name] = (decision, engineered)

    return results


__all__ = ["prepare_datasets_for_modes"]

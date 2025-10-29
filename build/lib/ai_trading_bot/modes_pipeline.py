"""Multi-mode dataset preparation helper."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd

from ai_trading_bot.data_pipeline import prepare_dataset as base_prepare_dataset


def _resolve_dates(end_iso: str | None, lookback_days: int) -> tuple[str, str]:
    if end_iso:
        end = pd.to_datetime(end_iso, utc=True)
    else:
        end = pd.Timestamp.utcnow().tz_localize("UTC")
    start = end - pd.Timedelta(days=lookback_days)
    return start.strftime("%Y-%m-%dT%H:%M:%SZ"), end.strftime("%Y-%m-%dT%H:%M:%SZ")


def prepare_datasets_for_modes(config: dict, force_download: bool = False):
    """Build a dataset for each mode listed in config['modes']."""
    results: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    base_cfg = deepcopy(config)
    symbol = base_cfg["data"]["symbol"]
    end_cfg = base_cfg["data"].get("end_date")
    engineered_dir = Path("data/engineered")
    engineered_dir.mkdir(parents=True, exist_ok=True)

    for mode in base_cfg.get("modes", []):
        mode_name = mode["name"]
        interval = mode["interval"]
        lookback_days = int(mode.get("lookback_days", 5))
        start_iso, end_iso = _resolve_dates(end_cfg, lookback_days)

        cfg = deepcopy(base_cfg)
        cfg["data"]["interval"] = interval
        cfg["data"]["start_date"] = start_iso
        cfg["data"]["end_date"] = end_iso

        cfg["pipeline"]["long_threshold"] = mode.get("long_threshold", cfg["pipeline"]["long_threshold"])
        cfg["pipeline"]["short_threshold"] = mode.get("short_threshold", cfg["pipeline"]["short_threshold"])
        cfg["pipeline"]["long_bands"] = mode.get("long_bands", cfg["pipeline"]["long_bands"])
        cfg["pipeline"]["short_bands"] = mode.get("short_bands", cfg["pipeline"]["short_bands"])

        decision, engineered = base_prepare_dataset(cfg, force_download=force_download)

        suffix = f"{symbol}_{mode_name}_{interval}_{start_iso}_{end_iso}"
        engineered.to_parquet(engineered_dir / f"{suffix}.parquet")
        engineered.to_csv(engineered_dir / f"{suffix}.csv")

        if "target_return" not in engineered.columns or "target" not in engineered.columns:
            raise ValueError(f"Missing target columns for mode '{mode_name}'.")

        results[mode_name] = (decision, engineered)

    return results


__all__ = ["prepare_datasets_for_modes"]

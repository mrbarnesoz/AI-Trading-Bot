"""Utilities for consuming ATR sweep outputs to drive runtime defaults."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional

_FLOAT_FIELDS = {
    "atr_mult",
    "trail_k",
    "ttp_k",
    "hyst_k",
    "cap_fraction",
    "cancel_spread_bps",
    "total_return",
    "annualised_return",
    "annualised_volatility",
    "calc_sharpe",
    "expectancy_after_costs",
    "max_drawdown",
    "avg_slippage_bps",
    "maker_fill_ratio",
}
_INT_FIELDS = {"min_hold", "trades_count"}
_BOOL_FIELDS = {"pass"}


def _coerce_value(key: str, value: str) -> object:
    if value is None or value == "":
        return None
    if key in _INT_FIELDS:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None
    if key in _FLOAT_FIELDS:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    if key in _BOOL_FIELDS:
        return str(value).strip().lower() in {"1", "true", "yes", "y"}
    return value


def load_top_candidate(path: Path | str = "Top_ATR_Sweep_Candidates.csv") -> Optional[Dict[str, object]]:
    """Return the first candidate row from the supplied CSV if available."""
    csv_path = Path(path)
    if not csv_path.exists():
        return None

    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        try:
            row = next(reader)
        except StopIteration:
            return None

    if row is None:
        return None

    parsed = {key: _coerce_value(key, value) for key, value in row.items()}
    return {k: v for k, v in parsed.items() if v is not None}


__all__ = ["load_top_candidate"]

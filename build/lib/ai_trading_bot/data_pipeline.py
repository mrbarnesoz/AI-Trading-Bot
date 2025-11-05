"""Compatibility wrapper around :mod:`ai_trading_bot.data.pipeline`."""

from __future__ import annotations

from typing import Dict, Tuple, Union

import pandas as pd

from ai_trading_bot.config import AppConfig
from ai_trading_bot.data.pipeline import prepare_dataset as _prepare_dataset


def prepare_dataset(config: Union[AppConfig, Dict], force_download: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Backwards-compatible entry point for dataset preparation."""
    return _prepare_dataset(config, force_download=force_download)


__all__ = ["prepare_dataset"]

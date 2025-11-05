"""Shared abstractions for strategy modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd


@dataclass
class StrategySlice:
    """Container for strategy-specific signals and diagnostics."""

    name: str
    signals: pd.Series
    probabilities: pd.Series
    diagnostics: Dict[str, float] = field(default_factory=dict)

    def ensure_index(self, index: pd.Index) -> None:
        """Align internal series to the provided index."""
        if not self.signals.index.equals(index):
            self.signals = self.signals.reindex(index).fillna(0.0)
        if not self.probabilities.index.equals(index):
            self.probabilities = self.probabilities.reindex(index).fillna(0.5)


def _series(name: str, index: pd.Index, values: Optional[pd.Series] = None, default: float = 0.0) -> pd.Series:
    if values is not None:
        series = values.reindex(index).astype(float)
    else:
        series = pd.Series(default, index=index)
    series.name = name
    return series

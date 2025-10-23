"""Strategy that uses the trained ML model to generate trading signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ai_trading_bot.config import ModelConfig, PipelineConfig
from ai_trading_bot.models.predictor import generate_probabilities, load_model

_TOL = 1e-12


@dataclass
class StrategyOutput:
    signals: pd.Series
    probabilities: pd.Series
    data: pd.DataFrame


def _prepare_long_bands(bands: Sequence[float], override: Optional[float]) -> list[float]:
    cleaned = sorted({float(b) for b in bands})
    if override is not None:
        cleaned = [b for b in cleaned if b >= override - _TOL]
        cleaned.append(float(override))
    return sorted(set(cleaned))


def _prepare_short_bands(bands: Sequence[float], override: Optional[float]) -> list[float]:
    cleaned = sorted({float(b) for b in bands})
    if override is not None:
        cleaned = [b for b in cleaned if b <= override + _TOL]
        cleaned.append(float(override))
    return sorted(set(cleaned))


def generate_signals(
    engineered_data: pd.DataFrame,
    model_cfg: ModelConfig,
    pipeline_cfg: PipelineConfig,
    probability_long_threshold: Optional[float] = None,
    probability_short_threshold: Optional[float] = None,
) -> StrategyOutput:
    """Use the trained model to create long/short/flat signals with probabilistic banding."""
    long_thr = probability_long_threshold if probability_long_threshold is not None else pipeline_cfg.long_threshold
    short_thr = probability_short_threshold if probability_short_threshold is not None else pipeline_cfg.short_threshold

    if not 0 <= long_thr <= 1 or not 0 <= short_thr <= 1:
        raise ValueError("Probability thresholds must lie between 0 and 1.")
    if short_thr >= long_thr:
        raise ValueError("Short threshold must be strictly lower than long threshold.")

    model, feature_columns = load_model(model_cfg)
    probs = generate_probabilities(model, feature_columns, engineered_data, pipeline_cfg)
    probabilities = pd.Series(probs, index=engineered_data.index, name="probability")

    long_bands = _prepare_long_bands(pipeline_cfg.long_bands, long_thr) if pipeline_cfg.long_bands else []
    short_bands = _prepare_short_bands(pipeline_cfg.short_bands, short_thr) if pipeline_cfg.short_bands else []
    use_bands = bool(long_bands or short_bands)

    prob_array = probabilities.to_numpy()
    signal_values = np.zeros_like(prob_array, dtype=int)

    if use_bands:
        if long_bands:
            long_arr = np.array(long_bands, dtype=float)
            long_counts = np.searchsorted(long_arr, prob_array, side="right")
        else:
            long_counts = np.zeros_like(prob_array, dtype=int)

        if short_bands:
            short_arr = np.array(short_bands, dtype=float)
            short_counts = short_arr.size - np.searchsorted(short_arr, prob_array, side="left")
        else:
            short_counts = np.zeros_like(prob_array, dtype=int)

        signal_values = long_counts - short_counts
    else:
        signal_values[prob_array >= long_thr - _TOL] = 1
        signal_values[prob_array <= short_thr + _TOL] = -1

    signals = pd.Series(signal_values, index=engineered_data.index, name="signal")
    return StrategyOutput(signals=signals, probabilities=probabilities, data=engineered_data.copy())

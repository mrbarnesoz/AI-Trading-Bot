"""Strategy that uses the trained ML model to generate trading signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ai_trading_bot.config import ModelConfig, PipelineConfig
from ai_trading_bot.models.predictor import generate_probabilities, load_model


@dataclass
class StrategyOutput:
    signals: pd.Series
    probabilities: pd.Series
    data: pd.DataFrame


def generate_signals(
    engineered_data: pd.DataFrame,
    model_cfg: ModelConfig,
    pipeline_cfg: PipelineConfig,
    probability_threshold: float = 0.55,
) -> StrategyOutput:
    """Use the trained model to create long/flat signals."""
    model, feature_columns = load_model(model_cfg)
    probs = generate_probabilities(model, feature_columns, engineered_data, pipeline_cfg)
    signals = (probs >= probability_threshold).astype(int)
    # Align index to original data
    signals = pd.Series(signals, index=engineered_data.index, name="signal")
    probabilities = pd.Series(probs, index=engineered_data.index, name="probability")
    return StrategyOutput(signals=signals, probabilities=probabilities, data=engineered_data.copy())

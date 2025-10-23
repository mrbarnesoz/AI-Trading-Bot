from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ai_trading_bot.config import ModelConfig, PipelineConfig
from ai_trading_bot.strategies.ml_strategy import generate_signals


class DummyProbabilityModel:
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = X["probability"].to_numpy()
        return np.column_stack([1 - probs, probs])


def test_generate_signals_supports_probability_banding(tmp_path: Path) -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    engineered = pd.DataFrame(
        {
            "probability": [0.2, 0.35, 0.5, 0.65, 0.85],
            "target": 0,
            "target_return": 0.0,
        },
        index=index,
    )

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    artifact = model_dir / "price_direction_model.joblib"
    joblib.dump({"model": DummyProbabilityModel(), "feature_columns": ["probability"]}, artifact)

    model_cfg = ModelConfig(model_dir=model_dir)
    pipeline_cfg = PipelineConfig(
        long_threshold=0.55,
        short_threshold=0.45,
        long_bands=[0.55, 0.65, 0.8],
        short_bands=[0.45, 0.35, 0.25],
    )

    output = generate_signals(engineered, model_cfg, pipeline_cfg)

    assert output.signals.tolist() == [-3, -2, 0, 2, 3]
    assert np.allclose(output.probabilities.to_numpy(), engineered["probability"].to_numpy())

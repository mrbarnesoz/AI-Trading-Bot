from __future__ import annotations

import pandas as pd

from ai_trading_bot.config import FeatureConfig, PipelineConfig
from ai_trading_bot.features.indicators import engineer_features


def make_price_frame(length: int = 100) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=length, freq="D")
    data = {
        "Open": pd.Series(range(length), index=index).astype(float) + 100,
        "High": pd.Series(range(length), index=index).astype(float) + 101,
        "Low": pd.Series(range(length), index=index).astype(float) + 99,
        "Close": pd.Series(range(length), index=index).astype(float) + 100.5,
        "Adj Close": pd.Series(range(length), index=index).astype(float) + 100.5,
        "Volume": pd.Series(1000 + i for i in range(length)),
    }
    return pd.DataFrame(data, index=index)


def test_engineer_features_creates_expected_columns():
    price_data = make_price_frame()
    feature_cfg = FeatureConfig()
    pipeline_cfg = PipelineConfig(lookahead=1, target_column="target_return")

    engineered = engineer_features(price_data, feature_cfg, pipeline_cfg)

    assert "target" in engineered.columns
    assert "target_return" in engineered.columns
    assert engineered.index.is_monotonic_increasing
    assert len(engineered) < len(price_data)  # dropping NA rows

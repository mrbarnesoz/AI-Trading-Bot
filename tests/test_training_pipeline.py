from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

mlflow = pytest.importorskip("mlflow")

from models.gbm_classifier import GBMConfig
from training.datasets import DatasetSpec, load_dataset
from training.rolling_hft import RollingConfig, rolling_hft_training
from training.walkforward import WalkForwardConfig, run_walkforward_training


def _write_parquet_dataset(tmp_path: Path, regime: str, rows: int = 120) -> None:
    ts = pd.date_range("2024-01-01", periods=rows, freq="1min", tz="UTC")
    symbols = ["XBTUSD"] * rows
    labels = np.resize(np.array([-1, 0, 1]), rows)
    data = pl.DataFrame(
        {
            "ts": ts,
            "symbol": symbols,
            "feature_a": np.random.randn(rows),
            "feature_b": np.random.randn(rows),
            "label_sign": labels,
        }
    )
    output_dir = tmp_path / "features" / regime / "dt=2024-01-01"
    output_dir.mkdir(parents=True, exist_ok=True)
    data.write_parquet(output_dir / "part-000.parquet")


def test_load_dataset(tmp_path: Path) -> None:
    _write_parquet_dataset(tmp_path, "intraday", rows=60)
    spec = DatasetSpec(root=tmp_path, regime="intraday")
    features, labels, weights = load_dataset(spec)
    assert weights is None
    assert "feature_a" in features.columns
    assert len(features) == len(labels)
    assert isinstance(features.index, pd.DatetimeIndex)
    assert labels.isin([-1, 0, 1]).all()


def test_walkforward_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ml_path = tmp_path / "mlruns"
    mlflow.set_tracking_uri(f"file://{ml_path}")
    _write_parquet_dataset(tmp_path, "intraday", rows=180)
    spec = DatasetSpec(root=tmp_path, regime="intraday")
    features, labels, weights = load_dataset(spec)

    gbm_config = GBMConfig(
        params={
            "objective": "multiclass",
            "learning_rate": 0.2,
            "num_leaves": 15,
            "min_data_in_leaf": 5,
            "num_class": 3,
            "verbosity": -1,
        },
        num_boost_round=30,
        early_stopping_rounds=5,
    )
    wf_config = WalkForwardConfig(folds=3, embargo=1, params=gbm_config, experiment="test-walkforward")
    results = run_walkforward_training(features, labels, weights, wf_config)
    assert len(results) == 3
    for result in results:
        assert "long_sharpe" in result.metrics
        assert result.model_uri.startswith("runs:/")


def test_rolling_hft_training(tmp_path: Path) -> None:
    ml_path = tmp_path / "mlruns"
    mlflow.set_tracking_uri(f"file://{ml_path}")
    _write_parquet_dataset(tmp_path, "hft", rows=200)
    spec = DatasetSpec(root=tmp_path, regime="hft")
    features, labels, _ = load_dataset(spec)
    config = RollingConfig(
        window_days=1,
        step_days=1,
        sequence_length=10,
        batch_size=8,
        max_epochs=1,
        patience=1,
        experiment="test-hft",
    )
    results = rolling_hft_training(features, labels, config)
    assert results, "Expected at least one rolling window result"
    for entry in results:
        assert "metrics" in entry
        assert "long_sharpe" in entry["metrics"]

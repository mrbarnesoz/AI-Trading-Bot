"""Purged walk-forward training for intraday and swing regimes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Tuple

import mlflow
import numpy as np
import pandas as pd

from models.calibrate import calibrate_probabilities, save_calibration
from models.gbm_classifier import GBMConfig, train_lightgbm
from training import metrics

CLASS_MAPPING = {-1: "short", 0: "flat", 1: "long"}


@dataclass
class FoldResult:
    fold_id: int
    run_id: str
    model_uri: str
    metrics: Dict[str, float]


@dataclass
class WalkForwardConfig:
    folds: int
    embargo: int
    params: GBMConfig
    early_stop_metric: str = "long_sharpe"
    experiment: str = "walkforward"


def generate_purged_folds(idx: pd.DatetimeIndex, folds: int, embargo: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate purged walk-forward train/validation splits."""
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    fold_size = len(idx) // folds
    for i in range(folds):
        start = i * fold_size
        end = start + fold_size
        val_idx = np.arange(start, min(end, len(idx)))
        train_idx = np.concatenate(
            [
                np.arange(0, max(start - embargo, 0)),
                np.arange(min(end + embargo, len(idx)), len(idx)),
            ]
        )
        splits.append((train_idx, val_idx))
    return splits


def run_walkforward_training(
    features: pd.DataFrame,
    labels: pd.Series,
    weights: pd.Series | None,
    config: WalkForwardConfig,
) -> List[FoldResult]:
    mlflow.set_experiment(config.experiment)

    label_values = sorted(labels.unique().tolist())
    label_to_index = {value: idx for idx, value in enumerate(label_values)}
    encoded_labels = labels.map(label_to_index).to_numpy()
    numeric_labels = labels.astype(float).to_numpy()
    num_classes = len(label_values)

    params = dict(config.params.params)
    if num_classes <= 2:
        params.setdefault("objective", "binary")
    else:
        params.setdefault("objective", "multiclass")
        params["num_class"] = num_classes

    idx = features.index
    splits = generate_purged_folds(idx, config.folds, config.embargo)
    results: List[FoldResult] = []
    for fold_id, (train_idx, val_idx) in enumerate(splits):
        X_train, y_train = features.iloc[train_idx], labels.iloc[train_idx]
        X_val, y_val = features.iloc[val_idx], labels.iloc[val_idx]
        y_train_encoded = encoded_labels[train_idx]
        y_val_encoded = encoded_labels[val_idx]
        train_weights = weights.iloc[train_idx] if weights is not None else None

        booster = train_lightgbm(
            X_train,
            pd.Series(y_train_encoded, index=X_train.index),
            train_weights,
            (X_val, pd.Series(y_val_encoded, index=X_val.index)),
            GBMConfig(params=params, num_boost_round=config.params.num_boost_round, early_stopping_rounds=config.params.early_stopping_rounds),
        )

        val_preds = booster.predict(X_val)
        if num_classes <= 2:
            long_scores = np.asarray(val_preds).reshape(-1)
            short_scores = 1.0 - long_scores
        else:
            val_preds = np.asarray(val_preds)
            long_scores = val_preds[:, label_to_index.get(1, label_values[-1])]
            short_scores = val_preds[:, label_to_index.get(-1, label_values[0])]

        y_val_numeric = numeric_labels[val_idx]
        long_targets = (y_val == 1).astype(int).to_numpy()
        short_targets = (y_val == -1).astype(int).to_numpy()

        long_calibrated, long_calibrator = calibrate_probabilities(long_scores, long_targets)
        short_calibrated, short_calibrator = calibrate_probabilities(short_scores, short_targets)

        long_sharpe = metrics.compute_sharpe(long_calibrated, y_val_numeric)
        short_sharpe = metrics.compute_sharpe(short_calibrated, -y_val_numeric)
        calmar = metrics.compute_calmar(long_calibrated, y_val_numeric)
        brier_long = metrics.brier_score(long_calibrated, long_targets)
        brier_short = metrics.brier_score(short_calibrated, short_targets)

        feature_importance = dict(zip(features.columns, booster.feature_importance().tolist()))

        metrics_payload = {
            "long_sharpe": long_sharpe,
            "short_sharpe": short_sharpe,
            "calmar": calmar,
            "brier_long": brier_long,
            "brier_short": brier_short,
        }

        with mlflow.start_run(run_name=f"walkforward_fold_{fold_id}") as run:
            mlflow.log_metrics(metrics_payload)
            mlflow.log_params(params)
            mlflow.log_dict(feature_importance, artifact_file="feature_importance.json")

            with TemporaryDirectory() as tmp:
                model_path = Path(tmp) / "lightgbm.txt"
                booster.save_model(str(model_path))
                mlflow.log_artifact(str(model_path), artifact_path="model")
                long_cal_path = Path(tmp) / "calibrator_long.joblib"
                short_cal_path = Path(tmp) / "calibrator_short.joblib"
                save_calibration(long_calibrator, long_cal_path)
                save_calibration(short_calibrator, short_cal_path)
                mlflow.log_artifact(str(long_cal_path), artifact_path="model")
                mlflow.log_artifact(str(short_cal_path), artifact_path="model")

            model_uri = f"runs:/{run.info.run_id}/model"
            results.append(
                FoldResult(
                    fold_id=fold_id,
                    run_id=run.info.run_id,
                    model_uri=model_uri,
                    metrics=metrics_payload,
                )
            )
    return results

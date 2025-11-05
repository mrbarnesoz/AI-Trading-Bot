"""Walk-forward validation utilities."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from ai_trading_bot.config import AppConfig, ModelConfig, PipelineConfig
from ai_trading_bot.backtesting.simulator import run_backtest as simulate_backtest
from ai_trading_bot.models.predictor import (
    _initialise_model,
    _prepare_features,
    generate_probabilities,
)
from ai_trading_bot.pipeline import prepare_dataset
from ai_trading_bot.strategies.ml_strategy import signals_from_probabilities

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardSegment:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    summary: Dict[str, float]


@dataclass
class WalkForwardReport:
    segments: List[WalkForwardSegment]
    aggregate: Dict[str, float]


def walk_forward_backtest(
    config: AppConfig,
    *,
    train_period: pd.Timedelta,
    test_period: pd.Timedelta,
    step_period: Optional[pd.Timedelta] = None,
    force_download: bool = False,
) -> WalkForwardReport:
    """Run walk-forward validation using configured dataset."""
    decision, engineered = prepare_dataset(config, force_download=force_download)
    return walk_forward_on_dataset(
        config,
        decision,
        engineered,
        train_period=train_period,
        test_period=test_period,
        step_period=step_period,
    )


def walk_forward_on_dataset(
    base_config: AppConfig,
    decision_frame: pd.DataFrame,
    engineered_frame: pd.DataFrame,
    *,
    train_period: pd.Timedelta,
    test_period: pd.Timedelta,
    step_period: Optional[pd.Timedelta] = None,
) -> WalkForwardReport:
    """Walk forward using provided dataset (useful for testing)."""
    if decision_frame.empty or engineered_frame.empty:
        return WalkForwardReport([], {})

    decision = decision_frame.sort_index()
    engineered = engineered_frame.sort_index()
    index = decision.index
    start = index[0]
    end = index[-1]
    step = step_period or test_period

    segments: List[WalkForwardSegment] = []
    window_start = start
    iteration = 0

    while True:
        train_start = window_start
        train_end = train_start + train_period
        test_start = train_end
        test_end = test_start + test_period

        if test_end > end:
            break

        train_mask = (engineered.index >= train_start) & (engineered.index < train_end)
        test_mask = (engineered.index >= test_start) & (engineered.index < test_end)
        price_mask = (decision.index >= test_start) & (decision.index < test_end)

        if train_mask.sum() < 20 or test_mask.sum() < 10 or price_mask.sum() < 10:
            window_start += step
            continue

        train_df = engineered.loc[train_mask]
        test_df = engineered.loc[test_mask]
        price_df = decision.loc[price_mask].copy()

        model, feature_columns = _fit_model(train_df, base_config.model, base_config.pipeline)
        probabilities = generate_probabilities(model, feature_columns, test_df, base_config.pipeline)
        probability_series = pd.Series(probabilities, index=test_df.index, name="probability")
        signals = signals_from_probabilities(probability_series, base_config.pipeline)
        aligned_signals = signals.reindex(price_df.index, method="ffill").fillna(0.0)

        backtest_cfg = copy.deepcopy(base_config.backtest)
        trailing_cfg = copy.deepcopy(base_config.risk.trailing)
        result = simulate_backtest(
            price_df,
            aligned_signals.astype(float),
            backtest_cfg,
            trailing_cfg=trailing_cfg,
            symbol=base_config.data.symbol,
            regime="walk_forward",
        )
        summary = dict(result.summary)
        summary["train_rows"] = float(train_mask.sum())
        summary["test_rows"] = float(test_mask.sum())
        summary["train_start"] = train_start.isoformat()
        summary["train_end"] = train_end.isoformat()
        summary["test_start"] = test_start.isoformat()
        summary["test_end"] = test_end.isoformat()

        segments.append(
            WalkForwardSegment(
                index=iteration,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                summary=summary,
            )
        )
        iteration += 1
        window_start += step

    aggregate = _aggregate_segments(segments)
    return WalkForwardReport(segments=segments, aggregate=aggregate)


def _aggregate_segments(segments: List[WalkForwardSegment]) -> Dict[str, float]:
    if not segments:
        return {}
    returns = [seg.summary.get("total_return", 0.0) for seg in segments]
    cumulative_return = float(np.prod([1.0 + r for r in returns]) - 1.0)
    sharpe_values = [seg.summary.get("sharpe_ratio", 0.0) for seg in segments]
    average_sharpe = float(np.mean(sharpe_values)) if sharpe_values else 0.0
    drawdowns = [seg.summary.get("max_drawdown", 0.0) for seg in segments]
    worst_drawdown = float(min(drawdowns)) if drawdowns else 0.0
    total_trades = float(sum(seg.summary.get("trades_count", 0) for seg in segments))
    return {
        "segments": float(len(segments)),
        "cumulative_return": cumulative_return,
        "average_sharpe": average_sharpe,
        "worst_drawdown": worst_drawdown,
        "total_trades": total_trades,
    }


def _fit_model(
    engineered_data: pd.DataFrame,
    model_cfg: ModelConfig,
    pipeline_cfg: PipelineConfig,
):
    features, target, feature_columns = _prepare_features(engineered_data, pipeline_cfg)
    model = _initialise_model(model_cfg)

    if len(features) < 4:
        model.fit(features, target)
        return model, feature_columns

    stratify = target if target.nunique() > 1 else None
    test_size = min(max(model_cfg.test_size, 0.1), 0.4)
    try:
        X_train, X_valid, y_train, y_valid = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=model_cfg.random_state,
            stratify=stratify if stratify is not None and stratify.nunique() > 1 else None,
        )
    except ValueError:
        X_train, X_valid, y_train, y_valid = features, None, target, None

    model.fit(X_train, y_train)
    if X_valid is None or len(X_valid) == 0:
        return model, feature_columns

    try:
        calibrator = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
        calibrator.fit(X_valid, y_valid)
        return calibrator, feature_columns
    except Exception as exc:  # pragma: no cover - calibration best effort
        logger.debug("Calibration skipped: %s", exc)
        return model, feature_columns

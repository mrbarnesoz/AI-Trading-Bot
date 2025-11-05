"""Strategy engine wrappers for live trading."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ai_trading_bot.config import AppConfig, FeatureConfig, ModelConfig, PipelineConfig
from ai_trading_bot.features.indicators import engineer_features
from ai_trading_bot.models.predictor import generate_probabilities, load_model
from ai_trading_bot.orchestrator import MultiStrategyOrchestrator
from ai_trading_bot.strategies.base import StrategySlice
from ai_trading_bot.strategies.ml_strategy import StrategyOutput, signals_from_probabilities
from ai_trading_bot.strategies.rule_based import mean_reversion_signals

from .data import BarEvent

logger = logging.getLogger(__name__)


@dataclass
class StrategyDecision:
    symbol: str
    timestamp: datetime
    signal: float
    probability: float
    strategy: str = ""
    regime: str = ""
    metadata: Dict[str, float] = field(default_factory=dict)
    raw_output: StrategyOutput | None = None


class StrategyEngine:
    """Abstract strategy engine."""

    def on_bar(self, bar: BarEvent) -> Optional[StrategyDecision]:
        raise NotImplementedError


class MLStrategyEngine(StrategyEngine):
    """Thin wrapper around the ML pipeline for incremental bar processing."""

    def __init__(self, config: AppConfig, *, component_weights: Optional[Dict[str, float]] = None) -> None:
        self.config = config
        self.model_cfg: ModelConfig = config.model
        self.feature_cfg: FeatureConfig = config.features
        self.pipeline_cfg: PipelineConfig = config.pipeline
        self._model, self._feature_columns = load_model(self.model_cfg)
        self._bars: Dict[str, pd.DataFrame] = defaultdict(self._empty_frame)
        self.component_weights = component_weights or {}
        self._orchestrator = MultiStrategyOrchestrator()
        logger.info("ML strategy engine initialised with %s features.", len(self._feature_columns))

    def _empty_frame(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"], dtype=float)

    def on_bar(self, bar: BarEvent) -> Optional[StrategyDecision]:
        frame = self._bars[bar.symbol]
        ts = pd.Timestamp(bar.end)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        row = pd.DataFrame(
            {
                "Open": [bar.open],
                "High": [bar.high],
                "Low": [bar.low],
                "Close": [bar.close],
                "Volume": [bar.volume],
            },
            index=[ts],
        )
        frame = pd.concat([frame, row]).sort_index()
        self._bars[bar.symbol] = frame.tail(5000)  # keep rolling window

        if len(frame) < max(50, self.pipeline_cfg.lookahead + 10):
            return None

        engineered = engineer_features(frame.copy(), self.feature_cfg, self.pipeline_cfg)
        if engineered.empty:
            return None
        try:
            probabilities = generate_probabilities(
                self._model,
                self._feature_columns,
                engineered,
                self.pipeline_cfg,
            )
        except Exception as exc:  # pragma: no cover - guardrails
            logger.exception("Failed to score probabilities for %s: %s", bar.symbol, exc)
            return None

        probabilities_series = pd.Series(probabilities, index=engineered.index, name="probability")
        signals = signals_from_probabilities(probabilities_series, self.pipeline_cfg)

        components: Dict[str, pd.Series] = {
            "ml_signal": signals.astype(float),
            "trend": pd.Series(np.sign(signals.to_numpy()).astype(float), index=signals.index, name="trend_signal"),
        }
        component_probabilities: Dict[str, pd.Series] = {
            "ml_probability": probabilities_series.rename("ml_probability"),
            "trend": probabilities_series.rename("trend_probability"),
        }

        mean_rev_slice = mean_reversion_signals(engineered)
        components["mean_reversion"] = mean_rev_slice.signals
        component_probabilities["mean_reversion"] = mean_rev_slice.probabilities

        base_output = StrategyOutput(
            signals=signals.astype(float),
            probabilities=probabilities_series.astype(float),
            data=engineered.copy(),
            components=components,
            component_probabilities=component_probabilities,
        )

        orchestrated_output, regimes, slices = self._orchestrator.orchestrate(engineered, base_output)

        final_signals = orchestrated_output.signals.copy()
        active_strategy_series = orchestrated_output.data.get("active_strategy", pd.Series("ml_trend", index=engineered.index))

        if self.component_weights:
            weighted = pd.Series(0.0, index=final_signals.index, dtype=float)
            for name, weight in self.component_weights.items():
                key = f"{name}_signal"
                series = orchestrated_output.components.get(key)
                if series is not None:
                    weighted = weighted.add(series * float(weight), fill_value=0.0)
            weighted = weighted.clip(-1.0, 1.0)
            mask = weighted.abs() >= 1e-3
            if mask.any():
                final_signals.loc[mask] = weighted.loc[mask]
                active_strategy_series.loc[mask] = "weighted"

        latest_index = final_signals.index[-1]
        latest_signal = float(final_signals.iloc[-1])
        latest_prob = float(orchestrated_output.probabilities.iloc[-1])
        regime_value = regimes.loc[latest_index] if latest_index in regimes.index else ""
        strategy_name = str(active_strategy_series.iloc[-1]) if len(active_strategy_series) else "ml_trend"
        diagnostics: Dict[str, float] = {}
        if strategy_name in slices:
            diagnostics = {k: float(v) for k, v in slices[strategy_name].diagnostics.items()}
        diagnostics.setdefault("regime_trend", float((regimes == "trend").mean() if len(regimes) else 0.0))
        diagnostics.setdefault("regime_breakout", float((regimes == "breakout").mean() if len(regimes) else 0.0))
        diagnostics.setdefault("regime_range", float((regimes == "range").mean() if len(regimes) else 0.0))

        decision = StrategyDecision(
            symbol=bar.symbol,
            timestamp=pd.Timestamp(latest_index).to_pydatetime(),
            signal=latest_signal,
            probability=latest_prob,
            strategy=strategy_name,
            regime=regime_value,
            metadata=diagnostics,
            raw_output=orchestrated_output,
        )
        return decision

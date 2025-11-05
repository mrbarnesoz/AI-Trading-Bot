"""Multi-strategy orchestration for combining complementary signals."""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from ai_trading_bot.regime.detector import detect_regime, summarise_regime
from ai_trading_bot.strategies.base import StrategySlice
from ai_trading_bot.strategies.ml_strategy import StrategyOutput
from ai_trading_bot.strategies.rule_based import (
    breakout_signals,
    mean_reversion_signals,
    momentum_signals,
)


class MultiStrategyOrchestrator:
    """Toggle between strategy modules depending on detected regime."""

    def __init__(self) -> None:
        self._cache: Dict[str, StrategySlice] = {}

    def _generate_slices(self, engineered: pd.DataFrame, base_output: StrategyOutput) -> Dict[str, StrategySlice]:
        slices: Dict[str, StrategySlice] = {}
        base_slice = StrategySlice(
            name="ml_trend",
            signals=base_output.signals.astype(float),
            probabilities=base_output.probabilities.astype(float),
            diagnostics={"source": "ml"},
        )
        base_slice.ensure_index(engineered.index)

        slices["ml_trend"] = base_slice
        slices["mean_reversion"] = mean_reversion_signals(engineered)
        slices["momentum"] = momentum_signals(engineered)
        slices["breakout"] = breakout_signals(engineered)
        return slices

    def orchestrate(self, engineered: pd.DataFrame, base_output: StrategyOutput) -> Tuple[StrategyOutput, pd.Series, Dict[str, StrategySlice]]:
        index = engineered.index
        slices = self._generate_slices(engineered, base_output)
        regimes = detect_regime(engineered)
        selection = pd.Series("ml_trend", index=index, dtype=object)
        final_signal = pd.Series(0.0, index=index, name="signal")
        final_prob = pd.Series(0.5, index=index, name="probability")

        for ts in index:
            regime = regimes.loc[ts]
            if regime == "trend":
                chosen = slices["momentum"]
            elif regime == "breakout":
                chosen = slices["breakout"]
            else:
                chosen = slices["mean_reversion"]

            if abs(chosen.signals.loc[ts]) < 1e-9:
                chosen = slices["ml_trend"]

            final_signal.loc[ts] = float(chosen.signals.loc[ts])
            final_prob.loc[ts] = float(chosen.probabilities.loc[ts])
            selection.loc[ts] = chosen.name

        # Build component dictionaries
        components = base_output.components.copy()
        component_probabilities = base_output.component_probabilities.copy()
        for name, slice_ in slices.items():
            components[f"{name}_signal"] = slice_.signals.rename(f"{name}_signal")
            component_probabilities[f"{name}_probability"] = slice_.probabilities.rename(f"{name}_probability")

        for strategy_name in slices.keys():
            indicator = (selection == strategy_name).astype(float)
            components[f"selected_{strategy_name}"] = indicator.rename(f"selected_{strategy_name}")

        data = base_output.data.copy()
        data["active_strategy"] = selection
        orchestrated_output = StrategyOutput(
            signals=final_signal,
            probabilities=final_prob,
            data=data,
            components=components,
            component_probabilities=component_probabilities,
        )
        return orchestrated_output, regimes, slices


def orchestrate_multi_strategy(engineered: pd.DataFrame, base_output: StrategyOutput) -> Tuple[StrategyOutput, pd.Series, Dict[str, StrategySlice]]:
    orchestrator = MultiStrategyOrchestrator()
    return orchestrator.orchestrate(engineered, base_output)

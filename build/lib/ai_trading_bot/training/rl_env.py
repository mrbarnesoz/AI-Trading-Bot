"""Prototype reinforcement-learning environments for threshold & strategy weighting."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ai_trading_bot.config import AppConfig
from ai_trading_bot.pipeline import execute_backtest, prepare_dataset
from ai_trading_bot.strategies.ml_strategy import StrategyOutput, generate_signals
from ai_trading_bot.backtesting.simulator import BacktestResult, run_backtest as simulate_backtest


@dataclass
class RLStep:
    action_index: int
    long_threshold: float
    short_threshold: float
    reward: float
    summary: Dict[str, float] = field(default_factory=dict)


class BacktestTradingEnv:
    """Minimal gym-like environment that maps discrete threshold adjustments to backtest rewards."""

    def __init__(
        self,
        base_config: AppConfig | Dict[str, object],
        *,
        action_grid: Optional[Sequence[Tuple[float, float]]] = None,
        reward_metric: str = "sharpe",
        enforce_gates: bool = False,
        mode: str = "threshold",
        component_actions: Optional[Sequence[Dict[str, float]]] = None,
    ) -> None:
        if not isinstance(base_config, AppConfig):
            base_config = AppConfig.from_dict(dict(base_config))
        self._base_config = base_config
        self.reward_metric = reward_metric
        self.enforce_gates = enforce_gates
        self.mode = mode
        self.action_grid: List[Tuple[float, float]] = (
            list(action_grid) if action_grid is not None else [(-0.02, 0.02), (0.0, 0.0), (0.02, -0.02)]
        )
        self.component_actions = list(component_actions or [])
        if mode == "weight" and not self.component_actions:
            self.component_actions = [
                {"trend": 1.0, "mean_reversion": 0.0},
                {"trend": 0.7, "mean_reversion": 0.3},
                {"trend": 0.5, "mean_reversion": 0.5},
                {"trend": 0.3, "mean_reversion": 0.7},
                {"trend": 0.0, "mean_reversion": 1.0},
            ]
        if mode not in {"threshold", "weight"}:
            raise ValueError("mode must be 'threshold' or 'weight'.")
        if mode == "threshold" and not self.action_grid:
            raise ValueError("action_grid must contain at least one adjustment tuple.")
        if mode == "weight" and not self.component_actions:
            raise ValueError("component_actions must contain at least one weight mapping.")
        self.history: List[RLStep] = []
        self._state: Dict[str, float] = {}
        self._price_frame: Optional[pd.DataFrame] = None
        self._engineered_frame: Optional[pd.DataFrame] = None
        self._base_strategy_output: Optional[StrategyOutput] = None
        self.reset()

    @property
    def action_space(self) -> int:
        return len(self.action_grid)

    def reset(self) -> Dict[str, float]:
        self.history.clear()
        self._state = {
            "long_threshold": float(self._base_config.pipeline.long_threshold),
            "short_threshold": float(self._base_config.pipeline.short_threshold),
            "last_reward": 0.0,
        }
        if self.mode == "weight":
            self._prepare_dataset()
        return dict(self._state)

    def step(
        self,
        action_index: int,
        *,
        force_download: bool = False,
        long_bounds: Tuple[float, float] = (0.01, 0.99),
        short_bounds: Tuple[float, float] = (0.0, 0.98),
    ) -> Tuple[Dict[str, float], float, bool, Dict[str, object]]:
        if self.mode == "threshold":
            return self._step_threshold(action_index, force_download, long_bounds, short_bounds)
        return self._step_weights(action_index)

    def _compute_reward(self, result: BacktestResult, config: AppConfig) -> float:
        summary = result.summary
        if self.reward_metric == "total_return":
            return float(summary.get("total_return", 0.0))
        if self.reward_metric == "expectancy":
            return float(summary.get("expectancy_after_costs", 0.0))

        sharpe = float(summary.get("calc_sharpe", 0.0))
        dd = float(summary.get("max_drawdown", 0.0))
        penalty = 0.0
        if dd < config.backtest.max_drawdown_floor:
            penalty += abs(dd - config.backtest.max_drawdown_floor)
        trades = float(summary.get("trades_count", 0.0))
        if trades == 0:
            penalty += 0.1
        return sharpe - penalty


    def _step_threshold(
        self,
        action_index: int,
        force_download: bool,
        long_bounds: Tuple[float, float],
        short_bounds: Tuple[float, float],
    ) -> Tuple[Dict[str, float], float, bool, Dict[str, object]]:
        if not 0 <= action_index < len(self.action_grid):
            raise IndexError(f"action_index {action_index} outside action_grid size {len(self.action_grid)}")

        delta_long, delta_short = self.action_grid[action_index]
        long_thr = float(
            np.clip(self._base_config.pipeline.long_threshold + delta_long, long_bounds[0], long_bounds[1])
        )
        short_thr = float(
            np.clip(self._base_config.pipeline.short_threshold + delta_short, short_bounds[0], short_bounds[1])
        )

        env_config = AppConfig.from_dict(self._base_config.to_nested_dict())
        env_config.pipeline.long_threshold = long_thr
        env_config.pipeline.short_threshold = min(short_thr, long_thr - 1e-3)

        strategy_output, result, metadata = execute_backtest(
            env_config,
            force_download=force_download,
            long_threshold=long_thr,
            short_threshold=short_thr,
            enforce_gates=self.enforce_gates,
        )

        reward = self._compute_reward(result, env_config)
        self.history.append(
            RLStep(
                action_index=action_index,
                long_threshold=long_thr,
                short_threshold=short_thr,
                reward=reward,
                summary=dict(result.summary),
            )
        )

        state = {
            "long_threshold": long_thr,
            "short_threshold": short_thr,
            "last_reward": reward,
            "trades": float(result.summary.get("trades_count", 0.0)),
            "sharpe": float(result.summary.get("calc_sharpe", 0.0)),
        }
        self._state = state
        info = {"summary": result.summary, "metadata": metadata, "strategy_output": strategy_output}
        return dict(state), reward, True, info

    def _step_weights(self, action_index: int) -> Tuple[Dict[str, float], float, bool, Dict[str, object]]:
        if not self._base_strategy_output or self._price_frame is None:
            raise RuntimeError("Dataset not prepared; call reset() before stepping in weight mode.")
        if not 0 <= action_index < len(self.component_actions):
            raise IndexError(
                f"action_index {action_index} outside component_actions size {len(self.component_actions)}"
            )
        weights = self.component_actions[action_index]
        base_output = self._base_strategy_output
        components = base_output.components
        weighted = pd.Series(0.0, index=base_output.signals.index, dtype=float)
        for name, weight in weights.items():
            comp = components.get(name)
            if comp is not None:
                weighted = weighted.add(comp * float(weight), fill_value=0.0)
        weighted = weighted.clip(-1.0, 1.0)
        final_signals = weighted.apply(lambda x: 0.0 if abs(x) < 1e-9 else float(np.sign(x)))

        env_config = AppConfig.from_dict(self._base_config.to_nested_dict())
        result = simulate_backtest(
            self._price_frame,
            final_signals.reindex(self._price_frame.index).ffill().fillna(0.0),
            env_config.backtest,
            trailing_cfg=env_config.risk.trailing,
            symbol=env_config.data.symbol,
            regime="rl_weight",
        )

        reward = self._compute_reward(result, env_config)
        self.history.append(
            RLStep(
                action_index=action_index,
                long_threshold=float(env_config.pipeline.long_threshold),
                short_threshold=float(env_config.pipeline.short_threshold),
                reward=reward,
                summary=dict(result.summary),
            )
        )

        state = {
            "long_threshold": float(env_config.pipeline.long_threshold),
            "short_threshold": float(env_config.pipeline.short_threshold),
            "last_reward": reward,
            "trades": float(result.summary.get("trades_count", 0.0)),
            "sharpe": float(result.summary.get("calc_sharpe", 0.0)),
        }
        self._state = state
        info = {"summary": result.summary, "weights": weights}
        return dict(state), reward, True, info

    def _prepare_dataset(self) -> None:
        price_frame, engineered = prepare_dataset(self._base_config, force_download=False)
        strategy_output = generate_signals(engineered, self._base_config.model, self._base_config.pipeline)
        reindexed = price_frame.reindex(strategy_output.signals.index).ffill().bfill()
        self._price_frame = reindexed
        self._engineered_frame = engineered
        strategy_output.components.setdefault("trend", strategy_output.signals.astype(float))
        self._base_strategy_output = strategy_output


__all__ = ["BacktestTradingEnv", "RLStep"]

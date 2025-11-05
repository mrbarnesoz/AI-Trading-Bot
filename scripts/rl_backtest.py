"""Evaluate backtest rewards across threshold adjustments or strategy weighting actions."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from ai_trading_bot.config import load_config
from ai_trading_bot.training.rl_env import BacktestTradingEnv


def _parse_threshold_actions(raw_values: Iterable[str] | None) -> List[Tuple[float, float]]:
    if not raw_values:
        return []
    actions: List[Tuple[float, float]] = []
    for value in raw_values:
        if not value:
            continue
        for segment in value.split(","):
            segment = segment.strip()
            if not segment:
                continue
            try:
                long_delta, short_delta = segment.split(":", 1)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"Invalid action format '{segment}'. Expected '<delta_long>:<delta_short>'."
                ) from exc
            actions.append((float(long_delta), float(short_delta)))
    return actions


def _parse_weight_actions(raw_values: Iterable[str] | None) -> List[Dict[str, float]]:
    if not raw_values:
        return []
    actions: List[Dict[str, float]] = []
    for value in raw_values:
        if not value:
            continue
        weight_map: Dict[str, float] = {}
        for token in value.split(","):
            token = token.strip()
            if not token:
                continue
            if "=" not in token:
                raise argparse.ArgumentTypeError(
                    f"Invalid weight entry '{token}'. Expected 'component=value'."
                )
            key, raw_val = token.split("=", 1)
            weight_map[key.strip()] = float(raw_val)
        actions.append(weight_map)
    return actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RL environment sweeps for threshold or component-weight actions."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file.")
    parser.add_argument(
        "--mode",
        choices=("threshold", "weight"),
        default="threshold",
        help="Action mode: threshold adjustments or component weighting.",
    )
    parser.add_argument(
        "--actions",
        nargs="*",
        help="For threshold mode: '<delta_long>:<delta_short>'. For weight mode: 'trend=0.7,mean_reversion=0.3'.",
    )
    parser.add_argument(
        "--reward-metric",
        choices=("sharpe", "total_return", "expectancy"),
        default="sharpe",
        help="Reward metric used to score each action.",
    )
    parser.add_argument("--enforce-gates", action="store_true", help="Apply backtest gating rules.")
    parser.add_argument("--force-download", action="store_true", help="Refresh data before evaluation.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.mode == "threshold":
        threshold_actions = _parse_threshold_actions(args.actions)
        env = BacktestTradingEnv(
            config,
            action_grid=threshold_actions if threshold_actions else None,
            reward_metric=args.reward_metric,
            enforce_gates=args.enforce_gates,
            mode="threshold",
        )
        effective_actions = env.action_grid
    else:
        weight_actions = _parse_weight_actions(args.actions)
        env = BacktestTradingEnv(
            config,
            action_grid=None,
            reward_metric=args.reward_metric,
            enforce_gates=args.enforce_gates,
            mode="weight",
            component_actions=weight_actions if weight_actions else None,
        )
        effective_actions = env.component_actions

    results = []
    for idx, action in enumerate(effective_actions):
        env.reset()
        state, reward, done, info = env.step(idx, force_download=args.force_download)
        summary = info.get("summary", {})
        metadata = info.get("metadata", {})
        results.append(
            {
                "action_index": idx,
                "action": action,
                "reward": reward,
                "done": done,
                "state": state,
                "summary": summary,
                "metadata": metadata,
            }
        )

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_path = args.output or Path("results") / f"rl_backtest_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "generated_at": timestamp,
                "config_path": str(Path(args.config).resolve()),
                "mode": args.mode,
                "reward_metric": args.reward_metric,
                "actions": effective_actions,
                "results": results,
            },
            handle,
            indent=2,
        )

    top_reward = max((entry["reward"] for entry in results), default=None)
    print(
        json.dumps(
            {
                "output": str(out_path.resolve()),
                "mode": args.mode,
                "reward_metric": args.reward_metric,
                "actions": effective_actions,
                "top_reward": top_reward,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

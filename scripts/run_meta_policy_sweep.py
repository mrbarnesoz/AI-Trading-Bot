"""Meta-policy sweep that exercises the RL threshold/weight environment."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, Tuple

from ai_trading_bot.config import AppConfig, load_config
from ai_trading_bot.training.rl_env import BacktestTradingEnv

logger = logging.getLogger("meta_policy_sweep")

GRID = {
    "reward_metric": ["sharpe", "expectancy"],
    "delta": [0.01, 0.02, 0.04],
    "mode": ["threshold"],
    "max_steps": [3, 5],
}

CSV_FIELDS = [
    "reward_metric",
    "mode",
    "delta",
    "max_steps",
    "total_reward",
    "last_reward",
    "last_sharpe",
    "last_expectancy",
    "last_max_drawdown",
    "last_trades",
    "action_sequence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RL/meta-policy sweep over threshold adjustments.")
    parser.add_argument("--config", required=True, help="Path to the base YAML config.")
    parser.add_argument("--output-dir", default="sweeps", help="Directory for CSV output.")
    parser.add_argument("--max-workers", type=int, default=1, help="Reserved for future parallelism.")
    parser.add_argument("--grid-override", help="Optional JSON file describing custom grid values.")
    parser.add_argument("--steps", type=int, default=None, help="Override number of steps per combo.")
    return parser.parse_args()


def _load_grid(override_path: str | None) -> Dict[str, Iterable[object]]:
    grid = {**GRID}
    if override_path:
        override_file = Path(override_path)
        if not override_file.exists():
            raise FileNotFoundError(f"Grid override file not found: {override_file}")
        overrides = json.loads(override_file.read_text(encoding="utf-8"))
        for key, values in overrides.items():
            grid[key] = list(values)
    return grid


def _run_combo(base_cfg: AppConfig, combo: Dict[str, object], steps_override: int | None) -> Dict[str, object]:
    reward_metric = str(combo["reward_metric"])
    delta = float(combo["delta"])
    mode = str(combo["mode"])
    max_steps = int(steps_override or combo["max_steps"])
    action_grid: Tuple[Tuple[float, float], ...]

    if mode == "threshold":
        action_grid = (
            (-delta, delta),
            (0.0, 0.0),
            (delta, -delta),
        )
        component_actions = None
    else:
        action_grid = tuple()
        component_actions = [
            {"trend": 1.0, "mean_reversion": 0.0},
            {"trend": 0.5, "mean_reversion": 0.5},
            {"trend": 0.0, "mean_reversion": 1.0},
        ]

    env = BacktestTradingEnv(
        base_cfg,
        action_grid=action_grid if action_grid else None,
        reward_metric=reward_metric,
        enforce_gates=False,
        mode=mode,
        component_actions=component_actions,
    )
    state = env.reset()
    total_reward = 0.0
    last_reward = 0.0
    last_summary: Dict[str, float] = {}
    actions_taken: list[int] = []

    for step in range(max_steps):
        action_index = step % max(env.action_space, 1)
        next_state, reward, _done, info = env.step(action_index, force_download=False)
        total_reward += reward
        last_reward = reward
        state = next_state
        last_summary = {k: float(v) for k, v in (info.get("summary") or {}).items()}
        actions_taken.append(action_index)

    return {
        "reward_metric": reward_metric,
        "mode": mode,
        "delta": delta,
        "max_steps": max_steps,
        "total_reward": total_reward,
        "last_reward": last_reward,
        "last_sharpe": last_summary.get("calc_sharpe"),
        "last_expectancy": last_summary.get("expectancy_after_costs"),
        "last_max_drawdown": last_summary.get("max_drawdown"),
        "last_trades": last_summary.get("trades_count"),
        "action_sequence": actions_taken,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    base_cfg = load_config(args.config)
    if not isinstance(base_cfg, AppConfig):
        base_cfg = AppConfig.from_dict(base_cfg)

    grid = _load_grid(args.grid_override)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_csv = out_dir / f"meta_policy_{timestamp}.csv"

    keys = list(grid.keys())
    combos = product(*(grid[k] for k in keys))

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for values in combos:
            combo = dict(zip(keys, values))
            result = _run_combo(base_cfg, combo, args.steps)
            writer.writerow(
                {
                    **{k: combo[k] for k in keys},
                    **{k: result[k] for k in CSV_FIELDS if k in result},
                    "action_sequence": json.dumps(result["action_sequence"]),
                }
            )
            logger.info(
                "Meta-policy sweep reward=%s delta=%.3f steps=%s -> total_reward=%.4f sharpe=%s drawdown=%s",
                combo["reward_metric"],
                combo["delta"],
                result["max_steps"],
                result["total_reward"],
                result["last_sharpe"],
                result["last_max_drawdown"],
            )

    logger.info("Meta-policy sweep results written to %s", out_csv)


if __name__ == "__main__":
    main()


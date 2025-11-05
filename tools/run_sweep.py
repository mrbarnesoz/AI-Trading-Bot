"""Command-line wrapper for strategy parameter sweeps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

# Ensure project root on path when executed from tools/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_trading_bot.experiments.atr_sweep import main as run_atr_sweep
from ai_trading_bot.experiments.maker_reversion_band_sweep import (
    main as run_reversion_sweep,
)
from ai_trading_bot.experiments.maker_trend_breakout_sweep import (
    main as run_breakout_sweep,
)

SWEEP_DISPATCH = {
    "atr": run_atr_sweep,
    "maker_trend_breakout": run_breakout_sweep,
    "maker_reversion_band": run_reversion_sweep,
    "maker_vol_expansion": run_vol_expansion_sweep,
    "maker_trend_pullback": run_trend_pullback_sweep,
    "maker_vwap_reversion": run_vwap_reversion_sweep,
    "maker_rsi_divergence": run_rsi_divergence_sweep,
    "maker_keltner_ride": run_keltner_ride_sweep,
    "maker_range_scalper": run_range_scalper_sweep,
    "taker_momo_burst": run_taker_momo_sweep,
}


def _load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _parse_grid(entries: Iterable[str]) -> Dict[str, list[object]]:
    grid: Dict[str, list[object]] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid grid spec '{entry}'. Expected key=value1,value2")
        key, values = entry.split("=", 1)
        parsed_values: list[object] = []
        for raw in values.split(","):
            raw = raw.strip()
            if raw.lower() == "off":
                parsed_values.append("off")
                continue
            try:
                if "." in raw:
                    parsed_values.append(float(raw))
                else:
                    parsed_values.append(int(raw))
            except ValueError:
                parsed_values.append(raw)
        grid[key.strip()] = parsed_values
    return grid


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run parameter sweeps for trading strategies.")
    parser.add_argument("config", type=str, help="Path to base configuration (YAML/JSON).")
    parser.add_argument(
        "--strategy",
        choices=SWEEP_DISPATCH.keys(),
        default="atr",
        help="Strategy sweep to execute.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sweeps",
        help="Output directory for sweep CSV results.",
    )
    parser.add_argument(
        "--grid",
        action="append",
        default=[],
        help="Grid override in key=v1,v2 format. Repeat for multiple keys.",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return 2

    try:
        base_config = _load_config(config_path)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"Failed to parse config: {exc}", file=sys.stderr)
        return 3

    grid_override: Dict[str, list[object]] | None = None
    if args.grid:
        try:
            grid_override = _parse_grid(args.grid)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 4

    output_dir = Path(args.output)
    sweep_fn = SWEEP_DISPATCH[args.strategy]
    result_path = sweep_fn(base_config, output_dir=output_dir, grid_override=grid_override)
    print(result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())

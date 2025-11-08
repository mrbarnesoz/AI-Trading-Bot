"""Batch parameter sweep launcher for strategy configs."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "sweeps"

try:
    sys.path.insert(0, str(PROJECT_ROOT))
    from ai_trading_bot.config import load_config
    from ai_trading_bot.pipeline import prepare_dataset
except Exception as exc:  # pragma: no cover - handled via CLI output
    print(f"Unable to import ai_trading_bot (did you install the repo in editable mode?): {exc}", file=sys.stderr)
    raise


logger = logging.getLogger("param_sweeps")

SWEEP_MODULES = {
    "maker_reversion_band": "ai_trading_bot.experiments.maker_reversion_band_sweep",
    "maker_keltner_ride": "ai_trading_bot.experiments.maker_keltner_ride_sweep",
    "maker_range_scalper": "ai_trading_bot.experiments.maker_range_scalper_sweep",
    "maker_rsi_divergence": "ai_trading_bot.experiments.maker_rsi_divergence_sweep",
    "maker_trend_breakout": "ai_trading_bot.experiments.maker_trend_breakout_sweep",
    "maker_trend_pullback": "ai_trading_bot.experiments.maker_trend_pullback_sweep",
    "maker_vol_expansion": "ai_trading_bot.experiments.maker_vol_expansion_sweep",
    "maker_vwap_reversion": "ai_trading_bot.experiments.maker_vwap_reversion_sweep",
    "maker_swing": "ai_trading_bot.experiments.maker_reversion_band_sweep",
    "maker_keltner": "ai_trading_bot.experiments.maker_keltner_ride_sweep",
    "taker_momo_burst": "ai_trading_bot.experiments.taker_momo_burst_sweep",
    "ml_trend": "ai_trading_bot.experiments.ml_trend_sweep",
}


def _discover_configs(include: Iterable[str] | None = None) -> List[Path]:
    candidates = []
    include_set = {Path(path).resolve() for path in include or []}
    if include_set:
        for path in include_set:
            if path.exists():
                candidates.append(path)
        return sorted(set(candidates))

    for child in CONFIG_DIR.glob("*.yaml"):
        if child.name == "strategy_manifest.yaml":
            continue
        candidates.append(child)
    return sorted(candidates)


def _load_grid_override(path: Optional[str]) -> Dict[str, Dict[str, Iterable[object]]]:
    if not path:
        return {}
    override_path = Path(path)
    if not override_path.exists():
        raise FileNotFoundError(f"Grid override file not found: {override_path}")
    data = yaml.safe_load(override_path.read_text(encoding="utf-8")) or {}
    cleaned: Dict[str, Dict[str, Iterable[object]]] = {}
    for strategy_name, grid in data.items():
        if not isinstance(grid, dict):
            continue
        cleaned[str(strategy_name)] = {str(k): list(v) for k, v in grid.items()}
    return cleaned


def _resolve_module(strategy_name: str):
    module_path = SWEEP_MODULES.get(strategy_name)
    if not module_path:
        raise KeyError(
            f"No sweep module registered for strategy '{strategy_name}'. "
            "Update SWEEP_MODULES in scripts/run_param_sweeps.py."
        )
    return importlib.import_module(module_path)


def _run_single(task: tuple[str, Path, Path, Dict[str, Dict[str, Iterable[object]]]]) -> dict:
    strategy_name, config_path, output_dir, grid_override = task
    module = _resolve_module(strategy_name)
    config = load_config(str(config_path))
    config.sweep_mode = True
    overrides = grid_override.get(strategy_name)
    out_path = module.main(config, output_dir=output_dir, grid_override=overrides)
    return {
        "strategy": strategy_name,
        "config": str(config_path),
        "output": str(out_path),
    }


def _detect_strategy_name(config_path: Path) -> str:
    config = load_config(str(config_path))
    return str(config.strategy.name or "").strip()


def _precache_data(config_path: Path) -> None:
    """Fetch datasets once so later sweeps hit the local cache."""
    config = load_config(str(config_path))
    try:
        prepare_dataset(config, force_download=False)
        logger.info(
            "Cached data for %s (%s %s)",
            config_path.name,
            config.data.symbol,
            config.data.interval,
        )
    except Exception as exc:
        logger.warning("Failed to precache data for %s: %s", config_path, exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parameter sweeps for strategy configs.")
    parser.add_argument(
        "--configs",
        nargs="+",
        help="Explicit config paths to sweep. Defaults to every YAML in configs/ (except manifest).",
    )
    parser.add_argument(
        "--filter",
        nargs="*",
        help="Only run configs whose filename contains one of these substrings (case-insensitive).",
    )
    parser.add_argument(
        "--include-strategies",
        nargs="*",
        help="Restrict to strategy names (e.g. maker_reversion_band).",
    )
    parser.add_argument(
        "--grid-override",
        help="Optional YAML/JSON file mapping strategy names to grid overrides.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for sweep CSV outputs.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="How many configs to sweep in parallel. Use higher values on multi-core/cloud machines.",
    )
    parser.add_argument(
        "--skip-precache",
        action="store_true",
        help="Skip the pre-cache step (not recommended if BitMEX rate-limits are an issue).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without running sweeps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    grid_override = _load_grid_override(args.grid_override)
    configs = _discover_configs(args.configs)
    if args.filter:
        lowered = [token.lower() for token in args.filter]
        configs = [
            path for path in configs if any(token in path.name.lower() for token in lowered)
        ]

    targets: List[tuple[str, Path, Path, Dict[str, Dict[str, Iterable[object]]]]] = []
    include_strategies = {name.strip() for name in (args.include_strategies or []) if name.strip()}

    for cfg_path in configs:
        try:
            strategy_name = _detect_strategy_name(cfg_path)
        except Exception as exc:
            logger.error("Failed to load %s: %s", cfg_path, exc)
            continue
        if not strategy_name:
            logger.warning("Skipping %s (strategy.name missing)", cfg_path)
            continue
        if include_strategies and strategy_name not in include_strategies:
            continue
        if strategy_name not in SWEEP_MODULES:
            logger.warning("No sweep module registered for %s (%s)", cfg_path, strategy_name)
            continue
        targets.append((strategy_name, cfg_path, Path(args.output_dir), grid_override))

    if not targets:
        logger.warning("No configs matched the provided filters.")
        return

    logger.info("Prepared %s sweep task(s). Output -> %s", len(targets), args.output_dir)
    if args.dry_run:
        for strategy_name, cfg_path, *_ in targets:
            logger.info("Would sweep %s (%s)", cfg_path, strategy_name)
        return

    if not args.skip_precache:
        logger.info("Pre-caching datasets to avoid repeated API downloads...")
        for _, cfg_path, *_ in targets:
            _precache_data(cfg_path)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    results: List[dict] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_task = {executor.submit(_run_single, task): task for task in targets}
        for future in concurrent.futures.as_completed(future_to_task):
            strategy_name, cfg_path, *_ = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                logger.info("Sweep finished for %s (%s) -> %s", cfg_path, strategy_name, result["output"])
            except Exception as exc:
                logger.exception("Sweep failed for %s (%s): %s", cfg_path, strategy_name, exc)

    summary_path = Path(args.output_dir) / "sweep_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Sweep summary written to %s", summary_path)


if __name__ == "__main__":
    main()

"""Command-line entry point for the AI Trading Bot."""

from __future__ import annotations

import argparse
import json
import logging
import math
from copy import deepcopy
from pathlib import Path
from typing import Optional

from ai_trading_bot.config import AppConfig, load_config, save_config
from ai_trading_bot.pipeline import (
    execute_backtest,
    prepare_dataset,
    run_backtest,
    train,
)
from ai_trading_bot.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Trading Bot command-line interface.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_download = subparsers.add_parser("download", help="Download OHLCV market data.")
    parser_download.add_argument("--config", default="config.yaml", help="Path to configuration file.")
    parser_download.add_argument("--force", action="store_true", help="Force download even if cached data exists.")
    parser_download.add_argument("--verbose", action="store_true", help="Enable verbose logging output.")

    parser_train = subparsers.add_parser("train", help="Train the machine learning model.")
    parser_train.add_argument("--config", default="config.yaml", help="Path to configuration file.")
    parser_train.add_argument("--force-download", action="store_true", help="Refresh downloaded data before training.")
    parser_train.add_argument("--verbose", action="store_true", help="Enable verbose logging output.")

    parser_backtest = subparsers.add_parser("backtest", help="Run the backtest using the trained model.")
    parser_backtest.add_argument("--config", default="config.yaml", help="Path to configuration file.")
    parser_backtest.add_argument("--force-download", action="store_true", help="Refresh data before backtesting.")
    parser_backtest.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="(Deprecated) Alias for --long-threshold when only long trades are desired.",
    )
    parser_backtest.add_argument(
        "--long-threshold",
        type=float,
        default=None,
        help="Probability required to enter a long position (defaults to config).",
    )
    parser_backtest.add_argument(
        "--short-threshold",
        type=float,
        default=None,
        help="Probability at or below which to enter a short position (defaults to config).",
    )
    parser_backtest.add_argument("--verbose", action="store_true", help="Enable verbose logging output.")
    parser_backtest.add_argument("--ab-invert", action="store_true", help="Run A/B backtests with and without invert_side.")
    parser_backtest.add_argument("--atr-sweep", action="store_true", help="Sweep over ATR fraction thresholds.")
    parser_backtest.add_argument("--atr-values", type=str, help="Comma-separated list of ATR fraction thresholds for sweep.")
    parser_backtest.add_argument("--save-config", type=str, help="Persist the evaluated configuration to this path.")

    return parser


def _validate_summary(summary: dict) -> None:
    trades = summary.get("trades_count", 0)
    if trades < 300:
        raise AssertionError(f"trades_count too low: {trades}")
    maker_ratio = summary.get("maker_fill_ratio", 0.0)
    if maker_ratio < 0.90:
        raise AssertionError(f"maker_fill_ratio below threshold: {maker_ratio:.3f}")
    slippage = summary.get("avg_slippage_bps", float("inf"))
    if slippage > 2.5:
        raise AssertionError(f"avg_slippage_bps too high: {slippage:.3f}")
    max_dd = summary.get("max_drawdown", 0.0)
    if max_dd < -0.12:
        raise AssertionError(f"max_drawdown too deep: {max_dd:.3f}")
    expectancy = summary.get("expectancy_after_costs", 0.0)
    if expectancy <= 0:
        raise AssertionError(f"expectancy_after_costs non-positive: {expectancy}")
    ann_ret = summary.get("annualised_return", 0.0)
    ann_vol = summary.get("annualised_volatility", 0.0)
    calc_sharpe = summary.get("calc_sharpe")
    target_sharpe = float("inf") if ann_vol == 0 and ann_ret > 0 else (ann_ret / ann_vol if ann_vol else 0.0)
    if calc_sharpe is None:
        raise AssertionError("calc_sharpe missing from summary")
    if math.isfinite(target_sharpe):
        if abs(calc_sharpe - target_sharpe) > 1e-6:
            raise AssertionError(
                f"calc_sharpe mismatch: got {calc_sharpe:.6f}, expected {target_sharpe:.6f}"
            )
    else:
        if math.isfinite(calc_sharpe):
            raise AssertionError(
                f"calc_sharpe should be infinite (target {target_sharpe}), got {calc_sharpe}"
            )


def _parse_atr_values(arg: Optional[str]) -> list[float]:
    if not arg:
        return [0.0015, 0.0020, 0.0025, 0.0030]
    values: list[float] = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError as exc:  # pragma: no cover - CLI validation
            raise ValueError(f"Invalid ATR fraction '{part}'") from exc
    return values


def _run_variant(
    label: str,
    config: AppConfig,
    *,
    force_download: bool,
    long_threshold: Optional[float],
    short_threshold: Optional[float],
    verbose: bool,
) -> dict:
    summary = metadata = None
    try:
        _, result, metadata = execute_backtest(
            config,
            force_download=force_download,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            verbose=verbose,
            enforce_gates=False,
        )
        summary = result.summary
        _validate_summary(summary)
        status = "pass"
        message = ""
    except AssertionError as exc:
        status = "fail"
        message = str(exc)
    except Exception as exc:  # pragma: no cover - defensive
        status = "error"
        message = str(exc)
    payload = {
        "label": label,
        "status": status,
        "summary": summary,
        "metadata": metadata,
    }
    if status != "pass":
        payload["message"] = message
    return payload


def _run_ab_invert(
    base_config: AppConfig,
    *,
    force_download: bool,
    long_threshold: Optional[float],
    short_threshold: Optional[float],
    verbose: bool,
) -> dict:
    runs = []
    cfg_a = deepcopy(base_config)
    cfg_a.backtest.invert_side = False
    runs.append(
        _run_variant(
            "base",
            cfg_a,
            force_download=force_download,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            verbose=verbose,
        )
    )

    cfg_b = deepcopy(base_config)
    cfg_b.backtest.invert_side = True
    runs.append(
        _run_variant(
            "inverted",
            cfg_b,
            force_download=force_download,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            verbose=verbose,
        )
    )

    return {"mode": "ab_invert", "runs": runs}


def _run_atr_sweep(
    base_config: AppConfig,
    atr_values: list[float],
    *,
    force_download: bool,
    long_threshold: Optional[float],
    short_threshold: Optional[float],
    verbose: bool,
) -> dict:
    runs = []
    for value in atr_values:
        cfg = deepcopy(base_config)
        cfg.filters.min_atr_frac = value
        run = _run_variant(
            f"atr={value}",
            cfg,
            force_download=force_download,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            verbose=verbose,
        )
        run["atr_value"] = value
        runs.append(run)
    return {"mode": "atr_sweep", "runs": runs}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "download":
        config = load_config(args.config)
        configure_logging(logging.DEBUG if args.verbose else logging.INFO)
        prepare_dataset(config, force_download=args.force)
    elif args.command == "train":
        metrics = train(args.config, force_download=args.force_download, verbose=args.verbose)
        print(json.dumps(metrics, indent=2))
    elif args.command == "backtest":
        config = load_config(args.config)
        configure_logging(logging.DEBUG if args.verbose else logging.INFO)
        long_threshold = args.long_threshold if args.long_threshold is not None else args.threshold
        short_threshold = args.short_threshold

        payload: dict[str, object] = {}
        auto_ab = args.ab_invert or (not args.atr_sweep and config.data.symbol.upper() == "XBTUSD")

        if auto_ab:
            payload["ab_invert"] = _run_ab_invert(
                config,
                force_download=args.force_download,
                long_threshold=long_threshold,
                short_threshold=short_threshold,
                verbose=args.verbose,
            )
        if args.atr_sweep:
            atr_values = _parse_atr_values(args.atr_values)
            payload["atr_sweep"] = _run_atr_sweep(
                config,
                atr_values,
                force_download=args.force_download,
                long_threshold=long_threshold,
                short_threshold=short_threshold,
                verbose=args.verbose,
            )
        if payload:
            print(json.dumps(payload, indent=2, default=str))
            return

        strategy_output, result, metadata = execute_backtest(
            config,
            force_download=args.force_download,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            verbose=args.verbose,
            enforce_gates=True,
        )

        save_path: Optional[str] = args.save_config
        if not save_path and config.data.symbol.upper() == "XBTUSD":
            save_path = "configs/xbtusd_1h_maker_swing.yaml"
        if save_path:
            save_config(config, save_path)
        preview_series = strategy_output.signals.tail(5)
        preview_dict = {
            (ts.isoformat() if hasattr(ts, "isoformat") else str(ts)): float(value)
            for ts, value in preview_series.items()
        }

        payload = {
            "metadata": metadata,
            "summary": result.summary,
            "signal_preview": preview_dict,
        }
        print(json.dumps(payload, indent=2, default=str))
    else:  # pragma: no cover - argparse prevents this
        parser.print_help()


if __name__ == "__main__":
    main()


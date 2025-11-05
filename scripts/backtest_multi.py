"""Run multi-symbol backtests with portfolio-level exposure caps."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from ai_trading_bot.config import load_config
from ai_trading_bot.orchestration.multi_symbol import run_multi_symbol_backtests


def _parse_symbols(values: Iterable[str] | None) -> List[str]:
    if not values:
        return []
    symbols: List[str] = []
    for value in values:
        if not value:
            continue
        if "," in value:
            symbols.extend(part.strip() for part in value.split(",") if part.strip())
        else:
            symbols.append(value.strip())
    return [symbol for symbol in symbols if symbol]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run backtests across multiple BitMEX symbols.")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file.")
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Symbols to backtest (space or comma separated). Defaults to the config data.symbol.",
    )
    parser.add_argument(
        "--max-cap",
        type=float,
        default=0.6,
        help="Maximum total capital fraction across all symbols.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Refresh data before evaluating each symbol.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path. Defaults to results/multi_backtest_<timestamp>.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        symbols = [config.data.symbol]

    results = run_multi_symbol_backtests(
        config,
        symbols,
        force_download=args.force_download,
        max_portfolio_cap_fraction=float(args.max_cap),
    )
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_path = args.output or Path("results") / f"multi_backtest_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "symbols": symbols,
                "aggregate": results["aggregate"],
                "summaries": results["summaries"],
                "metadata": results["metadata"],
                "generated_at": timestamp,
                "config_path": str(Path(args.config).resolve()),
            },
            handle,
            indent=2,
        )
    print(json.dumps({"output": str(out_path.resolve()), **results["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()

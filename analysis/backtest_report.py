from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

from ai_trading_bot.config import AppConfig, load_config, save_config
from ai_trading_bot.pipeline import backtest


def compute_monthly_stats(equity: pd.Series) -> Tuple[float, float]:
    equity = equity.sort_index()
    monthly_equity = equity.resample("ME").last()
    monthly_returns = monthly_equity.pct_change().dropna()
    avg_monthly = monthly_returns.mean()
    positive_ratio = (monthly_returns > 0).mean()
    return float(avg_monthly), float(positive_ratio)


def run_symbol_backtest(symbol: str, start: str, interval: str, config_path: Path) -> dict:
    config: AppConfig = load_config(config_path)
    config.data.symbol = symbol
    config.data.start_date = start
    config.data.interval = interval
    config.data.end_date = None

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as temp_config:
        tmp_path = Path(temp_config.name)
        save_config(config, tmp_path)

    try:
        _, _, result = backtest(config_path=tmp_path, force_download=True)
    finally:
        tmp_path.unlink(missing_ok=True)

    equity = result.equity_curve
    avg_monthly, positive_ratio = compute_monthly_stats(equity)
    return {
        "symbol": symbol,
        "average_monthly_gain": avg_monthly,
        "monthly_positive_ratio": positive_ratio,
        "trade_win_rate": result.summary.get("win_rate", 0.0),
        "total_return": result.summary.get("total_return", 0.0),
        "annualised_return": result.summary.get("annualised_return", 0.0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-symbol backtests and summarise monthly stats.")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "GOOG"])
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics: List[dict] = []
    for symbol in args.symbols:
        metrics.append(run_symbol_backtest(symbol, args.start, args.interval, args.config))

    df = pd.DataFrame(metrics)
    aggregate = {
        "symbols_tested": args.symbols,
        "average_monthly_gain_mean": float(df["average_monthly_gain"].mean()),
        "average_monthly_gain_std": float(df["average_monthly_gain"].std(ddof=0)),
        "monthly_positive_ratio_mean": float(df["monthly_positive_ratio"].mean()),
        "trade_win_rate_mean": float(df["trade_win_rate"].mean()),
        "total_return_mean": float(df["total_return"].mean()),
    }
    output = {"per_symbol": metrics, "aggregate": aggregate}
    Path("results").mkdir(exist_ok=True)
    report_path = Path("results/backtest_report.json")
    report_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

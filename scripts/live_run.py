"""CLI entrypoint for running the live trading controller."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time

from ai_trading_bot.config import load_config
from ai_trading_bot.live import (
    BitmexExecutionRouter,
    BitmexWebsocketDataFeed,
    ControllerSettings,
    LiveTradingController,
    MLStrategyEngine,
    RiskLimits,
    RiskManager,
)
from ai_trading_bot.monitoring import MetricsCollector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AI trading bot in live (or paper) mode.")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file.")
    parser.add_argument("--symbols", nargs="*", help="Symbols to trade (defaults to config data.symbol).")
    parser.add_argument("--bar-interval", type=int, default=60, help="Bar aggregation interval in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Avoid sending live orders (paper mode).")
    parser.add_argument("--testnet", action="store_true", help="Use BitMEX testnet endpoint.")
    parser.add_argument("--contract-size", type=float, default=1.0, help="Contracts per signal unit.")
    parser.add_argument(
        "--max-position",
        type=float,
        default=50_000.0,
        help="Maximum per-symbol notional exposure in USD terms.",
    )
    parser.add_argument(
        "--max-leverage",
        type=float,
        default=3.0,
        help="Maximum portfolio leverage (notional / equity).",
    )
    parser.add_argument(
        "--max-daily-loss",
        type=float,
        default=1_000.0,
        help="Maximum allowed daily loss before halting trading.",
    )
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=5_000.0,
        help="Maximum cumulative drawdown before halting trading.",
    )
    parser.add_argument(
        "--initial-equity",
        type=float,
        default=0.0,
        help="Initial account equity used for drawdown monitoring.",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        help="Expose Prometheus metrics on this port.",
    )
    parser.add_argument(
        "--strategy-weights",
        help="Comma separated component weights, e.g. 'trend=0.7,mean_reversion=0.3'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    symbols = args.symbols or [config.data.symbol]
    feed = BitmexWebsocketDataFeed(symbols, bar_interval_seconds=args.bar_interval)
    component_weights = None
    if args.strategy_weights:
        component_weights = {}
        for token in args.strategy_weights.split(","):
            token = token.strip()
            if not token:
                continue
            if "=" not in token:
                raise ValueError(f"Invalid strategy weight token '{token}'. Expected 'name=value'.")
            key, raw_val = token.split("=", 1)
            component_weights[key.strip()] = float(raw_val)

    strategy = MLStrategyEngine(config, component_weights=component_weights)
    base_equity = args.initial_equity or config.backtest.initial_capital
    max_daily_loss = args.max_daily_loss if args.max_daily_loss > 0 else base_equity * abs(config.backtest.max_daily_loss_pct or 0.0)
    max_drawdown = args.max_drawdown if args.max_drawdown > 0 else base_equity * abs(config.backtest.max_drawdown_pct or 0.0)
    limits = RiskLimits(
        max_position_notional=args.max_position,
        max_portfolio_leverage=args.max_leverage,
        max_daily_loss=max_daily_loss,
        max_drawdown=max_drawdown,
    )
    risk_manager = RiskManager(limits)

    api_key = os.getenv("BITMEX_API_KEY")
    api_secret = os.getenv("BITMEX_API_SECRET")
    execution = BitmexExecutionRouter(
        api_key=api_key,
        api_secret=api_secret,
        testnet=args.testnet,
        dry_run=args.dry_run or not (api_key and api_secret),
    )

    metrics = None
    if args.metrics_port is not None:
        metrics = MetricsCollector()
        metrics.start_server(args.metrics_port)

    controller = LiveTradingController(
        data_feed=feed,
        strategy_engine=strategy,
        risk_manager=risk_manager,
        execution_router=execution,
        settings=ControllerSettings(default_contract_size=args.contract_size),
        metrics_collector=metrics,
    )

    stop_flag = {"stop": False}

    def _signal_handler(signum, frame):
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    if args.initial_equity:
        controller.update_account_state(equity=float(args.initial_equity), daily_pnl=0.0, drawdown=0.0)

    controller.start()
    try:
        while not stop_flag["stop"]:
            time.sleep(1)
    finally:
        controller.stop()
        execution.close()
        print("Live trading controller stopped.")


if __name__ == "__main__":
    sys.exit(main())

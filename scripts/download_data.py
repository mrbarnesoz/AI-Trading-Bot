"""CLI helper to download and cache market data."""

from __future__ import annotations

import argparse

from ai_trading_bot.config import load_config
from ai_trading_bot.data.fetch import download_price_data
from ai_trading_bot.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OHLCV market data.")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file.")
    parser.add_argument("--symbol", help="Override symbol defined in the config.")
    parser.add_argument("--start", help="Override start date (YYYY-MM-DD).")
    parser.add_argument("--end", help="Override end date (YYYY-MM-DD).")
    parser.add_argument("--interval", help="Override bar interval (e.g. 1d, 1h).")
    parser.add_argument("--force", action="store_true", help="Force refresh and ignore cached data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    config = load_config(args.config)

    if args.symbol:
        config.data.symbol = args.symbol
    if args.start:
        config.data.start_date = args.start
    if args.end:
        config.data.end_date = args.end
    if args.interval:
        config.data.interval = args.interval

    download_price_data(config.data, force_refresh=args.force)


if __name__ == "__main__":
    main()

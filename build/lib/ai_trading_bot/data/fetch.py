"""Utilities for retrieving market data."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from ai_trading_bot.config import DataConfig

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _cache_path(cfg: DataConfig) -> Path:
    end = cfg.end_date or datetime.utcnow().strftime("%Y%m%d")
    filename = f"{cfg.symbol}_{cfg.interval}_{cfg.start_date}_{end}.csv"
    return Path(cfg.cache_dir) / filename


def _normalise_price_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Ensure price data has a flat column index and required fields."""
    df = frame.copy()

    if isinstance(df.columns, pd.MultiIndex):
        if df.columns.nlevels >= 2 and symbol in df.columns.get_level_values(-1):
            df = df.droplevel(axis=1, level=-1)
        else:
            df.columns = ["_".join(str(part) for part in col if part) for col in df.columns.to_flat_index()]

    df.columns = [str(col).strip() for col in df.columns]
    df.columns.name = None

    # Rename any columns that keep the ticker suffix (e.g. "Close AAPL").
    rename_map = {}
    suffix = f" {symbol}"
    for col in df.columns:
        if col.endswith(suffix):
            rename_map[col] = col[: -len(suffix)]
    if rename_map:
        df = df.rename(columns=rename_map)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")

    if "Price" in df.columns and df["Price"].iloc[0] in {"Ticker", symbol}:
        # Occurs when reading legacy CSVs written with MultiIndex headers.
        df = df[df["Price"].astype(str).str.len() > 0]
        df = df[df["Price"] != "Ticker"]
        df = df.rename(columns={"Price": "Date"})
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")

    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.sort_index()

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Input price data is missing required columns after normalisation: {missing}")

    df[REQUIRED_COLUMNS] = df[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=REQUIRED_COLUMNS)
    return df


def download_price_data(cfg: DataConfig, force_refresh: bool = False) -> pd.DataFrame:
    """Download OHLCV data for the configured symbol."""
    cache_file = _cache_path(cfg)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.exists() and not force_refresh:
        logger.info("Loading price data from cache %s", cache_file)
        raw = _read_price_csv(cache_file)
        return _normalise_price_frame(raw, cfg.symbol)

    logger.info("Downloading price data for %s from %s", cfg.symbol, cfg.source)
    data = yf.download(
        tickers=cfg.symbol,
        start=cfg.start_date,
        end=cfg.end_date,
        interval=cfg.interval,
        progress=False,
        auto_adjust=False,
    )

    if data.empty:
        raise ValueError(f"No data retrieved for symbol {cfg.symbol}")

    data = _normalise_price_frame(data.rename_axis("Date"), cfg.symbol)
    data.to_csv(cache_file)
    logger.info("Saved downloaded data to %s", cache_file)
    return data


def _read_price_csv(path: Path) -> pd.DataFrame:
    """Read price data supporting both single and multi-index column variants."""
    try:
        df = pd.read_csv(path, header=[0, 1], index_col=0)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(axis=1, level=-1)
    except ValueError:
        df = pd.read_csv(path, index_col=0)
    return df


def load_price_data(cfg: DataConfig, path: Optional[Path | str] = None) -> pd.DataFrame:
    """Load price data from disk, using the cache path by default."""
    target = Path(path) if path else _cache_path(cfg)
    if not target.exists():
        raise FileNotFoundError(f"Price data file {target} not found. Run the download step first.")
    raw = _read_price_csv(target)
    return _normalise_price_frame(raw, cfg.symbol)

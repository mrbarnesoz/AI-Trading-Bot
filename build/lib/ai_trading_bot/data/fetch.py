"""BitMEX-first data acquisition utilities."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from ai_trading_bot.config import DataConfig

logger = logging.getLogger(__name__)

BITMEX_BASE_URL = "https://www.bitmex.com/api/v1/trade/bucketed"
SUPPORTED_INTERVALS = {"1m", "5m", "15m", "1h", "4h", "1d"}
RESAMPLE_RULES = {"15m": ("1m", "15min"), "4h": ("1h", "4h")}


def _ensure_cache_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _format_label(ts: Optional[pd.Timestamp], *, default: str) -> str:
    if ts is None:
        return default
    return ts.strftime("%Y%m%dT%H%M%SZ")


def _parse_timestamp(value: Optional[str]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    ts = pd.to_datetime(value, utc=True)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _build_cache_path(
    cache_dir: Path,
    symbol: str,
    interval: str,
    start_ts: Optional[pd.Timestamp],
    end_ts: Optional[pd.Timestamp],
) -> Path:
    start_label = _format_label(start_ts, default="start")
    end_label = _format_label(end_ts, default="now")
    filename = f"bitmex_{symbol}_{interval}_{start_label}_{end_label}.csv"
    return cache_dir / filename


def _request_bitmex_page(
    symbol: str,
    bin_size: str,
    start_time: Optional[pd.Timestamp],
    end_time: Optional[pd.Timestamp],
    limit: int,
) -> list[dict]:
    params = {
        "binSize": bin_size,
        "symbol": symbol,
        "partial": "false",
        "count": str(limit),
        "reverse": "false",
    }
    if start_time is not None:
        params["startTime"] = start_time.isoformat().replace("+00:00", "Z")
    if end_time is not None:
        params["endTime"] = end_time.isoformat().replace("+00:00", "Z")

    response = requests.get(BITMEX_BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected response from BitMEX: {payload!r}")
    return payload


def _postprocess_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame["Date"] = pd.to_datetime(frame["Date"], utc=True)
    frame = frame.set_index("Date").sort_index()
    frame.index.name = "Date"
    for column in ("Open", "High", "Low", "Close", "Volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
    frame["Volume"] = frame["Volume"].fillna(0.0)
    return frame


def fetch_bitmex_ohlcv(
    symbol: str,
    interval: str,
    start: Optional[str],
    end: Optional[str],
    out_csv: Path,
) -> pd.DataFrame:
    """Download BitMEX OHLCV data and persist it to ``out_csv``."""
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"Unsupported interval '{interval}'. Supported: {sorted(SUPPORTED_INTERVALS)}")

    api_interval, resample_rule = RESAMPLE_RULES.get(interval, (interval, None))
    start_ts = _parse_timestamp(start)
    end_ts = _parse_timestamp(end)

    rows: list[dict] = []
    next_start = start_ts
    while True:
        batch = _request_bitmex_page(symbol, api_interval, next_start, end_ts, limit=1000)
        if not batch:
            break

        for entry in batch:
            rows.append(
                {
                    "Date": entry["timestamp"],
                    "Open": entry["open"],
                    "High": entry["high"],
                    "Low": entry["low"],
                    "Close": entry["close"],
                    "Volume": entry.get("volume", entry.get("homeNotional", 0.0)),
                }
            )

        if len(batch) < 1000:
            break

        last_ts = pd.to_datetime(batch[-1]["timestamp"], utc=True)
        next_start = last_ts + pd.Timedelta(microseconds=1)

        if end_ts is not None and next_start >= end_ts:
            break

    if not rows:
        raise ValueError(f"No BitMEX OHLCV data for {symbol} interval={interval} start={start} end={end}")

    df = _postprocess_frame(pd.DataFrame(rows))
    df = df[~df.index.duplicated(keep="last")]

    if resample_rule is not None:
        df = df.resample(
            resample_rule,
            label="right",
            closed="left",
            origin=df.index[0],
        ).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        df = df.dropna(subset=["Open", "High", "Low", "Close"])

    df.to_csv(out_csv)
    logger.info("Saved BitMEX data for %s (%s) to %s", symbol, interval, out_csv)
    return df


def get_price_data(cfg: DataConfig, force_download: bool = False) -> pd.DataFrame:
    """Retrieve price data according to the configuration."""
    if cfg.source.lower() != "bitmex":
        raise ValueError(
            f"Unsupported data source '{cfg.source}'. Set data.source to 'bitmex' in config.yaml."
        )

    cache_dir = Path(cfg.cache_dir or "data/raw")
    _ensure_cache_dir(cache_dir)

    start_ts = _parse_timestamp(cfg.start_date)
    end_ts = _parse_timestamp(cfg.end_date)
    cache_path = _build_cache_path(cache_dir, cfg.symbol, cfg.interval, start_ts, end_ts)

    if cache_path.exists() and not force_download:
        logger.info("Loading price data for %s from cache %s", cfg.symbol, cache_path)
        data = pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
        if data.index.tzinfo is None:
            data.index = data.index.tz_localize("UTC")
        else:
            data.index = data.index.tz_convert("UTC")
        return data.sort_index()

    logger.info("Downloading price data for %s from bitmex", cfg.symbol)
    df = fetch_bitmex_ohlcv(cfg.symbol, cfg.interval, cfg.start_date, cfg.end_date, cache_path)
    return df.sort_index()


__all__ = ["fetch_bitmex_ohlcv", "get_price_data"]

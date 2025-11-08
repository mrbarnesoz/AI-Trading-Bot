"""BitMEX-first data acquisition utilities."""

from __future__ import annotations

import logging
import os
import time
from datetime import timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import requests

from ai_trading_bot.config import DataConfig

logger = logging.getLogger(__name__)

BITMEX_BASE_URL = "https://www.bitmex.com/api/v1/trade/bucketed"
SUPPORTED_INTERVALS = {"1m", "5m", "15m", "1h", "4h", "1d"}
RESAMPLE_RULES = {"15m": ("1m", "15min"), "4h": ("1h", "4h")}
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
BITMEX_MAX_RETRIES = int(os.getenv("BITMEX_MAX_RETRIES", "6"))
BITMEX_RETRY_BACKOFF = float(os.getenv("BITMEX_RETRY_BACKOFF", "1.5"))
BITMEX_MAX_BACKOFF = float(os.getenv("BITMEX_MAX_BACKOFF", "30"))
BITMEX_REQUESTS_PER_SECOND = float(os.getenv("BITMEX_REQUESTS_PER_SECOND", "1.5"))
BITMEX_CHUNK_SLEEP = float(os.getenv("BITMEX_CHUNK_SLEEP", "0.25"))
DEFAULT_CHUNK_DAYS = {
    "1m": 3,
    "5m": 7,
    "15m": 30,
    "1h": 120,
    "4h": 365,
    "1d": 365,
}
REQUIRED_OHLC_KEYS = frozenset({"timestamp", "open", "high", "low", "close"})

_LAST_REQUEST_TS = 0.0


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


def _respect_rate_limit() -> None:
    global _LAST_REQUEST_TS
    if BITMEX_REQUESTS_PER_SECOND <= 0:
        return
    min_interval = 1.0 / BITMEX_REQUESTS_PER_SECOND
    now = time.perf_counter()
    wait = (_LAST_REQUEST_TS + min_interval) - now
    if wait > 0:
        time.sleep(wait)
        now = time.perf_counter()
    _LAST_REQUEST_TS = now


def _retry_delay(response: Optional[requests.Response], attempt: int) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(float(retry_after), 0.0)
            except ValueError:
                logger.debug("Invalid Retry-After header value %s", retry_after)
    backoff = BITMEX_RETRY_BACKOFF * (2 ** (attempt - 1))
    return min(backoff, BITMEX_MAX_BACKOFF)


def _chunk_size_for_interval(bin_size: str) -> pd.Timedelta:
    env_key = f"BITMEX_CHUNK_DAYS_{bin_size.upper()}"
    override = os.getenv(env_key)
    if override:
        try:
            days = max(1, int(override))
        except ValueError:
            logger.warning("Invalid %s override '%s'. Falling back to default.", env_key, override)
            days = DEFAULT_CHUNK_DAYS.get(bin_size, 90)
    else:
        days = DEFAULT_CHUNK_DAYS.get(bin_size, 90)
    return pd.Timedelta(days=days)


def _iter_chunk_ranges(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    chunk_size: pd.Timedelta,
) -> Iterable[Tuple[pd.Timestamp, pd.Timestamp]]:
    cursor = start_ts
    while cursor < end_ts:
        chunk_end = min(cursor + chunk_size, end_ts)
        yield cursor, chunk_end
        cursor = chunk_end


def _preview_entry(entry: object, max_len: int = 200) -> str:
    text = repr(entry)
    if len(text) > max_len:
        text = text[:max_len] + "..."
    return text


def _validate_payload_rows(payload: List[dict]) -> Tuple[bool, Optional[dict]]:
    """Ensure every entry contains the required BitMEX OHLC fields."""
    for entry in payload:
        if not isinstance(entry, dict):
            return False, entry
        missing = REQUIRED_OHLC_KEYS - set(entry.keys())
        if missing:
            return False, entry
    return True, None


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

    attempt = 0
    while True:
        attempt += 1
        _respect_rate_limit()
        try:
            response = requests.get(BITMEX_BASE_URL, params=params, timeout=30)
        except requests.RequestException as exc:
            if attempt > BITMEX_MAX_RETRIES:
                raise
            delay = _retry_delay(None, attempt)
            logger.warning("BitMEX request error (%s). Retrying in %.1fs", exc, delay)
            time.sleep(delay)
            continue

        if response.status_code in RETRYABLE_STATUS:
            if attempt > BITMEX_MAX_RETRIES:
                response.raise_for_status()
            delay = _retry_delay(response, attempt)
            logger.warning(
                "BitMEX rate/availability limit (%s) for %s %s. Sleeping %.1fs before retry.",
                response.status_code,
                symbol,
                bin_size,
                delay,
            )
            time.sleep(delay)
            continue

        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError as exc:  # pragma: no cover - network failures
            raise ValueError("Failed to decode BitMEX response as JSON") from exc

        if isinstance(payload, dict) and "error" in payload:
            error = payload["error"]
            message = error.get("message") if isinstance(error, dict) else str(error)
            raise ValueError(f"BitMEX error: {message}")

        if not isinstance(payload, list):
            raise ValueError(f"Unexpected response from BitMEX: {payload!r}")

        valid_rows, bad_entry = _validate_payload_rows(payload)
        if not valid_rows:
            if attempt > BITMEX_MAX_RETRIES:
                raise ValueError(
                    f"BitMEX payload missing OHLC keys after retries: {_preview_entry(bad_entry)}"
                )
            delay = _retry_delay(response, attempt)
            logger.warning(
                "BitMEX payload missing OHLC fields (sample=%s). Retrying in %.1fs",
                _preview_entry(bad_entry),
                delay,
            )
            time.sleep(delay)
            continue
        return payload


def _download_chunk(
    symbol: str,
    bin_size: str,
    chunk_start: Optional[pd.Timestamp],
    chunk_end: Optional[pd.Timestamp],
) -> List[dict]:
    rows: List[dict] = []
    next_start = chunk_start
    while True:
        if chunk_end is not None and next_start is not None and next_start >= chunk_end:
            break

        batch = _request_bitmex_page(symbol, bin_size, next_start, chunk_end, limit=1000)
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
        if chunk_end is not None and next_start >= chunk_end:
            break

    return rows


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
    download_end_ts = end_ts or pd.Timestamp.now(tz=timezone.utc)

    rows: List[dict] = []
    if start_ts is not None and download_end_ts is not None:
        chunk_size = _chunk_size_for_interval(api_interval)
        chunk_ranges = list(_iter_chunk_ranges(start_ts, download_end_ts, chunk_size))
        if not chunk_ranges:
            chunk_ranges = [(start_ts, download_end_ts)]
    else:
        chunk_ranges = [(start_ts, download_end_ts)]

    total_chunks = len(chunk_ranges)
    for idx, (chunk_start, chunk_end) in enumerate(chunk_ranges, start=1):
        logger.debug(
            "Downloading BitMEX chunk %s/%s for %s (%s -> %s)",
            idx,
            total_chunks,
            symbol,
            chunk_start or "start",
            chunk_end or "now",
        )
        rows.extend(_download_chunk(symbol, api_interval, chunk_start, chunk_end))
        if BITMEX_CHUNK_SLEEP > 0 and idx < total_chunks:
            time.sleep(BITMEX_CHUNK_SLEEP)

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

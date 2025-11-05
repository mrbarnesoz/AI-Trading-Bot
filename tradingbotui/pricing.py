"""Utility helpers for fetching asset and FX prices."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

try:  # pragma: no cover - optional dependency
    import yfinance as yf
except Exception:  # pragma: no cover - environments without yfinance
    yf = None

logger = logging.getLogger("tradingbot.ui.pricing")

SYMBOL_TO_TICKER = {
    "XBTUSD": "BTC-USD",
    "BTCUSD": "BTC-USD",
    "BTCUSDT": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "ETHUSDT": "ETH-USD",
    "SOLUSD": "SOL-USD",
    "XRPUSD": "XRP-USD",
    "DOGEUSD": "DOGE-USD",
    "ADAUSD": "ADA-USD",
}

DEFAULT_PRICE_USD = {
    "XBTUSD": float(os.getenv("TRADINGBOT_DEFAULT_BTC_USD", "35000")),
    "BTCUSD": float(os.getenv("TRADINGBOT_DEFAULT_BTC_USD", "35000")),
    "BTCUSDT": float(os.getenv("TRADINGBOT_DEFAULT_BTC_USD", "35000")),
    "ETHUSD": float(os.getenv("TRADINGBOT_DEFAULT_ETH_USD", "1800")),
    "ETHUSDT": float(os.getenv("TRADINGBOT_DEFAULT_ETH_USD", "1800")),
    "SOLUSD": float(os.getenv("TRADINGBOT_DEFAULT_SOL_USD", "35")),
    "XRPUSD": float(os.getenv("TRADINGBOT_DEFAULT_XRP_USD", "0.6")),
    "DOGEUSD": float(os.getenv("TRADINGBOT_DEFAULT_DOGE_USD", "0.1")),
    "ADAUSD": float(os.getenv("TRADINGBOT_DEFAULT_ADA_USD", "0.35")),
}

DEFAULT_USD_TO_AUD = float(os.getenv("TRADINGBOT_DEFAULT_USD_TO_AUD", "1.5"))

PRICE_CACHE: Dict[Tuple[str, str], Dict[str, float]] = {}
FX_CACHE: Dict[str, float] = {}


def _parse_timestamp(value: Optional[str]) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        try:
            return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)


def _fetch_price_yf(ticker: str, timestamp: datetime) -> Optional[float]:
    if yf is None:
        return None

    try:
        start = (timestamp - timedelta(days=2)).strftime("%Y-%m-%d")
        end = (timestamp + timedelta(days=1)).strftime("%Y-%m-%d")
        data = yf.download(ticker, start=start, end=end, interval="60m", progress=False)
        if data.empty:
            data = yf.download(ticker, start=start, end=end, interval="1d", progress=False)
        if data.empty:
            return None
        ts = timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        ts = ts.astimezone(timezone.utc)
        try:
            row = data.loc[data.index.get_loc(ts, method="nearest")]
        except KeyError:
            row = data.iloc[-1]
        price = float(row["Close"])
        return price if price > 0 else None
    except Exception as exc:  # pragma: no cover - network errors
        logger.debug("yfinance price fetch failed for %s: %s", ticker, exc)
        return None


def _fallback_price(symbol: str) -> float:
    return DEFAULT_PRICE_USD.get(symbol.upper(), float(os.getenv("TRADINGBOT_DEFAULT_PRICE_USD", "30000")))


def _get_price_usd(symbol: str, timestamp: datetime) -> float:
    symbol = (symbol or "").upper()
    date_key = timestamp.strftime("%Y-%m-%d")
    cache_key = (symbol, date_key)
    cached = PRICE_CACHE.get(cache_key)
    if cached and "price_usd" in cached:
        return cached["price_usd"]

    ticker = SYMBOL_TO_TICKER.get(symbol)
    price = _fetch_price_yf(ticker, timestamp) if ticker else None
    if price is None:
        price = _fallback_price(symbol)
    PRICE_CACHE[cache_key] = {"price_usd": price}
    return price


def _fetch_fx_yf(timestamp: datetime) -> Optional[float]:
    if yf is None:
        return None
    try:
        start = (timestamp - timedelta(days=2)).strftime("%Y-%m-%d")
        end = (timestamp + timedelta(days=1)).strftime("%Y-%m-%d")
        data = yf.download("AUDUSD=X", start=start, end=end, interval="1d", progress=False)
        if data.empty:
            return None
        rate = float(data["Close"].iloc[-1])
        return rate if rate > 0 else None
    except Exception as exc:  # pragma: no cover - network errors
        logger.debug("yfinance FX fetch failed: %s", exc)
        return None


def _get_usd_to_aud(timestamp: datetime) -> float:
    date_key = timestamp.strftime("%Y-%m-%d")
    if date_key in FX_CACHE:
        return FX_CACHE[date_key]
    rate = _fetch_fx_yf(timestamp)
    if rate is None or rate <= 0:
        rate = DEFAULT_USD_TO_AUD
    else:
        rate = 1.0 / rate  # AUDUSD gives USD per AUD; invert for USD->AUD
    FX_CACHE[date_key] = rate
    return rate


def get_price_info(symbol: str, timestamp: Optional[str]) -> Dict[str, float]:
    """Return pricing information for the given symbol at the provided timestamp."""
    ts = _parse_timestamp(timestamp)
    price_usd = _get_price_usd(symbol, ts)
    usd_to_aud = _get_usd_to_aud(ts)
    return {
        "price_usd": price_usd,
        "usd_to_aud": usd_to_aud,
    }

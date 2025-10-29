"""Dataset preparation pipeline that supports yfinance and BitMEX OHLCV sources."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import requests

from ai_trading_bot.config import FeatureConfig, PipelineConfig
from ai_trading_bot.features.indicators import engineer_features

# yfinance only imported when needed


def _ensure_dir(path: Path | str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _utc_index(df: pd.DataFrame, ts_col: str = "Date") -> pd.DataFrame:
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.set_index(ts_col).sort_index()
    df.index.name = "Date"
    return df


def _pct_change_forward(series: pd.Series, periods: int = 1) -> pd.Series:
    return series.shift(-periods) / series - 1.0


def _label_binary_from_return(ret: pd.Series) -> pd.Series:
    return (ret > 0).astype(np.int8)


def _coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _as_numeric_series(value: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(value, pd.DataFrame):
        series = value.squeeze(axis="columns")
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
    else:
        series = value
    series = pd.to_numeric(series, errors="coerce")
    return series


def _sma(series: pd.Series | pd.DataFrame, window: int) -> pd.Series:
    series = _as_numeric_series(series)
    return series.rolling(window, min_periods=window).mean()


def _ema(series: pd.Series | pd.DataFrame, window: int) -> pd.Series:
    series = _as_numeric_series(series)
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def _rsi(close: pd.Series | pd.DataFrame, window: int = 14) -> pd.Series:
    close = _as_numeric_series(close)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    roll_up = gain.rolling(window, min_periods=window).mean()
    roll_down = loss.rolling(window, min_periods=window).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series | pd.DataFrame, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    close = _as_numeric_series(close)
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow
    sig = _ema(macd, signal)
    hist = macd - sig
    return macd, sig, hist


def _cache_path(cache_dir: Path, prefix: str, symbol: str, interval: str, start: str, end: str | None) -> Path:
    suffix = end or "now"
    return cache_dir / f"{prefix}_{symbol}_{interval}_{start}_{suffix}.csv"


def _download_yfinance(symbol: str, interval: str, start: str, end: str | None, cache_dir: Path) -> pd.DataFrame:
    import yfinance as yf

    cache_path = _cache_path(cache_dir, "yf", symbol, interval, start, end)
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if "Date" in cached.columns:
            cached = cached.dropna(subset=["Date"])
        cached = _coerce_numeric_columns(cached, ("Open", "High", "Low", "Close", "Adj Close", "Volume"))
        return _utc_index(cached, "Date")

    data = yf.download(
        symbol,
        interval=interval,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )
    if data.empty:
        raise ValueError(f"No yfinance data for {symbol} interval={interval} start={start} end={end}")

    if getattr(data, "columns", pd.Index([])).nlevels > 1:
        data.columns = [str(col).title() for col in data.columns.get_level_values(0)]
    else:
        data.columns = [str(col).title() for col in data.columns]

    data = data.reset_index()
    data = _coerce_numeric_columns(data, ("Open", "High", "Low", "Close", "Adj Close", "Volume"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(cache_path, index=False)
    return _utc_index(data, "Date")


def _download_bitmex(
    symbol: str,
    interval: str,
    start: str,
    end: str | None,
    cache_dir: Path,
    limit: int = 1000,
) -> pd.DataFrame:
    cache_path = _cache_path(cache_dir, "bitmex", symbol, interval, start, end)
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if "Date" in cached.columns:
            cached = cached.dropna(subset=["Date"])
        cached = _coerce_numeric_columns(cached, ("Open", "High", "Low", "Close", "Volume"))
        return _utc_index(cached, "Date")

    base = "https://www.bitmex.com/api/v1/trade/bucketed"
    params: Dict[str, str] = {
        "binSize": interval,
        "symbol": symbol,
        "partial": "false",
        "count": str(limit),
        "reverse": "false",
    }
    if start:
        params["startTime"] = start
    if end:
        params["endTime"] = end

    rows: list[dict] = []
    while True:
        response = requests.get(base, params=params, timeout=30)
        response.raise_for_status()
        batch = response.json()
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
        if len(batch) < limit:
            break
        params["startTime"] = batch[-1]["timestamp"]

    if not rows:
        raise ValueError(f"No BitMEX OHLCV data for {symbol} interval={interval} start={start} end={end}")

    df = _utc_index(pd.DataFrame(rows), "Date")
    df = _coerce_numeric_columns(df, ("Open", "High", "Low", "Close", "Volume"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=True)
    return df


def _aggregate_interval(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    if interval in {"1m", "5m", "1h", "4h", "1d"}:
        return df
    if interval == "15m":
        rule = "15T"
    elif interval == "4h":
        rule = "4H"
    else:
        raise ValueError(f"Unsupported interval aggregation: {interval}")

    agg = df.resample(rule).agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    return agg.dropna()


def prepare_dataset(config: dict, force_download: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_cfg = config["data"]
    source = data_cfg["source"].lower()
    symbol = data_cfg["symbol"]
    interval = data_cfg["interval"]
    start = data_cfg["start_date"]
    end = data_cfg["end_date"]
    cache_dir = Path(data_cfg["cache_dir"])
    _ensure_dir(cache_dir)

    if force_download:
        for path in cache_dir.glob(f"*{symbol}*{interval}*"):
            try:
                path.unlink()
            except OSError:
                pass

    if source == "yfinance":
        raw_df = _download_yfinance(symbol, interval, start, end, cache_dir)
    elif source in {"bitmex", "bitmex_ohlcv"}:
        interval_map = {"1m": "1m", "5m": "5m", "15m": "5m", "1h": "1h", "4h": "1h", "1d": "1d"}
        bitmex_interval = interval_map.get(interval, interval)
        raw_df = _download_bitmex(symbol, bitmex_interval, start, end, cache_dir)
        if interval in {"15m", "4h"}:
            raw_df = _aggregate_interval(raw_df, interval)
    else:
        raise ValueError(f"Unknown data source: {source}")

    if "Adj Close" not in raw_df.columns:
        raw_df["Adj Close"] = raw_df["Close"]

    feature_cfg = config["features"]
    pipeline_cfg = config["pipeline"]
    feature_config = FeatureConfig(**feature_cfg)
    pipeline_config = PipelineConfig(**pipeline_cfg)
    engineered_df = engineer_features(raw_df, feature_config, pipeline_config)

    float_cols = engineered_df.select_dtypes(include=["float64"]).columns
    engineered_df[float_cols] = engineered_df[float_cols].astype(np.float32)

    engineered_dir = Path("data/engineered")
    _ensure_dir(engineered_dir)
    out_name = f"{symbol}_{interval}_{start}_{end or 'now'}"
    engineered_df.to_csv(engineered_dir / f"{out_name}.csv", index=True)
    engineered_df.to_parquet(engineered_dir / f"{out_name}.parquet")

    decision_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "target_return", "target"]
    decision = engineered_df[decision_columns].copy()
    return decision, engineered_df


__all__ = ["prepare_dataset"]

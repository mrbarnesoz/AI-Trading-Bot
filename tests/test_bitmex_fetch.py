from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from ai_trading_bot.config import DataConfig
from ai_trading_bot.data.fetch import fetch_bitmex_ohlcv, get_price_data


class _FakeResponse:
    def __init__(self, payload: List[Dict[str, Any]]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:  # pragma: no cover - nothing to raise in tests
        return

    def json(self) -> List[Dict[str, Any]]:
        return self._payload


def test_fetch_bitmex_ohlcv_basic(monkeypatch, tmp_path):
    batches = [
        [
            {
                "timestamp": "2024-01-01T00:00:00.000Z",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 5000.0,
            },
            {
                "timestamp": "2024-01-01T01:00:00.000Z",
                "open": 100.5,
                "high": 102.0,
                "low": 100.0,
                "close": 101.5,
                "volume": 4200.0,
            },
        ],
        [],
    ]
    calls = {"count": 0}

    def fake_get(url, params, timeout):
        idx = min(calls["count"], len(batches) - 1)
        calls["count"] += 1
        return _FakeResponse(batches[idx])

    monkeypatch.setattr("ai_trading_bot.data.fetch.requests.get", fake_get)

    out_csv = tmp_path / "cache.csv"
    df = fetch_bitmex_ohlcv("XBTUSD", "1h", "2024-01-01T00:00:00Z", "2024-01-01T02:00:00Z", out_csv)

    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.index.tzinfo is not None and df.index.tzinfo.utcoffset(None).total_seconds() == 0
    assert len(df) == 2
    assert out_csv.exists()


def test_fetch_bitmex_ohlcv_resample_15m(monkeypatch, tmp_path):
    base_rows = []
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    for i in range(15):
        ts = start + pd.Timedelta(minutes=i)
        base_rows.append(
            {
                "timestamp": ts.isoformat().replace("+00:00", "Z"),
                "open": 100 + i,
                "high": 101 + i,
                "low": 99 + i,
                "close": 100.5 + i,
                "volume": 1000 + i,
            }
        )
    batches = [base_rows, []]
    calls = {"count": 0}

    def fake_get(url, params, timeout):
        idx = min(calls["count"], len(batches) - 1)
        calls["count"] += 1
        return _FakeResponse(batches[idx])

    monkeypatch.setattr("ai_trading_bot.data.fetch.requests.get", fake_get)

    out_csv = tmp_path / "resampled.csv"
    df = fetch_bitmex_ohlcv("XBTUSD", "15m", "2024-01-01T00:00:00Z", "2024-01-01T04:00:00Z", out_csv)

    assert len(df) == 1
    row = df.iloc[0]
    assert pytest.approx(row["Open"]) == 100.0
    assert pytest.approx(row["Close"]) == 114.5
    assert pytest.approx(row["Volume"]) == sum(1000 + i for i in range(15))


def test_get_price_data_uses_cache(monkeypatch, tmp_path, caplog):
    cache_file = tmp_path / "bitmex_XBTUSD_1h_20240101T000000Z_now.csv"
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
            "Open": [1, 2],
            "High": [2, 3],
            "Low": [0, 1],
            "Close": [1.5, 2.5],
            "Volume": [10, 20],
        }
    )
    df.to_csv(cache_file, index=False)

    cfg = DataConfig(
        symbol="XBTUSD",
        interval="1h",
        start_date="2024-01-01T00:00:00Z",
        end_date=None,
        source="bitmex",
        cache_dir=str(tmp_path),
    )

    data = get_price_data(cfg, force_download=False)
    assert len(data) == 2
    assert list(data.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert data.index.tzinfo is not None

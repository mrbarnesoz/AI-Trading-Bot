"""Ingest BitMEX funding and open interest data into bronze Parquet tables."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable

import httpx
import polars as pl

from utils import io as io_utils
from utils.time import ensure_utc_ns

logger = logging.getLogger(__name__)


class FundingOpenInterestIngestor:
    """Fetch and persist BitMEX funding and open interest series."""

    def __init__(self, base_url: str, output_root: Path, timeout_s: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.output_root = output_root
        self.timeout_s = timeout_s

    def fetch_funding(self, symbols: Iterable[str], start: datetime, end: datetime) -> Path:
        params = {"symbol": ",".join(symbols), "startTime": start.isoformat(), "endTime": end.isoformat()}
        url = f"{self.base_url}/funding"
        logger.info("Requesting funding data %s", url)
        response = httpx.get(url, params=params, timeout=self.timeout_s)
        response.raise_for_status()
        data = response.json()
        df = pl.DataFrame(data).rename({"timestamp": "ts"})
        df = df.with_columns(
            [
                ensure_utc_ns(pl.col("ts")),
                pl.col("fundingRate").cast(pl.Float64).alias("funding_rate"),
                pl.col("predictedFundingRate").cast(pl.Float64).alias("predicted_rate"),
            ]
        ).select(["ts", "symbol", "funding_rate", "predicted_rate"])
        path = io_utils.write_partitioned(df, self.output_root / "funding", partition_cols=["dt", "symbol"])
        logger.info("Stored funding series to %s", path)
        return path

    def fetch_open_interest(self, symbols: Iterable[str], start: datetime, end: datetime) -> Path:
        params = {"symbol": ",".join(symbols), "startTime": start.isoformat(), "endTime": end.isoformat()}
        url = f"{self.base_url}/openInterest"
        logger.info("Requesting open interest data %s", url)
        response = httpx.get(url, params=params, timeout=self.timeout_s)
        response.raise_for_status()
        data = response.json()
        df = pl.DataFrame(data).rename({"timestamp": "ts"})
        df = df.with_columns(
            [
                ensure_utc_ns(pl.col("ts")),
                pl.col("openInterest").cast(pl.Float64).alias("open_interest"),
            ]
        ).select(["ts", "symbol", "open_interest"])
        path = io_utils.write_partitioned(df, self.output_root / "open_interest", partition_cols=["dt", "symbol"])
        logger.info("Stored open interest series to %s", path)
        return path

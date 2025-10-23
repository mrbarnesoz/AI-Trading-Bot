#!/usr/bin/env python
"""Download, decompress, and store BitMEX trade/quote data as Parquet.

Example:
    python scripts/download_bitmex_data.py \
        --symbols XBTUSD ETHUSD \
        --start 2023-09-01 --end 2023-09-05 \
        --datasets trade quote \
        --output-root data/raw
"""

from __future__ import annotations

import argparse
import gzip
import io
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import polars as pl
import requests

BASE_URL = "https://public.bitmex.com/data"
VALID_DATASETS = {"trade", "quote"}


def iter_dates(start: date, end: date) -> Iterator[date]:
    current = start
    step = timedelta(days=1)
    while current <= end:
        yield current
        current += step


def candidate_urls(dataset: str, symbol: str, day: date) -> List[str]:
    day_str = day.isoformat()
    return [
        f"{BASE_URL}/{dataset}/{symbol}/{day_str}.csv.gz",
        f"{BASE_URL}/{dataset}/{day_str}.csv.gz",
        f"{BASE_URL}/{dataset}/{symbol}/{day_str}.json.gz",
        f"{BASE_URL}/{dataset}/{day_str}.json.gz",
    ]


def download_file(urls: Iterable[str], timeout: float = 30.0) -> Optional[bytes]:
    for url in urls:
        try:
            response = requests.get(url, stream=True, timeout=timeout)
        except requests.RequestException as exc:
            logging.warning("Request failed for %s: %s", url, exc)
            continue

        if response.status_code == 404:
            logging.debug("Not found: %s", url)
            continue
        response.raise_for_status()
        logging.info("Downloading %s (%s bytes)", url, response.headers.get("Content-Length", "?"))
        buffer = io.BytesIO()
        for chunk in response.iter_content(chunk_size=1 << 20):
            if chunk:
                buffer.write(chunk)
        return buffer.getvalue()
    return None


def parse_to_dataframe(data: bytes) -> pl.DataFrame:
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
        df = pl.read_csv(gz, try_parse_dates=False, ignore_errors=True)
    if "timestamp" in df.columns:
        df = df.with_columns(
            pl.col("timestamp")
            .cast(pl.Utf8)
            .str.replace("D", " ")
            .str.replace("Z", "")
            .strptime(pl.Datetime(time_unit="us", time_zone="UTC"), format="%Y-%m-%d %H:%M:%S%.f", strict=False)
        )
    return df


def normalise_for_dataset(dataset: str, df: pl.DataFrame) -> pl.DataFrame:
    if dataset == "trade":
        expected_cols = [
            "timestamp",
            "symbol",
            "side",
            "size",
            "price",
            "trdMatchID",
            "grossValue",
            "homeNotional",
            "foreignNotional",
        ]
        for col in expected_cols:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).alias(col))
        df = df.select(expected_cols)
    elif dataset == "quote":
        expected_cols = [
            "timestamp",
            "symbol",
            "bidSize",
            "bidPrice",
            "askPrice",
            "askSize",
        ]
        for col in expected_cols:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).alias(col))
        df = df.select(expected_cols)
    return df


def write_parquet(df: pl.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path, compression="zstd")
    logging.info("Wrote %s (%d rows)", output_path, df.height)


def process_day(dataset: str, symbol: str, day: date, output_root: Path, skip_existing: bool = True) -> None:
    rel_path = Path(dataset) / f"dt={day.isoformat()}" / f"symbol={symbol}" / "data.parquet"
    output_path = output_root / rel_path
    if skip_existing and output_path.exists():
        logging.info("Skipping existing %s", output_path)
        return

    data = download_file(candidate_urls(dataset, symbol, day))
    if data is None:
        logging.warning("No data found for %s %s %s", dataset, symbol, day)
        return

    df = parse_to_dataframe(data)
    if df.is_empty():
        logging.warning("Empty dataset for %s %s %s", dataset, symbol, day)
        return
    df = normalise_for_dataset(dataset, df)
    write_parquet(df, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download BitMEX trade/quote data and store as Parquet.")
    parser.add_argument("--symbols", nargs="+", required=True, help="Symbols to download (e.g., XBTUSD)")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(VALID_DATASETS),
        default=["trade"],
        help="Datasets to download (default: trade)",
    )
    parser.add_argument(
        "--output-root",
        default="data/raw/bitmex",
        help="Root directory for storing Parquet files (default: data/raw/bitmex)",
    )
    parser.add_argument("--no-skip", dest="skip_existing", action="store_false", help="Re-download even if files exist")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    start_date = datetime.fromisoformat(args.start).date()
    end_date = datetime.fromisoformat(args.end).date()
    output_root = Path(args.output_root)

    for dataset in args.datasets:
        logging.info("=== Dataset: %s ===", dataset)
        for symbol in args.symbols:
            logging.info("--- Symbol: %s ---", symbol)
            for day in iter_dates(start_date, end_date):
                process_day(dataset, symbol, day, output_root, skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()

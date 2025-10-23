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
from typing import Iterable, Iterator, List, Optional, Sequence

import polars as pl
import requests

BASE_URLS = [
    "https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data",
    "https://public.bitmex.com/data",
]
VALID_DATASETS = {"trade", "quote"}


def iter_dates(start: date, end: date) -> Iterator[date]:
    current = start
    step = timedelta(days=1)
    while current <= end:
        yield current
        current += step


def candidate_urls(dataset: str, day: date, symbols: Optional[Sequence[str]] = None) -> List[str]:
    """Return potential archive URLs for a dataset/day combination.

    BitMEX stores archives in YYYYMMDD format but some historical mirrors
    include ISO dates. We try both along with CSV/JSON variants.
    """
    day_iso = day.isoformat()
    day_compact = day.strftime("%Y%m%d")
    urls: List[str] = []
    for base in BASE_URLS:
        urls.extend(
            [
                f"{base}/{dataset}/{day_iso}.csv.gz",
                f"{base}/{dataset}/{day_compact}.csv.gz",
                f"{base}/{dataset}/{day_iso}.json.gz",
                f"{base}/{dataset}/{day_compact}.json.gz",
            ]
        )
        if symbols:
            for symbol in symbols:
                urls.extend(
                    [
                        f"{base}/{dataset}/{symbol}/{day_iso}.csv.gz",
                        f"{base}/{dataset}/{symbol}/{day_compact}.csv.gz",
                        f"{base}/{dataset}/{symbol}/{day_iso}.json.gz",
                        f"{base}/{dataset}/{symbol}/{day_compact}.json.gz",
                    ]
                )
    # Preserve order while dropping duplicates.
    seen = set()
    deduped = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


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
            .str.strptime(
                pl.Datetime(time_unit="us", time_zone="UTC"),
                format="%Y-%m-%d %H:%M:%S%.f",
                strict=False,
            )
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


def process_day(dataset: str, symbols: List[str], day: date, output_root: Path, skip_existing: bool = True) -> None:
    day_dir = Path(dataset) / f"dt={day.isoformat()}"
    outputs = {
        symbol: output_root / day_dir / f"symbol={symbol}" / "data.parquet" for symbol in symbols
    }

    if skip_existing and all(path.exists() for path in outputs.values()):
        logging.info("Skipping %s %s (all symbols exist)", dataset, day)
        return

    data = download_file(candidate_urls(dataset, day, symbols))
    if data is None:
        logging.warning("No archive found for %s %s", dataset, day)
        return

    df = parse_to_dataframe(data)
    if df.is_empty():
        logging.warning("Archive empty for %s %s", dataset, day)
        return
    df = normalise_for_dataset(dataset, df)

    for symbol, output_path in outputs.items():
        if skip_existing and output_path.exists():
            logging.info("Skipping existing %s", output_path)
            continue

        symbol_df = df.filter(pl.col("symbol") == symbol)
        if symbol_df.is_empty():
            logging.warning("No data found for %s %s %s", dataset, symbol, day)
            continue
        logging.info("Writing %d rows for %s %s %s", symbol_df.height, dataset, symbol, day)
        write_parquet(symbol_df, output_path)


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
        for day in iter_dates(start_date, end_date):
            logging.info("--- Day: %s ---", day)
            process_day(dataset, args.symbols, day, output_root, skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()

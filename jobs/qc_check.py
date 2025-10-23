from __future__ import annotations

"""Run QC checks on bronze and silver datasets, optionally producing a summary report."""

import argparse
import json
from datetime import datetime
import logging
from pathlib import Path
import polars as pl

from etl.qc_validations import (
    check_trade_minute_completeness,
    validate_monotonic_timestamps,
    validate_spread_non_negative,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QC checks on bronze/silver datasets.")
    parser.add_argument("--bronze-root", type=Path, required=True)
    parser.add_argument("--silver-root", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--ignore-before", type=str, default="2018-01-01")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ignore_before = None
    if args.ignore_before:
        try:
            ignore_before = datetime.fromisoformat(args.ignore_before).date()
        except ValueError:
            logger.warning("Invalid ignore-before value %s; ignoring.", args.ignore_before)
    summary = {"trade_warnings": [], "snapshot_warnings": []}
    trade_files = sorted((args.bronze_root / "trades").rglob("*.parquet"))
    for trade_file in trade_files:
        df = pl.read_parquet(trade_file)
        validate_monotonic_timestamps(df, "ts")
        result = check_trade_minute_completeness(df, threshold=args.threshold)
        trade_date = None
        for part in trade_file.parts:
            if part.startswith("dt="):
                try:
                    trade_date = datetime.fromisoformat(part.split("=", 1)[1]).date()
                except ValueError:
                    logger.debug("Unable to parse trade date from %s", part)
                break
        if not result.passed():
            if ignore_before and trade_date and trade_date < ignore_before:
                logger.info("Ignoring coverage warning for %s (date %s < %s)", trade_file, trade_date, ignore_before)
                continue
            warning = {"file": str(trade_file), "coverage": result.coverage}
            summary["trade_warnings"].append(warning)
            logger.warning("Coverage below threshold %s: %.3f", trade_file, result.coverage)
    snapshot_files = sorted((args.silver_root / "l2_snapshots").rglob("*.parquet"))
    for snapshot_file in snapshot_files:
        df = pl.read_parquet(snapshot_file)
        validate_monotonic_timestamps(df, "ts")
        try:
            validate_spread_non_negative(df)
        except ValueError as exc:
            summary["snapshot_warnings"].append({"file": str(snapshot_file), "error": str(exc)})
            logger.warning("Spread validation failed for %s: %s", snapshot_file, exc)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "QC checks completed with %d trade warnings, %d snapshot warnings.",
        len(summary["trade_warnings"]),
        len(summary["snapshot_warnings"]),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

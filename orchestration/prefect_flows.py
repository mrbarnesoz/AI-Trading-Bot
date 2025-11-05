"""Prefect flows for orchestrating BitMEX ETL, snapshots, bars, and QC."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

from prefect import flow, get_run_logger, task
from prefect.events import emit_event

from monitoring.alerts import send_notification
from orchestration.auto_research import run_auto_research
from orchestration.block_helpers import load_json_block
from orchestration.config_utils import build_config
from utils.config import load_yaml

DEFAULT_STORAGE_CONFIG = Path("conf/storage.yaml")
DEFAULT_CADENCE_CONFIG = Path("conf/cadence.yaml")
DEFAULT_SOURCES_CONFIG = Path("conf/sources.yaml")


def _run_command(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end='')
    if result.stderr:
        print(result.stderr, file=sys.stderr, end='')
    result.check_returncode()
    return result


def _prepare_runtime_configs() -> tuple[Path, Path, Path, list[Path]]:
    tmp_files: list[Path] = []
    storage_override = load_json_block("bitmex-storage-config", {})
    cadence_override = load_json_block("bitmex-cadence-config", {})
    sources_override = load_json_block("bitmex-sources-config", {})

    storage_path = DEFAULT_STORAGE_CONFIG
    if storage_override:
        storage_path = build_config(DEFAULT_STORAGE_CONFIG, storage_override)
        tmp_files.append(storage_path)

    cadence_path = DEFAULT_CADENCE_CONFIG
    if cadence_override:
        cadence_path = build_config(DEFAULT_CADENCE_CONFIG, cadence_override)
        tmp_files.append(cadence_path)

    sources_path = DEFAULT_SOURCES_CONFIG
    if sources_override:
        sources_path = build_config(DEFAULT_SOURCES_CONFIG, sources_override)
        tmp_files.append(sources_path)

    return storage_path, cadence_path, sources_path, tmp_files


@task(name="download-archives", retries=3, retry_delay_seconds=60)
def download_archives(start: str, end: str, symbols: Iterable[str], storage_config: Path, sources_config: Path) -> None:
    logger = get_run_logger()
    cmd = [
        sys.executable,
        "-m",
        "jobs.download_archives",
        "--storage-config",
        str(storage_config),
        "--sources-config",
        str(sources_config),
        "--start",
        start,
        "--end",
        end,
        "--symbols",
        *symbols,
    ]
    logger.info("Running %s", " ".join(cmd))
    logger.info("Symbols provided: %s", list(symbols))
    _run_command(cmd)


@task(name="run-pipeline", retries=3, retry_delay_seconds=60)
def run_pipeline(start: str, end: str, symbols: Iterable[str], storage_config: Path, cadence_config: Path, sources_config: Path) -> None:
    logger = get_run_logger()
    cmd = [
        sys.executable,
        "-m",
        "jobs.run_pipeline",
        "--storage-config",
        str(storage_config),
        "--cadence-config",
        str(cadence_config),
        "--sources-config",
        str(sources_config),
        "--start",
        start,
        "--end",
        end,
        "--symbols",
        *symbols,
    ]
    logger.info("Running %s", " ".join(cmd))
    logger.info("Symbols provided: %s", list(symbols))
    _run_command(cmd)


@task(name="build-snapshots", retries=3, retry_delay_seconds=60)
def build_snapshots() -> None:
    logger = get_run_logger()
    cmd = [sys.executable, "-m", "jobs.build_snapshots"]
    logger.info("Running %s", " ".join(cmd))
    _run_command(cmd)


@task(name="resample-bars", retries=3, retry_delay_seconds=60)
def resample_bars() -> None:
    logger = get_run_logger()
    cmd = [sys.executable, "-m", "jobs.run_pipeline"]
    logger.info("Running %s", " ".join(cmd))
    _run_command(cmd)


@task(name="run-qc", retries=2, retry_delay_seconds=30)
def run_qc(storage_config: Path) -> dict:
    logger = get_run_logger()
    storage_cfg = load_yaml(storage_config)
    bronze_root = storage_cfg["bronze_root"]
    silver_root = storage_cfg["silver_root"]
    qc_path = Path("results/qc_summary_prefect.json")
    cmd = [
        sys.executable,
        "-m",
        "jobs.qc_check",
        "--bronze-root",
        bronze_root,
        "--silver-root",
        silver_root,
        "--output-json",
        str(qc_path),
    ]
    logger.info("Running %s", " ".join(cmd))
    _run_command(cmd)
    if qc_path.exists():
        summary = json.loads(qc_path.read_text(encoding="utf-8"))
    else:
        summary = {"trade_warnings": [], "snapshot_warnings": []}
    logger.info("QC summary: %s", summary)
    return summary


@task(name="daily-report")
def generate_daily_report(summary: dict) -> Path:
    from reports.daily_report import report_from_metrics, save_report

    report = report_from_metrics(
        {"pnl": 0.0, "sharpe": 0.0, "latency_ms": 45.0, "feature_psi": 0.1}, summary
    )
    output = Path("results/daily_report_prefect.txt")
    save_report(report, output)
    return output


@task(name="circuit-breaker")
def circuit_breaker(summary: dict) -> None:
    logger = get_run_logger()
    trade_warnings = len(summary.get("trade_warnings", []))
    snapshot_warnings = len(summary.get("snapshot_warnings", []))
    if trade_warnings or snapshot_warnings:
        emit_event(
            event="qc.alert",
            resource={"prefect.resource.name": "bitmex-qc"},
            payload={"trade_warnings": trade_warnings, "snapshot_warnings": snapshot_warnings},
        )
        message = f"QC circuit breaker triggered - trade warnings: {trade_warnings}, snapshot warnings: {snapshot_warnings}"
        logger.error(message)
        send_notification(message)
        raise RuntimeError("QC circuit breaker triggered")
    logger.info("QC circuit breaker passed")


@task(name="auto-strategy-research")
def auto_strategy_research_task(
    symbols: Iterable[str] | None = None,
    timeframes: Iterable[str] | None = None,
    strategies: Iterable[str] | None = None,
    min_trades: int = 100,
    top_k: int = 3,
) -> str:
    logger = get_run_logger()
    logger.info(
        "Running auto strategy research (symbols=%s timeframes=%s strategies=%s, min_trades=%s, top_k=%s)",
        symbols,
        timeframes,
        strategies,
        min_trades,
        top_k,
    )
    result = run_auto_research(
        symbols=list(symbols) if symbols else None,
        timeframes=list(timeframes) if timeframes else None,
        strategies=list(strategies) if strategies else None,
        min_trades=min_trades,
        top_k=top_k,
    )
    logger.info("Auto strategy research summary stored at %s", result)
    return str(result)


def _date_range_for_stage(days_back: int = 1) -> tuple[str, str]:
    end_date = datetime.now(tz=timezone.utc).date() - timedelta(days=days_back - 1)
    start_date = end_date - timedelta(days=days_back - 1)
    return start_date.isoformat(), end_date.isoformat()


@flow(name="bitmex-etl-flow")
def bitmex_etl_flow(
    start: str | None = None,
    end: str | None = None,
    symbols: Iterable[str] | None = None,
) -> None:
    logger = get_run_logger()
    logger.info("bitmex_etl_flow received symbols: %s", symbols)
    if symbols is None:
        symbols = ["XBTUSD"]
    else:
        symbols = list(symbols)
    if not start or not end:
        start, end = _date_range_for_stage()
    storage_cfg, cadence_cfg, sources_cfg, tmp_files = _prepare_runtime_configs()
    try:
        download_archives(start, end, symbols, storage_cfg, sources_cfg)
        run_pipeline(start, end, symbols, storage_cfg, cadence_cfg, sources_cfg)
    finally:
        for tmp in tmp_files:
            tmp.unlink(missing_ok=True)


@flow(name="bitmex-qc-flow")
def bitmex_qc_flow() -> None:
    storage_cfg, _, _, tmp_files = _prepare_runtime_configs()
    try:
        summary = run_qc(storage_cfg)
        circuit_breaker(summary)
        generate_daily_report(summary)
    finally:
        for tmp in tmp_files:
            tmp.unlink(missing_ok=True)


@flow(name="bitmex-daily-flow")
def bitmex_daily_flow(symbols: Iterable[str] | None = None) -> None:
    if symbols is None:
        symbols = ["XBTUSD"]
    else:
        symbols = list(symbols)
    start, end = _date_range_for_stage()
    storage_cfg, cadence_cfg, sources_cfg, tmp_files = _prepare_runtime_configs()
    try:
        download_archives(start, end, symbols, storage_cfg, sources_cfg)
        run_pipeline(start, end, symbols, storage_cfg, cadence_cfg, sources_cfg)
        summary = run_qc(storage_cfg)
        circuit_breaker(summary)
        generate_daily_report(summary)
    finally:
        for tmp in tmp_files:
            tmp.unlink(missing_ok=True)


@flow(name="bitmex-auto-strategy")
def bitmex_auto_strategy_flow(
    symbols: Iterable[str] | None = None,
    timeframes: Iterable[str] | None = None,
    strategies: Iterable[str] | None = None,
    min_trades: int = 100,
    top_k: int = 3,
) -> str:
    """Run automated backtests across configs and emit best candidates."""
    result_path = auto_strategy_research_task(symbols, timeframes, strategies, min_trades, top_k)
    logger = get_run_logger()
    logger.info("bitmex_auto_strategy_flow completed. Summary at %s", result_path)
    return result_path

"""Daily scheduler orchestrating ETL pipeline, QC, and alerts."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
from functools import partial
import sys
from datetime import datetime, timedelta, time, timezone
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from deployment.monitoring import AlertRule, MonitoringSuite
from monitoring.alerts import send_notification
from reports.daily_report import report_from_metrics, save_report
from utils.config import load_yaml

logger = logging.getLogger(__name__)


def parse_time_string(value: str) -> time:
    hour, minute = value.split(":")
    return time(int(hour), int(minute))


async def wait_until(target: time) -> None:
    now = datetime.now(timezone.utc)
    target_dt = now.replace(hour=target.hour, minute=target.minute, second=0, microsecond=0)
    if target_dt <= now:
        target_dt += timedelta(days=1)
    await asyncio.sleep((target_dt - now).total_seconds())


async def run_daily(tasks: List[Tuple[time, Callable[[], None]]]) -> None:
    while True:
        for target_time, func in tasks:
            await wait_until(target_time)
            logger.info("Running scheduled task at %s", target_time)
            func()


def run_command(args: List[str]) -> bool:
    logger.info("Executing: %s", " ".join(args))
    try:
        subprocess.run(args, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        logger.exception("Command failed: %s", exc)
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily scheduler for ETL pipeline.")
    parser.add_argument("--storage-config", type=Path, default=Path("conf/storage.yaml"))
    parser.add_argument("--cadence-config", type=Path, default=Path("conf/cadence.yaml"))
    parser.add_argument("--sources-config", type=Path, default=Path("conf/sources.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    storage_cfg = load_yaml(args.storage_config)
    cadence_cfg = load_yaml(args.cadence_config)
    sources_cfg = load_yaml(args.sources_config)

    symbols = sources_cfg.get("default_symbols", ["XBTUSD"])
    auto_ingest_l2 = bool(storage_cfg.get("auto_ingest_l2", False))
    qc_output = Path("results/qc_summary.json")
    report_output = Path("results/daily_report.txt")

    monitoring = MonitoringSuite(notifier=lambda name, metrics: send_notification(f"{name} alert: {metrics}", title="BitMEX Pipeline Alert"))
    monitoring.register_alert(
        AlertRule(
            name="pipeline_failure",
            condition=lambda m: m.get("pipeline_success") == 0,
            action=lambda m: logger.critical("Pipeline failure alert triggered."),
        )
    )
    monitoring.register_alert(
        AlertRule(
            name="qc_trade_warning",
            condition=lambda m: m.get("qc_trade_warnings", 0) > 0,
            action=lambda m: logger.warning("QC trade warnings detected: %s", m["qc_trade_warnings"]),
        )
    )
    monitoring.register_alert(
        AlertRule(
            name="qc_snapshot_warning",
            condition=lambda m: m.get("qc_snapshot_warnings", 0) > 0,
            action=lambda m: logger.warning("QC snapshot warnings detected: %s", m["qc_snapshot_warnings"]),
        )
    )

    def pipeline_task() -> None:
        date = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
        cmd = [
            sys.executable,
            "-m",
            "jobs.run_pipeline",
            "--storage-config",
            str(args.storage_config),
            "--cadence-config",
            str(args.cadence_config),
            "--sources-config",
            str(args.sources_config),
            "--start",
            date,
            "--end",
            date,
        ]
        if symbols:
            cmd += ["--symbols", *symbols]
        success = run_command(cmd)
        monitoring.evaluate({"pipeline_success": 1 if success else 0})

        # Run daily incremental roll after pipeline
        roll_cmd = [
            sys.executable,
            "-m",
            "jobs.daily_incremental",
        ]
        run_command(roll_cmd)

    def qc_task() -> None:
        qc_cmd = [
            sys.executable,
            "-m",
            "jobs.qc_check",
            "--bronze-root",
            storage_cfg["bronze_root"],
            "--silver-root",
            storage_cfg["silver_root"],
            "--output-json",
            str(qc_output),
        ]
        run_command(qc_cmd)
        if qc_output.exists():
            summary = json.loads(qc_output.read_text(encoding="utf-8"))
        else:
            summary = {"trade_warnings": [], "snapshot_warnings": []}
        report = report_from_metrics(
            {"pnl": 0.0, "sharpe": 0.0, "latency_ms": 45.0, "feature_psi": 0.1},
            summary,
        )
        save_report(report, report_output)
        monitoring.evaluate(
            {
                "qc_trade_warnings": len(summary.get("trade_warnings", [])),
                "qc_snapshot_warnings": len(summary.get("snapshot_warnings", [])),
            }
        )

    schedule_times = [
        (parse_time_string(cadence_cfg["daily_schedule"]["roll_raw"]), pipeline_task),
        (parse_time_string(cadence_cfg["daily_schedule"]["qc_publish"]), qc_task),
    ]

    logger.info("Starting daily scheduler (auto_ingest_l2=%s)", auto_ingest_l2)
    asyncio.run(run_daily(schedule_times))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

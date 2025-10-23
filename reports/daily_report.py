"""Daily operational report generator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict


@dataclass
class DailyReport:
    pnl: float
    sharpe: float
    latency_ms: float
    feature_psi: float
    qc_trade_warnings: int = 0
    qc_snapshot_warnings: int = 0


def render_report(report: DailyReport) -> str:
    return (
        f"Daily Report ({datetime.now(timezone.utc).date()} UTC)\n"
        f"PnL: {report.pnl:.2f}\n"
        f"Sharpe: {report.sharpe:.2f}\n"
        f"Latency (ms): {report.latency_ms:.2f}\n"
        f"Feature PSI: {report.feature_psi:.2f}\n"
        f"QC trade warnings: {report.qc_trade_warnings}\n"
        f"QC snapshot warnings: {report.qc_snapshot_warnings}\n"
    )


def save_report(report: DailyReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(report), encoding="utf-8")


def report_from_metrics(metrics: Dict[str, float], qc_summary: Dict[str, list]) -> DailyReport:
    return DailyReport(
        pnl=float(metrics.get("pnl", 0.0)),
        sharpe=float(metrics.get("sharpe", 0.0)),
        latency_ms=float(metrics.get("latency_ms", 0.0)),
        feature_psi=float(metrics.get("feature_psi", 0.0)),
        qc_trade_warnings=len(qc_summary.get("trade_warnings", [])),
        qc_snapshot_warnings=len(qc_summary.get("snapshot_warnings", [])),
    )

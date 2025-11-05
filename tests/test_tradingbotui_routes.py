from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("DUCKDB_PATH", ":memory:")

from tradingbotui import routes
from tradingbotui import tasks
from tradingbotui import create_app


@pytest.fixture
def app(monkeypatch, tmp_path):
    logs_dir = tmp_path / "logs"
    core_results_dir = tmp_path / "results"
    results_dir = core_results_dir / "ui" / "backtest"
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(routes, "_update_job_registry", lambda *args, **kwargs: None)
    monkeypatch.setattr(routes, "_JOB_REGISTRY_PATH", logs_dir / "job_registry.json", raising=False)
    monkeypatch.setattr(routes, "_BACKTEST_RESULTS_DIR", results_dir, raising=False)
    monkeypatch.setattr(routes, "_BACKTEST_ARCHIVE_DIR", results_dir / "archive", raising=False)
    monkeypatch.setattr(tasks, "LOG_DIR", logs_dir, raising=False)
    monkeypatch.setattr(tasks, "RESULTS_DIR", core_results_dir, raising=False)
    monkeypatch.setattr(tasks, "STATE_PATH", logs_dir / "trading_state.json", raising=False)
    monkeypatch.setattr(tasks, "JOB_REGISTRY_PATH", logs_dir / "job_registry.json", raising=False)
    monkeypatch.setattr(tasks, "KAFKA_STATE_PATH", logs_dir / "kafka_state.json", raising=False)
    trade_log_path = logs_dir / "trades.jsonl"
    open_positions_path = logs_dir / "open_positions.json"
    latest_trades_path = results_dir / "latest_trades.json"
    monkeypatch.setattr(routes, "_TRADE_LOG_PATH", trade_log_path, raising=False)
    monkeypatch.setattr(tasks, "TRADE_LOG_PATH", trade_log_path, raising=False)
    monkeypatch.setattr(tasks, "OPEN_POSITIONS_PATH", open_positions_path, raising=False)
    monkeypatch.setattr(tasks, "LATEST_TRADES_PATH", latest_trades_path, raising=False)
    application = create_app()
    application.config["TESTING"] = True
    return application


def test_api_backtests_rejects_invalid_timeframe(app, monkeypatch):
    manifest = {
        "trend_breakout": {
            "timeframes": ["15m", "1h"],
            "default": "15m",
        }
    }
    monkeypatch.setattr(routes, "get_strategy_manifest", lambda: manifest)

    def fail(*_args, **_kwargs):
        raise AssertionError("backtest launch should not be attempted for invalid timeframe")

    monkeypatch.setattr(tasks, "start_backtest", fail)
    monkeypatch.setattr(tasks, "start_backtest_batch", fail)

    client = app.test_client()
    response = client.post(
        "/api/backtests",
        json={
            "strategies": ["trend_breakout"],
            "symbols": ["XBTUSD"],
            "timeframes": {"trend_breakout": "5m"},
            "capital_pct": 10,
        },
    )
    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"] == "invalid_timeframes"
    combined_details = " ".join(payload.get("details") or [])
    assert "trend_breakout" in combined_details
    assert "5m" in combined_details


def test_api_backtests_uses_manifest_default_timeframe(app, monkeypatch):
    manifest = {
        "trend_breakout": {
            "timeframes": ["15m", "1h"],
            "default": "1h",
        }
    }
    monkeypatch.setattr(routes, "get_strategy_manifest", lambda: manifest)

    recorded_calls = []

    def fake_start_backtest(strategy: str, symbol: str, timeframe: str, capital_pct: str):
        recorded_calls.append((strategy, symbol, timeframe, capital_pct))
        return "job-001"

    def fake_start_backtest_batch(*_args, **_kwargs):
        raise AssertionError("batch backtest should not be invoked for single symbol payloads")

    monkeypatch.setattr(tasks, "start_backtest", fake_start_backtest)
    monkeypatch.setattr(tasks, "start_backtest_batch", fake_start_backtest_batch)

    client = app.test_client()
    response = client.post(
        "/api/backtests",
        json={
            "strategies": ["trend_breakout"],
            "symbols": [],
            "capital_pct": 5,
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["job_ids"] == ["job-001"]
    assert payload["launch_plan"][0]["timeframe"] == "1h"
    assert payload["launch_plan"][0]["strategy"] == "trend_breakout"
    assert payload["launch_plan"][0]["symbols"] == []
    assert recorded_calls == [("trend_breakout", "", "1h", "5")]


def test_api_backtests_normalises_job_timestamps(app, monkeypatch):
    monkeypatch.setattr(routes, "_read_job_registry", lambda: {
        "job-legacy": {
            "id": "job-legacy",
            "type": "backtest",
            "status": "queued",
            "submitted_at": "20250701-010203",
        }
    })
    monkeypatch.setattr(routes, "_list_recent_results", lambda limit=20: [])
    client = app.test_client()
    response = client.get("/api/backtests")
    assert response.status_code == 200
    jobs = response.get_json()["jobs"]
    assert jobs[0]["submitted_at"].endswith("+00:00")
    assert "T" in jobs[0]["submitted_at"]


def test_api_backtest_result_detail_reads_file(app, tmp_path, monkeypatch):
    result_dir = routes._BACKTEST_RESULTS_DIR
    result_file = result_dir / "sample.json"
    result_payload = {
        "summary": {"calc_sharpe": 0.42},
        "metadata": {"symbol": "XBTUSD"},
    }
    result_file.write_text(json.dumps(result_payload), encoding="utf-8")
    client = app.test_client()
    response = client.get(f"/api/backtests/results/{result_file.name}")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["summary"]["calc_sharpe"] == 0.42


def test_api_archive_job_removes_entry_and_moves_results(app, tmp_path):
    jobs_path = routes._JOB_REGISTRY_PATH
    jobs_path.write_text(json.dumps({
        "job-123": {
            "id": "job-123",
            "type": "backtest",
            "status": "completed",
            "strategy": "mean_reversion",
            "config": "configs/xbtusd_mean_reversion.yaml",
            "timeframe": "15m",
            "submitted_at": "20250701-010203",
        }
    }), encoding="utf-8")
    result_dir = routes._BACKTEST_RESULTS_DIR
    result_payload = {
        "config": "configs/xbtusd_mean_reversion.yaml",
        "summary": {"calc_sharpe": 0.9},
        "metadata": {"symbol": "XBTUSD", "timeframe": "15m"},
    }
    result_file = result_dir / "xbtusd_mean_reversion.json"
    result_file.write_text(json.dumps(result_payload), encoding="utf-8")
    client = app.test_client()
    response = client.post("/api/jobs/job-123/archive", json={"archive_results": True})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "archived"
    assert payload["job_id"] == "job-123"
    assert jobs_path.exists()
    jobs_after = json.loads(jobs_path.read_text(encoding="utf-8"))
    assert "job-123" not in jobs_after
    archive_dir = routes._BACKTEST_ARCHIVE_DIR
    assert archive_dir.exists()
    archived_files = list(archive_dir.glob("*.json"))
    assert len(archived_files) == 1
    assert not result_file.exists()



def test_api_trading_status_returns_payload(app, monkeypatch):
    sample_status = {"paper": {"running": False, "pending": False}}
    monkeypatch.setattr(tasks, "get_trading_status", lambda: sample_status)
    sample_positions = {"count": 0, "notional_usd": 0.0, "pnl_usd": 0.0, "symbols": {}}
    monkeypatch.setattr(tasks, "get_open_positions_summary", lambda: sample_positions)
    client = app.test_client()
    response = client.get("/api/trading/status")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["paper"]["running"] is False
    assert payload["open_positions"] == sample_positions


def test_api_trading_paper_uses_manifest_default(app, monkeypatch):
    manifest = {"mean_reversion": {"timeframes": ["15m"], "default": "15m"}}
    monkeypatch.setattr(routes, "get_strategy_manifest", lambda: manifest)

    captured = {}

    def fake_schedule(mode, plan, capital_pct):
        captured["mode"] = mode
        captured["plan"] = plan
        captured["capital_pct"] = capital_pct
        return {"plan_id": "paper-1", "count": len(plan)}

    monkeypatch.setattr(tasks, "schedule_trading_plan", fake_schedule)
    client = app.test_client()
    response = client.post(
        "/api/trading/paper",
        json={"strategies": ["mean_reversion"], "symbols": ["xbtusd"]},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "queued"
    assert captured["mode"] == "paper"
    assert captured["capital_pct"] == "10"
    assert captured["plan"] == [{"strategy": "mean_reversion", "symbol": "XBTUSD", "timeframe": "15m"}]


def test_api_trading_live_supports_multiple_symbols(app, monkeypatch):
    manifest = {"trend": {"timeframes": ["1h"], "default": "1h"}}
    monkeypatch.setattr(routes, "get_strategy_manifest", lambda: manifest)

    captured = {}

    def fake_schedule(mode, plan, capital_pct):
        captured["mode"] = mode
        captured["plan"] = plan
        return {"plan_id": "live-1", "count": len(plan)}

    monkeypatch.setattr(tasks, "schedule_trading_plan", fake_schedule)
    client = app.test_client()
    response = client.post(
        "/api/trading/live",
        json={
            "strategies": ["trend"],
            "symbols": ["XBTUSD", "ETHUSD"],
            "timeframes": {"trend": "1h"},
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "queued"
    assert captured["mode"] == "live"
    assert len(captured["plan"]) == 2
    assert captured["plan"][0]["timeframe"] == "1h"


def test_api_trading_live_inherits_legacy_timeframe(app, monkeypatch):
    manifest = {
        "trend_breakout": {"timeframes": ["15m", "1h"], "default": "15m"},
        "mean_reversion": {"timeframes": ["5m", "15m"], "default": "15m"},
    }
    monkeypatch.setattr(routes, "get_strategy_manifest", lambda: manifest)

    captured = {}

    def fake_schedule(mode, plan, capital_pct):
        captured["mode"] = mode
        captured["plan"] = plan
        captured["capital_pct"] = capital_pct
        return {"plan_id": "live-legacy", "count": len(plan)}

    monkeypatch.setattr(tasks, "schedule_trading_plan", fake_schedule)
    client = app.test_client()
    response = client.post(
        "/api/trading/live",
        json={
            "strategies": ["trend_breakout", "mean_reversion"],
            "symbols": ["XBTUSD"],
            "timeframe": "15m",
        },
    )
    assert response.status_code == 200
    plan = captured["plan"]
    assert len(plan) == 2
    assert all(entry["timeframe"] == "15m" for entry in plan)


def test_api_trading_live_rejects_invalid_timeframe(app, monkeypatch):
    manifest = {"trend": {"timeframes": ["1h"], "default": "1h"}}
    monkeypatch.setattr(routes, "get_strategy_manifest", lambda: manifest)
    client = app.test_client()
    response = client.post(
        "/api/trading/live",
        json={"strategy": "trend", "symbol": "XBTUSD", "timeframe": "5m"},
    )
    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"] == "invalid_timeframe"
    assert any("Allowed timeframes" in detail for detail in payload.get("details", []))


def test_api_trading_stop_handles_modes(app, monkeypatch):
    calls = {}

    def fake_stop(mode=None):
        calls["mode"] = mode
        return {"stopped": [mode or "paper"], "errors": []}

    monkeypatch.setattr(tasks, "stop_trading", fake_stop)
    client = app.test_client()
    response = client.post("/api/trading/stop", json={"mode": "paper"})
    assert response.status_code == 200
    assert response.get_json()["status"] == "stopped"
    assert calls["mode"] == "paper"


def test_api_trading_stop_invalid_mode(app, monkeypatch):
    def fake_stop(mode=None):
        raise ValueError("bad mode")

    monkeypatch.setattr(tasks, "stop_trading", fake_stop)
    client = app.test_client()
    response = client.post("/api/trading/stop", json={"mode": "unknown"})
    assert response.status_code == 400
    assert response.get_json()["error"] == "invalid_mode"


def test_api_trades_infers_status_and_timeframe(app):
    trade_log_path = routes._TRADE_LOG_PATH
    trade_log_path.write_text(
        json.dumps(
            {
                "timestamp": "2025-11-05T10:00:00Z",
                "strategy": "mean_reversion",
                "symbol": "XBTUSD",
                "meta": {"interval": "15m"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    client = app.test_client()
    response = client.get("/api/trades")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["records"], "Expected at least one trade record"
    record = payload["records"][0]
    assert record["timeframe"] == "15m"
    assert record["status"] == "closed"


def test_api_trades_clear_endpoint(app):
    trade_log_path = routes._TRADE_LOG_PATH
    trade_log_path.write_text(json.dumps({"timestamp": "2025-11-05T10:00:00Z", "pnl": 1.0}) + "\n", encoding="utf-8")
    latest_trades_path = tasks.LATEST_TRADES_PATH
    latest_trades_path.parent.mkdir(parents=True, exist_ok=True)
    latest_trades_path.write_text(json.dumps([0.1, -0.2]), encoding="utf-8")

    client = app.test_client()
    response = client.post("/api/trades/clear")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert trade_log_path.exists() is False
    assert latest_trades_path.exists() is False
    assert payload["records"] == []

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

BASE_DIR = Path.cwd()
LOG_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STATE_PATH = LOG_DIR / "trading_state.json"
JOB_REGISTRY_PATH = LOG_DIR / "job_registry.json"
TRADE_LOG_PATH = LOG_DIR / "trades.jsonl"
OPEN_POSITIONS_PATH = LOG_DIR / "open_positions.json"
KAFKA_STATE_PATH = LOG_DIR / "kafka_state.json"
LATEST_TRADES_PATH = RESULTS_DIR / "latest_trades.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        pass
    return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _default_mode_state() -> Dict[str, Any]:
    return {
        "running": False,
        "pending": False,
        "started_at": None,
        "ended_at": None,
        "capital_pct": None,
        "plan": [],
        "last_updated": None,
    }


def _load_state() -> Dict[str, Any]:
    state = _read_json(STATE_PATH, {})
    state.setdefault("paper", _default_mode_state())
    state.setdefault("live", _default_mode_state())
    guardrails = state.setdefault("guardrails", {})
    guardrails.setdefault("last_paper", None)
    return state


def _save_state(state: Dict[str, Any]) -> None:
    _write_json(STATE_PATH, state)


def _guardrail_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    last_paper = state.get("guardrails", {}).get("last_paper")
    paper_is_recent = False
    if last_paper:
        try:
            dt = datetime.fromisoformat(last_paper)
            paper_is_recent = (datetime.now(timezone.utc) - dt).total_seconds() < 24 * 3600
        except ValueError:
            paper_is_recent = False
    return {"paper_is_recent": paper_is_recent, "last_paper": last_paper}


# ---------------------------------------------------------------------------
# Trade status helpers
# ---------------------------------------------------------------------------

def infer_trade_status(meta: Optional[Dict[str, Any]] = None, status: Optional[Any] = None) -> str:
    meta = meta or {}

    def _normalise(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, bool):
            return "open" if value else "closed"
        if isinstance(value, str):
            text = value.strip().lower()
            if not text:
                return None
            if text in {"open", "opened", "opening", "active"}:
                return "open"
            if text in {"closed", "closing", "closed_out", "exit", "exited", "flat"}:
                return "closed"
            return text
        return None

    resolved = _normalise(status)
    if resolved:
        return resolved

    for key in ("status", "state", "position_status"):
        resolved = _normalise(meta.get(key))
        if resolved:
            return resolved

    if "is_open" in meta:
        resolved = _normalise(meta.get("is_open"))
        if resolved:
            return resolved

    if "closed" in meta:
        resolved = _normalise(not meta.get("closed"))
        if resolved:
            return resolved

    for key in ("exit_time", "exit_at", "close_time", "closed_at", "exit_timestamp", "closed_timestamp"):
        if meta.get(key):
            return "closed"

    return "closed"


# ---------------------------------------------------------------------------
# Trade log helpers
# ---------------------------------------------------------------------------

def _append_trade_records(records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    TRADE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRADE_LOG_PATH.open("a", encoding="utf-8") as handle:
        for record in records:
            payload = dict(record)
            if "timestamp" not in payload or not payload["timestamp"]:
                payload["timestamp"] = _timestamp()
            payload.setdefault("status", infer_trade_status(payload.get("meta") if isinstance(payload.get("meta"), dict) else {}, payload.get("status")))
            handle.write(json.dumps(payload))
            handle.write("\n")


def _save_trade_history(
    pnl: Iterable[float],
    metadata: Optional[Dict[str, Any]] = None,
    trades: Optional[Iterable[Dict[str, Any]]] = None,
) -> None:
    history = list(pnl)
    LATEST_TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATEST_TRADES_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")

    meta = dict(metadata or {})
    meta.setdefault("pair", meta.get("symbol"))
    meta.setdefault("timeframe", meta.get("timeframe") or meta.get("interval") or meta.get("mode_interval") or meta.get("resolution"))
    meta.setdefault("strategy", meta.get("strategy") or meta.get("mode") or meta.get("strategy_name"))

    base_info = {
        "strategy": meta.get("strategy"),
        "symbol": meta.get("pair"),
        "timeframe": meta.get("timeframe"),
        "source": meta.get("config"),
    }

    records: List[Dict[str, Any]] = []
    if trades:
        for raw in trades:
            raw_dict = raw if isinstance(raw, dict) else {}
            records.append(
                {
                    **base_info,
                    "timestamp": raw_dict.get("timestamp") or _timestamp(),
                    "pnl": raw_dict.get("pnl"),
                    "confidence": raw_dict.get("confidence"),
                    "volatility": raw_dict.get("volatility"),
                    "position_size": raw_dict.get("position_size"),
                    "timeframe": raw_dict.get("timeframe") or base_info.get("timeframe"),
                    "status": infer_trade_status(raw_dict),
                    "meta": raw_dict,
                }
            )
    else:
        for idx, value in enumerate(history, start=1):
            records.append(
                {
                    **base_info,
                    "timestamp": _timestamp(),
                    "pnl": value,
                    "confidence": None,
                    "volatility": None,
                    "position_size": None,
                    "timeframe": base_info.get("timeframe"),
                    "status": "closed",
                    "meta": {"index": idx},
                }
            )

    _append_trade_records(records)


# ---------------------------------------------------------------------------
# Public API used by routes.py
# ---------------------------------------------------------------------------

def get_trading_status() -> Dict[str, Any]:
    state = _load_state()
    guardrails = _guardrail_snapshot(state)
    return {
        "paper": state["paper"],
        "live": state["live"],
        "guardrails": guardrails,
    }


def get_open_positions_summary() -> Dict[str, Any]:
    payload = _read_json(OPEN_POSITIONS_PATH, {})
    if isinstance(payload, dict):
        return {
            "count": int(payload.get("count", 0) or 0),
            "notional_usd": float(payload.get("notional_usd", 0.0) or 0.0),
            "pnl_usd": float(payload.get("pnl_usd", 0.0) or 0.0),
            "symbols": payload.get("symbols", {}),
        }
    return {"count": 0, "notional_usd": 0.0, "pnl_usd": 0.0, "symbols": {}}


def schedule_trading_plan(mode: str, plan: List[Dict[str, Any]], capital_pct: str | float | int) -> Dict[str, Any]:
    if mode not in {"paper", "live"}:
        raise ValueError(f"Unknown trading mode '{mode}'")
    state = _load_state()
    bucket = state.setdefault(mode, _default_mode_state())
    now = _timestamp()
    bucket.update(
        {
            "running": True,
            "pending": False,
            "started_at": now,
            "ended_at": None,
            "plan": plan,
            "capital_pct": capital_pct,
            "last_updated": now,
        }
    )
    if mode == "paper":
        state.setdefault("guardrails", {})["last_paper"] = now
    _save_state(state)
    plan_id = f"{mode}-{int(time.time() * 1000)}"
    return {"plan_id": plan_id, "count": len(plan), "mode": mode, "capital_pct": capital_pct}


def stop_trading(mode: Optional[str] = None) -> Dict[str, Any]:
    state = _load_state()
    modes = [mode] if mode in {"paper", "live"} else ["paper", "live"]
    stopped = []
    for entry in modes:
        bucket = state.setdefault(entry, _default_mode_state())
        if bucket.get("running") or bucket.get("pending"):
            bucket.update({"running": False, "pending": False, "ended_at": _timestamp(), "last_updated": _timestamp()})
            stopped.append(entry)
    _save_state(state)
    return {"stopped": stopped, "errors": []}


def stop_all() -> None:
    stop_trading()


def _register_job(entry: Dict[str, Any]) -> str:
    jobs = _read_json(JOB_REGISTRY_PATH, {})
    job_id = entry.get("id") or f"job-{int(time.time() * 1000)}"
    entry["id"] = job_id
    jobs[job_id] = entry
    _write_json(JOB_REGISTRY_PATH, jobs)
    return job_id


def start_backtest(strategy: str, symbol: str, timeframe: str, capital_pct: str | float | int) -> str:
    job = {
        "type": "backtest",
        "strategy": strategy,
        "symbols": [symbol] if symbol else [],
        "timeframe": timeframe,
        "capital_pct": capital_pct,
        "status": "queued",
        "submitted_at": _timestamp(),
    }
    return _register_job(job)


def start_backtest_batch(strategies: List[str], symbols: List[str], timeframe: str, capital_pct: str | float | int) -> str:
    job = {
        "type": "backtest_batch",
        "strategies": strategies,
        "symbols": symbols,
        "timeframe": timeframe,
        "capital_pct": capital_pct,
        "status": "queued",
        "submitted_at": _timestamp(),
    }
    return _register_job(job)


def kafka_stack_status() -> Dict[str, Any]:
    state = _read_json(KAFKA_STATE_PATH, {"running": False, "last_started": None, "last_stopped": None})
    return {
        "running": bool(state.get("running")),
        "last_started": state.get("last_started"),
        "last_stopped": state.get("last_stopped"),
    }


def start_kafka_stack() -> Dict[str, Any]:
    state = kafka_stack_status()
    state.update({"running": True, "last_started": _timestamp()})
    _write_json(KAFKA_STATE_PATH, state)
    return {"success": True, **state}


def stop_kafka_stack() -> Dict[str, Any]:
    state = kafka_stack_status()
    state.update({"running": False, "last_stopped": _timestamp()})
    _write_json(KAFKA_STATE_PATH, state)
    return {"success": True, **state}


def clear_trade_history() -> Dict[str, Any]:
    removed: List[str] = []
    for path in (TRADE_LOG_PATH, LATEST_TRADES_PATH):
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return {"status": "cleared", "removed": removed}


__all__ = [
    "get_trading_status",
    "get_open_positions_summary",
    "schedule_trading_plan",
    "stop_trading",
    "stop_all",
    "start_backtest",
    "start_backtest_batch",
    "kafka_stack_status",
    "start_kafka_stack",
    "stop_kafka_stack",
    "infer_trade_status",
    "clear_trade_history",
]

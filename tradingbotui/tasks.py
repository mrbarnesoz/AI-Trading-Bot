from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

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
DECISION_LOG_PATH = LOG_DIR / "decision_events.jsonl"
LATEST_TRADES_PATH = RESULTS_DIR / "latest_trades.json"
GUARDRAIL_LOG_PATH = LOG_DIR / "guardrail_events.jsonl"
GUARDRAIL_SNAPSHOT_DIR = RESULTS_DIR / "guardrail_snapshots"
RESULTS_UI_DIR = RESULTS_DIR / "ui" / "backtest"
RESULTS_UI_DIR.mkdir(parents=True, exist_ok=True)

BACKTEST_SCRIPT = Path("scripts") / "backtest.py"

SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")
GUARDRAIL_LOG = GUARDRAIL_LOG_PATH

from tradingbotui.strategy_manifest import get_strategy_manifest

logger = logging.getLogger("tradingbot.ui.tasks")


STRATEGY_EXIT_GUIDANCE: Dict[str, str] = {
    "mean_reversion": "Closes once price mean reverts toward VWAP/ATR midline or when the ATR trailing stop flips after the minimum hold window.",
    "reversion_band": "Takes profit when price re-enters the volatility band; ATR trailing locks cap losses if the squeeze persists.",
    "keltner_channel": "Rides the Keltner channel trend and exits when price loses the band or the swing-mode trailing stop is hit.",
    "momentum_burst": "Targets quick bursts; exits on momentum decay or if the bar-limit expires while the taker stop remains intact.",
    "rsi_divergence": "Lets divergence resolve, then closes on opposing momentum or at the protective stop anchored to the prior swing.",
    "swing_trading": "Follows higher-timeframe swings and exits when market structure breaks or the max drawdown floor is tagged.",
    "trend_breakout": "Holds the breakout until price closes back inside the base or the breakout trailing stop is breached.",
    "vwap_reversion": "Flips flat at VWAP retests; otherwise the hysteresis/ATR stop handles the exit once momentum fades.",
}


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


def _strategy_label(strategy: str) -> str:
    entry = get_strategy_manifest().get(strategy, {})
    label = entry.get("label")
    if label:
        return label
    return strategy.replace("_", " ").title() if strategy else "Unknown"


def _build_plan_summary(plan_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    manifest = get_strategy_manifest()
    summary: List[Dict[str, Any]] = []
    for entry in plan_entries:
        strategy = str(entry.get("strategy") or "")
        timeframe = str(entry.get("timeframe") or "")
        symbols = [str(symbol) for symbol in entry.get("symbols") or []]
        manifest_entry = manifest.get(strategy, {})
        label = manifest_entry.get("label") or _strategy_label(strategy)
        description = manifest_entry.get("description") or ""
        allowed = manifest_entry.get("timeframes") or []
        default_tf = manifest_entry.get("default")
        if default_tf and timeframe and timeframe == default_tf:
            tf_reason = f"{timeframe} is the manifest default for {label}."
        elif timeframe and timeframe in allowed:
            tf_reason = f"{timeframe} sits inside the recommended window ({', '.join(allowed)})."
        elif timeframe and allowed:
            tf_reason = (
                f"{timeframe} was requested even though the manifest lists {', '.join(allowed)}; "
                "treating it as an operator override."
            )
        elif timeframe:
            tf_reason = f"No manifest guidance available; keeping operator-supplied timeframe {timeframe}."
        else:
            tf_reason = "No timeframe provided; run will rely on the config fallback."
        symbol_reason = f"Symbols: {', '.join(symbols)}" if symbols else "Symbols fallback to strategy defaults."
        entry_reason = " ".join(segment for segment in (description, tf_reason, symbol_reason) if segment).strip()
        exit_reason = STRATEGY_EXIT_GUIDANCE.get(
            strategy, "Exits follow the configured trailing stops, hysteresis, and risk limits for this strategy."
        )
        summary.append(
            {
                "strategy": strategy,
                "label": label,
                "timeframe": timeframe,
                "symbols": symbols,
                "entry_reason": entry_reason,
                "exit_reason": exit_reason,
            }
        )
    return summary


def _read_job_registry() -> Dict[str, dict]:
    try:
        if JOB_REGISTRY_PATH.exists():
            payload = json.loads(JOB_REGISTRY_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
    except json.JSONDecodeError:
        pass
    return {}


def _write_job_registry(jobs: Dict[str, dict]) -> None:
    JOB_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    JOB_REGISTRY_PATH.write_text(json.dumps(jobs, indent=2, sort_keys=True), encoding="utf-8")


def _update_job_entry(job_id: str, **updates: Any) -> None:
    if not job_id:
        return
    jobs = _read_job_registry()
    entry = jobs.get(job_id)
    if entry is None:
        return
    entry.update({k: v for k, v in updates.items() if v is not None})
    _write_job_registry(jobs)


def _append_job_result(job_id: str, result: Dict[str, Any]) -> None:
    if not job_id:
        return
    jobs = _read_job_registry()
    entry = jobs.get(job_id)
    if entry is None:
        return
    results = entry.setdefault("results", [])
    results.append(result)
    _write_job_registry(jobs)


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
# Decision metrics helpers
# ---------------------------------------------------------------------------

def _parse_timestamp(value: Optional[str]) -> Tuple[Optional[datetime], Optional[str]]:
    if not value:
        return None, None
    text = str(value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed, parsed.isoformat()
    except ValueError:
        return None, text


def _load_decision_events(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if not DECISION_LOG_PATH.exists():
        return []
    events: List[Dict[str, Any]] = []
    try:
        with DECISION_LOG_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []

    def sort_key(item: Dict[str, Any]) -> Tuple[int, float]:
        parsed, _ = _parse_timestamp(item.get("timestamp"))
        if parsed:
            return (0, parsed.timestamp())
        return (1, 0.0)

    events.sort(key=sort_key, reverse=True)
    if limit is not None:
        return events[:limit]
    return events


def record_trade_decision(event: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(event or {})
    stage_raw = str(payload.get("stage") or "unknown").strip()
    if not stage_raw:
        raise ValueError("stage is required")
    stage = stage_raw.lower()

    strategy = str(payload.get("strategy") or "").strip() or "unknown"
    symbol = str(payload.get("symbol") or "").strip() or "?"
    timestamp = payload.get("timestamp")
    _, iso_ts = _parse_timestamp(timestamp)
    if not iso_ts:
        iso_ts = _timestamp()

    record = {
        "timestamp": iso_ts,
        "stage": stage,
        "stage_label": stage_raw,
        "strategy": strategy,
        "symbol": symbol,
        "details": payload.get("details") or {},
    }
    DECISION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DECISION_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record))
        handle.write("\n")
    return record


def get_trade_metrics(limit: int = 50) -> Dict[str, Any]:
    events = _load_decision_events()
    stage_counts: Dict[str, int] = defaultdict(int)
    strategy_counts: Dict[str, Dict[str, Any]] = {}

    for entry in events:
        stage = str(entry.get("stage") or "unknown").lower()
        strategy = str(entry.get("strategy") or "unknown")
        stage_counts[stage] += 1
        strat_bucket = strategy_counts.setdefault(
            strategy, {"total": 0, "by_stage": defaultdict(int)}
        )
        strat_bucket["total"] += 1
        strat_bucket["by_stage"][stage] += 1

    last_event_at = events[0].get("timestamp") if events else None

    metrics = {
        "total_events": len(events),
        "by_stage": dict(stage_counts),
        "by_strategy": {
            strategy: {
                "total": info["total"],
                "by_stage": dict(info["by_stage"]),
            }
            for strategy, info in strategy_counts.items()
        },
        "last_event_at": last_event_at,
    }

    recent_events = events[:limit] if limit is not None else events
    state = _load_state()
    mode_snapshot = {
        name: {
            "running": bool(bucket.get("running")),
            "pending": bool(bucket.get("pending")),
            "started_at": bucket.get("started_at"),
            "ended_at": bucket.get("ended_at"),
            "last_updated": bucket.get("last_updated"),
        }
        for name, bucket in (("paper", state.get("paper", {})), ("live", state.get("live", {})))
    }
    active_modes = [
        name for name, info in mode_snapshot.items() if info["running"] or info["pending"]
    ]
    status = "scanning" if active_modes else "idle"

    return {
        "status": status,
        "active_modes": active_modes,
        "modes": mode_snapshot,
        "heartbeat": _timestamp(),
        "last_event_at": last_event_at,
        "metrics": metrics,
        "events": recent_events,
    }


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
    jobs = _read_job_registry()
    job_id = entry.get("id") or f"job-{int(time.time() * 1000)}"
    entry["id"] = job_id
    jobs[job_id] = entry
    _write_job_registry(jobs)
    return job_id


def start_backtest(
    strategy: str,
    symbol: str,
    timeframe: str,
    capital_pct: str | float | int,
    *,
    parent_id: Optional[str] = None,
    hidden: bool = False,
) -> str:
    job = {
        "type": "backtest",
        "strategy": strategy,
        "symbols": [symbol] if symbol else [],
        "timeframe": timeframe,
        "capital_pct": capital_pct,
        "status": "queued",
        "submitted_at": _timestamp(),
    }
    if parent_id:
        job["parent_id"] = parent_id
    if hidden:
        job["hidden"] = True
    return _register_job(job)


def start_backtest_batch(
    strategies: List[str],
    symbols: List[str],
    timeframe: str,
    capital_pct: str | float | int,
    *,
    parent_id: Optional[str] = None,
    hidden: bool = False,
) -> str:
    job = {
        "type": "backtest_batch",
        "strategies": strategies,
        "symbols": symbols,
        "timeframe": timeframe,
        "capital_pct": capital_pct,
        "status": "queued",
        "submitted_at": _timestamp(),
    }
    if parent_id:
        job["parent_id"] = parent_id
    if hidden:
        job["hidden"] = True
    return _register_job(job)


def _coerce_date(value: Optional[str], *, end_of_day: bool = False) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if len(text) == 10:
            dt = datetime.strptime(text, "%Y-%m-%d")
            if end_of_day:
                dt = datetime.combine(dt.date(), datetime.max.time()).replace(microsecond=0)
            else:
                dt = datetime.combine(dt.date(), datetime.min.time())
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
    except ValueError:
        return text
    return dt.isoformat(timespec="seconds")


def _safe_filename(*parts: str) -> str:
    raw = "-".join(part for part in parts if part)
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in raw)
    return f"{cleaned}.json"


def _run_backtest_subprocess(config_path: Path) -> Dict[str, Any]:
    cmd = [sys.executable, str(BACKTEST_SCRIPT), "--config", str(config_path)]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(BASE_DIR),
        )
    except subprocess.CalledProcessError as exc:
        stdout = (exc.stdout or "").strip()
        stderr = (exc.stderr or "").strip()
        message = stderr or stdout or "Backtest failed without output."
        raise RuntimeError(f"Backtest process failed ({exc.returncode}): {message}") from exc
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    stdout_stripped = stdout.strip()
    if not stdout_stripped:
        raise RuntimeError("Backtest produced no output.")

    def _decode(candidate: str) -> Dict[str, Any]:
        candidate = candidate.strip()
        if not candidate:
            raise ValueError("empty candidate")
        return json.loads(candidate)

    try:
        return _decode(stdout_stripped)
    except Exception:
        pass

    start = stdout.find("{")
    end = stdout.rfind("}")
    if start != -1 and end != -1 and end >= start:
        candidate = stdout[start:end + 1]
        try:
            return _decode(candidate)
        except Exception:
            pass

    error_tail = "\n".join(stdout.splitlines()[-10:])
    stderr_tail = "\n".join(stderr.splitlines()[-10:])
    raise RuntimeError(
        f"Unable to parse backtest output. stdout tail:\n{error_tail or '<empty>'}\n"
        f"stderr tail:\n{stderr_tail or '<empty>'}"
    )


def _render_record(
    *,
    plan_job_id: str,
    child_job_id: str,
    strategy: str,
    label: str,
    symbol: str,
    timeframe: str,
    capital_pct: Optional[float],
    config_path: Path,
    entry_reason: str,
    exit_reason: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    summary = payload.get("summary") or {}
    metadata = {
        "strategy": strategy,
        "label": label,
        "symbol": payload.get("symbol") or symbol,
        "interval": payload.get("interval") or timeframe,
        "rows": payload.get("rows"),
        "meta_score": payload.get("meta_score"),
        "meta_metrics": payload.get("meta_metrics"),
        "latest_decision": payload.get("latest_decision"),
    }
    record = {
        "kind": "backtest",
        "capital_pct": capital_pct,
        "config": str(config_path.resolve()),
        "generated_at": _timestamp(),
        "identity": {
            "config_stem": config_path.stem,
            "strategy": strategy,
            "label": label,
            "symbol": symbol,
            "timeframe": timeframe,
            "plan_job_id": plan_job_id,
            "child_job_id": child_job_id,
        },
        "strategy": strategy,
        "strategy_label": label,
        "symbol": symbol,
        "timeframe": timeframe,
        "metadata": metadata,
        "summary": summary,
        "explanations": {
            "entry": entry_reason,
            "exit": exit_reason,
        },
    }
    return record


def _write_result_record(record: Dict[str, Any]) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    filename = _safe_filename(
        record.get("strategy_label", record.get("strategy", "strategy")).lower(),
        record.get("symbol", "symbol").lower(),
        record.get("timeframe", "tf"),
        timestamp,
    )
    target_path = RESULTS_UI_DIR / filename
    target_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return target_path


def _run_strategy_symbol_backtest(
    *,
    plan_job_id: str,
    child_job_id: str,
    strategy: str,
    timeframe: str,
    symbol: str,
    capital_pct: Optional[float],
    config_path: Path,
    base_config: Dict[str, Any],
    summary_entry: Dict[str, Any],
    start_date: Optional[str],
    end_date: Optional[str],
) -> Dict[str, Any]:
    config_clone = copy.deepcopy(base_config)
    data_section = config_clone.setdefault("data", {})
    if symbol:
        data_section["symbol"] = symbol
    if timeframe:
        data_section["interval"] = timeframe
    if start_date:
        data_section["start_date"] = start_date
    if end_date:
        data_section["end_date"] = end_date

    backtest_section = config_clone.setdefault("backtest", {})
    backtest_section["enforce_gates"] = False
    try:
        current_gate = float(backtest_section.get("min_gate_trades", 1) or 1)
    except (TypeError, ValueError):
        current_gate = 1
    backtest_section["min_gate_trades"] = min(current_gate, 1)

    tmp_path = Path(tempfile.gettempdir()) / f"ui_backtest_{uuid.uuid4().hex}.yaml"
    with tmp_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_clone, handle, sort_keys=False)

    try:
        payload = _run_backtest_subprocess(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    label = summary_entry.get("label") or _strategy_label(strategy)
    record = _render_record(
        plan_job_id=plan_job_id,
        child_job_id=child_job_id,
        strategy=strategy,
        label=label,
        symbol=symbol,
        timeframe=timeframe,
        capital_pct=capital_pct,
        config_path=config_path,
        entry_reason=summary_entry.get("entry_reason", ""),
        exit_reason=summary_entry.get("exit_reason", ""),
        payload=payload,
    )
    result_path = _write_result_record(record)
    return {
        "path": str(result_path),
        "strategy": strategy,
        "symbol": symbol,
        "timeframe": timeframe,
        "summary": record.get("summary"),
    }


def run_backtest_plan(
    plan_job_id: str,
    plan_jobs: List[Dict[str, Any]],
    capital_pct: str | float | int,
    plan_summary: List[Dict[str, Any]],
    start_date: Optional[str],
    end_date: Optional[str],
) -> None:
    try:
        capital_value = float(capital_pct)
    except (TypeError, ValueError):
        capital_value = None
    summary_lookup = {
        (str(item.get("strategy") or "").lower(), str(item.get("timeframe") or "").lower()): item
        for item in (plan_summary or [])
    }
    manifest = get_strategy_manifest()
    start_iso = _coerce_date(start_date, end_of_day=False)
    end_iso = _coerce_date(end_date, end_of_day=True)
    _update_job_entry(plan_job_id, status="running", started_at=_timestamp())
    try:
        for job_meta in plan_jobs:
            child_id = job_meta.get("job_id")
            plan_entry = job_meta.get("plan_entry", {})
            if not child_id:
                continue
            _update_job_entry(child_id, status="running", started_at=_timestamp())
            strategy = str(plan_entry.get("strategy") or "")
            timeframe = str(plan_entry.get("timeframe") or "")
            manifest_entry = manifest.get(strategy, {})
            config_path = Path(manifest_entry.get("config") or "config.yaml")
            if not config_path.exists():
                raise FileNotFoundError(f"Config template missing at {config_path}")
            base_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            template_symbol = str(base_config.get("data", {}).get("symbol") or "").upper()
            symbols = [str(symbol).upper() for symbol in plan_entry.get("symbols") or [] if symbol]
            if not symbols and template_symbol:
                symbols = [template_symbol]
            if not symbols:
                symbols = ["XBTUSD"]
            summary_entry = summary_lookup.get((strategy.lower(), timeframe.lower()), {})
            results: List[Dict[str, Any]] = []
            for symbol in symbols:
                try:
                    result_info = _run_strategy_symbol_backtest(
                        plan_job_id=plan_job_id,
                        child_job_id=child_id,
                        strategy=strategy,
                        timeframe=timeframe,
                        symbol=symbol,
                        capital_pct=capital_value,
                        config_path=config_path,
                        base_config=base_config,
                        summary_entry=summary_entry,
                        start_date=start_iso,
                        end_date=end_iso,
                    )
                    results.append(result_info)
                    _append_job_result(child_id, result_info)
                    parent_result = dict(result_info)
                    parent_result["child_job_id"] = child_id
                    _append_job_result(plan_job_id, parent_result)
                except Exception as exc:
                    _update_job_entry(child_id, status="failed", completed_at=_timestamp(), error=str(exc))
                    logger.exception("Backtest failed for %s/%s: %s", strategy, symbol, exc)
                    raise
            _update_job_entry(
                child_id,
                status="completed",
                completed_at=_timestamp(),
                results=results,
            )
        _update_job_entry(plan_job_id, status="completed", completed_at=_timestamp())
    except Exception as exc:
        _update_job_entry(plan_job_id, status="failed", completed_at=_timestamp(), error=str(exc))
        logger.exception("Backtest plan %s failed: %s", plan_job_id, exc)
        raise


def spawn_plan_runner(
    plan_job_id: str,
    plan_jobs: List[Dict[str, Any]],
    capital_pct: str | float | int,
    plan_summary: List[Dict[str, Any]],
    start_date: Optional[str],
    end_date: Optional[str],
) -> None:
    thread = threading.Thread(
        target=run_backtest_plan,
        args=(plan_job_id, plan_jobs, capital_pct, plan_summary, start_date, end_date),
        daemon=True,
    )
    thread.start()



def create_backtest_plan_job(
    plan: List[Dict[str, Any]],
    capital_pct: str | float | int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    normalized_plan: List[Dict[str, Any]] = []
    for entry in plan:
        normalized_plan.append(
            {
                "strategy": entry.get("strategy"),
                "timeframe": entry.get("timeframe"),
                "symbols": entry.get("symbols") or [],
            }
        )
    plan_summary = _build_plan_summary(normalized_plan)
    job = {
        "type": "backtest_plan",
        "status": "queued",
        "submitted_at": _timestamp(),
        "capital_pct": capital_pct,
        "plan": normalized_plan,
        "plan_summary": plan_summary,
        "start_date": start_date,
        "end_date": end_date,
        "child_jobs": [],
    }
    job_id = _register_job(job)
    return job_id, plan_summary


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


def _post_slack_message(text: str, **kwargs: Any) -> None:
    if not SLACK_WEBHOOK_URL:
        return
    try:
        import requests  # type: ignore
    except ImportError:
        return
    payload = {"text": text}
    payload.update(kwargs.get("extra_payload", {}))
    try:
        requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=10)
    except Exception:
        pass


def _record_guardrail_violation(event: Dict[str, Any]) -> None:
    payload = dict(event or {})
    payload.setdefault("timestamp", _timestamp())
    log_path: Path = globals().get("GUARDRAIL_LOG", GUARDRAIL_LOG_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload))
        handle.write("\n")
    stage = payload.get("stage") or "unknown"
    reason = payload.get("reason") or payload.get("message") or "Guardrail triggered."
    _post_slack_message(f"Guardrail violation ({stage}): {reason}", extra_payload={"stage": stage})


def clear_job_registry() -> Dict[str, Any]:
    removed = 0
    if JOB_REGISTRY_PATH.exists():
        try:
            jobs = _read_json(JOB_REGISTRY_PATH, {})
            removed = len(jobs) if isinstance(jobs, dict) else 0
        finally:
            JOB_REGISTRY_PATH.unlink(missing_ok=True)
    return {"status": "cleared", "removed": removed}


def clear_guardrail_logs() -> Dict[str, Any]:
    removed_files: List[str] = []
    if GUARDRAIL_LOG_PATH.exists():
        GUARDRAIL_LOG_PATH.unlink()
        removed_files.append(str(GUARDRAIL_LOG_PATH))
    if GUARDRAIL_SNAPSHOT_DIR.exists():
        for child in GUARDRAIL_SNAPSHOT_DIR.glob("**/*"):
            if child.is_file():
                removed_files.append(str(child))
        shutil.rmtree(GUARDRAIL_SNAPSHOT_DIR, ignore_errors=True)
    return {"status": "cleared", "removed": removed_files}


__all__ = [
    "get_trading_status",
    "get_open_positions_summary",
    "schedule_trading_plan",
    "stop_trading",
    "stop_all",
    "create_backtest_plan_job",
    "spawn_plan_runner",
    "start_backtest",
    "start_backtest_batch",
    "kafka_stack_status",
    "start_kafka_stack",
    "stop_kafka_stack",
    "infer_trade_status",
    "record_trade_decision",
    "get_trade_metrics",
    "clear_trade_history",
    "clear_job_registry",
    "clear_guardrail_logs",
]

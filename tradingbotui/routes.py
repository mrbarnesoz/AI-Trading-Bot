"""Flask routes for the trading bot UI."""



from __future__ import annotations



import csv

import difflib

import json

import math

import os

import shutil

import subprocess

import sys

import tempfile

import types

import zipfile

from datetime import datetime, timedelta, timezone

from pathlib import Path

from statistics import mean, pstdev

from typing import Dict, List, Optional, Tuple


import yaml

from flask import (

    Blueprint,

    Response,

    current_app,

    flash,

    jsonify,

    redirect,

    render_template,

    request,

    send_file,

    send_from_directory,

    url_for,

)


from tradingbotui import tasks
from tradingbotui.strategy_manifest import (
    get_strategy_manifest,
    get_timeframes_for_strategy,
)



try:

    from ai_trading_bot.config import load_config

    from ai_trading_bot.risk.guardrails import GuardrailViolation, validate_app_config

except Exception:

    load_config = None  # type: ignore

    GuardrailViolation = None  # type: ignore

    def validate_app_config(*_args, **_kwargs):  # type: ignore

        return None





main_bp = Blueprint("main", __name__)

api_bp = Blueprint("api", __name__, url_prefix="/api")



_MODULE_PATH = Path(__file__).resolve()
_STATIC_DIR = _MODULE_PATH.parent / "static"
_APP_BUNDLE_PATH = _STATIC_DIR / "js" / "app.bundle.js"
_VENDOR_DIR = _STATIC_DIR / "vendor"


def _vendor_mtime() -> float:
    if not _VENDOR_DIR.exists():
        return _MODULE_PATH.stat().st_mtime
    mtimes = [path.stat().st_mtime for path in _VENDOR_DIR.glob("*.js") if path.exists()]
    return max(mtimes) if mtimes else _MODULE_PATH.stat().st_mtime


_ASSET_VERSION = str(
    int(
        max(
            _MODULE_PATH.stat().st_mtime,
            _APP_BUNDLE_PATH.stat().st_mtime if _APP_BUNDLE_PATH.exists() else _MODULE_PATH.stat().st_mtime,
            _vendor_mtime(),
        )
    )
)

_ALLOCATION_PATH = Path(os.getcwd()) / "logs" / "strategy_allocations.json"

_JOB_REGISTRY_PATH = Path(os.getcwd()) / "logs" / "job_registry.json"

_TRADE_LOG_PATH = Path(os.getcwd()) / "logs" / "trades.jsonl"

_GUARDRAIL_LOG_PATH = Path(os.getcwd()) / "logs" / "guardrail_events.jsonl"

_PROMOTION_DIR = Path(os.getcwd()) / "results" / "promotions"

_GUARDRAIL_SNAPSHOT_DIR = Path(os.getcwd()) / "results" / "guardrail_snapshots"

_BACKTEST_EXPORT_DIR = Path(os.getcwd()) / "backtests"
_BACKTEST_RESULTS_DIR = Path(os.getcwd()) / "results" / "ui" / "backtest"
_BACKTEST_ARCHIVE_DIR = _BACKTEST_RESULTS_DIR / "archive"





@main_bp.app_context_processor

def _inject_template_globals() -> dict:

    return {"asset_version": _ASSET_VERSION}


@main_bp.route("/", methods=["GET"])
def dashboard() -> str:

    """Render the single-page dashboard shell."""

    return render_template("dashboard.html")




def _available_configs(config_dir: Path) -> List[str]:

    if not config_dir.exists():

        return []

    return sorted(

        [f.name for f in config_dir.iterdir() if f.suffix in {".yaml", ".yml"}]

    )





def _discover_ui_options() -> dict:

    config_dir = Path(os.getcwd()) / "configs"

    strategies = set()

    pairs = set()

    timeframes = set()



    def _normalise(val: str | None, *, default: str = "") -> str:

        if not val:

            return default

        return str(val).strip()



    if config_dir.exists():

        for path in config_dir.glob("*.y*ml"):

            stem = path.stem

            tokens = stem.split("_")

            if len(tokens) >= 3:

                pairs.add(tokens[0].upper())

                timeframes.add(tokens[1])

                strategies.add("_".join(tokens[2:]))

            elif len(tokens) == 2:

                pairs.add(tokens[0].upper())

                strategies.add(tokens[1])

            else:

                strategies.add(stem)

            try:

                with path.open("r", encoding="utf-8") as handle:

                    cfg = yaml.safe_load(handle)

            except Exception:

                cfg = {}

            data_cfg = cfg.get("data") if isinstance(cfg, dict) else {}

            if isinstance(data_cfg, dict):

                symbol = _normalise(data_cfg.get("symbol"))

                interval = _normalise(data_cfg.get("interval"))

                if symbol:

                    pairs.add(symbol.upper())

                if interval:

                    timeframes.add(interval)



    if not strategies:

        strategies.update({"maker_swing", "trend_follow", "mean_reversion"})

    if not pairs:

        pairs.update({"XBTUSD", "ETHUSD"})

    if not timeframes:

        timeframes.update({"1m", "5m", "1h", "1d"})



    return {

        "strategies": sorted(strategies),

        "pairs": sorted(pairs),

        "timeframes": sorted(timeframes, key=lambda tf: (len(tf), tf)),

    }





def _load_metrics() -> List[dict]:

    """Load recent backtest metrics exported by the UI task runner."""



    def _format_pair(meta: dict) -> str:

        symbol = meta.get("symbol") or meta.get("pair") or "-"

        interval = meta.get("interval") or meta.get("timeframe")

        return f"{symbol} {interval}" if interval else symbol



    base_results = Path(os.getcwd()) / "results"

    ui_backtests = base_results / "ui" / "backtest"

    candidates: List[Path] = []



    if ui_backtests.exists():

        candidates.extend(

            sorted(

                [

                    path

                    for path in ui_backtests.glob("*.json")

                    if "-latest" not in path.name

                ],

                key=lambda p: p.stat().st_mtime,

                reverse=True,

            )

        )



    if not candidates and base_results.exists():

        candidates.extend(

            sorted(

                base_results.glob("**/*.json"),

                key=lambda p: p.stat().st_mtime,

                reverse=True,

            )

        )



    summaries: List[dict] = []

    for result_file in candidates:

        try:

            with result_file.open("r", encoding="utf-8") as handle:

                payload = json.load(handle)

        except (json.JSONDecodeError, OSError):

            continue



        summary = payload.get("summary") or payload.get("aggregate")

        if not isinstance(summary, dict):

            continue



        metadata = payload.get("metadata", {})

        if not isinstance(metadata, dict):

            metadata = {}

        identity = payload.get("identity", {})

        if not isinstance(identity, dict):

            identity = {}

        record = {

            "path": str(result_file.relative_to(base_results)) if base_results in result_file.parents else result_file.name,

            "strategy": metadata.get("strategy") or identity.get("strategy") or "-",

            "pair": _format_pair(metadata or identity),

            "sharpe": summary.get("calc_sharpe") or summary.get("sharpe_ratio"),

            "max_drawdown": summary.get("max_drawdown"),

            "expectancy": summary.get("expectancy_after_costs"),

            "win_rate": summary.get("win_rate"),

        }

        summaries.append(record)

        if len(summaries) >= 10:

            break

    return summaries





def _load_trade_history() -> List[float]:

    """Return a list of recent trade P&L values for the chart."""

    trades_file = Path(os.getcwd()) / "results" / "latest_trades.json"

    if trades_file.exists():

        try:

            with trades_file.open("r", encoding="utf-8") as handle:

                data = json.load(handle)

            if isinstance(data, list):

                return [float(x) for x in data[-20:]]

        except (json.JSONDecodeError, OSError, ValueError):

            pass

    # Fallback dummy data

    return [50, -20, 30, -10, 80, -40, 60, -15]





def _load_auto_research() -> List[dict]:

    """Load the latest auto strategy research summary if available."""

    auto_dir = Path(os.getcwd()) / "results" / "auto"

    if not auto_dir.exists():

        return []

    latest_file = None

    for path in auto_dir.glob("*.json"):

        if latest_file is None or path.stat().st_mtime > latest_file.stat().st_mtime:

            latest_file = path

    if latest_file is None:

        return []

    try:

        with latest_file.open("r", encoding="utf-8") as handle:

            payload = json.load(handle)

    except (json.JSONDecodeError, OSError):

        return []



    selections = payload.get("selections", [])

    rows: List[dict] = []

    for entry in selections:

        best = entry.get("best") or {}

        if not isinstance(best, dict):

            continue

        rows.append(

            {

                "symbol": entry.get("symbol"),

                "timeframe": entry.get("timeframe"),

                "strategy": best.get("strategy"),

                "sharpe": best.get("calc_sharpe"),

                "trades": best.get("trades_count"),

                "return": best.get("total_return"),

                "path": (Path('auto') / latest_file.name).as_posix(),

            }

        )

    rows.sort(

        key=lambda item: (

            (item.get("sharpe") if item.get("sharpe") is not None else float("-inf")),

            (item.get("return") if item.get("return") is not None else float("-inf")),

        ),

        reverse=True,

    )

    return rows





UI_STATE_PATH = Path(os.getcwd()) / "logs" / "ui_state.json"





def _load_ui_state() -> dict:

    default = {

        "periodic_retraining": False,

        "last_checkpoint": None,

        "mode": "idle",

        "capital_pct": 10,

        "multi_strategy": True,

        "last_pairs": [],

        "last_strategy": None,

        "last_timeframe": None,

        "kill_switch": False,

        "drawdown_threshold": 0.2,

        "online_learning": False,

        "notifications": [],

    }

    try:

        if UI_STATE_PATH.exists():

            data = json.loads(UI_STATE_PATH.read_text(encoding="utf-8"))

            if isinstance(data, dict):

                default.update(data)

    except Exception:

        pass

    return default





def _save_ui_state(state: dict) -> None:

    try:

        UI_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

        UI_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")

    except Exception:  # pragma: no cover - log only

        current_app.logger.exception("Failed to persist UI state.")





def _build_equity_curve(trade_history: List[float]) -> List[float]:

    equity = []

    total = 0.0

    for pnl in trade_history:

        total += pnl

        equity.append(total)

    return equity





def _build_drawdown_series(equity_curve: List[float]) -> List[float]:

    drawdown = []

    running_max = float("-inf")

    for value in equity_curve:

        running_max = max(running_max, value)

        drawdown.append(value - running_max)

    return drawdown





def _strategy_orchestration_metrics(auto_data: List[dict]) -> List[dict]:

    metrics = []

    for entry in auto_data:

        best = entry.get("best")

        if not best:

            continue

        metrics.append(

            {

                "symbol": entry.get("symbol"),

                "timeframe": entry.get("timeframe"),

                "strategy": best.get("strategy"),

                "sharpe": best.get("calc_sharpe"),

                "win_rate": best.get("win_rate"),

                "max_drawdown": best.get("max_drawdown"),

                "trades_count": best.get("trades_count"),

            }

        )

    return metrics





def _list_recent_results(limit: int = 10) -> List[dict]:
    records: List[dict] = []
    if not _BACKTEST_RESULTS_DIR.exists():
        return records
    try:
        paths = sorted(
            _BACKTEST_RESULTS_DIR.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return records

    seen_keys: set[str] = set()
    max_records = max(int(limit or 0), 0) if limit is not None else None

    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        summary = payload.get("summary", {})
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        identity = _result_identity(payload)
        identity_key = _strategy_key(identity.get("symbol"), identity.get("timeframe"), identity.get("strategy"))
        seen_token = identity_key or f"file::{path.name.lower()}"
        if identity_key and identity_key in seen_keys:
            continue
        if seen_token in seen_keys:
            continue
        seen_keys.add(seen_token)

        flags: List[str] = []
        trades = summary.get("trades_count") or summary.get("total_trades")
        sharpe = summary.get("calc_sharpe")
        try:
            if trades is not None and float(trades) < 200:
                flags.append("low_trades")
        except (ValueError, TypeError):
            pass
        try:
            if sharpe is not None and float(sharpe) < 0:
                flags.append("negative_sharpe")
        except (ValueError, TypeError):
            pass

        generated_at = (
            payload.get("generated_at")
            or metadata.get("generated_at")
            or summary.get("generated_at")
            or _to_iso_timestamp(path.stat().st_mtime)
        )
        generated_iso = _to_iso_timestamp(generated_at) or _to_iso_timestamp(path.stat().st_mtime)

        records.append(
            {
                "file": path.name,
                "strategy": identity.get("strategy") or payload.get("strategy"),
                "strategy_label": payload.get("strategy_label") or identity.get("label") or metadata.get("label"),
                "sharpe": summary.get("calc_sharpe"),
                "trades": summary.get("trades_count"),
                "return": summary.get("total_return"),
                "path": str(path),
                "config": payload.get("config")
                or (
                    payload.get("metadata", {}).get("config")
                    if isinstance(payload.get("metadata"), dict)
                    else None
                ),
                "flags": flags,
                "summary": summary,
                "metadata": metadata,
                "capital_pct": payload.get("capital_pct"),
                "generated_at": generated_iso,
                "identity": identity,
                "identity_key": identity_key,
                "plan_job_id": identity.get("plan_job_id"),
                "child_job_id": identity.get("child_job_id"),
                "explanations": payload.get("explanations") if isinstance(payload.get("explanations"), dict) else {},
            }
        )
        if max_records is not None and len(records) >= max_records:
            break
    return records


def _match_backtest_results(job: dict) -> List[Path]:
    if not _BACKTEST_RESULTS_DIR.exists() or not isinstance(job, dict):
        return []

    candidates: List[Path] = []
    job_config = str(job.get("config") or "").lower()
    job_strategy = str(job.get("strategy") or "").lower()
    job_timeframe = str(job.get("timeframe") or "").lower()
    job_pairs_raw: List[str] = []
    for key in ("pairs", "symbols"):
        raw = job.get(key)
        if isinstance(raw, (list, tuple, set)):
            job_pairs_raw.extend(raw)
    job_pair_single = job.get("pair")
    if job_pair_single:
        job_pairs_raw.append(job_pair_single)
    job_symbol_single = job.get("symbol")
    if job_symbol_single:
        job_pairs_raw.append(job_symbol_single)
    job_pairs = {str(pair).upper() for pair in job_pairs_raw if pair}
    job_id = str(job.get("id") or "").lower()
    job_parent_id = str(job.get("parent_id") or "").lower()
    related_child_ids = {job_id} if job_id else set()
    for child_id in job.get("child_jobs") or []:
        if child_id:
            related_child_ids.add(str(child_id).lower())
    plan_link_ids = set()
    if job_parent_id:
        plan_link_ids.add(job_parent_id)
    plan_hint = str(job.get("plan_job_id") or "").lower()
    if plan_hint:
        plan_link_ids.add(plan_hint)
    job_type = str(job.get("type") or "").lower()
    if job_type == "backtest_plan" and job_id:
        plan_link_ids.add(job_id)

    try:
        files = sorted(
            _BACKTEST_RESULTS_DIR.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        files = []

    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload_config = str(payload.get("config") or "").lower()
        if job_config and payload_config and payload_config == job_config:
            candidates.append(path)
            continue
        file_name = path.name.lower()
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        identity = _result_identity(payload)
        meta_strategy = str(identity.get("strategy") or metadata.get("mode") or "").lower()
        meta_timeframe = str(identity.get("timeframe") or metadata.get("interval") or summary.get("interval") or "").lower()
        meta_symbol = str(identity.get("symbol") or metadata.get("symbol") or metadata.get("pair") or "").upper()
        identity_plan_id = str(identity.get("plan_job_id") or "").lower()
        identity_child_id = str(identity.get("child_job_id") or "").lower()

        matched = False

        if job_id and identity_child_id and identity_child_id == job_id:
            matched = True
        if not matched and related_child_ids and identity_child_id in related_child_ids:
            matched = True
        if not matched and plan_link_ids and identity_plan_id and identity_plan_id in plan_link_ids:
            matched = True

        if job_strategy and meta_strategy and (meta_strategy == job_strategy or job_strategy in meta_strategy):
            if not job_timeframe or not meta_timeframe or meta_timeframe == job_timeframe:
                matched = True
        if not matched and job_pairs and meta_symbol and meta_symbol in job_pairs:
            if not job_timeframe or not meta_timeframe or meta_timeframe == job_timeframe:
                matched = True
        if not matched and job_strategy and job_strategy in file_name:
            matched = True
        if not matched and job_pairs:
            lower_pairs = {pair.lower() for pair in job_pairs}
            if any(pair in file_name for pair in lower_pairs):
                matched = True
        if not matched and job_id and job_id.lower() in file_name:
            matched = True

        if matched:
            candidates.append(path)

    return candidates





def _strategy_key(symbol: str | None, timeframe: str | None, strategy: str | None) -> str:

    safe_symbol = (symbol or "").upper()

    safe_timeframe = (timeframe or "").lower()

    safe_strategy = (strategy or "").lower().replace(" ", "_")

    return f"{safe_symbol}::{safe_timeframe}::{safe_strategy}"


def _result_identity(payload: dict) -> dict:
    """Extract normalized identity attributes from a result payload."""
    identity = payload.get("identity")
    if not isinstance(identity, dict):
        identity = {}
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        summary = {}

    def _pick(*values: Optional[str]) -> Optional[str]:
        for value in values:
            if isinstance(value, str):
                trimmed = value.strip()
                if trimmed:
                    return trimmed
        return None

    strategy = _pick(
        identity.get("strategy"),
        metadata.get("strategy"),
        metadata.get("mode"),
        summary.get("strategy"),
        summary.get("mode"),
        payload.get("strategy"),
    )
    label = _pick(
        identity.get("label"),
        metadata.get("label"),
        payload.get("strategy_label"),
        strategy.title() if isinstance(strategy, str) else None,
    )
    symbol = _pick(
        identity.get("symbol"),
        metadata.get("symbol"),
        metadata.get("pair"),
        summary.get("symbol"),
        summary.get("pair"),
    )
    timeframe = _pick(
        identity.get("timeframe"),
        metadata.get("interval"),
        metadata.get("timeframe"),
        summary.get("interval"),
        summary.get("timeframe"),
        identity.get("mode_interval"),
        metadata.get("mode_interval"),
    )
    plan_job_id = _pick(identity.get("plan_job_id"), payload.get("plan_job_id"))
    child_job_id = _pick(identity.get("child_job_id"), payload.get("child_job_id"))
    config_stem = _pick(identity.get("config_stem"))

    return {
        "strategy": strategy,
        "label": label,
        "symbol": symbol,
        "timeframe": timeframe,
        "plan_job_id": plan_job_id,
        "child_job_id": child_job_id,
        "config_stem": config_stem,
    }





def _load_strategy_allocations() -> dict:

    default = {"weights": {}, "disabled": []}

    try:

        if _ALLOCATION_PATH.exists():

            data = json.loads(_ALLOCATION_PATH.read_text(encoding="utf-8"))

            if isinstance(data, dict):

                weights = data.get("weights")

                disabled = data.get("disabled")

                if isinstance(weights, dict):

                    default["weights"] = {

                        str(key): float(value)

                        for key, value in weights.items()

                        if isinstance(key, str) and isinstance(value, (int, float))

                    }

                if isinstance(disabled, list):

                    default["disabled"] = [str(item) for item in disabled if isinstance(item, str)]

    except Exception:

        pass

    return default





def _save_strategy_allocations(data: dict) -> None:

    try:

        _ALLOCATION_PATH.parent.mkdir(parents=True, exist_ok=True)

        payload = {

            "weights": data.get("weights", {}),

            "disabled": data.get("disabled", []),

        }

        _ALLOCATION_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    except Exception:  # pragma: no cover - diagnostic only

        current_app.logger.exception("Failed to persist strategy allocations.")


def _to_iso_timestamp(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
            return dt.isoformat(timespec="seconds")
        except (OSError, OverflowError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        for parser in (
            lambda s: datetime.fromisoformat(s),
            lambda s: datetime.strptime(s, "%Y%m%d-%H%M%S"),
            lambda s: datetime.strptime(s, "%Y%m%d%H%M%S"),
        ):
            try:
                dt = parser(text)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                return dt.isoformat(timespec="seconds")
            except (ValueError, TypeError):
                continue
    return None


def _normalize_job_entry(job: dict) -> dict:
    if not isinstance(job, dict):
        return {}
    normalized = dict(job)
    for field in ("submitted_at", "started_at", "completed_at"):
        iso_value = _to_iso_timestamp(job.get(field))
        if iso_value:
            normalized[field] = iso_value
    return normalized


def _resolve_timeframe_for_strategy(
    strategy: str,
    requested_timeframe: Optional[str],
    manifest: Optional[dict] = None,
) -> Tuple[Optional[str], Optional[str], List[str], Optional[str]]:
    manifest = manifest or get_strategy_manifest()
    entry = manifest.get(strategy, {})
    allowed = [
        str(tf)
        for tf in entry.get("timeframes", [])
        if isinstance(tf, str) and tf.strip()
    ]
    lookup = {tf.lower(): tf for tf in allowed}
    default_raw = entry.get("default")
    default_tf = None
    if isinstance(default_raw, str) and default_raw:
        default_tf = lookup.get(default_raw.lower(), default_raw)

    resolved = None
    error = None
    if lookup:
        if isinstance(requested_timeframe, str) and requested_timeframe:
            key = requested_timeframe.lower()
            if key in lookup:
                resolved = lookup[key]
            else:
                error = f"Allowed timeframes for {strategy} are {allowed}"
        else:
            resolved = lookup.get(default_raw.lower()) if isinstance(default_raw, str) and default_raw else None
            if not resolved and allowed:
                resolved = allowed[0]
    else:
        if isinstance(requested_timeframe, str) and requested_timeframe:
            resolved = requested_timeframe
        elif default_tf:
            resolved = default_tf

    if not resolved:
        resolved = requested_timeframe or default_tf or "1h"

    return resolved, error, allowed, default_tf or (allowed[0] if allowed else None)


def _read_job_registry() -> Dict[str, dict]:
    if not _JOB_REGISTRY_PATH.exists():
        return {}
    try:
        return json.loads(_JOB_REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _write_job_registry(jobs: Dict[str, dict]) -> None:
    _JOB_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _JOB_REGISTRY_PATH.write_text(json.dumps(jobs, indent=2, sort_keys=True), encoding="utf-8")


def _update_job_registry(job_id: str, **updates) -> None:
    jobs = _read_job_registry()
    entry = jobs.get(job_id)
    if entry is None:
        return
    entry.update(updates)
    _write_job_registry(jobs)


def _remove_job(job_id: str) -> Optional[dict]:
    jobs = _read_job_registry()
    entry = jobs.pop(job_id, None)
    if entry is None:
        return None
    _write_job_registry(jobs)
    return entry



def _list_jobs() -> List[dict]:
    jobs = _read_job_registry()
    normalized: List[dict] = []
    for entry in jobs.values():
        if entry.get("hidden"):
            continue
        normalized.append(_normalize_job_entry(entry))
    return sorted(normalized, key=lambda item: item.get("submitted_at", ""), reverse=True)


def _load_guardrail_logs(limit: Optional[int] = None) -> List[dict]:
    if not _GUARDRAIL_LOG_PATH.exists():
        return []
    entries: List[dict] = []
    try:
        with _GUARDRAIL_LOG_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    entries.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
    if limit is not None:
        return entries[:limit]
    return entries


def _load_guardrail_snapshots(limit: int = 20) -> List[dict]:
    if not _GUARDRAIL_SNAPSHOT_DIR.exists():
        return []
    results: List[dict] = []
    # Search for JSON summaries in descending mtime order.
    files = sorted(
        _GUARDRAIL_SNAPSHOT_DIR.glob("**/*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        results.append(
            {
                "name": path.name,
                "path": str(path),
                "created_at": datetime.utcfromtimestamp(path.stat().st_mtime).isoformat(),
                "summary": payload,
            }
        )
        if len(results) >= limit:
            break
    return results


def _cancel_inflight_jobs() -> int:
    jobs = _read_job_registry()
    if not jobs:
        return 0
    now = datetime.utcnow().isoformat()
    updated = False
    cancelled_count = 0
    for record in jobs.values():
        status = (record.get("status") or "").lower()
        if status in {"queued", "running"}:
            record["status"] = "cancelled"
            record.setdefault("completed_at", now)
            record["completed_at"] = now
            record["error"] = record.get("error") or "Cancelled via restart endpoint."
            updated = True
            cancelled_count += 1
    if updated:
        _JOB_REGISTRY_PATH.write_text(json.dumps(jobs, indent=2), encoding="utf-8")
    return cancelled_count


def _read_trade_records(limit: Optional[int] = None) -> List[dict]:
    if not _TRADE_LOG_PATH.exists():
        return []
    records: List[dict] = []
    try:
        with _TRADE_LOG_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []

    normalised: List[dict] = []
    for record in records:
        entry = dict(record) if isinstance(record, dict) else {}
        meta = entry.get("meta") if isinstance(entry.get("meta"), dict) else {}
        if not entry.get("timeframe"):
            entry["timeframe"] = (
                meta.get("timeframe")
                or meta.get("interval")
                or meta.get("mode_interval")
                or meta.get("resolution")
                or entry.get("timeframe")
            )
        if not entry.get("strategy"):
            entry["strategy"] = (
                entry.get("strategy")
                or meta.get("strategy")
                or meta.get("mode")
            )
        if not entry.get("status"):
            entry["status"] = tasks.infer_trade_status(meta, entry.get("status"))
        normalised.append(entry)

    if limit:
        return normalised[-limit:]
    return normalised


def _calculate_drawdown(pnl_series: List[float]) -> float:
    max_equity = float("-inf")
    equity = 0.0
    max_drawdown = 0.0
    for value in pnl_series:
        equity += value
        if equity > max_equity:
            max_equity = equity
        drawdown = max_equity - equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown


def _rolling_statistics(
    records: List[dict],
    *,
    days: int = 30,
    strategy: str | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> dict:
    if not records:
        return {"sharpe": 0.0, "trades": 0, "win_rate": 0.0, "drawdown": 0.0, "pnl_pct": 0.0, "pnl_usd": 0.0}

    horizon = timedelta(days=days)
    now = datetime.utcnow()
    filtered: List[float] = []
    wins = 0
    pnl_total = 0.0
    notional_total = 0.0

    for entry in records:
        ts = entry.get("timestamp")
        try:
            ts_dt = datetime.fromisoformat(ts.replace("Z", "")) if ts else None
        except Exception:
            ts_dt = None
        if ts_dt and now - ts_dt > horizon:
            continue
        if strategy and entry.get("strategy") != strategy:
            continue
        if symbol and entry.get("symbol") != symbol:
            continue
        if timeframe and entry.get("timeframe") != timeframe:
            continue
        pnl = float(entry.get("pnl", 0.0) or 0.0)
        pnl_total += pnl
        filtered.append(pnl)
        if pnl > 0:
            wins += 1
        try:
            notional = float(entry.get("notional_usd") or entry.get("position_size") or 0.0)
        except (TypeError, ValueError):
            notional = 0.0
        notional_total += abs(notional)

    if not filtered:
        return {"sharpe": 0.0, "trades": 0, "win_rate": 0.0, "drawdown": 0.0, "pnl_pct": 0.0, "pnl_usd": 0.0}

    trades = len(filtered)
    avg = mean(filtered)
    std = pstdev(filtered) if trades > 1 else 0.0
    sharpe = (avg / std * math.sqrt(trades)) if std else 0.0
    win_rate = wins / trades if trades else 0.0
    drawdown = _calculate_drawdown(filtered)
    pnl_pct = (pnl_total / notional_total * 100.0) if notional_total else 0.0
    return {
        "sharpe": round(sharpe, 4),
        "trades": trades,
        "win_rate": round(win_rate, 4),
        "drawdown": round(drawdown, 4),
        "pnl_pct": round(pnl_pct, 4),
        "pnl_usd": round(pnl_total, 4),
        "notional_usd": round(notional_total, 4),
    }


def _parameter_warnings(config_data: dict, config_path: Optional[str] = None) -> List[str]:
    warnings: List[str] = []
    backtest = config_data.get("backtest", {}) if isinstance(config_data, dict) else {}
    filters = config_data.get("filters", {}) if isinstance(config_data, dict) else {}

    if backtest:
        pos_frac = float(backtest.get("position_capital_fraction", 0.0) or 0.0)
        if pos_frac > 0.2:
            warnings.append(f"position_capital_fraction {pos_frac:.2f} exceeds recommended 0.20")
        total_frac = float(backtest.get("max_total_capital_fraction", 0.0) or 0.0)
        if total_frac > 0.9:
            warnings.append(f"max_total_capital_fraction {total_frac:.2f} exceeds recommended 0.90")
        min_gate = float(backtest.get("min_gate_trades", 0) or 0)
        if min_gate < 10:
            warnings.append("min_gate_trades below safety threshold (10)")
    if filters:
        confidence = filters.get("min_confidence")
        if confidence is not None and confidence < 0.05:
            warnings.append(f"min_confidence {confidence:.2f} below recommended 0.05")
    if config_path and load_config and validate_app_config:
        try:
            cfg_obj = load_config(config_path)
            validate_app_config(cfg_obj, stage="ui")
        except GuardrailViolation as exc:  # pragma: no cover - guardrail feedback
            messages = getattr(exc, "violations", None)
            warnings.extend(messages if messages else [str(exc)])
        except Exception:
            pass
    return warnings

def _resolve_config_path(strategy: str | None, symbol: str | None, timeframe: str | None) -> Optional[str]:
    mapping = _available_config_map()
    candidates = []
    strategy_key = (strategy or "").lower().replace(" ", "_")
    symbol_key = (symbol or "").lower() if symbol else ""
    timeframe_key = (timeframe or "").lower() if timeframe else ""
    if symbol_key and timeframe_key and strategy_key:
        candidates.append(f"{symbol_key}_{timeframe_key}_{strategy_key}")
    if symbol_key and strategy_key:
        candidates.append(f"{symbol_key}_{strategy_key}")
    if symbol_key and timeframe_key:
        candidates.append(f"{symbol_key}_{timeframe_key}")
    if strategy_key:
        candidates.append(strategy_key)
    for candidate in candidates:
        if candidate in mapping:
            return str(mapping[candidate])
    return None


def _load_config_dict(config_path: Optional[str]) -> Optional[dict]:
    if not config_path:
        return None
    path = Path(config_path)
    if not path.exists():
        return None
    if load_config is not None:
        try:
            cfg = load_config(str(path))
            if hasattr(cfg, "to_nested_dict"):
                return cfg.to_nested_dict()
        except Exception:
            pass
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None


def _strategy_rows() -> List[dict]:
    auto_data = _load_auto_research()
    orchestration = _strategy_orchestration_metrics(auto_data)
    recent_results = _list_recent_results(50)
    recent_by_config = {row.get("config"): row for row in recent_results if row.get("config")}
    recent_by_identity: Dict[str, dict] = {}
    for row in recent_results:
        identity_key = row.get("identity_key")
        if identity_key and identity_key not in recent_by_identity:
            recent_by_identity[identity_key] = row
    trade_records = _read_trade_records()
    jobs_by_config: Dict[str, dict] = {}
    for job in _list_jobs():
        cfg = job.get("config")
        if cfg:
            jobs_by_config[cfg] = job
    manifest = get_strategy_manifest()
    rows: List[dict] = []
    for metric in orchestration:
        strategy = metric.get("strategy")
        symbol = metric.get("symbol")
        timeframe = metric.get("timeframe")
        config_path = _resolve_config_path(strategy, symbol, timeframe)
        config_data = _load_config_dict(config_path)
        warnings = _parameter_warnings(config_data or {}, config_path) if config_data is not None else []
        row_id = "::".join(filter(None, [strategy, symbol, timeframe]))
        rolling = _rolling_statistics(
            trade_records,
            days=30,
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe,
        )
        sharpe = metric.get("sharpe") or 0.0
        trades = metric.get("trades_count") or 0
        underperforming = (sharpe < 0.5) or (trades < 50)
        manifest_entry = manifest.get(strategy, {})
        recommended_timeframes = manifest_entry.get("timeframes") or []
        default_timeframe = manifest_entry.get("default") or (recommended_timeframes[0] if recommended_timeframes else timeframe)
        strategy_label = manifest_entry.get("label") or strategy.replace("_", " ").title()
        identity_key = _strategy_key(symbol, timeframe, strategy)
        recent_result = recent_by_identity.get(identity_key)
        if recent_result is None and config_path:
            recent_result = recent_by_config.get(config_path)
        rows.append(
            {
                "id": row_id,
                "strategy": strategy,
                "strategy_label": strategy_label,
                "symbol": symbol,
                "timeframe": timeframe,
                "sharpe": sharpe,
                "win_rate": metric.get("win_rate"),
                "drawdown": metric.get("max_drawdown"),
                "trades": trades,
                "config": config_path,
                "latest_result": recent_result,
                "job": jobs_by_config.get(config_path),
                "rolling": rolling,
                "underperforming": underperforming,
                "warnings": warnings,
                "recommended_timeframes": recommended_timeframes,
                "default_timeframe": default_timeframe,
            }
        )
    return rows



def _build_alerts(
    rows: List[dict],
    ui_state: dict,
    portfolio_stats: dict,
    kafka_status: Optional[dict] = None,
) -> List[dict]:
    alerts: List[dict] = []
    if ui_state.get("kill_switch"):
        alerts.append({"level": "critical", "message": "Kill switch is active. Trading halted."})
    threshold = ui_state.get("drawdown_threshold") or 0.0
    drawdown = portfolio_stats.get("drawdown") or 0.0
    if threshold and drawdown > threshold:
        alerts.append({
            "level": "critical",
            "message": f"Portfolio drawdown {drawdown:.2f} exceeds threshold {threshold:.2f}.",
        })
    for row in rows:
        strategy = row.get("strategy")
        symbol = row.get("symbol")
        timeframe = row.get("timeframe")
        label = " / ".join(filter(None, [strategy, symbol, timeframe]))
        if row.get("underperforming"):
            alerts.append({
                "level": "warning",
                "message": f"Strategy {label} flagged as underperforming.",
            })
        job = row.get("job")
        if job and job.get("status") == "failed":
            alerts.append({
                "level": "error",
                "message": f"Job {job.get('id')} for {label} failed: {job.get('error', 'unknown error')}",
            })
        for warning in row.get("warnings", []):
            alerts.append({
                "level": "warning",
                "message": f"{label}: {warning}",
            })
    if kafka_status:
        kafka_error = kafka_status.get("error")
        running = kafka_status.get("running", False)
        manage_enabled = kafka_status.get("manage_enabled", True)
        if kafka_error:
            alerts.append({
                "level": "warning",
                "message": f"Kafka stack issue: {kafka_error}",
            })
        elif manage_enabled and not running:
            alerts.append({
                "level": "warning",
                "message": "Kafka/ZooKeeper stack is not running.",
            })
    return alerts


# API endpoints

@api_bp.route('/strategies', methods=['GET'])
def api_get_strategies():
    rows = _strategy_rows()
    min_sharpe = request.args.get('min_sharpe', type=float)
    if min_sharpe is not None:
        rows = [row for row in rows if (row.get('sharpe') or 0.0) >= min_sharpe]
    if request.args.get('underperforming') == 'true':
        rows = [row for row in rows if row.get('underperforming')]
    trade_records = _read_trade_records()
    portfolio_stats = _rolling_statistics(trade_records)
    kafka_status = tasks.kafka_stack_status()
    alerts = _build_alerts(rows, _load_ui_state(), portfolio_stats, kafka_status)
    response = {
        'timestamp': datetime.utcnow().isoformat(),
        'strategies': rows,
        'manifest': get_strategy_manifest(),
        'portfolio': portfolio_stats,
        'jobs': _list_jobs(),
        'alerts': alerts,
        'kafka': kafka_status,
    }
    return jsonify(response)


@api_bp.route('/strategies/promote', methods=['POST'])
def api_promote_strategy():
    payload = request.get_json(force=True) or {}
    config_path = payload.get('config')
    if not config_path:
        return jsonify({'error': 'config path is required'}), 400
    source = Path(config_path)
    if not source.exists():
        return jsonify({'error': f'config not found at {config_path}'}), 404
    destination = Path(os.getcwd()) / 'config.yaml'
    timestamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
    snapshot_dir = _PROMOTION_DIR / timestamp
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    try:
        if destination.exists():
            shutil.copy2(destination, snapshot_dir / f'previous-{destination.name}')
        shutil.copy2(source, snapshot_dir / source.name)
        shutil.copy2(source, destination)
    except OSError as exc:
        return jsonify({'error': f'failed to promote config: {exc}'}), 500

    rows = _strategy_rows()
    promoted_row = next((row for row in rows if row.get('config') == str(source)), None)
    summary = {
        'timestamp': timestamp,
        'source': str(source),
        'destination': str(destination),
        'strategy': promoted_row,
    }
    (snapshot_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (snapshot_dir / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['key', 'value'])
        for key, value in summary.items():
            if isinstance(value, dict):
                writer.writerow([key, json.dumps(value)])
            else:
                writer.writerow([key, value])

    state = _load_ui_state()
    state['mode'] = 'live'
    state['active_config'] = str(destination)
    state['last_promotion'] = timestamp
    _save_ui_state(state)
    return jsonify({'status': 'ok', 'summary': summary})


@api_bp.route('/backtests', methods=['GET'])
def api_list_backtests():
    limit = request.args.get('limit', default=20, type=int)
    jobs = _list_jobs()[:limit]
    results = _list_recent_results(limit)
    guardrail_logs = _load_guardrail_logs()
    guardrail_snapshots = _load_guardrail_snapshots()
    return jsonify(
        {
            'jobs': jobs,
            'results': results,
            'guardrail_logs': guardrail_logs,
            'guardrail_snapshots': guardrail_snapshots,
            'guardrails': guardrail_logs,  # backwards compatibility with legacy UI
        }
    )


@api_bp.route('/backtests/results/<path:result_file>', methods=['GET'])
def api_get_backtest_result(result_file: str):
    safe_name = Path(result_file).name
    target_path = _BACKTEST_RESULTS_DIR / safe_name
    if not target_path.exists():
        return jsonify({'error': 'result_not_found'}), 404
    try:
        payload = json.loads(target_path.read_text(encoding='utf-8'))
    except Exception as exc:  # pragma: no cover - diagnostic
        current_app.logger.exception("Failed to read backtest result %s", target_path)
        return jsonify({'error': 'result_read_failed', 'details': str(exc)}), 500
    return jsonify(payload)


@api_bp.route('/backtests/results', methods=['DELETE'])
def api_clear_backtest_results():
    archive = request.args.get('archive', default='1')
    archive_results = str(archive).lower() not in {'0', 'false', 'no'}
    if not _BACKTEST_RESULTS_DIR.exists():
        return jsonify({'status': 'cleared', 'removed': 0, 'archived': []})
    files = sorted(_BACKTEST_RESULTS_DIR.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    archived_paths: List[str] = []
    removed = 0
    if archive_results and files:
        _BACKTEST_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    for path in files:
        if archive_results:
            target = _BACKTEST_ARCHIVE_DIR / path.name
            counter = 1
            while target.exists():
                target = _BACKTEST_ARCHIVE_DIR / f"{path.stem}-{counter}{path.suffix}"
                counter += 1
            try:
                shutil.move(str(path), str(target))
                archived_paths.append(str(target))
            except OSError:
                continue
        else:
            try:
                path.unlink()
            except OSError:
                continue
        removed += 1
    return jsonify({'status': 'cleared', 'removed': removed, 'archived': archived_paths})


@api_bp.route('/trading/status', methods=['GET'])
def api_trading_status():
    status = tasks.get_trading_status()
    status["open_positions"] = tasks.get_open_positions_summary()
    return jsonify(status)


def _parse_strategy_list(payload: dict) -> List[str]:
    raw = payload.get('strategies')
    strategies: List[str] = []
    if isinstance(raw, str):
        strategies = [raw]
    elif isinstance(raw, (list, tuple, set)):
        strategies = [str(item).strip() for item in raw if str(item).strip()]
    if not strategies and payload.get('strategy'):
        strategies = [str(payload['strategy']).strip()]
    return [strategy for strategy in strategies if strategy]


def _parse_symbol_list(payload: dict) -> List[str]:
    raw = payload.get('symbols')
    symbols: List[str] = []
    if isinstance(raw, str):
        symbols = [raw]
    elif isinstance(raw, (list, tuple, set)):
        symbols = [str(item).upper() for item in raw if str(item).strip()]
    if not symbols and payload.get('symbol'):
        symbols = [str(payload['symbol']).upper()]
    return [symbol for symbol in symbols if symbol]


def _normalize_timeframe_mapping(strategies: List[str], raw_timeframes: object) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if isinstance(raw_timeframes, dict):
        for key, value in raw_timeframes.items():
            if isinstance(key, str) and isinstance(value, str) and value:
                mapping[key] = value
    elif isinstance(raw_timeframes, (list, tuple)):
        for idx, value in enumerate(raw_timeframes):
            if idx < len(strategies) and isinstance(value, str) and value:
                mapping[strategies[idx]] = value
    return mapping


def _build_trading_plan(
    strategies: List[str],
    symbols: List[str],
    timeframe_overrides: Dict[str, str],
) -> Tuple[List[dict], List[str]]:
    manifest = get_strategy_manifest()
    plan: List[dict] = []
    errors: List[str] = []
    target_symbols = symbols or []
    for strategy in strategies:
        requested_tf = timeframe_overrides.get(strategy)
        resolved_tf, error, _, _ = _resolve_timeframe_for_strategy(strategy, requested_tf, manifest)
        if error:
            errors.append(error)
            continue
        if target_symbols:
            for symbol in target_symbols:
                plan.append({
                    "strategy": strategy,
                    "symbol": symbol,
                    "timeframe": resolved_tf,
                })
        else:
            plan.append({
                "strategy": strategy,
                "symbol": "",
                "timeframe": resolved_tf,
            })
    return plan, errors


@api_bp.route('/trading/paper', methods=['POST'])
def api_trading_paper():
    payload = request.get_json(force=True) or {}
    strategies = _parse_strategy_list(payload)
    if not strategies:
        return jsonify({'error': 'strategy_required'}), 400
    symbols = _parse_symbol_list(payload)
    timeframe_overrides = _normalize_timeframe_mapping(strategies, payload.get('timeframes'))
    legacy_timeframe = payload.get('timeframe')
    if legacy_timeframe and strategies:
        if isinstance(legacy_timeframe, (list, tuple)):
            for idx, strategy in enumerate(strategies):
                if idx < len(legacy_timeframe):
                    value = legacy_timeframe[idx]
                    if isinstance(value, str) and value:
                        timeframe_overrides.setdefault(strategy, value)
        else:
            legacy_value = str(legacy_timeframe)
            if legacy_value:
                for strategy in strategies:
                    timeframe_overrides.setdefault(strategy, legacy_value)
    plan, errors = _build_trading_plan(strategies, symbols, timeframe_overrides)
    if errors:
        return jsonify({'error': 'invalid_timeframe', 'details': errors}), 400
    if not plan:
        return jsonify({'error': 'no_plan_generated'}), 400
    capital_pct = str(payload.get('capital_pct', '10'))
    schedule = tasks.schedule_trading_plan('paper', plan, capital_pct)
    response = {
        'status': 'queued',
        'launch_plan': plan,
        'capital_pct': capital_pct,
        'schedule': schedule,
    }
    return jsonify(response)


@api_bp.route('/trading/live', methods=['POST'])
def api_trading_live():
    payload = request.get_json(force=True) or {}
    strategies = _parse_strategy_list(payload)
    if not strategies:
        return jsonify({'error': 'strategy_required'}), 400
    symbols = _parse_symbol_list(payload)
    timeframe_overrides = _normalize_timeframe_mapping(strategies, payload.get('timeframes'))
    legacy_timeframe = payload.get('timeframe')
    if legacy_timeframe and strategies:
        if isinstance(legacy_timeframe, (list, tuple)):
            for idx, strategy in enumerate(strategies):
                if idx < len(legacy_timeframe):
                    value = legacy_timeframe[idx]
                    if isinstance(value, str) and value:
                        timeframe_overrides.setdefault(strategy, value)
        else:
            legacy_value = str(legacy_timeframe)
            if legacy_value:
                for strategy in strategies:
                    timeframe_overrides.setdefault(strategy, legacy_value)
    plan, errors = _build_trading_plan(strategies, symbols, timeframe_overrides)
    if errors:
        return jsonify({'error': 'invalid_timeframe', 'details': errors}), 400
    if not plan:
        return jsonify({'error': 'no_plan_generated'}), 400
    capital_pct = str(payload.get('capital_pct', '10'))
    schedule = tasks.schedule_trading_plan('live', plan, capital_pct)
    response = {
        'status': 'queued',
        'launch_plan': plan,
        'capital_pct': capital_pct,
        'schedule': schedule,
    }
    return jsonify(response)


@api_bp.route('/trading/stop', methods=['POST'])
def api_trading_stop():
    payload = request.get_json(silent=True) or {}
    mode = payload.get('mode')
    try:
        result = tasks.stop_trading(mode)
    except ValueError as exc:
        return jsonify({'error': 'invalid_mode', 'details': str(exc)}), 400
    if result.get('errors'):
        return jsonify({'status': 'partial', 'details': result}), 500
    return jsonify({'status': 'stopped', 'details': result})


@api_bp.route('/jobs/<job_id>/archive', methods=['POST'])
def api_archive_job(job_id: str):
    payload = request.get_json(silent=True) or {}
    archive_results = bool(payload.get('archive_results'))
    jobs = _read_job_registry()
    job_entry = jobs.get(job_id)
    if job_entry is None:
        return jsonify({'error': 'job_not_found'}), 404
    status = str(job_entry.get('status') or '').lower()
    if status not in {'completed', 'failed', 'stopped', 'cancelled'}:
        return jsonify({'error': 'job_not_finished', 'status': status}), 400

    removed_entry = _remove_job(job_id)
    normalized_entry = _normalize_job_entry(removed_entry or {})
    archived_files: List[str] = []
    if archive_results and removed_entry:
        matches = _match_backtest_results(removed_entry)
        if matches:
            _BACKTEST_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        for path in matches:
            if not path.exists():
                continue
            target = _BACKTEST_ARCHIVE_DIR / path.name
            if target.exists():
                timestamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
                target = _BACKTEST_ARCHIVE_DIR / f"{path.stem}-{timestamp}{path.suffix}"
            try:
                shutil.move(str(path), str(target))
                archived_files.append(str(target))
            except OSError as exc:  # pragma: no cover - filesystem errors
                current_app.logger.exception("Failed to archive result %s: %s", path, exc)
                continue
    return jsonify({
        'status': 'archived',
        'job_id': job_id,
        'job': normalized_entry,
        'archived_results': archived_files,
    })


@api_bp.route('/backtests', methods=['POST'])
def api_launch_backtest():
    payload = request.get_json(force=True) or {}
    raw_strategies = payload.get('strategies')
    if isinstance(raw_strategies, str):
        strategies = [raw_strategies]
    else:
        strategies = [str(item) for item in (raw_strategies or []) if item]
    if not strategies and payload.get('strategy'):
        strategies = [str(payload['strategy'])]

    raw_symbols = payload.get('symbols')
    if isinstance(raw_symbols, str):
        symbols = [raw_symbols]
    else:
        symbols = [str(item) for item in (raw_symbols or []) if item]
    if not symbols and payload.get('symbol') is not None:
        symbol_value = payload.get('symbol')
        if isinstance(symbol_value, str) and symbol_value:
            symbols = [symbol_value]
        elif symbol_value:
            symbols = [str(symbol_value)]

    raw_timeframes = payload.get('timeframes')
    timeframe_overrides: Dict[str, str] = {}
    if isinstance(raw_timeframes, dict):
        timeframe_overrides = {
            str(key): str(value)
            for key, value in raw_timeframes.items()
            if isinstance(key, str) and isinstance(value, str) and value
        }
    elif isinstance(raw_timeframes, (list, tuple)):
        for idx, value in enumerate(raw_timeframes):
            if idx < len(strategies) and isinstance(value, str) and value:
                timeframe_overrides[strategies[idx]] = value
    legacy_timeframe = payload.get('timeframe')
    if isinstance(legacy_timeframe, str):
        legacy_timeframe = legacy_timeframe.strip()
    else:
        legacy_timeframe = None

    manifest = get_strategy_manifest()

    capital_pct = str(payload.get('capital_pct', '10'))
    start_date = payload.get('start_date')
    end_date = payload.get('end_date')

    current_app.logger.info(
        "Backtest request received: strategies=%s symbols=%s overrides=%s legacy_timeframe=%s capital_pct=%s start=%s end=%s",
        strategies,
        symbols,
        timeframe_overrides,
        legacy_timeframe,
        capital_pct,
        start_date,
        end_date,
    )

    if not strategies:
        return jsonify({'error': 'strategies are required'}), 400

    job_ids: List[str] = []
    launch_plan: List[Dict[str, object]] = []
    warnings: List[str] = []
    errors: List[str] = []

    for strategy in strategies:
        manifest_entry = manifest.get(strategy, {})
        allowed_timeframes = [
            tf for tf in manifest_entry.get("timeframes", []) if isinstance(tf, str) and tf
        ]
        allowed_lookup = {tf.lower(): tf for tf in allowed_timeframes}
        default_tf = manifest_entry.get("default")
        preferred_default = None
        if isinstance(default_tf, str):
            preferred_default = allowed_lookup.get(default_tf.lower(), default_tf)
        if not preferred_default and allowed_timeframes:
            preferred_default = allowed_timeframes[0]

        requested_tf = timeframe_overrides.get(strategy) or legacy_timeframe
        requested_tf_clean = requested_tf.strip().lower() if isinstance(requested_tf, str) else ""

        resolved_timeframe = None
        if allowed_lookup:
            if requested_tf_clean and requested_tf_clean not in allowed_lookup:
                error = (
                    f"Strategy {strategy} only permits timeframes {allowed_timeframes}; "
                    f"received '{requested_tf}'."
                )
                errors.append(error)
                current_app.logger.error(error)
                continue
            if requested_tf_clean and requested_tf_clean in allowed_lookup:
                resolved_timeframe = allowed_lookup[requested_tf_clean]
            else:
                resolved_timeframe = preferred_default or next(iter(allowed_lookup.values()))
        else:
            if requested_tf_clean:
                resolved_timeframe = requested_tf if isinstance(requested_tf, str) else None
            elif preferred_default:
                resolved_timeframe = preferred_default

        if not resolved_timeframe:
            resolved_timeframe = legacy_timeframe or preferred_default or "1h"
            if not allowed_lookup:
                warning = (
                    f"Strategy {strategy}: manifest missing timeframe info; using fallback '{resolved_timeframe}'."
                )
                warnings.append(warning)
                current_app.logger.warning(
                    "No manifest timeframe for strategy %s; falling back to %s",
                    strategy,
                    resolved_timeframe,
                )

        plan_entry = {
            "strategy": strategy,
            "timeframe": resolved_timeframe,
            "symbols": list(symbols),
        }
        launch_plan.append(plan_entry)
        current_app.logger.info(
            "Backtest launch plan entry: strategy=%s timeframe=%s symbols=%s",
            strategy,
            resolved_timeframe,
            plan_entry["symbols"] or ["(default)"],
        )

    if errors:
        return jsonify({'error': 'invalid_timeframes', 'details': errors}), 400

    bundle_job_id: Optional[str] = None
    plan_summary: List[dict] = []
    plan_jobs_meta: List[dict] = []
    child_job_ids: List[str] = []
    if launch_plan:
        bundle_job_id, plan_summary = tasks.create_backtest_plan_job(
            plan=launch_plan,
            capital_pct=capital_pct,
            start_date=start_date,
            end_date=end_date,
        )
        if bundle_job_id:
            _update_job_registry(bundle_job_id, start_date=start_date, end_date=end_date)

    for plan_entry in launch_plan:
        strategy = str(plan_entry.get("strategy") or "")
        plan_symbols = plan_entry.get("symbols") or []
        timeframe = str(plan_entry.get("timeframe") or legacy_timeframe or "1h")
        hidden = bool(bundle_job_id)
        plan_meta = {
            "strategy": strategy,
            "symbols": plan_symbols,
            "timeframe": timeframe,
        }
        if len(plan_symbols) > 1:
            job_id = tasks.start_backtest_batch(
                strategy,
                plan_symbols,
                timeframe,
                capital_pct,
                parent_id=bundle_job_id,
                hidden=hidden,
            )
            if job_id:
                child_job_ids.append(job_id)
                plan_jobs_meta.append({"job_id": job_id, "plan_entry": plan_meta})
        else:
            symbol = plan_symbols[0] if plan_symbols else ''
            job_id = tasks.start_backtest(
                strategy,
                symbol,
                timeframe,
                capital_pct,
                parent_id=bundle_job_id,
                hidden=hidden,
            )
            if job_id:
                child_job_ids.append(job_id)
                plan_jobs_meta.append({"job_id": job_id, "plan_entry": plan_meta})
            else:
                current_app.logger.warning(
                    "Backtest launch failed for strategy=%s symbol=%s timeframe=%s",
                    strategy,
                    symbol,
                    timeframe,
                )
    if bundle_job_id and child_job_ids:
        _update_job_registry(bundle_job_id, child_jobs=child_job_ids)
        tasks.spawn_plan_runner(
            bundle_job_id,
            plan_jobs_meta,
            capital_pct,
            plan_summary,
            start_date,
            end_date,
        )
        job_ids = [bundle_job_id]
    else:
        job_ids = child_job_ids

    current_app.logger.info("Backtest jobs queued: %s", job_ids)
    response = {
        'status': 'queued',
        'job_ids': job_ids,
        'launch_plan': launch_plan,
    }
    if warnings:
        response['warnings'] = warnings
    return jsonify(response)

@main_bp.route('/api/backtests', methods=['GET'])
def main_proxy_backtests_get():
    """Legacy route compatibility for /api/backtests GET."""
    return api_list_backtests()


@main_bp.route('/api/backtests', methods=['POST'])
def main_proxy_backtests_post():
    """Legacy route compatibility for /api/backtests POST."""
    return api_launch_backtest()


@api_bp.route('/jobs', methods=['GET'])
def api_jobs():
    return jsonify({'jobs': _list_jobs()})


@api_bp.route('/configs', methods=['GET'])
def api_list_configs():
    configs_dir = Path(os.getcwd()) / 'configs'
    entries: List[dict] = []
    for path in sorted(configs_dir.glob('*.y*ml')):
        data = _load_config_dict(str(path)) or {}
        entries.append(
            {
                'name': path.name,
                'path': str(path),
                'warnings': _parameter_warnings(data, str(path)),
                'updated_at': datetime.utcfromtimestamp(path.stat().st_mtime).isoformat(),
            }
        )
    root_config = Path(os.getcwd()) / 'config.yaml'
    if root_config.exists():
        data = _load_config_dict(str(root_config)) or {}
        entries.insert(
            0,
            {
                'name': 'config.yaml',
                'path': str(root_config),
                'warnings': _parameter_warnings(data, str(root_config)),
                'updated_at': datetime.utcfromtimestamp(root_config.stat().st_mtime).isoformat(),
            },
        )
    return jsonify({'configs': entries})

@main_bp.route('/api/configs', methods=['GET'])
def main_proxy_configs():
    """Expose the configs listing for legacy clients without the blueprint prefix."""
    return api_list_configs()


@api_bp.route('/configs/<path:name>', methods=['GET'])
def api_get_config(name: str):
    base_dir = Path(os.getcwd())
    path = (base_dir / name).resolve()
    if not path.exists():
        path = (base_dir / 'configs' / name).resolve()
    if not path.exists():
        return jsonify({'error': 'config not found'}), 404
    try:
        content = path.read_text(encoding='utf-8')
    except OSError as exc:
        return jsonify({'error': str(exc)}), 500
    return jsonify({'name': name, 'path': str(path), 'content': content})


@api_bp.route('/configs/<path:name>', methods=['PUT'])
def api_update_config(name: str):
    payload = request.get_json(force=True) or {}
    content = payload.get('content')
    if content is None:
        return jsonify({'error': 'content is required'}), 400
    base_dir = Path(os.getcwd())
    path = (base_dir / name).resolve()
    if not path.exists():
        path = (base_dir / 'configs' / name).resolve()
    if not path.exists():
        return jsonify({'error': 'config not found'}), 404
    backup = path.with_suffix(path.suffix + '.bak')
    try:
        shutil.copy2(path, backup)
        path.write_text(content, encoding='utf-8')
    except OSError as exc:
        return jsonify({'error': f'failed to write config: {exc}'}), 500

    warnings = []
    if load_config and validate_app_config:
        try:
            cfg_obj = load_config(str(path))
            validate_app_config(cfg_obj, stage='ui')
        except GuardrailViolation as exc:
            warnings = list(getattr(exc, 'violations', [])) or [str(exc)]
        except Exception as exc:
            warnings = [str(exc)]
    return jsonify({'status': 'ok', 'warnings': warnings})


@api_bp.route('/configs/<path:name>/revert', methods=['POST'])
def api_revert_config(name: str):
    base_dir = Path(os.getcwd())
    path = (base_dir / name).resolve()
    if not path.exists():
        path = (base_dir / 'configs' / name).resolve()
    backup = path.with_suffix(path.suffix + '.bak')
    if not backup.exists():
        return jsonify({'error': 'backup not found'}), 404
    try:
        shutil.copy2(backup, path)
    except OSError as exc:
        return jsonify({'error': f'failed to restore backup: {exc}'}), 500
    return jsonify({'status': 'ok'})


@api_bp.route('/trades', methods=['GET'])
def api_trades():
    limit = request.args.get('limit', default=500, type=int)
    records = _read_trade_records(limit=limit)
    stats = _rolling_statistics(records, days=request.args.get('days', default=30, type=int))
    return jsonify({'records': records, 'stats': stats})


@api_bp.route('/trades/export', methods=['GET'])
def api_trades_export():
    records = _read_trade_records()

    def generate():
        header = [
            'timestamp', 'strategy', 'symbol', 'timeframe', 'status', 'pnl', 'pnl_pct', 'confidence', 'volatility',
            'position_size', 'notional_usd', 'notional_base', 'notional_aud', 'price_usd', 'usd_to_aud',
            'pnl_value_base', 'pnl_value_aud'
        ]
        yield ','.join(header) + '\n'
        for record in records:
            row = {key: record.get(key, '') for key in header}
            yield ','.join(str(row.get(col, '')) for col in header) + '\n'

    filename = f"trades-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.csv"
    return Response(generate(), mimetype='text/csv', headers={'Content-Disposition': f'attachment; filename={filename}'})


@api_bp.route('/trades/clear', methods=['POST'])
def api_trades_clear():
    result = tasks.clear_trade_history()
    trade_records = _read_trade_records()
    stats = _rolling_statistics(trade_records)
    return jsonify({'status': 'ok', 'result': result, 'records': trade_records, 'stats': stats})


@api_bp.route('/trades/metrics', methods=['GET'])
def api_trades_metrics():
    limit = request.args.get('limit', type=int) or 50
    payload = tasks.get_trade_metrics(limit=limit)
    return jsonify(payload)


@api_bp.route('/trades/metrics', methods=['POST'])
def api_trades_metrics_record():
    payload = request.get_json(force=True) or {}
    try:
        event = tasks.record_trade_decision(payload)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    limit = request.args.get('limit', type=int) or 50
    metrics = tasks.get_trade_metrics(limit=limit)
    metrics['last_event'] = event
    return jsonify(metrics)


@api_bp.route('/backtests/jobs/clear', methods=['POST'])
def api_clear_job_queue():
    result = tasks.clear_job_registry()
    jobs = _list_jobs()
    return jsonify({'status': 'ok', 'result': result, 'jobs': jobs})


@api_bp.route('/guardrails/clear', methods=['POST'])
def api_guardrails_clear():
    result = tasks.clear_guardrail_logs()
    guardrail_logs = _load_guardrail_logs()
    guardrail_snapshots = _load_guardrail_snapshots()
    return jsonify(
        {
            'status': 'ok',
            'result': result,
            'guardrail_logs': guardrail_logs,
            'guardrail_snapshots': guardrail_snapshots,
        }
    )


@api_bp.route('/learning/toggle', methods=['POST'])
def api_toggle_learning():
    payload = request.get_json(force=True) or {}
    state = _load_ui_state()
    if 'online_learning' in payload:
        state['online_learning'] = bool(payload['online_learning'])
    if 'periodic_retraining' in payload:
        state['periodic_retraining'] = bool(payload['periodic_retraining'])
    _save_ui_state(state)
    return jsonify({'status': 'ok', 'state': state})

@main_bp.route('/learning/toggle', methods=['POST'])
@main_bp.route('/api/learning/toggle', methods=['POST'])
def main_proxy_toggle_learning():
    """Allow both legacy and API-prefixed callers to toggle learning settings."""
    return api_toggle_learning()

@api_bp.route('/system/kill-switch', methods=['POST'])
def api_kill_switch():
    payload = request.get_json(force=True) or {}
    armed = bool(payload.get('armed'))
    state = _load_ui_state()
    state['kill_switch'] = armed
    _save_ui_state(state)
    if armed:
        tasks.stop_all()
    return jsonify({'status': 'ok', 'armed': armed})


@api_bp.route('/system/restart', methods=['POST'])
def api_system_restart():
    """Stop running jobs and reset orchestration state."""
    tasks.stop_all()
    cancelled = _cancel_inflight_jobs()
    timestamp = datetime.utcnow().isoformat()
    state = _load_ui_state()
    state.update(
        {
            'mode': 'idle',
            'kill_switch': False,
            'last_restart': timestamp,
        }
    )
    _save_ui_state(state)
    current_app.logger.info("Restart triggered; cancelled_jobs=%s", cancelled)
    return jsonify({'status': 'ok', 'timestamp': timestamp, 'cancelled_jobs': cancelled, 'state': state})


@api_bp.route('/system/kafka', methods=['GET'])
def api_kafka_status():
    return jsonify(tasks.kafka_stack_status())


@api_bp.route('/system/kafka', methods=['POST'])
def api_kafka_control():
    payload = request.get_json(force=True) or {}
    action = (payload.get('action') or '').lower()
    try:
        current_app.logger.info("Kafka control requested: action=%s", action or "status")
        if action in {"", "status"}:
            result = tasks.kafka_stack_status()
        elif action == "start":
            result = tasks.start_kafka_stack()
        elif action == "stop":
            result = tasks.stop_kafka_stack()
        elif action == "restart":
            stop_result = tasks.stop_kafka_stack()
            start_result = tasks.start_kafka_stack()
            success = bool(stop_result.get("success")) and bool(start_result.get("success"))
            result = {
                "success": success,
                "stop": stop_result,
                "start": start_result,
            }
        else:
            return jsonify({'success': False, 'error': f"Unknown action '{action}'."}), 400
    except Exception as exc:  # pragma: no cover - defensive logging
        current_app.logger.exception("Kafka control failed for action %s", action)
        return jsonify({'success': False, 'error': str(exc)}), 500
    current_app.logger.info("Kafka action response: %s", result)
    return jsonify(result)

@main_bp.route('/api/system/kafka', methods=['GET'])
def main_proxy_kafka_status():
    """Allow UI calls to /api/system/kafka even if api blueprint is unreachable."""
    return api_kafka_status()


@main_bp.route('/api/system/kafka', methods=['POST'])
def main_proxy_kafka_control():
    return api_kafka_control()


@api_bp.route('/system/alerts', methods=['GET'])
def api_system_alerts():
    rows = _strategy_rows()
    trade_records = _read_trade_records()
    portfolio_stats = _rolling_statistics(trade_records)
    kafka_status = tasks.kafka_stack_status()
    alerts = _build_alerts(rows, _load_ui_state(), portfolio_stats, kafka_status)
    return jsonify({'alerts': alerts, 'kafka': kafka_status})


@api_bp.route('/system/state', methods=['GET'])
def api_system_state():
    rows = _strategy_rows()
    trade_records = _read_trade_records()
    portfolio_stats = _rolling_statistics(trade_records)
    kafka_status = tasks.kafka_stack_status()
    alerts = _build_alerts(rows, _load_ui_state(), portfolio_stats, kafka_status)
    return jsonify(
        {
            'state': _load_ui_state(),
            'portfolio': portfolio_stats,
            'alerts': alerts,
            'strategies': rows,
            'jobs': _list_jobs(),
            'kafka': kafka_status,
        }
    )


@api_bp.route('/ui/diagnostic', methods=['POST'])
def api_ui_diagnostic():
    payload = request.get_json(silent=True) or {}
    kind = payload.get('kind', 'diagnostic')
    message = payload.get('message', '')
    stack = payload.get('stack')
    current_app.logger.error("UI %s: %s %s", kind, message, stack or '')
    return jsonify({'status': 'ok'})

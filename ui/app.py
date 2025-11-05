from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

from ai_trading_bot.config import load_config, save_config
from ai_trading_bot.pipeline import prepare_dataset, train as pipeline_train, backtest as pipeline_backtest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
LOG_ROOT = PROJECT_ROOT / "logs"
RESULTS_ROOT = PROJECT_ROOT / "results"
MODELS_ROOT = PROJECT_ROOT / "models"
HISTORY_ROOT = RESULTS_ROOT / "history"
TRAIN_HISTORY_DIR = MODELS_ROOT / "history"
BACKTEST_HISTORY_DIR = HISTORY_ROOT / "backtests"
LATEST_MODEL_METADATA = MODELS_ROOT / "latest_metadata.json"
LATEST_BACKTEST_FILE = RESULTS_ROOT / "backtest_results.json"
STATE_FILE = LOG_ROOT / "runbot-state.json"


################################################################################
# Styling helpers
################################################################################


def apply_theme() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg-0: #0f1115;
                --bg-1: #1a1d23;
                --bg-2: #22252c;
                --bg-3: #2a2d35;
                --accent-blue: #0066ff;
                --accent-green: #00cc88;
                --accent-amber: #ff9500;
                --accent-red: #ff3366;
                --text-primary: #e5e7eb;
                --text-secondary: #9ca3af;
                --text-muted: #6b7280;
            }
            body {
                background-color: var(--bg-0);
            }
            .stApp {
                background: linear-gradient(180deg, #10131a 0%, #0b0d11 100%);
                color: var(--text-primary);
                font-family: "Inter", "Segoe UI", sans-serif;
            }
            .block-container {
                padding-top: 1.5rem;
            }
            .main-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 1rem;
            }
            .card {
                background: var(--bg-1);
                border: 1px solid #2a2d35;
                border-radius: 16px;
                padding: 20px;
                margin-bottom: 1rem;
            }
            .card:hover {
                border-color: rgba(0, 102, 255, 0.45);
            }
            .card-title {
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 1.05rem;
                font-weight: 600;
                margin-bottom: 1rem;
            }
            .status-dot {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                background: var(--bg-2);
                border-radius: 999px;
                padding: 4px 10px;
                font-size: 0.85rem;
            }
            .status-dot span {
                width: 10px;
                height: 10px;
                border-radius: 50%;
                display: inline-block;
            }
            .kpi-card {
                background: var(--bg-1);
                border: 1px solid #2a2d35;
                border-radius: 12px;
                padding: 18px;
                margin-bottom: 1rem;
            }
            .kpi-title {
                font-size: 0.85rem;
                color: var(--text-secondary);
                margin-bottom: 6px;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }
            .kpi-value {
                font-size: 1.85rem;
                font-weight: 600;
                margin: 0;
            }
            .log-output {
                font-family: "JetBrains Mono", "Fira Code", monospace;
                font-size: 13px;
                background: var(--bg-2);
                border-radius: 12px;
                border: 1px solid rgba(255,255,255,0.05);
                padding: 16px;
                height: 400px;
                overflow-y: auto;
                white-space: pre-wrap;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


################################################################################
# Utility
################################################################################


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except json.JSONDecodeError:
        return {}


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def format_timestamp(value: str | None) -> str:
    if not value:
        return "--"
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except ValueError:
        return value


def append_action_log(title: str, status: str, message: str) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "title": title,
        "status": status,
        "message": message,
    }
    st.session_state.setdefault("action_logs", [])
    st.session_state["action_logs"].insert(0, entry)
    st.session_state["action_logs"] = st.session_state["action_logs"][:8]


def render_status_dot(label: str, color: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="status-dot" title="{text}">
            <span style="background:{color};"></span>{label}
        </div>
        """,
        unsafe_allow_html=True,
    )


def fetch_runbot_state() -> Dict[str, Any]:
    return load_json(STATE_FILE)


def runbot_status_dot(name: str, processes: List[Dict[str, Any]]) -> Dict[str, str]:
    entry = next((p for p in processes if p.get("Name") == name), None)
    if not entry:
        return {"label": "Stopped", "color": "var(--accent-red)", "tooltip": "Process not running"}
    pid = entry.get("Id")
    status = "Running" if pid else "Unknown"
    color = "var(--accent-green)" if pid else "var(--accent-amber)"
    tooltip = f"PID {pid}" if pid else "Status unknown"
    return {"label": status, "color": color, "tooltip": tooltip}


def get_last_etl_info() -> Dict[str, Any] | None:
    info = st.session_state.get("last_etl_info")
    if info:
        return info
    return None


def get_last_training_metrics() -> Dict[str, Any]:
    metrics = st.session_state.get("last_training_metrics")
    if metrics:
        return metrics
    return load_json(LATEST_MODEL_METADATA)


def get_last_backtest_payload() -> Dict[str, Any]:
    payload = st.session_state.get("last_backtest_payload")
    if payload:
        return payload
    return load_json(LATEST_BACKTEST_FILE)


def get_last_backtest_equity() -> pd.DataFrame | None:
    equity = st.session_state.get("last_backtest_equity")
    if equity is not None:
        return equity
    return None


def get_last_qc_summary() -> Dict[str, Any]:
    summary = st.session_state.get("last_qc_summary")
    if summary:
        return summary
    return load_json(RESULTS_ROOT / "qc_summary.json")


def run_qc_checks() -> Dict[str, Any]:
    bronze = PROJECT_ROOT / "data" / "bronze"
    silver = PROJECT_ROOT / "data" / "silver"
    qc_result_path = RESULTS_ROOT / "qc_summary.json"
    if not bronze.exists() or not silver.exists():
        message = "Bronze and silver folders not found. Run a data update first."
        append_action_log("Quality Check", "warning", message)
        raise FileNotFoundError(message)
    command = [
        sys.executable,
        "-m",
        "jobs.qc_check",
        "--bronze-root",
        str(bronze),
        "--silver-root",
        str(silver),
        "--output-json",
        str(qc_result_path),
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        error = completed.stderr.strip() or completed.stdout.strip() or "Quality check failed."
        append_action_log("Quality Check", "error", error)
        raise RuntimeError(error)
    summary = load_json(qc_result_path)
    st.session_state["last_qc_summary"] = summary
    append_action_log("Quality Check", "success", "Quality check finished successfully.")
    return summary


def list_training_runs() -> List[Tuple[str, Dict[str, Any]]]:
    runs: list[Tuple[str, Dict[str, Any]]] = []
    latest = load_json(LATEST_MODEL_METADATA)
    if latest:
        runs.append(("Latest", latest))
    if TRAIN_HISTORY_DIR.exists():
        for file in sorted(TRAIN_HISTORY_DIR.glob("metadata_*.json"), reverse=True):
            data = load_json(file)
            if not data:
                continue
            label = data.get("trained_at", file.stem.replace("metadata_", ""))
            runs.append((label, data))
    seen: set[str] = set()
    unique: list[Tuple[str, Dict[str, Any]]] = []
    for label, data in runs:
        key = data.get("trained_at", label)
        if key in seen:
            continue
        seen.add(key)
        unique.append((label, data))
    return unique


def list_backtest_runs() -> List[Tuple[str, Dict[str, Any]]]:
    runs: list[Tuple[str, Dict[str, Any]]] = []
    latest = load_json(LATEST_BACKTEST_FILE)
    if latest:
        runs.append(("Latest", latest))
    if BACKTEST_HISTORY_DIR.exists():
        for file in sorted(BACKTEST_HISTORY_DIR.glob("backtest_*.json"), reverse=True):
            data = load_json(file)
            if not data:
                continue
            label = data.get("generated_at", file.stem.replace("backtest_", ""))
            runs.append((label, data))
    seen: set[str] = set()
    unique: list[Tuple[str, Dict[str, Any]]] = []
    for label, data in runs:
        key = data.get("generated_at", label)
        if key in seen:
            continue
        seen.add(key)
        unique.append((label, data))
    return unique


def run_etl(force_download: bool = True) -> Dict[str, Any]:
    config = load_config(CONFIG_PATH)
    price_frame, engineered = prepare_dataset(config, force_download=force_download)
    start_ts = pd.Timestamp(price_frame.index.min())
    end_ts = pd.Timestamp(price_frame.index.max())
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    start_text = start_ts.strftime("%Y-%m-%d %H:%M UTC")
    end_text = end_ts.strftime("%Y-%m-%d %H:%M UTC")
    info = {
        "symbol": config.data.symbol,
        "interval": config.data.interval,
        "rows": int(engineered.shape[0]),
        "start": start_text,
        "end": end_text,
    }
    st.session_state["last_etl_info"] = info
    st.session_state["last_price_frame"] = price_frame
    st.session_state["last_engineered_frame"] = engineered
    message = (
        f"Prepared {info['rows']} rows for {info['symbol']} ({info['interval']}). "
        f"Data spans {info['start']} to {info['end']}."
    )
    append_action_log("Run ETL", "success", message)
    st.success(f"Data update complete. {message}")
    return info


def run_training(force_download: bool = False) -> Dict[str, Any]:
    metrics = pipeline_train(CONFIG_PATH, force_download=force_download)
    st.session_state["last_training_metrics"] = metrics
    append_action_log(
        "Train Model",
        "success",
        f"Accuracy {metrics.get('accuracy', 0):.2%} | Mode {metrics.get('mode', '--')}"
    )
    st.session_state["selected_training_option"] = None
    return metrics


def save_backtest_summary(payload: Dict[str, Any]) -> None:
    payload.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
    save_json(LATEST_BACKTEST_FILE, payload)
    BACKTEST_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    history_path = BACKTEST_HISTORY_DIR / f"backtest_{payload['generated_at'].replace(':', '-')}.json"
    save_json(history_path, payload)


def run_backtest(
    force_download: bool = False,
    long_threshold: float | None = None,
    short_threshold: float | None = None,
) -> Dict[str, Any]:
    strategy_output, result, metadata = pipeline_backtest(
        CONFIG_PATH,
        force_download=force_download,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
    )
    payload = {
        "mode": metadata.get("interval", config.data.interval if (config := load_config(CONFIG_PATH)) else "--"),
        "mode_score": getattr(strategy_output, "meta_score", None),
        "mode_metrics": getattr(strategy_output, "meta_metrics", {}),
        "summary": result.summary,
    }
    save_backtest_summary(payload)
    sharpe = payload["summary"].get("sharpe_ratio", 0.0)
    append_action_log("Backtest", "success", f"Sharpe {sharpe:.2f}")
    st.session_state["last_backtest_payload"] = payload
    st.session_state["last_backtest_equity"] = result.equity_curve.to_frame("Equity")
    st.session_state["last_backtest_performance"] = result.performance
    if hasattr(strategy_output, "decisions"):
        st.session_state["last_meta_decisions"] = getattr(strategy_output, "decisions")
    st.session_state["selected_backtest_option"] = None
    return payload


def render_action_log_card() -> None:
    logs = st.session_state.get("action_logs", [])
    if not logs:
        return
    with st.expander("Recent activity", expanded=False):
        for entry in logs:
            st.markdown(f"**{entry['timestamp']}** - {entry['title']} ({entry['status']})")
            st.code(entry["message"], language="text")


def render_control_center() -> None:
    st.subheader("Control Center")
    render_health_cards()
    render_quick_wizard()
    render_scenario_sandbox()
    render_action_log_card()


def render_health_cards() -> None:
    st.markdown("#### At-a-Glance Health")
    state = fetch_runbot_state()
    processes = state.get("Processes") or state.get("processes") or []
    col_services, col_data, col_model, col_trading = st.columns(4)

    service_items: List[str] = []
    for internal_name, display_name in {
        "prefect-server": "Prefect Server",
        "prefect-worker": "Prefect Worker",
        "gui-streamlit": "GUI Process",
    }.items():
        meta = runbot_status_dot(internal_name, processes)
        service_items.append(
            f"<li><strong>{display_name}</strong>: {meta['label']} ({meta['tooltip']})</li>"
        )
    services_html = "".join(service_items) or "<li>No service information yet.</li>"
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">System Services</div>
            <ul>{services_html}</ul>
            <p style="color:var(--text-muted); margin-top:0.5rem;">All services should show Running before live trading.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    etl_info = get_last_etl_info()
    if etl_info:
        data_body = (
            f"<p>Last update prepared <strong>{etl_info['rows']:,}</strong> rows for {etl_info['symbol']} "
            f"({etl_info['interval']}).</p>"
            f"<p>Data window: {etl_info['start']} to {etl_info['end']} (UTC).</p>"
        )
    else:
        data_body = '<p>No data refresh recorded yet. Use the "Fresh Data Update" scenario below.</p>'
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">Data Freshness</div>
            {data_body}
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = get_last_training_metrics()
    if metrics:
        accuracy = metrics.get("accuracy")
        precision = metrics.get("precision")
        recall = metrics.get("recall")
        accuracy_text = f"{accuracy:.2%}" if isinstance(accuracy, (int, float)) else "--"
        precision_text = f"{precision:.2%}" if isinstance(precision, (int, float)) else "--"
        recall_text = f"{recall:.2%}" if isinstance(recall, (int, float)) else "--"
        trained_at = format_timestamp(metrics.get("trained_at"))
        model_body = (
            f"<p>Model type: {metrics.get('model', 'Unknown')}.</p>"
            f"<p>Accuracy: {accuracy_text} | Precision: {precision_text} | Recall: {recall_text}.</p>"
            f"<p>Most recent training run: {trained_at}.</p>"
        )
    else:
        model_body = "<p>No training run found. Run the model step in the wizard to create one.</p>"
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">Model Readiness</div>
            {model_body}
        </div>
        """,
        unsafe_allow_html=True,
    )

    backtest_payload = get_last_backtest_payload()
    summary = backtest_payload.get("summary", {}) if backtest_payload else {}
    sharpe = summary.get("sharpe_ratio")
    total_return = summary.get("total_return")
    sharpe_text = f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else "--"
    total_return_text = f"{total_return:.2%}" if isinstance(total_return, (int, float)) else "--"
    live_status = st.session_state.get("live_agent_status", "Not started")
    if summary:
        trading_body = (
            f"<p>Sharpe (risk-adjusted reward): {sharpe_text}.</p>"
            f"<p>Total return: {total_return_text}.</p>"
            f"<p>Live agent status: {live_status}.</p>"
        )
    else:
        trading_body = (
            f"<p>No backtest run yet. Run the backtest step first.</p>"
            f"<p>Live agent status: {live_status}.</p>"
        )
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">Trading Readiness</div>
            {trading_body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_quick_wizard() -> None:
    st.markdown("#### Guided Quick Start")
    st.caption("Work through each step in order. Every button explains what it does and stores the result for the dashboard.")
    steps = [
        {
            "title": "Step 1 - Download fresh BitMEX data",
            "description": "Fetch the latest candles and funding rates, then rebuild the clean dataset.",
            "button_label": "Run data update",
            "spinner": "Downloading and preparing data...",
            "success": "Data refresh complete.",
            "error": "Data refresh failed:",
            "key": "wizard_data",
            "action": lambda: run_etl(force_download=True),
        },
        {
            "title": "Step 2 - Run Quality Check (QC)",
            "description": "Scan the prepared data for gaps or anomalies and log a Quality Check (QC) report.",
            "button_label": "Run quality check",
            "spinner": "Running quality checks...",
            "success": "Quality check finished.",
            "error": "Quality check failed:",
            "key": "wizard_qc",
            "action": run_qc_checks,
        },
        {
            "title": "Step 3 - Train or refresh the model",
            "description": "Train the machine learning model with the latest engineered features.",
            "button_label": "Train model",
            "spinner": "Training model...",
            "success": "Model training complete.",
            "error": "Model training failed:",
            "key": "wizard_train",
            "action": lambda: run_training(force_download=False),
        },
        {
            "title": "Step 4 - Backtest with current settings",
            "description": "Evaluate recent performance using the configured thresholds and cost model.",
            "button_label": "Run backtest",
            "spinner": "Running backtest...",
            "success": "Backtest complete.",
            "error": "Backtest failed:",
            "key": "wizard_backtest",
            "action": run_backtest,
        },
        {
            "title": "Step 5 - Start or stop the live agent",
            "description": "Launch or stop live trading after the earlier steps succeed.",
            "button_label": None,
            "key": "wizard_live",
            "action": None,
        },
    ]

    st.session_state.setdefault("wizard_step", 0)
    if len(st.session_state.get("wizard_progress", [])) != len(steps):
        st.session_state["wizard_progress"] = [False] * len(steps)
    st.session_state.setdefault("wizard_results", {})

    progress = st.session_state["wizard_progress"]
    current_step = max(0, min(st.session_state["wizard_step"], len(steps) - 1))
    st.session_state["wizard_step"] = current_step
    step = steps[current_step]

    status_labels: List[str] = []
    for idx, step_info in enumerate(steps):
        if progress[idx]:
            marker = "[done]"
        elif idx == current_step:
            marker = "[in progress]"
        else:
            marker = "[pending]"
        status_labels.append(f"{marker} {step_info['title']}")
    st.markdown("\n".join(status_labels))

    st.markdown(f"**{step['title']}**")
    st.write(step["description"])

    if step.get("action"):
        if st.button(step["button_label"] or "Run step", key=f"wizard_run_{current_step}"):
            try:
                with st.spinner(step.get("spinner", "Working...")):
                    result = step["action"]()
                st.session_state["wizard_results"][step["key"]] = result
                progress[current_step] = True
                st.success(step.get("success", "Step completed."))
            except Exception as exc:
                progress[current_step] = False
                st.error(f"{step.get('error', 'Step failed:')} {exc}")
    else:
        render_live_agent_controls()
        progress[current_step] = st.session_state.get("live_agent_status") in {"running", "stopped"}

    result = st.session_state["wizard_results"].get(step["key"])
    if result and step["key"] == "wizard_data":
        st.caption("Latest data update stored. See the chart in the scenario section for a preview.")
    elif result and step["key"] == "wizard_qc":
        total_warnings = len(result.get("trade_warnings", [])) + len(result.get("snapshot_warnings", []))
        st.caption(f"Quality Check Warnings: {total_warnings}.")
    elif result and step["key"] == "wizard_train":
        st.caption(f"Model accuracy: {result.get('accuracy', 0):.2%}.")
    elif result and step["key"] == "wizard_backtest":
        sharpe = result.get("summary", {}).get("sharpe_ratio")
        st.caption(f"Backtest Sharpe: {sharpe:.2f}" if sharpe is not None else "Backtest summary saved.")

    prev_col, next_col = st.columns(2)
    if prev_col.button("Previous step", disabled=current_step == 0):
        st.session_state["wizard_step"] = current_step - 1
        st.experimental_rerun()
    can_advance = progress[current_step] or current_step == len(steps) - 1
    if next_col.button(
        "Next step" if current_step < len(steps) - 1 else "Stay here",
        disabled=not can_advance or current_step == len(steps) - 1,
    ):
        st.session_state["wizard_step"] = min(current_step + 1, len(steps) - 1)
        st.experimental_rerun()


def render_live_agent_controls() -> None:
    st.info(
        "To start the live agent, open a PowerShell 7 window and run:\n\n"
        "`prefect worker start --pool \"bitmex\"`\n\n"
        "Then, in another window, run:\n\n"
        "`python -m deployment.live_agent_runner --config conf/live_agent.yaml --api-client live.execution.bitmex:BitmexClient`\n\n"
        "Use the buttons below to record the agent status for the dashboard."
    )
    st.session_state.setdefault("live_agent_status", "Not started")
    col_start, col_stop = st.columns(2)
    if col_start.button("Mark live agent as running"):
        st.session_state["live_agent_status"] = "running"
        append_action_log("Live Agent", "success", "Live agent marked as running.")
        st.success("Status updated to running.")
    if col_stop.button("Mark live agent as stopped"):
        st.session_state["live_agent_status"] = "stopped"
        append_action_log("Live Agent", "warning", "Live agent marked as stopped.")
        st.info("Status updated to stopped.")
    st.caption(f"Recorded status: {st.session_state['live_agent_status']}")


def render_scenario_sandbox() -> None:
    st.markdown("#### Ready-Made Scenarios")
    st.caption("Pick a card to run a pre-configured workflow. Results and quick charts display in the card.")
    st.session_state.setdefault("scenario_results", {})

    scenarios = [
        {
            "title": "Fresh Data Update",
            "description": "Downloads new BitMEX bars and funding data, then rebuilds the dataset.",
            "button": "Run data update",
            "spinner": "Refreshing data...",
            "success": "Data update finished.",
            "error": "Unable to refresh data:",
            "key": "scenario_data",
            "action": lambda: {
                "info": run_etl(force_download=True),
                "price": st.session_state.get("last_price_frame"),
            },
            "renderer": render_scenario_data,
        },
        {
            "title": "Train and Test Model",
            "description": "Retrains the model with the latest features and shows key accuracy metrics.",
            "button": "Train model",
            "spinner": "Training model...",
            "success": "Model training complete.",
            "error": "Model training failed:",
            "key": "scenario_train",
            "action": lambda: {"metrics": run_training(force_download=False)},
            "renderer": render_scenario_training,
        },
        {
            "title": "Risk and Quality Check",
            "description": "Runs the Quality Check (QC) pipeline to spot data anomalies.",
            "button": "Run quality check",
            "spinner": "Running quality check...",
            "success": "Quality check complete.",
            "error": "Quality check failed:",
            "key": "scenario_qc",
            "action": lambda: {"summary": run_qc_checks()},
            "renderer": render_scenario_qc,
        },
        {
            "title": "Backtest with Safer Settings",
            "description": "Backtests using tighter entry/exit thresholds to emphasise capital preservation.",
            "button": "Run safer backtest",
            "spinner": "Running backtest...",
            "success": "Backtest complete.",
            "error": "Backtest failed:",
            "key": "scenario_backtest",
            "action": lambda: {
                "payload": run_backtest(long_threshold=0.65, short_threshold=0.35),
                "equity": get_last_backtest_equity(),
            },
            "renderer": render_scenario_backtest,
        },
    ]

    for i in range(0, len(scenarios), 2):
        columns = st.columns(2)
        for column, scenario in zip(columns, scenarios[i:i + 2]):
            with column:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f'<div class="card-title">{scenario["title"]}</div>', unsafe_allow_html=True)
                st.write(scenario["description"])
                if st.button(scenario["button"], key=f"{scenario['key']}_button"):
                    try:
                        with st.spinner(scenario["spinner"]):
                            result = scenario["action"]()
                        st.session_state["scenario_results"][scenario["key"]] = result
                        st.success(scenario["success"])
                    except Exception as exc:
                        st.error(f"{scenario['error']} {exc}")
                result = st.session_state["scenario_results"].get(scenario["key"])
                if result:
                    scenario["renderer"](result)
                st.markdown('</div>', unsafe_allow_html=True)


def render_scenario_data(result: Dict[str, Any]) -> None:
    info = result.get("info", {})
    price_frame = result.get("price")
    if info:
        st.caption(
            f"Rows: {info.get('rows', 0):,} | Window: {info.get('start')} to {info.get('end')} (UTC)."
        )
    if isinstance(price_frame, pd.DataFrame) and "Close" in price_frame.columns:
        preview = price_frame[["Close"]].tail(200)
        st.line_chart(preview, height=140)


def render_scenario_training(result: Dict[str, Any]) -> None:
    metrics = result.get("metrics", {})
    if not metrics:
        return
    cols = st.columns(3)
    cols[0].metric("Accuracy (correct predictions)", f"{metrics.get('accuracy', 0):.2%}")
    cols[1].metric("Precision (positive quality)", f"{metrics.get('precision', 0):.2%}")
    cols[2].metric("Recall (coverage)", f"{metrics.get('recall', 0):.2%}")
    st.caption(f"Latest training run: {format_timestamp(metrics.get('trained_at'))}")


def render_scenario_qc(result: Dict[str, Any]) -> None:
    summary = result.get("summary", {})
    trade_warnings = len(summary.get("trade_warnings", []))
    snapshot_warnings = len(summary.get("snapshot_warnings", []))
    data = pd.DataFrame(
        {"Warning Type": ["Trade alerts", "Snapshot alerts"], "Count": [trade_warnings, snapshot_warnings]}
    ).set_index("Warning Type")
    st.bar_chart(data, height=140)
    total = trade_warnings + snapshot_warnings
    if total == 0:
        st.caption("No data issues detected.")
    else:
        st.caption(f"{total} potential issues need review.")


def render_scenario_backtest(result: Dict[str, Any]) -> None:
    payload = result.get("payload", {})
    equity = result.get("equity")
    summary = payload.get("summary", {})
    cols = st.columns(2)
    sharpe = summary.get("sharpe_ratio")
    total_return = summary.get("total_return")
    cols[0].metric("Sharpe (risk-adjusted reward)", f"{sharpe:.2f}" if sharpe is not None else "--")
    cols[1].metric("Total return", f"{total_return:.2%}" if total_return is not None else "--")
    if isinstance(equity, pd.DataFrame):
        st.line_chart(equity.tail(200), height=140)


def render_pipeline_stage(stage: Dict[str, str]) -> None:
    color_map = {
        "success": "var(--accent-green)",
        "running": "var(--accent-blue)",
        "warning": "var(--accent-amber)",
        "failed": "var(--accent-red)",
        "pending": "var(--text-muted)",
    }
    color = color_map.get(stage.get("status", "pending"), "var(--text-muted)")
    label_map = {
        "success": "OK",
        "running": "RUN",
        "warning": "WARN",
        "failed": "ERR",
        "pending": "...",
    }
    label = label_map.get(stage.get("status", "pending"), "...")
    st.markdown(
        f"""
        <div class="card" style="text-align:center; border-left: 4px solid {color};">
            <div class="card-title" style="justify-content:center; gap:0.5rem;">
                <span style="color:{color}; font-size:1rem;">{label}</span>
                {stage.get('name', 'Stage')}
            </div>
            <p style="margin:0; color:var(--text-secondary);">{stage.get('detail', '')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipelines() -> None:
    st.subheader("Data and Training Jobs")
    raw_count = len(list((PROJECT_ROOT / "data" / "raw").rglob("*.csv")))
    bronze_count = len(list((PROJECT_ROOT / "data" / "bronze").rglob("*.parquet")))
    qc_summary = load_json(RESULTS_ROOT / "qc_summary.json")

    stages = [
        {
            "name": "Download raw data",
            "status": "success" if raw_count else "pending",
            "detail": f"Cached CSV files: {raw_count}",
        },
        {
            "name": "Prepare clean data",
            "status": "success" if bronze_count else "pending",
            "detail": f"Clean Parquet files: {bronze_count}",
        },
    ]
    if qc_summary:
        warn_count = len(qc_summary.get("trade_warnings", [])) + len(qc_summary.get("snapshot_warnings", []))
        stages.append(
            {
                "name": "Quality Check (QC)",
                "status": "warning" if warn_count else "success",
                "detail": f"Warnings to review: {warn_count}",
            }
        )
    else:
        stages.append(
            {
                "name": "Quality Check (QC)",
                "status": "pending",
                "detail": "No QC summary found yet. Run the check to generate one.",
            }
        )

    stage_cols = st.columns(len(stages))
    for col, stage in zip(stage_cols, stages):
        with col:
            render_pipeline_stage(stage)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Quality Check (QC) Summary</div>', unsafe_allow_html=True)
    if qc_summary:
        st.json(qc_summary)
    else:
        st.info("Run the quality check to create qc_summary.json.")
    st.markdown('</div>', unsafe_allow_html=True)


def build_option_map(runs: List[Tuple[str, Dict[str, Any]]]) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    options: List[str] = []
    mapping: Dict[str, Dict[str, Any]] = {}
    for idx, (label, data) in enumerate(runs):
        timestamp = data.get("trained_at") or data.get("generated_at") or label
        option = f"{idx + 1}. {timestamp}"
        options.append(option)
        mapping[option] = data
    return options, mapping


def render_model_results() -> None:
    st.subheader("Model Results and Backtests")

    training_runs = list_training_runs()
    training_options, training_map = build_option_map(training_runs)
    if training_options:
        default_training = st.session_state.get("selected_training_option")
        if default_training not in training_options:
            default_training = training_options[0]
        selected_training = st.selectbox(
            "Training runs",
            training_options,
            index=training_options.index(default_training),
            key="training_run_select",
        )
        st.session_state["selected_training_option"] = selected_training
        training_data = training_map[selected_training]
    else:
        training_data = {}

    backtest_runs = list_backtest_runs()
    backtest_options, backtest_map = build_option_map(backtest_runs)
    if backtest_options:
        default_backtest = st.session_state.get("selected_backtest_option")
        if default_backtest not in backtest_options:
            default_backtest = backtest_options[0]
        selected_backtest = st.selectbox(
            "Backtest history",
            backtest_options,
            index=backtest_options.index(default_backtest),
            key="backtest_run_select",
        )
        st.session_state["selected_backtest_option"] = selected_backtest
        backtest_data = backtest_map[selected_backtest]
    else:
        backtest_data = {}

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Most Recent Training Run</div>', unsafe_allow_html=True)
        if training_data:
            st.metric("Model type", training_data.get("model", "Unknown"))
            st.metric("Accuracy (correct predictions)", f"{training_data.get('accuracy', 0):.2%}")
            st.metric("Precision (positive quality)", f"{training_data.get('precision', 0):.2%}")
            st.metric("Recall (coverage)", f"{training_data.get('recall', 0):.2%}")
            st.caption(f"Trained on: {format_timestamp(training_data.get('trained_at'))}")
            st.markdown("**Input features used**")
            st.write(", ".join(training_data.get("features", [])) or "--")
        else:
            st.info("Train the model to populate this section.")
        if st.button("Train model again", key="retrain_now"):
            with st.spinner("Retraining model..."):
                run_training()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Mode Selection Summary</div>', unsafe_allow_html=True)
        if training_data:
            st.metric("Active mode", training_data.get("mode", "--"))
            mode_score = training_data.get("mode_score")
            st.metric("Mode score", f"{mode_score:.3f}" if mode_score is not None else "--")
        else:
            st.info("Train the model to evaluate mode selection.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Backtest Summary (explained numbers)</div>', unsafe_allow_html=True)
    if backtest_data:
        summary = backtest_data.get("summary", {})
        metric_cols = st.columns(4)
        labels = [
            ("Sharpe ratio (reward vs risk)", summary.get("sharpe_ratio")),
            ("Total return", summary.get("total_return")),
            ("Maximum drawdown", summary.get("max_drawdown")),
            ("Win rate", summary.get("win_rate")),
        ]
        for col, (label, value) in zip(metric_cols, labels):
            with col:
                st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="kpi-title">{label}</div>', unsafe_allow_html=True)
                if value is None:
                    display_value = "--"
                elif "Sharpe" in label:
                    display_value = f"{value:.2f}"
                else:
                    display_value = f"{value:.2%}"
                st.markdown(f'<div class="kpi-value">{display_value}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        st.caption(f"Generated: {format_timestamp(backtest_data.get('generated_at'))}")
    else:
        st.info("Run a backtest to populate metrics.")
    if st.button("Run Backtest", key="run_backtest_cta"):
        with st.spinner("Running backtest..."):
            run_backtest()
    st.markdown('</div>', unsafe_allow_html=True)


def render_live_trading() -> None:
    st.subheader("Live Trading Controls")
    st.caption("Use this page to review current live status and see any open trades captured to disk.")

    status = st.session_state.get("live_agent_status", "Not started")
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">Recorded live agent status</div>
            <p>The dashboard currently records the live agent as: <strong>{status}</strong>.</p>
            <p>If this does not match reality, return to the Control Center wizard and update the status there.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">How to start or stop live trading</div>', unsafe_allow_html=True)
    st.markdown(
        """
        1. Start a Prefect worker so deployments can hand out tasks:
           ```
           prefect worker start --pool "bitmex"
           ```
        2. Launch the live agent (replace the API client path if you have a custom client):
           ```
           python -m deployment.live_agent_runner --config conf/live_agent.yaml --api-client live.execution.bitmex:BitmexClient
           ```
        3. When you shut the agent down, mark it as stopped in the Control Center so the health cards stay accurate.
        """.strip()
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Open trades on disk</div>', unsafe_allow_html=True)
    trades_path = RESULTS_ROOT / "open_trades.json"
    trades = load_json(trades_path)
    if trades:
        rows = trades if isinstance(trades, list) else [trades]
        st.caption("Data sourced from results/open_trades.json.")
        st.table(rows)
    else:
        st.info("No open trade data available. Populate results/open_trades.json to view live positions.")
    st.markdown('</div>', unsafe_allow_html=True)


def tail_file(path: Path, lines: int) -> str:
    if not path.exists():
        return f"{path.name} not found."
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"Unable to read {path.name}: {exc}"
    chunk = content[-lines:] if lines > 0 else content
    return "\n".join(chunk) if chunk else "(no log output yet)"


def list_log_files() -> List[Path]:
    if not LOG_ROOT.exists():
        return []
    paths: List[Path] = []
    for pattern in ("*.log", "*.out.log", "*.err.log"):
        paths.extend(LOG_ROOT.glob(pattern))
    return sorted(set(paths))


def render_logs() -> None:
    st.subheader("Logs and Troubleshooting")
    log_files = list_log_files()
    if not log_files:
        st.warning("No log files found in the logs/ folder yet.")
        return
    options = [p.name for p in log_files]
    selected = st.selectbox("Choose a log file", options)
    lines = st.slider("Number of lines to show", 50, 1000, 200, 50)
    autorefresh = st.checkbox("Refresh automatically every 5 seconds", value=False)
    log_path = next(p for p in log_files if p.name == selected)
    st.markdown('<div class="log-output">', unsafe_allow_html=True)
    st.text(tail_file(log_path, lines))
    st.markdown('</div>', unsafe_allow_html=True)
    if autorefresh:
        st.experimental_rerun()




def render_trailing_controls(config) -> None:
    trailing = config.risk.trailing
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Trailing Stops and Take-Profit</div>', unsafe_allow_html=True)

    summary_rows = []
    for regime_key, label in [("HFT", "HFT"), ("intraday", "Intraday"), ("swing", "Swing")]:
        summary_rows.append({"Regime": label, "Stop ATR": trailing.stop_multiplier(regime_key), "Take ATR": trailing.take_multiplier(regime_key)})
    st.dataframe(pd.DataFrame(summary_rows).set_index("Regime"))
    st.caption("ATR stands for Average True Range and controls how far stops/trailing targets sit from price.")

    with st.form("trailing_form"):
        enabled = st.checkbox("Enable trailing risk controls", value=trailing.enabled)
        col_hft, col_intraday, col_swing = st.columns(3)
        regime_inputs = {}
        for col, regime_key, label in [(col_hft, "HFT", "HFT"), (col_intraday, "intraday", "Intraday"), (col_swing, "swing", "Swing")]:
            with col:
                stop_val = st.number_input(f"{label} stop ATR", min_value=0.1, value=float(trailing.stop_multiplier(regime_key)), step=0.1, format="%.2f", key=f"trail_stop_{regime_key}")
                take_val = st.number_input(f"{label} take ATR", min_value=0.1, value=float(trailing.take_multiplier(regime_key)), step=0.1, format="%.2f", key=f"trail_take_{regime_key}")
                regime_inputs[regime_key] = (stop_val, take_val)

        min_lock = st.number_input("Minimum profit (R multiple) before trailing activates", min_value=0.0, value=float(trailing.min_lock.get("R_multiple", 1.0)), step=0.1, format="%.2f")
        guard = st.number_input("Slippage guard (bps)", min_value=0.0, value=float(trailing.slippage_guard_bps), step=1.0)
        max_updates = st.number_input("Max updates per minute", min_value=1, value=int(trailing.max_updates_per_min), step=1)
        submitted = st.form_submit_button("Save trailing settings")
        if submitted:
            trailing.enabled = bool(enabled)
            trailing.k_atr.setdefault("stop", {})
            trailing.k_atr.setdefault("take", {})
            for regime_key, (stop_val, take_val) in regime_inputs.items():
                trailing.k_atr["stop"][regime_key] = float(stop_val)
                trailing.k_atr["take"][regime_key] = float(take_val)
            trailing.min_lock["R_multiple"] = float(min_lock)
            trailing.slippage_guard_bps = float(guard)
            trailing.max_updates_per_min = int(max_updates)
            save_config(config, CONFIG_PATH)
            st.success("Trailing settings updated. Check config.yaml for the saved values.")

    st.markdown('</div>', unsafe_allow_html=True)

def render_settings() -> None:
    st.subheader("Settings")
    config = load_config(CONFIG_PATH)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">config.yaml quick summary</div>', unsafe_allow_html=True)
    if config:
        st.write(f"Symbol: {config.data.symbol}")
        st.write(f"Interval: {config.data.interval}")
        st.write(f"Lookahead: {config.pipeline.lookahead}")
        st.write(f"Long threshold: {config.pipeline.long_threshold}")
        st.write(f"Short threshold: {config.pipeline.short_threshold}")
    else:
        st.warning("config.yaml not found or could not be parsed.")
    st.markdown('</div>', unsafe_allow_html=True)

    if config:
        render_trailing_controls(config)

    col_mode, col_runbot = st.columns(2)
    with col_mode:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Choose interface mode</div>', unsafe_allow_html=True)
        st.radio("Select interface mode", ["Streamlit", "FastAPI", "Gradio"], index=0, key="gui_mode_radio")
        st.info("Update GuiMode in RunBot.ps1 to apply these changes.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_runbot:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">RunBot options</div>', unsafe_allow_html=True)
        st.checkbox("Use Docker Agent", value=False)
        st.checkbox("Debug Mode", value=False)
        st.checkbox("Auto-start Worker", value=True)
        st.caption("Edit RunBot.ps1 to make these toggles permanent.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">.env quick look</div>', unsafe_allow_html=True)
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            mask = "******" if any(word in key.upper() for word in ("KEY", "SECRET", "TOKEN", "PASS")) else value
            st.write(f"`{key}` = {mask}")
    else:
        st.write(".env file not found.")
    st.markdown('</div>', unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="AI Trading Bot Control Center", layout="wide")
    apply_theme()

    st.title("AI Trading Bot Control Center")
    st.caption("Manage data, training, and evaluation for the trading bot from one place.")

    sidebar_option = st.sidebar.radio(
        "Navigate",
        (
            "Control Center",
            "Data and Training Jobs",
            "Model Results",
            "Live Trading",
            "Logs and Troubleshooting",
            "Settings",
        ),
    )

    if sidebar_option == "Control Center":
        render_control_center()
    elif sidebar_option == "Data and Training Jobs":
        render_pipelines()
    elif sidebar_option == "Model Results":
        render_model_results()
    elif sidebar_option == "Live Trading":
        render_live_trading()
    elif sidebar_option == "Logs and Troubleshooting":
        render_logs()
    elif sidebar_option == "Settings":
        render_settings()


if __name__ == "__main__":
    main()




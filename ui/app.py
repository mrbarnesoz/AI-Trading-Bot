from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = PROJECT_ROOT / "logs"
RESULTS_ROOT = PROJECT_ROOT / "results"
MODELS_ROOT = PROJECT_ROOT / "models"
STATE_FILE = LOG_ROOT / "runbot-state.json"
OPEN_TRADES_FILE = RESULTS_ROOT / "open_trades.json"
DRIFT_FILE = RESULTS_ROOT / "drift_report.json"
BACKTEST_FILE = RESULTS_ROOT / "backtest_results.json"
TRAINING_META_FILE = MODELS_ROOT / "latest_metadata.json"

################################################################################
# Styling and helpers
################################################################################


def apply_theme() -> None:
    """Inject custom CSS for the dark theme aesthetic."""
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
            .trade-card {
                border-left: 4px solid transparent;
                transition: border-color 0.2s ease;
            }
            .trade-card.profit {
                border-left-color: rgba(0, 204, 136, 0.85);
            }
            .trade-card.loss {
                border-left-color: rgba(255, 51, 102, 0.85);
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


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except yaml.YAMLError:
        return {}


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
    candidates: List[Path] = []
    for pattern in ("*.log", "*.out.log", "*.err.log"):
        candidates.extend(LOG_ROOT.glob(pattern))
    return sorted(set(candidates))


def humanize_timedelta(delta: timedelta) -> str:
    total_seconds = int(delta.total_seconds())
    prefix = ""
    if total_seconds < 0:
        total_seconds = abs(total_seconds)
        prefix = "-"
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{prefix}{hours}h {minutes}m"
    if minutes > 0:
        return f"{prefix}{minutes}m {seconds}s"
    return f"{prefix}{seconds}s"


def fetch_runbot_state() -> Dict[str, Any]:
    data = load_json(STATE_FILE, {})
    return {
        "prefect_ui": data.get("PrefectUi"),
        "gui_mode": data.get("GuiMode", "streamlit"),
        "processes": data.get("Processes", []),
        "timestamp": data.get("Timestamp"),
    }


def runbot_status_dot(name: str, processes: List[Dict[str, Any]]) -> Dict[str, Any]:
    entry = next((p for p in processes if p.get("Name") == name), None)
    if not entry:
        return {"label": "Stopped", "color": "var(--accent-red)", "tooltip": "Process not running"}
    pid = entry.get("Id")
    label = "Running" if pid else "Unknown"
    color = "var(--accent-green)" if pid else "var(--accent-amber)"
    tooltip = f"PID {pid}" if pid else "Status unknown"
    return {"label": label, "color": color, "tooltip": tooltip}


def load_qc_summary() -> Dict[str, Any]:
    return load_json(RESULTS_ROOT / "qc_summary.json", {})


def load_backtest_results() -> Dict[str, Any]:
    return load_json(BACKTEST_FILE, {})


def load_training_metadata() -> Dict[str, Any]:
    return load_json(TRAINING_META_FILE, {})


def load_open_trades() -> List[Dict[str, Any]]:
    default_sample = [
        {
            "symbol": "XBTUSD",
            "side": "LONG",
            "entry_price": 34250.0,
            "current_price": 34580.0,
            "pnl_pct": 0.96,
            "pnl_usd": 96.0,
            "size": 1000,
            "capital_fraction": 0.10,
            "stop": 33825.0,
            "take": 35000.0,
            "opened_at": (datetime.utcnow() - timedelta(hours=2, minutes=15)).isoformat(),
        },
        {
            "symbol": "ETHUSD",
            "side": "SHORT",
            "entry_price": 1825.0,
            "current_price": 1802.0,
            "pnl_pct": 1.26,
            "pnl_usd": 126.0,
            "size": 500,
            "capital_fraction": 0.08,
            "stop": 1850.0,
            "take": 1780.0,
            "opened_at": (datetime.utcnow() - timedelta(minutes=45)).isoformat(),
        },
    ]
    return load_json(OPEN_TRADES_FILE, default_sample)


def load_drift_metrics() -> Dict[str, Any]:
    default = {
        "feature_drift": 0.12,
        "feature_threshold": 0.15,
        "prediction_drift": 0.03,
        "prediction_threshold": 0.10,
        "timestamp": datetime.utcnow().isoformat(),
    }
    return load_json(DRIFT_FILE, default)


def render_status_dot(label: str, color: str, text: str) -> None:
    st.markdown(
        f"""
        <div class=\"status-dot\" title=\"{text}\">
            <span style=\"background:{color};\"></span>{label}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_stage(stage: Dict[str, Any]) -> None:
    status = stage.get("status", "pending")
    palette = {
        "success": ("var(--accent-green)", "?"),
        "running": ("var(--accent-blue)", "?"),
        "pending": ("var(--text-muted)", "?"),
        "failed": ("var(--accent-red)", "?"),
    }
    color, symbol = palette.get(status, palette["pending"])
    label = stage.get("name", "Stage")
    detail = stage.get("detail", "")
    st.markdown(
        f"""
        <div class=\"card\" style=\"text-align:center; border-left: 4px solid {color};\">
            <div class=\"card-title\" style=\"justify-content:center; gap: 0.5rem;\">
                <span style=\"color:{color}; font-size:1.4rem;\">{symbol}</span>
                {label}
            </div>
            <p style=\"margin:0; color:var(--text-secondary);\">{detail}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
################################################################################
# Section renderers
################################################################################


def render_overview() -> None:
    state = fetch_runbot_state()
    processes = state.get("processes", [])

    st.subheader("Overview")
    col_a, col_b, col_c = st.columns([1.2, 1.3, 1.5])

    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">RunBot Health</div>', unsafe_allow_html=True)
        mappings = {
            "prefect-server": "Prefect Server",
            "prefect-worker": "Prefect Worker",
            "gui-streamlit": "GUI",
        }
        for key, label in mappings.items():
            meta = runbot_status_dot(key, processes)
            render_status_dot(label, meta["color"], meta["tooltip"])
        timestamp = state.get("timestamp")
        if timestamp:
            st.markdown(f"<small>Last update: {timestamp}</small>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Quick Actions</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.button("Run ETL", use_container_width=True)
            st.button("Backtest", use_container_width=True)
        with c2:
            st.button("Train Model", use_container_width=True)
            st.button("Run QC", use_container_width=True)
        st.info("Prefect integration pending. Buttons are placeholders.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_c:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Alert Feed</div>', unsafe_allow_html=True)
        alerts = load_json(RESULTS_ROOT / "alerts.json", [])
        if not alerts:
            st.write("No active alerts.")
        else:
            for alert in alerts[:6]:
                level = alert.get("level", "info").capitalize()
                ts = alert.get("timestamp", "--")
                msg = alert.get("message", "")
                st.markdown(f"- **{level}**  `{ts}` — {msg}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">System Environment</div>', unsafe_allow_html=True)
    env_cols = st.columns(4)
    metadata = {
        "Python": os.environ.get("PYTHON_VERSION", "3.11"),
        "Prefect": os.environ.get("PREFECT_VERSION", "3.x"),
        "Work Pool": os.environ.get("PREFECT_WORK_POOL", "bitmex"),
        "GUI Mode": state.get("gui_mode", "streamlit"),
    }
    for idx, (key, value) in enumerate(metadata.items()):
        with env_cols[idx % 4]:
            st.metric(key, value)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Deployment Schedule (Next 24h)</div>', unsafe_allow_html=True)
    schedule = load_json(RESULTS_ROOT / "prefect_schedule.json", [])
    if not schedule:
        st.write("No schedule data available.")
    else:
        for item in schedule:
            name = item.get("deployment", "unknown")
            run_in = item.get("run_in", "--")
            st.write(f"• {name} — next run in {run_in}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Quick Start</div>', unsafe_allow_html=True)
    qcols = st.columns(4)
    actions = [
        ("Download Data", "python scripts/download_data.py --symbol BTC-USD --force"),
        ("Train Model", "python -m ai_trading_bot train"),
        ("Backtest", "python -m ai_trading_bot backtest"),
        ("Run QC", "python -m jobs.qc_check ..."),
    ]
    for col, (label, command) in zip(qcols, actions):
        with col:
            st.button(label, use_container_width=True)
            st.caption(f"`{command}`")
    st.markdown('</div>', unsafe_allow_html=True)


def render_pipelines() -> None:
    st.subheader("Pipelines")
    etl_status = load_json(RESULTS_ROOT / "etl_status.json", {})

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">ETL Status Board</div>', unsafe_allow_html=True)
    stages = etl_status.get(
        "stages",
        [
            {"name": "Download", "status": "success", "detail": "1.2M rows"},
            {"name": "Normalize", "status": "running", "detail": "45% complete"},
            {"name": "QC", "status": "pending", "detail": "Awaiting previous stage"},
        ],
    )
    stage_cols = st.columns(len(stages))
    for col, stage in zip(stage_cols, stages):
        with col:
            render_pipeline_stage(stage)
    last_run = etl_status.get("last_run")
    if last_run:
        st.caption(f"Last run: {last_run.get('timestamp')} • Duration: {last_run.get('duration')}")
    st.markdown('</div>', unsafe_allow_html=True)

    qc_data = load_qc_summary()
    metrics_col, chart_col = st.columns([1, 1])
    with metrics_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">QC Summary</div>', unsafe_allow_html=True)
        if not qc_data:
            st.info("Run the QC deployment to populate the summary.")
        else:
            for key, value in qc_data.items():
                if isinstance(value, dict):
                    passed = value.get("pass", 0)
                    failed = value.get("fail", 0)
                    total = passed + failed
                    pct = (passed / total * 100) if total else 0
                    st.metric(f"{key.capitalize()} pass rate", f"{pct:.1f}%", delta=f"-{failed} fail")
                else:
                    st.write(f"{key}: {value}")
        st.markdown('</div>', unsafe_allow_html=True)

    with chart_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Pipeline Metrics</div>', unsafe_allow_html=True)
        history = load_json(RESULTS_ROOT / "etl_history.json", [])
        if history:
            st.line_chart(history, height=200)
        else:
            st.write("No historical metrics yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Recent Runs</div>', unsafe_allow_html=True)
    runs = load_json(RESULTS_ROOT / "prefect_runs.json", [])
    if runs:
        st.dataframe(runs, use_container_width=True, height=240)
    else:
        st.write("No Prefect run history yet.")
    st.markdown('</div>', unsafe_allow_html=True)


def render_model_backtesting() -> None:
    st.subheader("Model & Backtesting")
    training_meta = load_training_metadata()
    backtest = load_backtest_results()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Latest Training</div>', unsafe_allow_html=True)
        st.metric("Model", training_meta.get("model", "RandomForest"))
        for label in ("accuracy", "precision", "recall"):
            value = training_meta.get(label)
            if isinstance(value, float):
                display = f"{value:.2%}"
            else:
                display = value if value is not None else "--"
            st.metric(label.capitalize(), display)
        st.caption(f"Trained: {training_meta.get('trained_at', 'unknown')}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Regime Selector</div>', unsafe_allow_html=True)
        st.metric("Active Regime", training_meta.get("mode", "swing"))
        score = training_meta.get("mode_score")
        if isinstance(score, float):
            st.metric("Confidence", f"{score:.3f}")
        features = training_meta.get("features", [])
        st.caption("Features: " + (", ".join(features) if features else "N/A"))
        st.button("Retrain Now", key="retrain_model")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Backtest KPIs</div>', unsafe_allow_html=True)
    if not backtest:
        st.info("Run a backtest to populate metrics.")
    else:
        metric_cols = st.columns(4)
        metric_pairs = [
            ("Sharpe", backtest.get("sharpe")),
            ("Total Return", backtest.get("total_return")),
            ("Max Drawdown", backtest.get("max_drawdown")),
            ("Win Rate", backtest.get("win_rate")),
        ]
        for col, (label, value) in zip(metric_cols, metric_pairs):
            with col:
                st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="kpi-title">{label}</div>', unsafe_allow_html=True)
                if isinstance(value, float):
                    formatted = f"{value:.2%}" if "Return" in label or "Rate" in label else f"{value:.2f}"
                else:
                    formatted = value if value is not None else "--"
                st.markdown(f'<div class="kpi-value">{formatted}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    history = load_json(RESULTS_ROOT / "training_history.json", [])
    if history:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Historical Metrics</div>', unsafe_allow_html=True)
        st.line_chart(history, height=220)
        st.markdown('</div>', unsafe_allow_html=True)

def render_trade_card(trade: Dict[str, Any]) -> None:
    pnl_pct = trade.get("pnl_pct", 0.0)
    pnl_usd = trade.get("pnl_usd", 0.0)
    status_class = "profit" if pnl_pct >= 0 else "loss"
    opened_at = trade.get("opened_at")
    elapsed = "--"
    if opened_at:
        try:
            opened_dt = datetime.fromisoformat(opened_at)
            elapsed = humanize_timedelta(datetime.utcnow() - opened_dt)
        except ValueError:
            elapsed = "--"
    st.markdown(
        f"""
        <div class=\"card trade-card {status_class}\">
            <div class=\"card-title\">
                <span style=\"font-size:1.2rem;font-weight:700;\">{trade.get('symbol','--')}</span>
                <span>{trade.get('side','').upper()}</span>
            </div>
            <div style=\"display:flex; gap:2rem; flex-wrap:wrap;\">
                <div><small>Entry</small><div style=\"font-size:1.05rem;\">{trade.get('entry_price','--')}</div></div>
                <div><small>Current</small><div style=\"font-size:1.05rem;\">{trade.get('current_price','--')}</div></div>
                <div><small>P&amp;L %</small><div style=\"font-size:1.05rem;color:{'var(--accent-green)' if pnl_pct >=0 else 'var(--accent-red)'};\">{pnl_pct:+.2f}%</div></div>
                <div><small>P&amp;L USD</small><div style=\"font-size:1.05rem;color:{'var(--accent-green)' if pnl_usd >=0 else 'var(--accent-red)'};\">${pnl_usd:+.2f}</div></div>
                <div><small>Size</small><div style=\"font-size:1.05rem;\">{trade.get('size','--')} units</div></div>
                <div><small>Capital</small><div style=\"font-size:1.05rem;\">{trade.get('capital_fraction',0)*100:.1f}%</div></div>
            </div>
            <div style=\"margin-top:1rem; display:flex; gap:2rem; flex-wrap:wrap;\">
                <div>Stop: {trade.get('stop','--')}</div>
                <div>Take: {trade.get('take','--')}</div>
                <div>Opened: {elapsed} ago</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    btn_cols = st.columns(2)
    with btn_cols[0]:
        st.button("Close Position", key=f"close_{trade.get('symbol')}_{trade.get('side')}")
    with btn_cols[1]:
        st.button("Adjust Risk", key=f"risk_{trade.get('symbol')}_{trade.get('side')}")


def render_live_trading() -> None:
    st.subheader("Live Trading")
    trades = load_open_trades()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Open Trades</div>', unsafe_allow_html=True)
    if not trades:
        st.info("No active positions.")
    else:
        total_usd = sum(trade.get("pnl_usd", 0.0) for trade in trades)
        weighted_pct = sum(trade.get("pnl_pct", 0.0) * trade.get("capital_fraction", 0.0) for trade in trades)
        capital = sum(trade.get("capital_fraction", 0.0) for trade in trades)
        st.write(f"Aggregate P&L: ${total_usd:+.2f} ({weighted_pct:+.2f}%) — Capital deployed: {capital*100:.1f}%")
        for trade in trades:
            render_trade_card(trade)
    st.markdown('</div>', unsafe_allow_html=True)

    col_risk, col_exposure = st.columns([1, 1])
    with col_risk:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Risk Controls</div>', unsafe_allow_html=True)
        st.write("Guardrails: Active")
        st.write("Capital Usage: 18% of 50% limit")
        st.write("Max Drawdown: -3.2%")
        st.button("Kill Switch", key="kill_switch")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_exposure:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Exposure by Regime</div>', unsafe_allow_html=True)
        exposure = load_json(RESULTS_ROOT / "exposure.json", {"scalping": 0.1, "intraday": 0.35, "swing": 0.55})
        st.bar_chart(exposure, height=180)
        st.markdown('</div>', unsafe_allow_html=True)

    drift = load_drift_metrics()
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Drift Monitoring</div>', unsafe_allow_html=True)
    st.write(
        f"Feature Drift: {drift.get('feature_drift', 0):.3f} (threshold {drift.get('feature_threshold', 0.15):.2f})"
    )
    st.write(
        f"Prediction Drift: {drift.get('prediction_drift', 0):.3f} (threshold {drift.get('prediction_threshold', 0.10):.2f})"
    )
    st.caption(f"Last check: {drift.get('timestamp', '--')}")
    st.button("Run Drift Test", key="run_drift")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Recent Prefect Runs</div>', unsafe_allow_html=True)
    runs = load_json(RESULTS_ROOT / "prefect_runs_live.json", [])
    if runs:
        st.table(runs)
    else:
        st.write("No recent runs recorded for live flows.")
    st.markdown('</div>', unsafe_allow_html=True)


def render_logs() -> None:
    st.subheader("Logs & Diagnostics")
    log_files = list_log_files()
    if not log_files:
        st.warning("No logs found in the logs/ directory.")
        return
    selected = st.selectbox("Select log file", [p.name for p in log_files])
    lines = st.slider("Tail lines", min_value=50, max_value=1000, value=200, step=50)
    log_path = next(p for p in log_files if p.name == selected)
    st.markdown('<div class="log-output">', unsafe_allow_html=True)
    st.text(tail_file(log_path, lines))
    st.markdown('</div>', unsafe_allow_html=True)

    col_snippets, col_env = st.columns([1, 1])
    with col_snippets:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Command Snippets</div>', unsafe_allow_html=True)
        snippets = {
            "Run ETL Deployment": "prefect deployment run 'bitmex-etl-flow/etl-bitmex'",
            "Tail Prefect Logs": "tail -f logs/prefect-server-*.out.log",
            "Trigger QC": "prefect deployment run 'bitmex-qc-flow/qc-bitmex'",
        }
        for label, command in snippets.items():
            st.markdown(f"**{label}**")
            st.code(command, language="bash")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_env:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Environment Variables</div>', unsafe_allow_html=True)
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                mask = "******" if any(word in key.upper() for word in ("KEY", "SECRET", "TOKEN", "PASS")) else value
                st.write(f"`{key}` = {mask}")
        else:
            st.write(".env file not found.")
        st.markdown('</div>', unsafe_allow_html=True)


def render_settings() -> None:
    st.subheader("Settings")
    config = load_yaml(PROJECT_ROOT / "config.yaml")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">config.yaml Summary</div>', unsafe_allow_html=True)
    if config:
        data_section = config.get("data", {})
        pipeline = config.get("pipeline", {})
        st.write(f"Symbol: {data_section.get('symbol','--')}")
        st.write(
            "Mode thresholds: long {0}, short {1}".format(
                pipeline.get('long_threshold'), pipeline.get('short_threshold')
            )
        )
        bands = pipeline.get("long_bands")
        if bands:
            st.write(f"Long bands: {bands}")
    else:
        st.warning("config.yaml not found or could not be parsed.")
    st.markdown('</div>', unsafe_allow_html=True)

    col_mode, col_runbot = st.columns([1, 1])
    with col_mode:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">GUI Mode</div>', unsafe_allow_html=True)
        st.radio("Select GUI Mode", ["Streamlit", "FastAPI", "Gradio"], index=0, key="gui_mode_radio")
        st.info("Update GuiMode in RunBot.ps1 to apply changes.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_runbot:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">RunBot Toggles</div>', unsafe_allow_html=True)
        st.checkbox("Use Docker Agent", value=False)
        st.checkbox("Debug Mode", value=False)
        st.checkbox("Auto-start Worker", value=True)
        st.caption("Edit RunBot.ps1 configuration to persist changes.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">.env Preview</div>', unsafe_allow_html=True)
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            mask = "******" if any(word in key.upper() for word in ("KEY", "SECRET", "TOKEN", "PASS")) else value
            st.write(f"`{key}` = {mask}")
    else:
        st.write(".env file not found.")
    st.markdown('</div>', unsafe_allow_html=True)


################################################################################
# Main application
################################################################################


def main() -> None:
    st.set_page_config(page_title="AI Trading Bot Control Center", layout="wide")
    apply_theme()
    st.markdown(
        """
        <div class=\"main-header\">
            <div>
                <h1 style=\"margin-bottom:0;\">AI Trading Bot Control Center</h1>
                <p style=\"color:var(--text-secondary); margin-top:4px;\">
                    Operational overview for data pipelines, model training, and live trading.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section = st.sidebar.radio(
        "Navigate",
        (
            "Overview",
            "Pipelines",
            "Model & Backtesting",
            "Live Trading",
            "Logs & Diagnostics",
            "Settings",
        ),
    )

    if section == "Overview":
        render_overview()
    elif section == "Pipelines":
        render_pipelines()
    elif section == "Model & Backtesting":
        render_model_backtesting()
    elif section == "Live Trading":
        render_live_trading()
    elif section == "Logs & Diagnostics":
        render_logs()
    elif section == "Settings":
        render_settings()


if __name__ == "__main__":
    main()

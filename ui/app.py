from __future__ import annotations

import json
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


def run_etl(force_download: bool = True) -> None:
    config = load_config(CONFIG_PATH)
    price_frame, engineered = prepare_dataset(config, force_download=force_download)
    message = (
        f"Symbol {config.data.symbol} | {engineered.shape[0]} rows | Interval {config.data.interval} | "
        f"Prices span {price_frame.index.min()} to {price_frame.index.max()}"
    )
    append_action_log("Run ETL", "success", message)
    st.success(f"ETL complete. {message}")


def run_training(force_download: bool = False) -> Dict[str, Any]:
    metrics = pipeline_train(CONFIG_PATH, force_download=force_download)
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
    st.session_state["selected_backtest_option"] = None
    return payload


def render_action_log_card() -> None:
    logs = st.session_state.get("action_logs", [])
    if not logs:
        return
    with st.expander("Recent actions", expanded=False):
        for entry in logs:
            st.markdown(f"**{entry['timestamp']}** — {entry['title']} ({entry['status']})")
            st.code(entry["message"], language="text")


def render_overview() -> None:
    st.subheader("Overview")
    state = fetch_runbot_state()
    processes = state.get("Processes") or state.get("processes") or []
    col_a, col_b, col_c = st.columns([1.2, 1.3, 1.5])

    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">RunBot Health</div>', unsafe_allow_html=True)
        for key, label in {
            "prefect-server": "Prefect Server",
            "prefect-worker": "Prefect Worker",
            "gui-streamlit": "GUI",
        }.items():
            meta = runbot_status_dot(key, processes)
            render_status_dot(label, meta["color"], meta["tooltip"])
        timestamp = state.get("Timestamp") or state.get("timestamp")
        if timestamp:
            st.caption(f"Last update: {format_timestamp(timestamp)}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Quick Actions</div>', unsafe_allow_html=True)
        qa_col1, qa_col2 = st.columns(2)
        with qa_col1:
            if st.button("Run ETL", key="qa_run_etl", use_container_width=True):
                with st.spinner("Running ETL..."):
                    run_etl(force_download=True)
            if st.button("Backtest", key="qa_backtest", use_container_width=True):
                with st.spinner("Running backtest..."):
                    run_backtest()
        with qa_col2:
            if st.button("Train Model", key="qa_train", use_container_width=True):
                with st.spinner("Training model..."):
                    run_training()
            if st.button("Run QC", key="qa_qc", use_container_width=True):
                qc_result_path = RESULTS_ROOT / "qc_summary.json"
                bronze = PROJECT_ROOT / "data" / "bronze"
                silver = PROJECT_ROOT / "data" / "silver"
                if bronze.exists() and silver.exists():
                    import subprocess

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
                    with st.spinner("Running QC checks..."):
                        completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
                    if completed.returncode == 0:
                        append_action_log("QC", "success", "QC checks complete.")
                        st.success("QC checks finished. Inspect qc_summary.json for details.")
                    else:
                        append_action_log("QC", "error", completed.stderr.strip() or "QC failed")
                        st.error("QC checks failed; see console for details.")
                else:
                    st.warning("Expected bronze/silver directories not found; run the ETL pipeline first.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_c:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">System Environment</div>', unsafe_allow_html=True)
        env_cols = st.columns(4)
        metadata = {
            "Python": sys.version.split()[0],
            "Prefect": state.get("prefect_version", "3.x"),
            "Work Pool": state.get("work_pool", "bitmex"),
            "GUI Mode": state.get("gui_mode", "streamlit"),
        }
        for idx, (key, value) in enumerate(metadata.items()):
            with env_cols[idx % 4]:
                st.metric(key, value)
        st.markdown('</div>', unsafe_allow_html=True)

    render_action_log_card()


def render_pipeline_stage(stage: Dict[str, str]) -> None:
    color_map = {
        "success": "var(--accent-green)",
        "running": "var(--accent-blue)",
        "warning": "var(--accent-amber)",
        "failed": "var(--accent-red)",
        "pending": "var(--text-muted)",
    }
    symbol_map = {
        "success": "✓",
        "running": "●",
        "warning": "!",
        "failed": "✗",
        "pending": "○",
    }
    color = color_map.get(stage.get("status", "pending"), "var(--text-muted)")
    symbol = symbol_map.get(stage.get("status", "pending"), "○")
    st.markdown(
        f"""
        <div class="card" style="text-align:center; border-left: 4px solid {color};">
            <div class="card-title" style="justify-content:center; gap:0.5rem;">
                <span style="color:{color}; font-size:1.4rem;">{symbol}</span>
                {stage.get('name', 'Stage')}
            </div>
            <p style="margin:0; color:var(--text-secondary);">{stage.get('detail', '')}</p>
        </div>
        """,
        unsafe_allow_html=True)


def render_pipelines() -> None:
    st.subheader("Pipelines")
    raw_count = len(list((PROJECT_ROOT / "data" / "raw").rglob("*.csv")))
    bronze_count = len(list((PROJECT_ROOT / "data" / "bronze").rglob("*.parquet")))
    qc_summary = load_json(RESULTS_ROOT / "qc_summary.json")

    stages = [
        {
            "name": "Download",
            "status": "success" if raw_count else "pending",
            "detail": f"Cached files: {raw_count}",
        },
        {
            "name": "Normalize",
            "status": "success" if bronze_count else "pending",
            "detail": f"Parquet files: {bronze_count}",
        },
    ]
    if qc_summary:
        warn_count = len(qc_summary.get("trade_warnings", [])) + len(qc_summary.get("snapshot_warnings", []))
        stages.append(
            {
                "name": "QC",
                "status": "warning" if warn_count else "success",
                "detail": f"Warnings: {warn_count}",
            }
        )
    else:
        stages.append(
            {
                "name": "QC",
                "status": "pending",
                "detail": "No QC summary on disk.",
            }
        )

    stage_cols = st.columns(len(stages))
    for col, stage in zip(stage_cols, stages):
        with col:
            render_pipeline_stage(stage)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">QC Summary</div>', unsafe_allow_html=True)
    if qc_summary:
        st.json(qc_summary)
    else:
        st.info("Run the QC check to generate qc_summary.json.")
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


def render_model_backtesting() -> None:
    st.subheader("Model & Backtesting")

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
            "Backtest runs",
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
        st.markdown('<div class="card-title">Latest Training</div>', unsafe_allow_html=True)
        if training_data:
            st.metric("Model", training_data.get("model", "Unknown"))
            st.metric("Accuracy", f"{training_data.get('accuracy', 0):.2%}")
            st.metric("Precision", f"{training_data.get('precision', 0):.2%}")
            st.metric("Recall", f"{training_data.get('recall', 0):.2%}")
            st.caption(f"Trained: {format_timestamp(training_data.get('trained_at'))}")
            st.markdown("**Features**")
            st.write(", ".join(training_data.get("features", [])) or "--")
        else:
            st.info("Train the model to populate this section.")
        if st.button("Retrain Now", key="retrain_now"):
            with st.spinner("Retraining model..."):
                run_training()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Regime Selector</div>', unsafe_allow_html=True)
        if training_data:
            st.metric("Active Regime", training_data.get("mode", "--"))
            mode_score = training_data.get("mode_score")
            st.metric("Mode Score", f"{mode_score:.3f}" if mode_score is not None else "--")
        else:
            st.info("Train the model to evaluate regime selection.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Backtest KPIs</div>', unsafe_allow_html=True)
    if backtest_data:
        summary = backtest_data.get("summary", {})
        metric_cols = st.columns(4)
        labels = [
            ("Sharpe", summary.get("sharpe_ratio")),
            ("Total Return", summary.get("total_return")),
            ("Max Drawdown", summary.get("max_drawdown")),
            ("Win Rate", summary.get("win_rate")),
        ]
        for col, (label, value) in zip(metric_cols, labels):
            with col:
                st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="kpi-title">{label}</div>', unsafe_allow_html=True)
                if value is None:
                    display_value = "--"
                elif label in {"Sharpe"}:
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
    st.subheader("Live Trading")
    trades_path = RESULTS_ROOT / "open_trades.json"
    trades = load_json(trades_path)
    if trades:
        rows = trades if isinstance(trades, list) else [trades]
        st.table(rows)
    else:
        st.info("No open trade data available. Populate results/open_trades.json to enable this view.")


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
    st.subheader("Logs & Diagnostics")
    log_files = list_log_files()
    if not log_files:
        st.warning("No logs found in the logs/ directory yet.")
        return
    options = [p.name for p in log_files]
    selected = st.selectbox("Select log file", options)
    lines = st.slider("Tail lines", 50, 1000, 200, 50)
    autorefresh = st.checkbox("Auto-refresh every 5 seconds", value=False)
    log_path = next(p for p in log_files if p.name == selected)
    st.markdown('<div class="log-output">', unsafe_allow_html=True)
    st.text(tail_file(log_path, lines))
    st.markdown('</div>', unsafe_allow_html=True)
    if autorefresh:
        st.experimental_rerun()




def render_trailing_controls(config) -> None:
    trailing = config.risk.trailing
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Trailing Stops & Take-Profit</div>', unsafe_allow_html=True)

    summary_rows = []
    for regime_key, label in [("HFT", "HFT"), ("intraday", "Intraday"), ("swing", "Swing")]:
        summary_rows.append({"Regime": label, "Stop ATR": trailing.stop_multiplier(regime_key), "Take ATR": trailing.take_multiplier(regime_key)})
    st.dataframe(pd.DataFrame(summary_rows).set_index("Regime"))

    with st.form("trailing_form"):
        enabled = st.checkbox("Enable trailing risk controls", value=trailing.enabled)
        col_hft, col_intraday, col_swing = st.columns(3)
        regime_inputs = {}
        for col, regime_key, label in [(col_hft, "HFT", "HFT"), (col_intraday, "intraday", "Intraday"), (col_swing, "swing", "Swing")]:
            with col:
                stop_val = st.number_input(f"{label} stop ATR", min_value=0.1, value=float(trailing.stop_multiplier(regime_key)), step=0.1, format="%.2f", key=f"trail_stop_{regime_key}")
                take_val = st.number_input(f"{label} take ATR", min_value=0.1, value=float(trailing.take_multiplier(regime_key)), step=0.1, format="%.2f", key=f"trail_take_{regime_key}")
                regime_inputs[regime_key] = (stop_val, take_val)

        min_lock = st.number_input("Min R multiple before trailing activates", min_value=0.0, value=float(trailing.min_lock.get("R_multiple", 1.0)), step=0.1, format="%.2f")
        guard = st.number_input("Slippage guard (bps)", min_value=0.0, value=float(trailing.slippage_guard_bps), step=1.0)
        max_updates = st.number_input("Max updates per minute", min_value=1, value=int(trailing.max_updates_per_min), step=1)
        submitted = st.form_submit_button("Save Trailing Settings")
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
            st.success("Trailing settings updated. Changes persist to config.yaml")

    st.markdown('</div>', unsafe_allow_html=True)

def render_settings() -> None:
    st.subheader("Settings")
    config = load_config(CONFIG_PATH)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">config.yaml Summary</div>', unsafe_allow_html=True)
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
            "Overview",
            "Pipelines",
            "Model & Backtesting",
            "Live Trading",
            "Logs & Diagnostics",
            "Settings",
        ),
    )

    if sidebar_option == "Overview":
        render_overview()
    elif sidebar_option == "Pipelines":
        render_pipelines()
    elif sidebar_option == "Model & Backtesting":
        render_model_backtesting()
    elif sidebar_option == "Live Trading":
        render_live_trading()
    elif sidebar_option == "Logs & Diagnostics":
        render_logs()
    elif sidebar_option == "Settings":
        render_settings()


if __name__ == "__main__":
    main()

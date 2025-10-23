# AI Trading Bot

An end-to-end reference implementation of an AI-assisted trading bot. The app provides utilities to download OHLCV data, engineer technical indicator features, train a machine learning classifier, and evaluate the strategy via vectorised backtesting.

> Keep this README up to date whenever you change the bot or its tooling.

## Key Capabilities
- Automated data download and caching via `yfinance`.
- Technical indicator feature engineering (SMA, EMA, RSI, MACD, volatility, volume).
- Random forest model trained to predict next-period positive returns.
- Adaptive mode selection that chooses between scalping (sub-minute), intraday (1-15 min), and swing (hourly/daily) trading styles.
- Probability-driven long/short/flat signal generation with multi-band position sizing.
- Backtesting module with trading costs, Sharpe ratio, win-rate, drawdown, and capital utilisation reporting.
- Capital allocation rules that cap risk per position so multiple trades can be open simultaneously.
- CLI scripts and VS Code tasks for repeatable workflows.

## Project Structure
```
|-- config.yaml              # Central configuration for data, model, and strategy
|-- pyproject.toml           # Python project metadata and dependencies
|-- scripts/                 # Convenience entrypoints for common workflows
|-- src/ai_trading_bot/      # Core package code (data, features, model, strategy, backtest)
|-- data/                    # Cached raw/processed data (ignored by git)
|-- models/                  # Persisted model artifacts (ignored by git)
|-- logs/                    # Log output (ignored by git)
`-- tests/                   # Pytest-based unit tests
```

## Getting Started
1. **Prerequisites**
   - Python 3.10+ (recommended 3.11 for best library support)
   - pip
   - (Optional) Git for version control

2. **Create a virtual environment**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1  # use source .venv/bin/activate on macOS/Linux
   ```

3. **Install dependencies**
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -e .[dev]
   ```

4. **(Optional) Configure VS Code**
   - Open the workspace folder in VS Code.
   - Accept the prompt to use the interpreter at `.venv\Scripts\python.exe`.
   - Use the provided tasks (`Terminal > Run Task...`) for training, backtesting, or running tests.

## Flash Up the Bot (Windows Quick Launch)
For everyday use on Windows, the `RunBot` scripts start everything (Prefect server, worker/agent, and the chosen GUI) and shut it down again.

1. Install PowerShell 7+ and Python 3.11 if you have not already.
2. From the repo root, double-click `RunBot.bat` or run it in a terminal:
   ```powershell
   .\RunBot.bat
   ```
   The script will build the virtual environment, deploy Prefect flows, and launch the default Streamlit UI at http://127.0.0.1:8501.
3. To stop all services, run:
   ```powershell
   .\RunBot.ps1 -Stop
   ```
4. Optional extras:
   - `.\RunBot.ps1 -Debug` streams live log tails and enables verbose Prefect logging.
   - Set `GuiMode` inside `RunBot.ps1` to `fastapi` or `gradio` if you prefer a different interface.
   - Logs are written under `logs\` with one file per service; inspect them if something fails.

Update this section whenever you add new services, ports, or environment requirements so day-to-day operators always have the latest steps.

## Usage
Run all commands from the project root with your virtual environment activated.

- **Download data**
  ```powershell
  python scripts/download_data.py --symbol AAPL --force
  ```
  Cached files are written to `data/raw/`. Adjust defaults in `config.yaml`.

- **Train the model**
  ```powershell
  python -m ai_trading_bot train
  ```
  Metrics are printed as JSON and the trained model is saved to `models/price_direction_model.joblib`.

- **Backtest the strategy**
  ```powershell
  python -m ai_trading_bot backtest --long-threshold 0.6 --short-threshold 0.4
  ```
  The output includes the automatically selected mode, diagnostic metrics, and the backtest summary. Detailed performance (including equity curve) lives in-memory, but you can persist it by extending the script.

- **Run tests**
  ```powershell
  python -m pytest
  ```

## BitMEX Data Pipeline Scaffold
Stage 1 introduces an ETL framework for BitMEX historical and live market data. Key directories:

- `etl/`: downloaders, normalizers, bar builders, QC helpers.
- `schemas/`: YAML contracts for bronze and silver datasets.
- `conf/`: Hydra-friendly configuration for storage, sources, and cadence.
- `jobs/`: backfill and daily incremental entrypoints.
- `utils/`: shared I/O, checksum, and time helpers.
- `docs/data_contracts.md`: canonical description of dataset guarantees and QC gates.

Bronze and silver data are stored under `data/` following the bronze/silver/gold convention, with DuckDB (`warehouse.duckdb`) scanning external Parquet files. L2 snapshots default to a 250 ms cadence; adjust in `conf/cadence.yaml`.

Run the Stage 1 pipeline end-to-end:

```powershell
# Download archives for the desired date range
python -m jobs.download_archives --start 2014-11-22 --end 2014-11-23 --symbols XBTUSD

# Normalize, build snapshots/bars, and register DuckDB views
python -m jobs.run_pipeline --start 2014-11-22 --end 2014-11-23 --symbols XBTUSD

# Run QC checks and emit summary
python -m jobs.qc_check --bronze-root data/bronze --silver-root data/silver --output-json results/qc_summary.json
```

```powershell
# Generate a daily report that includes QC counts
python - <<'PY'
import json
from pathlib import Path
from reports.daily_report import report_from_metrics, save_report
qc = json.load(open('results/qc_summary.json'))
report = report_from_metrics({'pnl': 0.0, 'sharpe': 0.0, 'latency_ms': 45, 'feature_psi': 0.1}, qc)
save_report(report, Path('results/daily_report.txt'))
PY
```

Set `auto_ingest_l2: true` in `conf/storage.yaml` to download order book archives (default is disabled). The async scheduler in `jobs/daily_scheduler.py` can automate the overnight routine:

```powershell
python -m jobs.daily_scheduler --storage-config conf/storage.yaml --cadence-config conf/cadence.yaml
```

Alerts fire if the pipeline fails or QC emits warnings; reports are written to `results/daily_report.txt`.

## Training Pipelines
Stage 3 scaffolding introduces model stacks per regime:

- `models/gbm_classifier.py`: LightGBM wrapper for intraday/swing (weighted logloss).
- `models/tcn_classifier.py`: PyTorch Lightning TCN module for HFT (focal loss ready).
- `models/calibrate.py`: isotonic/Platt calibration utilities.
- `training/walkforward.py`: purged walk-forward routine with embargoed folds.
- `training/rolling_hft.py`: rolling HFT retraining loop with per-window logging.
- `training/metrics.py`: Sharpe, Calmar, and Brier helpers.

These stubs hook into MLflow for experiment tracking and will be extended with full data loaders and evaluation logic as Stage 3 is completed.

## Evaluation & Simulation
Stage 4 - Backtesting & Evaluation toolkit:

- `eval/backtest_engine.py`: vectorized backtester with configurable BitMEX cost models.
- `eval/costs.py`: taker/slippage/latency cost model utilities.
- `eval/metrics.py`: Sharpe, Sortino, Calmar, max drawdown, hit rate.
- `eval/forward_test.py`: shadow-testing helper for canary runs.
- `monitoring/drift.py`: PSI calculation for feature drift.

Outputs (metrics, trades) can be routed to `results/` and MLflow for auditability.

## Live Agent
Stage 5 introduces the embedded agent skeleton:

- `live/agent/core.py`: async per-symbol loop coordinating policy, risk, and execution.
- `live/policy/decision.py`: regime-specific thresholds for maker-first decisioning.
- `live/execution/order_router.py`: placeholder order router supporting conditional spread crossing.
- `live/risk/guardrails.py`: leverage, loss-stop, and checkpoint stubs.
- `live/state/checkpoint.py`: periodic state persistence.
- `logging/structured.py`: structured JSON logging for audit trails.

These modules set the stage for wiring BitMEX APIs, latency watchdogs, and the maker-first strategy thresholds defined in Stage 5.

## Stage 3 - Walk-Forward Model Training
Gold datasets in `data/gold/` feed the LightGBM walk-forward trainer (`training/walkforward.py`) and the HFT TCN rolling loop (`training/rolling_hft.py`). Launch end-to-end training per regime with:

```powershell
python jobs/train_models.py --gold-root data/gold --regimes intraday swing hft --symbols XBTUSD
```

- Walk-forward folds log Sharpe/Calmar/Brier scores plus feature importances to MLflow; calibrators serialise alongside the model artifacts.
- HFT windows train the Lightning TCN with latency-friendly sequence sampling; metrics capture long/short Sharpe.
- Synthetic fixtures live in `tests/test_training_pipeline.py` (skipped automatically if MLflow is not installed).

## Stage 4 - Backtesting & Evaluation
`eval/backtest_engine.py` now simulates maker-first execution, turnover, and latency offsets. Combine it with the cost model to produce production-grade tear-downs:

```python
from pathlib import Path
from eval.backtest_engine import BacktestConfig, run_backtest, save_results
from eval.costs import CostModel

metrics, portfolio, trades = run_backtest(
    features_df,
    probabilities,
    realised_returns,
    CostModel(taker_fee_bps=2.5, slippage_bps=1.0),
    BacktestConfig(regime="intraday", long_threshold=0.60, short_threshold=0.60, cross_threshold=0.75),
)
save_results(metrics, trades, Path("results/intraday_backtest"))
```

The `metrics` payload includes Sharpe, Sortino, Calmar, max drawdown, hit rate, profit factor, and turnover. See `tests/test_eval_backtest_engine.py` for end-to-end coverage.

## Stage 5 - Live Agent, Risk Controls, and Persistence
`live/policy/decision.py` produces structured `Decision` objects (side, size, confidence, cross-spread trigger, and target exposure). Risk is enforced through:

- `live/state/positions.py`: multi-sub-position manager per symbol.
- `live/risk/guardrails.py`: leverage caps, ATR-based liquidation buffers, and daily loss kill switches.
- `live/state/checkpoint.py`: JSON persistence for positions + risk state, restoring automatically on boot.

Tests under `tests/test_live_risk_policy.py` validate both decision logic and guard rails.

## Deployment & Operations
Stage 6 scaffolding covers rollout, monitoring, and maintenance:

- `deployment/phases.py`: phased rollout plan (dry run → shadow → canary → expansion → full).
- `deployment/monitoring.py`: alert registration/evaluation hooks.
- `reports/daily_report.py`: daily PnL/latency/drift report generator.
- `maintenance/retrain_scheduler.py`: cadence-aware retrain scheduler.

Operational glue:

- Register Prefect deployments with `python prefect_deployments.py`, start a worker on the `bitmex` process pool, then trigger runs via `prefect deployment run bitmex-daily-flow/daily-bitmex`.
- Launch the embedded live agent once evaluation gates clear:

  ```powershell
  python deployment/live_agent_runner.py --config conf/live_agent.yaml --api-client live.execution.bitmex_client:BitmexClient --checkpoint data/checkpoints/live_state.json
  ```

  The YAML config declares symbols/regimes, threshold overrides, and risk limits. State is checkpointed automatically for crash recovery.

Dashboards and alerts will integrate here to enforce readiness gates before moving between deployment phases.

## Configuration
Fine-tune behaviour through `config.yaml`. Key fields include:
- `data`: symbol, date range, interval, and caching location.
- `features`: indicator types and window sizes.
- `model`: hyperparameters for the random forest classifier.
- `backtest`: capital, transaction cost, risk-free rate, and capital-allocation constraints (`position_capital_fraction`, `max_total_capital_fraction`, `max_position_units`).
- `pipeline`: prediction horizon (`lookahead`), target column name, and probability thresholds for long/short trades.
  - Optional `long_bands` / `short_bands` lists let you scale position size as probabilities move deeper into confident territory.
- `modes`: definitions for scalping, intraday, and swing regimes. Each mode can override thresholds, probability bands, and risk settings; the bot evaluates them and trades with the mode that scores highest for the current market regime.

### Adaptive Mode Selection
Each run scores every configured mode using recent price action:
- **Trend strength** (absolute mean return vs volatility) favours swing trading when markets trend cleanly.
- **Volatility** and **liquidity** favour scalping when intraday noise is elevated.
- Combined scores decide the mode; overrides (thresholds, bands, risk limits) are applied automatically.

Modes are declared in `config.yaml`:

```yaml
modes:
  - name: "scalping"
    interval: "1m"
    lookback_days: 2
    volatility_weight: 1.5
    trend_weight: 0.7
    position_fraction: 0.05
    max_total_fraction: 0.2
  - name: "intraday"
    interval: "5m"
    lookback_days: 7
  - name: "swing"
    interval: "1h"
    lookback_days: 60
    trend_weight: 1.4
```

You can tailor thresholds, banding, and capital limits per mode. The chosen mode is reported in the CLI/JSON output.

### Probability Banding
Set multiple probability bands to express conviction tiers. For example:

```yaml
pipeline:
  long_threshold: 0.55
  short_threshold: 0.45
  long_bands: [0.55, 0.65, 0.80]   # 1x when >=0.55, 2x when >=0.65, 3x when >=0.80
  short_bands: [0.45, 0.35, 0.25]  # -1x when <=0.45, -2x when <=0.35, -3x when <=0.25
```

CLI overrides such as `--long-threshold` or `--short-threshold` tighten the outer band without touching deeper tiers.

## Extending the Bot
- Swap in alternative models by modifying `src/ai_trading_bot/models/predictor.py`.
- Add new indicators to `features/indicators.py` and list them in `config.yaml`.
- Implement new strategies in `strategies/` and wire them into `pipeline.py`.
- Persist detailed backtest analytics (drawdown curves, trade list, etc.) from `backtesting/simulator.py`.

## Troubleshooting
- **`ModuleNotFoundError: No module named 'pytest'`** - ensure dependencies are installed with `python -m pip install -e .[dev]`.
- **`ValueError: No data retrieved for symbol`** - verify ticker/interval combinations are supported by Yahoo Finance and you have network connectivity (1-minute data is only available for recent sessions).
- **Signals look empty** - confirm the long/short thresholds leave a tradable band (e.g. `--long-threshold 0.55 --short-threshold 0.45`), or relax the probability bands for the selected mode.

## Licensing
This project is provided as reference code; adapt it for educational or personal trading experiments. Always validate strategies on out-of-sample data and understand the risks of live trading.
## Prefect Orchestration
Stage 6 optionally uses Prefect for orchestration. Flows live in `orchestration/prefect_flows.py` and cover archive download, ETL, QC, and circuit-breaker logic.

### Register deployments
```powershell
prefect deployment build orchestration/prefect_flows.py:bitmex_daily_flow -n daily-bitmex
prefect deployment apply bitmex_daily_flow-deployment.yaml
```

### Start a local agent (Docker)
```powershell
docker build -f docker/prefect-agent.Dockerfile -t bitmex-prefect-agent .
docker run --rm --network host -e PREFECT_API_URL="http://host.docker.internal:4200/api" bitmex-prefect-agent
```

Run individual flows as needed:
```powershell
prefect deployment run 'bitmex-etl-flow'
prefect deployment run 'bitmex-qc-flow'
```

Prefect blocks (e.g. S3, BitMEX credentials) can be registered via `prefect block register --file blocks/bitmex_blocks.py` when extending to managed secrets.
To scaffold sample Prefect blocks locally:
```powershell
python blocks/register_blocks.py
```

Set `APPRISE_URLS` or register the `alert-webhooks` secret block with one or more Apprise URLs (Slack, email, etc.) so alerts fire when QC or pipeline checks fail.

Readme.md
+162
-1

Readme
# AI Trading Bot

This repository implements a modular, AI-driven BitMEX trading system capable of fully automated leveraged trading with configurable paper/live modes. It accompanies the architectural and design plans stored in `docs/`.

## Features
- Microservice-inspired Python modules for data ingestion, AI strategy execution, risk management, and order routing.
- BitMEX Testnet/Mainnet support with authenticated REST order placement and WebSocket market data streaming.
- Enforced 3:1 minimum risk-reward ratio and leverage guard rails (10x–20x).
- Training pipeline for the bundled AI strategy model and reusable presets for discretionary and rules-based playbooks.
- FastAPI control plane with paper-trading toggle and signal submission endpoint.
- Command line interface for training, recording paper trades, and executing strategies manually with automatic selection support.
- Paper-trading learning loop that folds recorded outcomes into future AI training runs.
- Context-aware strategy selector that blends heuristic regime detection with paper-trade performance to deploy the right preset.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file at the repository root and set the following variables:

```env
BITMEX_API_KEY=your_key
BITMEX_API_SECRET=your_secret
BITMEX_ENVIRONMENT=testnet  # or mainnet once approved
PAPER_TRADING=true
```

## Running Tests

```bash
pytest
```

## Usage

### Train the Strategy Model

```bash
PYTHONPATH=src python -m trading_bot.main train \
  --data data/training_dataset.pkl \
  --output artifacts/ \
  --feature-dim 16 \
  --epochs 10
```

### Submit a Live Signal (Paper Mode by Default)

```bash
python -m trading_bot.main live \
  --symbol XBTUSD \
  --side Buy \
  --entry 30000 \
  --stop 29000 \
  --take 33999 \
  --size 0.5 \
  --strategy auto \
  --funding-rate 0.012 \
  --price-momentum 0.8 \
  --volatility 1.1 \
  --open-interest-change 0.6 \
  --orderbook-imbalance 0.2
```

Providing the optional context metrics allows the orchestrator to select an appropriate strategy automatically. Supply an explicit
`--strategy` slug if you want to bypass the selector.

### Inspect Built-in Strategy Presets

```bash
python -m trading_bot.main list-strategies
```

This command enumerates six preloaded strategies:

1. **Perpetual funding/basis cash-and-carry** – market-neutral carry program harvesting positive funding.
2. **Donchian/ATR trend breakout** – Turtle-style trend follower with ATR risk controls.
3. **Liquidation/open-interest squeeze** – momentum entries into dense liquidation bands with OI confirmation.
4. **VWAP reversion with session bands** – intraday mean-reversion fades toward session VWAP.
5. **Funding-rate contrarian swings** – low-frequency swings on extreme funding z-scores versus momentum.
6. **Order-book imbalance with CVD confirmation** – microstructure scalps using depth imbalance and CVD divergence.

Each preset describes timeframe, leverage guidelines, entry/exit rules, automation notes, and supporting references so UIs or automation tooling can surface them directly.

### Record Paper Trading Outcomes for Learning

```bash
python -m trading_bot.main paper-record \
  --strategy donchian_atr_breakout \
  --features 0.75,0.62,0.55,0.40 \
  --outcome 1 \
  --regime trend \
  --reward 0.018
```

This command appends the completed paper trade to `data/paper_trades.jsonl`. Subsequent `train` runs automatically fold these samples into the dataset and expose their aggregate statistics to the selector.

### Request a Strategy Recommendation

```bash
python -m trading_bot.main suggest-strategy \
  --funding-rate -0.004 \
  --price-momentum -0.2 \
  --volatility 0.35 \
  --open-interest-change -0.05 \
  --orderbook-imbalance -0.1
```

The CLI prints the recommended preset, detected regime, and paper-trade metrics that influenced the decision, providing transparency into how the AI routes strategies.

### Run the API Service

```bash
uvicorn trading_bot.services.api:app --reload
```

The API exposes:
- `GET /health`
- `POST /toggle-paper`
- `POST /signals`

## Next Steps

Review the operational hardening checklist in [`docs/next_steps.md`](docs/next_steps.md) for guidance on
environment provisioning, historical data preparation, paper-trading validation, and production rollout
procedures.

## Publishing to GitHub

If you cloned this workspace without an origin remote configured, the code will only exist locally. To make
the repository visible on GitHub:

1. Create an empty repository on GitHub (for example `your-user/ai-trading-bot`).
2. Add the GitHub remote to this project:

   ```bash
   git remote add origin git@github.com:your-user/ai-trading-bot.git
   ```

   Replace the URL with either the SSH or HTTPS URL provided by GitHub.
3. Push the current branch (named `work` in this workspace) and set it as the default upstream:

   ```bash
   git push -u origin work
   ```

   If you prefer `main` on GitHub, rename the branch locally before pushing:

   ```bash
   git branch -m work main
   git push -u origin main
   ```

After pushing, refresh the GitHub repository page and you should see all committed code and documentation.

## Compliance Notes
- The BitMEX connector uses only public endpoints and authenticated requests documented in BitMEX's API reference.
- Users must review BitMEX's Terms of Service and confirm jurisdictional eligibility before enabling mainnet trading.
- API keys are loaded from environment variables and never stored in the codebase.
ai_trading_bot_codebase.zip
New
Binary file not shown
artifacts/strategy_model.json
New
+5
-0

{
  "training_samples": 1024,
  "feature_dim": 16,
  "paper_trading_samples": 0
}
artifacts/strategy_model.pt
New
+27
-0

{
  "weights": [
    0.004722831833000512,
    -0.014067514508752908,
    -0.0056576721966890165,
    -0.007914829265242384,
    0.006341490126284249,
    0.005005827043695184,
    0.009452162600984547,
    -0.012733922142830843,
    -0.0019392656620221993,
    -0.01487486824916894,
    -0.009678335873838881,
    0.0014157895297066062,
    -0.015655875405115552,
    -0.01015320864619951,
    0.005208767676640779,
    0.0016704262704299827
  ],
  "bias": -0.006418313493719415,
  "config": {
    "feature_dim": 16,
    "epochs": 10,
    "batch_size": 64,
    "learning_rate": 0.001
  }
}
data/training_dataset.pkl
New
Binary file not shown
docs/ai_model_plan.md
New
+17
-0

# AI Model Design Plan

## Objectives
- Optimize for risk-adjusted return (e.g., Sharpe ratio) while maintaining a strong win rate.

## Feature Strategy
- Incorporate multi-time-frame candle features (e.g., 1m/5m/15m) for context across short- and medium-term horizons.
- Engineer order book–derived signals (depth imbalance, liquidity shifts, spread dynamics) to capture microstructure effects.

## Data Coverage & Timeframes (Recommended)
- Ingest **36 months** of historical BitMEX market data to capture multiple leverage cycles, liquidity regimes, and exchange microstructure shifts.
- Maintain overlapping candle sets at **1m, 5m, 15m, 1h, 4h, and 1d** resolutions, engineered with rolling windows (e.g., 20/50/200-period moving statistics) for both training and inference.
- Pair each live candle update with synchronized Level 2 order book snapshots aggregated at **100ms** intervals, deriving imbalance, queue position, and sweep metrics over **1s, 5s, and 30s** lookbacks.
- Refresh training datasets quarterly with the latest quarter of data while retaining at least two full years for model retraining and validation.

## Next Steps
- Await stakeholder confirmation on the recommended coverage/timeframe plan before finalizing feature engineering and model selection.
docs/architecture_plan.md
New
+51
-0

# AI-Powered Crypto Trading Bot Architecture Plan

## 1. Overview
A modular microservices architecture optimized for fully automated leveraged trading on BitMEX (with the ability to extend to additional exchanges) while supporting both desktop and cloud deployment. Emphasis is placed on low-latency execution, strong risk controls, and secure handling of API keys.

## 2. Service Topology
| Service | Responsibilities | Tech Stack | Deployment Notes |
| --- | --- | --- | --- |
| Data Ingestion Service | Subscribe to BitMEX WebSocket streams for trades, order book, and instrument metadata. Normalize to unified schema, enrich with computed indicators, store in TimescaleDB and Redis. | Python (FastAPI workers), AsyncIO, WebSockets, TimescaleDB client (psycopg2/SQLAlchemy), Redis-py. | Containerized worker replicas; horizontal scaling by symbol partitioning. |
| AI Strategy Service | Consume normalized data, run ML models (transformers, reinforcement learning), apply feature pipelines, generate trade signals. Includes backtesting harness for paper trading mode. | Python, PyTorch, scikit-learn, Celery for distributed jobs. | GPU-enabled optional; autoscale via Kubernetes HPA. |
| Risk & Compliance Service | Evaluate leverage, exposure, stop-loss/take-profit thresholds, volatility filters, compliance checks. Maintains account state snapshots. | Python, FastAPI, Pydantic for rule definitions, Redis streams. | Runs redundantly; persists risk events to PostgreSQL. |
| Execution Service | Translate approved signals into BitMEX REST/WebSocket orders, manage order lifecycle, enforce rate limiting and retries, handle position monitoring. | Python, AsyncIO, HTTPX, websockets, redis-queue. | Deploy in pairs (active/standby). Requires secure secret injection. |
| Configuration & API Gateway | REST/GraphQL API for user configuration, strategy deployment, paper/live toggle, audit logs. Authentication & RBAC. | FastAPI (Python) or Node.js (NestJS). PostgreSQL ORM (SQLAlchemy). | Fronts all services; integrates with OAuth provider if desired. |
| UI/Dashboard | Web SPA providing live metrics, strategy controls, risk alerts, and historical performance. | React + TypeScript, Material UI/Recharts, WebSocket dashboards. | Hosted via static CDN or container. |
| Messaging Backbone | Decouples services via pub/sub for events and commands. | Redis Streams (initially) with option to upgrade to Kafka. | Highly available Redis cluster. |

## 3. Data Storage Layer
- **TimescaleDB (Primary Time-Series)**: Candlesticks, order book snapshots, trade executions, indicator values. Partitioned by symbol & timeframe with retention policies.
- **PostgreSQL (Metadata & Config)**: Strategy definitions, user profiles, encrypted API keys (using libsodium-based envelope encryption), audit logs, backtest results.
- **Redis (Caching & Queues)**: Signal buffers, rate-limit counters, distributed locks, inter-service pub/sub.
- **Object Storage (Historical Archive/ML Datasets)**: S3-compatible storage for raw tick dumps, model artifacts, and training datasets with versioning.

## 4. Security & Compliance
- Secrets managed via Vault or AWS KMS; injected at runtime (no hardcoding). Local desktop deployments use an encrypted keystore (libsodium). 
- Role-based access control enforced in the API Gateway.
- TLS termination at ingress (Traefik/NGINX) with mTLS between internal services for cloud deployments.
- Logging & audit trails persisted to PostgreSQL and forwarded to ELK/OpenSearch for analysis.
- Compliance checks for jurisdictional restrictions before enabling live trading.

## 5. Deployment Targets
- **Desktop**: Docker Compose stack orchestrating all services; GPU pass-through optional for AI training. Local `.env` provides secrets path references.
- **Cloud**: Kubernetes cluster (EKS/GKE). Helm charts define deployments, ConfigMaps, Secrets, HPA policies, and stateful sets for databases. Utilize managed DB (RDS/Timescale Cloud) and Redis (ElastiCache/Upstash).

## 6. Observability
- Prometheus metrics exporters for each service; Grafana dashboards tailored to trading KPIs (PnL, Sharpe, drawdown, latency).
- Centralized logging via Fluent Bit → OpenSearch/ELK.
- Alerting rules for latency spikes, order rejections, risk limit breaches, and service health.

## 7. Extensibility
- Exchange connectors abstracted behind unified adapter interface to facilitate onboarding Binance Futures, Bitget, Bybit, OKX, Kraken Pro, Coinbase Advanced once TOS compliance verified.
- Strategy plug-in architecture supporting rule-based, ML, and reinforcement learning modules.
- Modular risk rules to incorporate additional controls (e.g., circuit breakers, daily loss limits).

## 8. Next Steps
1. Specify exchange API integration details (BitMEX first, followed by priority order of others).
2. Define AI model selection, data requirements, and training workflow.
3. Detail risk management logic and parameterization.
4. Design execution engine flow diagrams and state machines.
5. Outline UI/dashboard wireframes and API contracts.

Please confirm or request changes to this architecture so we can proceed to the next phase.
docs/exchange_api_plan.md
New
+31
-0

# BitMEX Exchange Integration Plan

## Environment Support
- Connectors must operate against both **BitMEX Testnet** and **Mainnet** environments.
- Environment selection is configurable at runtime; credentials are stored encrypted and never hardcoded.
- Shared codepath handles authentication, with domain-specific REST/WebSocket endpoints injected per environment.

## Order Functionality
- Core support for **Limit**, **Stop**, and **Take-Profit** order types.
- Each order submission enforces leverage bounds of **10x to 20x** per the project scope.
- Orders default to reduce-only when closing existing positions; configurable flags expose BitMEX-specific options (e.g., `execInst`).
- Order lifecycle management includes placement, amendment, cancellation, and status tracking via the WebSocket `order` stream.

## Connectivity & Data
- REST client covers trading (`/order`, `/position`, `/execution`) and account endpoints required for post-trade reconciliation.
- WebSocket client subscribes to live feeds for `orderBookL2`, `trade`, `position`, `margin`, and `order` channels to maintain real-time state.
- Automatic reconnection and exponential backoff handle transient disconnects while respecting BitMEX rate limits.
- Snapshot reconciliation pulls a **50x50 order book depth** (50 bids/50 asks) from `/orderBook/L2` every **5 seconds**, or immediately after a connection drop, to realign the in-memory book with BitMEX's canonical state without overwhelming bandwidth.

## Rate Limiting & Throttling
- Shared limiter module tracks REST and WebSocket burst limits, aligning with BitMEX's request weight scheme.
- Redis-based distributed counters ensure consistency across execution service replicas.
- Circuit breaker halts new order submissions when limit thresholds are approached, notifying the risk service.

## Testing & Simulation
- Unified test harness runs against the Testnet sandbox before enabling Mainnet.
- Mock adapters simulate REST/WebSocket responses to support offline unit tests.
- Paper-trading mode reuses live market data while routing orders to the simulator queue instead of BitMEX.

## Outstanding Clarifications
- None at this time; proceeding under the assumption that default reduce-only handling satisfies optional BitMEX order flag requirements.
docs/execution_engine_plan.md
New
+33
-0

# Execution Engine Design Plan

## Routing Redundancy & Failover Strategy
- Operate a **primary** and **secondary** execution worker per exchange environment (Testnet/Mainnet).
- Workers share a Redis-backed distributed lock so only one submits orders at a time; the standby monitors heartbeats and assumes control within <2 seconds if the primary fails.
- Maintain idempotent order intents stored in PostgreSQL so a takeover never duplicates submissions during failover.
- Use health-checked WebSocket connectivity; stale data beyond 3 seconds automatically triggers a controlled switchover to the standby.

## Core Responsibilities
- Consume normalized signals from the AI Strategy service via Kafka topics (`signals.live`, `signals.paper`).
- Enforce leverage and position sizing rules before issuing orders, delegating to the Risk service for cross-checks.
- Translate abstract intents into BitMEX REST/WebSocket commands, supporting limit, stop, and take-profit orders with reduce-only semantics when closing positions.
- Persist execution audit logs (intent, exchange response, fills) in PostgreSQL for traceability.

## Latency & Throughput Targets
- Sub-second signal-to-order latency (p50 < 500 ms, p95 < 900 ms) when market connectivity is healthy.
- Burst handling for up to 50 orders per minute while remaining inside BitMEX request weight limits.
- Priority queueing ensures cancel/replace actions pre-empt new entries when the risk service issues overrides.

## Reliability & Observability
- Heartbeat metrics emitted every second (`execution.heartbeat`, `execution.lag`) and scraped by Prometheus.
- Structured logs (JSON) forwarded to the central ELK stack with correlation IDs linking strategy signals, risk approvals, and exchange responses.
- Alert thresholds: no successful submission for 60 seconds, heartbeat lag > 2 seconds, or REST error rate > 5% over 1 minute triggers PagerDuty notifications.

## Deployment & Configuration
- Packaged as a stateless container with configuration supplied via environment variables (API keys read from Vault at startup).
- Supports paper-trading and live-trading modes via a configuration flag; paper mode routes intents to the simulator queue while still exercising latency metrics.
- Canary deployments: roll out updates to standby workers first, observe for anomalies, then promote to primary.

## Testing Approach
- Integration tests replay captured WebSocket streams to validate failover and idempotent order handling.
- Chaos testing injects simulated worker crashes and network partitions to ensure seamless standby promotion.
- Load tests verify throughput targets while coordinating with the rate-limiter module shared with the exchange connector.
docs/next_steps.md
New
+35
-0

# Next Steps for the AI Trading Bot

This guide outlines the recommended activities after cloning the repository and reviewing the architecture plans. It focuses on hardening the system prior to live execution and ensuring that all compliance and operational safeguards are in place.

## 1. Environment Preparation
- Provision Redis, PostgreSQL/TimescaleDB, and object storage according to `docs/architecture_plan.md`.
- Populate the `.env` file with BitMEX testnet credentials and database connection strings.
- Bootstrap database schemas and create encryption keys for storing API secrets at rest.

## 2. Data Acquisition
- Configure the `DataIngestionService` to write WebSocket events into TimescaleDB and Redis queues.
- Backfill historical candles and order-book snapshots using BitMEX bulk data APIs for the last 36 months.
- Validate data completeness with integrity checks (row counts per interval, gap detection routines).

## 3. Model Training Pipeline
- Transform the historical dataset into the tensor format expected by `MarketDataset`.
- Run `python -m trading_bot.main train` with representative hyperparameters and track metrics (loss, Sharpe, win rate) in experiment logs.
- Schedule quarterly retraining jobs and archive model artifacts to object storage.

## 4. Paper Trading Shakeout
- Keep `PAPER_TRADING=true` and replay live market feeds through the orchestrator.
- Monitor order flow, latency, and risk-manager decisions in Grafana/Prometheus dashboards.
- Iterate on strategy thresholds before promoting any configuration to mainnet.

## 5. Live Trading Readiness Checklist
- Perform BitMEX Testnet end-to-end tests: entry, stop-loss, and take-profit orders across leverage settings.
- Review legal and compliance requirements for your jurisdiction; obtain explicit consent before enabling live keys.
- Establish alerting (PagerDuty, Slack) for connectivity loss, risk breaches, or failed health checks.

## 6. Production Rollout
- Switch to `PAPER_TRADING=false` only after sign-off from compliance stakeholders.
- Enable the standby execution worker and verify failover handling via simulated outages.
- Document operational runbooks for incident response and recovery.

Following this checklist helps ensure the trading stack is thoroughly validated before any real capital is deployed.
docs/push_to_github.md
New
+46
-0

# Publishing the Repository to GitHub

The execution environment where this project was prepared does not have access to your GitHub account or any network credentials, so it cannot push code on your behalf. To publish the repository, perform the following steps from your local machine or any environment that has GitHub access:

1. **Clone or copy the project locally**
   ```bash
   git clone <path-to-this-workspace> ai-trading-bot
   cd ai-trading-bot
   ```
   If you already have the project locally, simply change into the repository directory.

2. **Create an empty repository on GitHub**
   Use the GitHub web interface to create a new repository (e.g., `your-user/ai-trading-bot`). Leave it empty—do not initialize with a README or `.gitignore`.

3. **Add the GitHub remote**
   ```bash
   git remote add origin git@github.com:your-user/ai-trading-bot.git
   ```
   Substitute the SSH URL with the HTTPS version if you prefer:
   ```bash
   git remote add origin https://github.com/your-user/ai-trading-bot.git
   ```

4. **Push the existing branch**
   If you want to keep the current branch name (`work`):
   ```bash
   git push -u origin work
   ```
   To publish the code on a `main` branch instead:
   ```bash
   git branch -m work main
   git push -u origin main
   ```

5. **Verify on GitHub**
   Refresh the GitHub repository page. The commits, files, and history from this project should now appear online.

6. **(Optional) Share the ZIP archive**
   The root of the repository contains `ai_trading_bot_codebase.zip`. Upload this file to GitHub Releases or distribute it directly if you need to provide a single-file download.

Once the remote is configured, subsequent updates only require committing your changes locally and running:
```bash
git push
```

If you encounter authentication issues, ensure that your SSH keys or personal access tokens are configured according to GitHub’s documentation.
docs/risk_management_plan.md
New
+27
-0

# Risk Management Plan

## 1. Objectives
Establish deterministic controls that cap downside risk, enforce disciplined leverage usage, and guarantee every automated trade targets at least a three-to-one reward-to-risk ratio between take-profit and stop-loss levels.

## 2. Core Constraints (Confirmed)
- **Minimum Reward-to-Risk Ratio**: Each strategy-generated order must set take-profit targets no closer than three times the distance of the stop-loss from the entry price. If a signal cannot satisfy the 3:1 ratio given current volatility or exchange constraints, it is rejected.

## 3. Exposure Limits (Confirmed)
- **Position Sizing**: Risk per trade capped at 1% of account equity (configurable).
- **Daily Loss Limit**: Trading halts once cumulative realized losses reach 4% of equity for the day; manual reset required.
- **Max Concurrent Positions**: Limit simultaneous open positions per instrument to one directional trade to avoid correlated drawdowns.

## 4. Volatility & Liquidity Filters (Confirmed)
- **ATR-Based Stop Calibration**: Minimum stop distance = 2× 14-period ATR on the dominant timeframe to avoid noise-triggered exits while respecting the 3:1 ratio for profit targets.
- **Order Book Depth Check**: Ensure aggregated quote size at intended entry price supports at least 3× the order quantity within 10 bps slippage; otherwise reduce size or skip trade.

## 5. Leverage Governance (Confirmed)
- **Dynamic Leverage Scaling**: Default leverage 10×, scalable up to 20× only when trailing 30-trade Sharpe ratio > 1.0 and drawdown < 5%.
- **Circuit Breaker**: If unrealized PnL drawdown exceeds 8% of equity, reduce leverage to 5× and shrink position sizing by half until recovery.

## 6. Compliance & Monitoring (Confirmed)
- **Risk Engine Enforcement**: Execution service must call Risk & Compliance API before order placement; API validates ratio, exposure, and leverage rules.
- **Audit Trails**: All rejected orders log rationale (e.g., insufficient reward-to-risk) to PostgreSQL for review.
- **Alerts**: Notify operators via dashboard and webhook when trades are skipped or when cumulative losses approach thresholds.

The user approved the above safeguards in the "lets proceed" session, allowing the team to proceed to execution engine design.
docs/ui_dashboard_plan.md
New
+52
-0

# UI & Dashboard Design Plan

## Core Objectives
- Provide real-time visibility into bot status, market conditions, and account performance for both paper and live trading modes.
- Enable safe manual overrides (pause, resume, flatten) without bypassing automated risk controls.
- Surface actionable analytics that align with the AI strategy's risk-adjusted return and win-rate targets.

## Key Views & Widgets
- **Global Overview Panel**
  - Environment switcher (Paper/Testnet/Mainnet) with status badges.
  - System health summary (data feeds, AI strategy, execution workers) driven by Prometheus metrics.
  - High-level equity curve sparkline with configurable timeframe shortcuts.
- **Open Positions & Orders Table**
  - Displays symbol, side, size, entry price, mark price, leverage, stop-loss, take-profit, and time-in-position.
  - Shows per-position floating P&L in both absolute and percentage terms, color-coded for instant recognition.
  - Inline controls for closing, adjusting stops/targets, or toggling reduce-only behavior (subject to risk guardrails).
- **AI Signal History**
  - Chronological list of recent strategy outputs with confidence scores, feature flags, and execution outcome (filled, rejected, cancelled).
  - Supports filtering by symbol, signal class (trend, mean reversion, volatility breakout), and confidence threshold.
- **Risk Monitor**
  - Displays current leverage usage vs. configured caps, active circuit breakers, and recent risk alerts.
  - Visualizes stop-loss vs. take-profit distances to enforce the minimum 3:1 ratio requirement.
- **Account Performance Tabs**
  - Tabbed summary for Day, Month, 6 Months, Year, and Lifetime horizons.
  - Each tab includes aggregated P&L, realized vs. unrealized breakdowns, win rate, Sharpe-like risk-adjusted metric, and drawdown stats.
  - Provide downloadable CSV reports per tab for compliance and analysis.

## Data & Refresh Cadence
- WebSocket channel from API service pushes updates for positions, orders, balances, and health metrics every second.
- Historical performance data fetched via REST with caching and delta updates to minimize payload sizes.
- All timestamps normalized to UTC and rendered in the user's local timezone on the client.

## Interaction & Controls
- Global kill-switch requiring MFA confirmation to halt all trading and cancel outstanding orders.
- Manual order entry modal for emergency hedging, inheriting default risk parameters and enforcing max leverage.
- Notification center displaying success/error toasts and persistent alerts for unresolved risk events.

## Technology Recommendations
- Frontend: React + TypeScript with component library (e.g., Material UI) for rapid layout of tables, charts, and tabs.
- State management via React Query for server synchronization and Zustand for local UI state.
- Backend API exposes GraphQL for flexible querying of nested metrics; subscriptions power real-time widgets.
- Charting using Recharts or TradingView Lightweight Charts for performance metrics and price overlays.

## Security & Compliance
- Enforce role-based access (viewer, trader, admin) with JWT-backed sessions stored securely in HttpOnly cookies.
- Sensitive actions (manual overrides, kill-switch) require step-up authentication (TOTP or WebAuthn challenge).
- Audit log view within the dashboard listing user actions, timestamp, and affected resources.

## Testing & Monitoring
- Component-level unit tests with React Testing Library and Jest, covering data tables, tabs, and modals.
- End-to-end workflows scripted in Playwright to validate real-time updates, manual overrides, and account summary accuracy.
- Frontend performance budgets (Time to Interactive < 3s on broadband) monitored via automated Lighthouse runs in CI.
pytest.ini
New
+3
-0

[pytest]
addopts = -q
pythonpath = src
requirements.txt
New
+9
-0

fastapi==0.110.2
httpx==0.27.0
python-dotenv==1.0.1
redis==5.0.3
SQLAlchemy==2.0.29
asyncpg==0.29.0
websockets==12.0
torch==2.2.2
uvicorn==0.29.0
src/__init__.py
New

No content
src/trading_bot/__init__.py
New

No content
src/trading_bot/config/settings.py
New
+66
-0

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Optional


@dataclass
class Settings:
    environment: Literal["development", "production", "test"] = "development"
    bitmex_api_key: Optional[str] = None
    bitmex_api_secret: Optional[str] = None
    bitmex_environment: Literal["testnet", "mainnet"] = "testnet"
    leverage_min: float = 10.0
    leverage_max: float = 20.0
    risk_reward_min_ratio: float = 3.0
    paper_trading: bool = True
    paper_trading_store: str = "data/paper_trades.jsonl"
    redis_url: str = "redis://localhost:6379/0"
    postgres_dsn: str = "postgresql+asyncpg://trader:trader@localhost:5432/trading_bot"
    timescale_dsn: str = "postgresql+asyncpg://trader:trader@localhost:5432/trading_market"

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.getenv("ENVIRONMENT", "development")
        bitmex_env = os.getenv("BITMEX_ENVIRONMENT", "testnet")
        leverage_min = float(os.getenv("LEVERAGE_MIN", 10))
        leverage_max = float(os.getenv("LEVERAGE_MAX", 20))
        risk_ratio = float(os.getenv("RISK_REWARD_MIN_RATIO", 3))
        paper_trading = os.getenv("PAPER_TRADING", "true").lower() in {"1", "true", "yes"}
        paper_store = os.getenv("PAPER_TRADING_STORE", "data/paper_trades.jsonl")

        if leverage_max < leverage_min:
            raise ValueError("leverage_max must be >= leverage_min")
        if leverage_max > 25:
            raise ValueError("leverage_max exceeds supported upper bound (25x)")
        if risk_ratio < 3:
            raise ValueError("Risk-reward ratio must be >= 3:1 as per specification")

        return cls(
            environment=env,  # type: ignore[arg-type]
            bitmex_api_key=os.getenv("BITMEX_API_KEY"),
            bitmex_api_secret=os.getenv("BITMEX_API_SECRET"),
            bitmex_environment=bitmex_env,  # type: ignore[arg-type]
            leverage_min=leverage_min,
            leverage_max=leverage_max,
            risk_reward_min_ratio=risk_ratio,
            paper_trading=paper_trading,
            paper_trading_store=paper_store,
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            postgres_dsn=os.getenv(
                "POSTGRES_DSN", "postgresql+asyncpg://trader:trader@localhost:5432/trading_bot"
            ),
            timescale_dsn=os.getenv(
                "TIMESCALE_DSN", "postgresql+asyncpg://trader:trader@localhost:5432/trading_market"
            ),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


__all__ = ["Settings", "get_settings"]
src/trading_bot/data/ingestion_service.py
New
+56
-0

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from trading_bot.config.settings import get_settings
from trading_bot.exchanges.bitmex import BitmexClient

logger = logging.getLogger(__name__)


class DataIngestionService:
    """Streams live market data and persists snapshots into storage backends."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = BitmexClient(environment=self.settings.bitmex_environment)
        self._task: asyncio.Task[None] | None = None
        self._running = asyncio.Event()

    async def start(self) -> None:
        if self._task and not self._task.done():
            logger.warning("Data ingestion already running")
            return
        self._running.set()
        self._task = asyncio.create_task(self._run())
        logger.info("Data ingestion service started")

    async def stop(self) -> None:
        self._running.clear()
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        await self.client.close()
        logger.info("Data ingestion service stopped")

    async def _run(self) -> None:
        queue = await self.client.stream(
            ["instrument:XBTUSD", "trade:XBTUSD", "orderBookL2_25:XBTUSD"]
        )
        while self._running.is_set():
            try:
                message = await asyncio.wait_for(queue.get(), timeout=5)
            except asyncio.TimeoutError:
                continue
            await self.process_message(message)

    async def process_message(self, message: dict[str, Any]) -> None:
        logger.debug("Received message: %s", message)
        # TODO: persist to TimescaleDB/Redis. Stub for now.


__all__ = ["DataIngestionService"]
src/trading_bot/data/paper_trading_repository.py
New
+126
-0

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from trading_bot.config.settings import get_settings


@dataclass(frozen=True)
class PaperTradeSample:
    """Represents the outcome of a completed paper trade."""

    strategy_slug: str
    features: List[float]
    outcome: float
    market_regime: str
    reward: float


@dataclass(frozen=True)
class StrategyPerformance:
    """Aggregated statistics for a strategy within a regime."""

    win_rate: float
    avg_reward: float
    trades: int


class PaperTradingRepository:
    """Lightweight persistence layer for paper trading samples."""

    def __init__(self, path: Path | None = None) -> None:
        settings = get_settings()
        default_path = Path(settings.paper_trading_store)
        self.path = path or default_path

    def add_sample(self, sample: PaperTradeSample) -> None:
        """Append a paper trade sample to the backing store."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(sample)) + "\n")

    def load_samples(self) -> List[PaperTradeSample]:
        """Return all recorded paper trading samples."""

        if not self.path.exists():
            return []
        samples: List[PaperTradeSample] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                samples.append(
                    PaperTradeSample(
                        strategy_slug=payload["strategy_slug"],
                        features=[float(value) for value in payload["features"]],
                        outcome=float(payload["outcome"]),
                        market_regime=payload["market_regime"],
                        reward=float(payload.get("reward", 0.0)),
                    )
                )
        return samples

    def to_dataset(self) -> Tuple[List[List[float]], List[float]]:
        """Convert recorded trades into training features and labels."""

        samples = self.load_samples()
        features = [sample.features for sample in samples]
        targets = [sample.outcome for sample in samples]
        return features, targets

    def compute_statistics(self) -> Dict[str, Dict[str, StrategyPerformance]]:
        """Aggregate performance metrics by strategy and regime."""

        samples = self.load_samples()
        summary: Dict[str, Dict[str, Dict[str, float]]] = {}
        for sample in samples:
            strategy_bucket = summary.setdefault(sample.strategy_slug, {})
            regime_bucket = strategy_bucket.setdefault(
                sample.market_regime, {"wins": 0.0, "trades": 0.0, "reward": 0.0}
            )
            regime_bucket["trades"] += 1
            regime_bucket["reward"] += sample.reward
            if sample.outcome >= 0.5:
                regime_bucket["wins"] += 1

            overall_bucket = strategy_bucket.setdefault(
                "__overall__", {"wins": 0.0, "trades": 0.0, "reward": 0.0}
            )
            overall_bucket["trades"] += 1
            overall_bucket["reward"] += sample.reward
            if sample.outcome >= 0.5:
                overall_bucket["wins"] += 1

        result: Dict[str, Dict[str, StrategyPerformance]] = {}
        for strategy_slug, buckets in summary.items():
            strat_stats: Dict[str, StrategyPerformance] = {}
            for regime, values in buckets.items():
                trades = int(values["trades"])
                if trades == 0:
                    continue
                win_rate = values["wins"] / trades
                avg_reward = values["reward"] / trades
                strat_stats[regime] = StrategyPerformance(
                    win_rate=win_rate,
                    avg_reward=avg_reward,
                    trades=trades,
                )
            result[strategy_slug] = strat_stats
        return result

    def clear(self) -> None:
        """Utility used by tests to reset the repository."""

        if self.path.exists():
            self.path.unlink()


__all__ = [
    "PaperTradeSample",
    "StrategyPerformance",
    "PaperTradingRepository",
]

src/trading_bot/exchanges/bitmex.py
New
+122
-0

from __future__ import annotations

import asyncio
import hmac
import json
import time
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Dict, Optional

import httpx
import websockets

from trading_bot.config.settings import get_settings


BITMEX_REST_BASE = {
    "testnet": "https://testnet.bitmex.com",
    "mainnet": "https://www.bitmex.com",
}
BITMEX_WS_BASE = {
    "testnet": "wss://ws.testnet.bitmex.com/realtime",
    "mainnet": "wss://ws.bitmex.com/realtime",
}


@dataclass
class BitmexOrder:
    symbol: str
    side: str
    order_qty: float
    ord_type: str
    price: Optional[float] = None
    stop_px: Optional[float] = None
    cl_ord_id: Optional[str] = None
    exec_inst: Optional[str] = None
    reduce_only: bool = False

    def as_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        aliases = {
            "order_qty": "orderQty",
            "ord_type": "ordType",
            "stop_px": "stopPx",
            "cl_ord_id": "clOrdID",
            "exec_inst": "execInst",
            "reduce_only": "reduceOnly",
        }
        transformed = {}
        for key, value in payload.items():
            if value is None:
                continue
            transformed[aliases.get(key, key)] = value
        return transformed


class BitmexClient:
    def __init__(self, *, environment: Optional[str] = None, timeout: float = 10.0) -> None:
        settings = get_settings()
        self.environment = environment or settings.bitmex_environment
        self.base_url = BITMEX_REST_BASE[self.environment]
        self.ws_url = BITMEX_WS_BASE[self.environment]
        self.api_key = settings.bitmex_api_key
        self.api_secret = settings.bitmex_api_secret
        self.timeout = timeout
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    def _auth_headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        if not self.api_key or not self.api_secret:
            raise RuntimeError("API key/secret required for authenticated BitMEX call")
        expires = int(time.time()) + 30
        message = method + path + str(expires) + body
        signature = hmac.new(
            self.api_secret.encode("utf-8"), message.encode("utf-8"), sha256
        ).hexdigest()
        return {
            "api-expires": str(expires),
            "api-key": self.api_key,
            "api-signature": signature,
            "content-type": "application/json",
        }

    async def place_order(self, order: BitmexOrder) -> dict[str, Any]:
        payload = order.as_payload()
        headers = self._auth_headers("POST", "/api/v1/order", json.dumps(payload))
        response = await self._client.post("/api/v1/order", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        payload = {"orderID": order_id}
        headers = self._auth_headers("DELETE", "/api/v1/order", json.dumps(payload))
        response = await self._client.delete(
            "/api/v1/order", json=payload, headers=headers
        )
        response.raise_for_status()
        return response.json()

    async def fetch_positions(self) -> list[dict[str, Any]]:
        headers = self._auth_headers("GET", "/api/v1/position")
        response = await self._client.get("/api/v1/position", headers=headers)
        response.raise_for_status()
        return response.json()

    async def stream(self, topics: list[str]) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        async def _listen() -> None:
            async with websockets.connect(self.ws_url) as ws:
                sub_request = {"op": "subscribe", "args": topics}
                await ws.send(json.dumps(sub_request))
                async for message in ws:
                    queue.put_nowait(json.loads(message))

        asyncio.create_task(_listen())
        return queue


__all__ = ["BitmexClient", "BitmexOrder"]
src/trading_bot/execution/execution_service.py
New
+59
-0

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Optional

from trading_bot.config.settings import get_settings
from trading_bot.exchanges.bitmex import BitmexClient, BitmexOrder
from trading_bot.risk.risk_manager import RiskAssessment, RiskManager, TradeSignal

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    accepted: bool
    order_id: Optional[str] = None
    reason: Optional[str] = None


class ExecutionService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = BitmexClient(environment=self.settings.bitmex_environment)
        self.risk_manager = RiskManager()
        self.paper_mode = self.settings.paper_trading

    async def execute_signal(self, signal: TradeSignal, account_equity: float) -> ExecutionResult:
        assessment: RiskAssessment = self.risk_manager.evaluate(signal, account_equity)
        if not assessment.valid:
            return ExecutionResult(False, reason=assessment.reason)

        if self.paper_mode:
            simulated_order_id = f"paper-{uuid.uuid4()}"
            logger.info("Paper trade order generated: %s", simulated_order_id)
            return ExecutionResult(True, order_id=simulated_order_id)

        order = BitmexOrder(
            symbol=signal.symbol,
            side="Buy" if signal.side.lower() == "buy" else "Sell",
            order_qty=signal.size,
            ord_type="Limit",
            price=signal.entry_price,
            stop_px=signal.stop_loss,
            cl_ord_id=str(uuid.uuid4()),
            exec_inst="ReduceOnly" if signal.side.lower() == "sell" else None,
            reduce_only=signal.side.lower() == "sell",
        )
        response = await self.client.place_order(order)
        order_id = response.get("orderID")
        return ExecutionResult(True, order_id=order_id)

    async def close(self) -> None:
        await self.client.close()


__all__ = ["ExecutionService", "ExecutionResult"]
src/trading_bot/main.py
New
+225
-0

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from trading_bot.data.paper_trading_repository import PaperTradeSample
from trading_bot.strategy.presets import get_strategy_presets
from trading_bot.strategy.selector import StrategyContext


def run_training(args: argparse.Namespace) -> None:
    from trading_bot.services.training_pipeline import train_model, TrainingConfig

    config = TrainingConfig(feature_dim=args.feature_dim, epochs=args.epochs)
    train_model(Path(args.data), Path(args.output), config)


def run_live(args: argparse.Namespace) -> None:
    from trading_bot.services.trading_orchestrator import TradingOrchestrator
    from trading_bot.risk.risk_manager import TradeSignal

    orchestrator = TradingOrchestrator()

    async def _run() -> None:
        strategy_slug = args.strategy
        if strategy_slug == "auto":
            context_args = [
                args.funding_rate,
                args.price_momentum,
                args.volatility,
                args.open_interest_change,
                args.orderbook_imbalance,
            ]
            if any(value is None for value in context_args):
                raise ValueError(
                    "Auto strategy selection requires funding-rate, price-momentum, "
                    "volatility, open-interest-change, and orderbook-imbalance"
                )
            context = StrategyContext(
                funding_rate=args.funding_rate,
                price_momentum=args.price_momentum,
                volatility=args.volatility,
                open_interest_change=args.open_interest_change,
                orderbook_imbalance=args.orderbook_imbalance,
            )
            decision = orchestrator.select_strategy(context)
            strategy_slug = decision.strategy.slug
            print(
                "Selected strategy:"
                f" {decision.strategy.name} ({decision.strategy.slug}) —"
                f" regime={decision.regime}, win_rate={decision.win_rate:.2f},"
                f" expected_reward={decision.expected_reward:.4f}"
            )
        signal = TradeSignal(
            symbol=args.symbol,
            side=args.side,
            entry_price=args.entry,
            stop_loss=args.stop,
            take_profit=args.take,
            confidence=0.5,
            size=args.size,
            strategy_slug=strategy_slug,
        )
        await orchestrator.process_signal(signal)
        await orchestrator.stop()

    asyncio.run(_run())


def run_list_presets(_: argparse.Namespace) -> None:
    for preset in get_strategy_presets():
        print(f"[{preset.slug}] {preset.name} — {preset.category}")
        print(f"  Timeframe: {preset.timeframe}")
        print(f"  Leverage: {preset.leverage}")
        print(f"  Entry: {preset.entry}")
        print(f"  Stop: {preset.stop_loss}")
        print(f"  Take profit: {preset.take_profit}")
        print(f"  Exit: {preset.exit}")
        print(f"  Automation: {preset.automation}")
        print(f"  References: {', '.join(preset.references)}")
        print()


def run_paper_record(args: argparse.Namespace) -> None:
    from trading_bot.services.trading_orchestrator import TradingOrchestrator

    orchestrator = TradingOrchestrator()
    features = [float(value) for value in args.features.split(",") if value.strip()]
    if not features:
        raise ValueError("At least one feature value must be provided")
    if not 0.0 <= args.outcome <= 1.0:
        raise ValueError("Outcome must be between 0 and 1 (inclusive)")
    sample = PaperTradeSample(
        strategy_slug=args.strategy,
        features=features,
        outcome=args.outcome,
        market_regime=args.regime,
        reward=args.reward,
    )
    orchestrator.record_paper_trade(sample)
    print(
        f"Recorded paper trade for {args.strategy} with outcome={args.outcome} "
        f"and reward={args.reward}"
    )


def run_suggest_strategy(args: argparse.Namespace) -> None:
    from trading_bot.services.trading_orchestrator import TradingOrchestrator

    orchestrator = TradingOrchestrator()
    context = StrategyContext(
        funding_rate=args.funding_rate,
        price_momentum=args.price_momentum,
        volatility=args.volatility,
        open_interest_change=args.open_interest_change,
        orderbook_imbalance=args.orderbook_imbalance,
    )
    decision = orchestrator.select_strategy(context)
    strategy = decision.strategy
    print(f"Recommended strategy: {strategy.name} ({strategy.slug})")
    print(f"Category: {strategy.category}; regime: {decision.regime}")
    print(f"Win rate: {decision.win_rate:.2f}; Expected reward: {decision.expected_reward:.4f}")
    print(f"Description: {strategy.description}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Trading Bot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    strategy_slugs = [preset.slug for preset in get_strategy_presets()]

    train_parser = subparsers.add_parser("train", help="Train the AI model")
    train_parser.add_argument("--data", required=True, help="Path to training dataset")
    train_parser.add_argument("--output", required=True, help="Output directory for the model")
    train_parser.add_argument("--feature-dim", type=int, required=True, help="Input feature dimension")
    train_parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    train_parser.set_defaults(func=run_training)

    live_parser = subparsers.add_parser("live", help="Execute a single live signal")
    live_parser.add_argument("--symbol", required=True)
    live_parser.add_argument("--side", choices=["Buy", "Sell"], required=True)
    live_parser.add_argument("--entry", type=float, required=True)
    live_parser.add_argument("--stop", type=float, required=True)
    live_parser.add_argument("--take", type=float, required=True)
    live_parser.add_argument("--size", type=float, required=True)
    live_parser.add_argument(
        "--strategy",
        choices=["auto", *strategy_slugs],
        default="auto",
        help="Strategy slug to execute or 'auto' to select dynamically",
    )
    live_parser.add_argument("--funding-rate", type=float, help="Current funding rate (decimal)")
    live_parser.add_argument("--price-momentum", type=float, help="Normalized price momentum")
    live_parser.add_argument("--volatility", type=float, help="Realized volatility score")
    live_parser.add_argument(
        "--open-interest-change",
        type=float,
        help="Normalized change in open interest",
    )
    live_parser.add_argument(
        "--orderbook-imbalance",
        type=float,
        help="Order book imbalance score (bid-ask skew)",
    )
    live_parser.set_defaults(func=run_live)

    preset_parser = subparsers.add_parser(
        "list-strategies", help="Display available strategy presets"
    )
    preset_parser.set_defaults(func=run_list_presets)

    paper_parser = subparsers.add_parser(
        "paper-record", help="Record the outcome of a paper trade for learning"
    )
    paper_parser.add_argument("--strategy", choices=strategy_slugs, required=True)
    paper_parser.add_argument(
        "--features",
        required=True,
        help="Comma separated feature vector captured during the trade",
    )
    paper_parser.add_argument(
        "--outcome",
        type=float,
        required=True,
        help="Trade outcome (1 for win, 0 for loss)",
    )
    paper_parser.add_argument(
        "--regime",
        default="custom",
        choices=[
            "carry",
            "trend",
            "squeeze",
            "mean_reversion",
            "funding_contrarian",
            "microstructure",
            "custom",
        ],
        help="Market regime observed during the paper trade",
    )
    paper_parser.add_argument(
        "--reward",
        type=float,
        default=0.0,
        help="Risk-adjusted reward achieved on the trade",
    )
    paper_parser.set_defaults(func=run_paper_record)

    suggest_parser = subparsers.add_parser(
        "suggest-strategy", help="Recommend a strategy for the current context"
    )
    suggest_parser.add_argument("--funding-rate", type=float, required=True)
    suggest_parser.add_argument("--price-momentum", type=float, required=True)
    suggest_parser.add_argument("--volatility", type=float, required=True)
    suggest_parser.add_argument("--open-interest-change", type=float, required=True)
    suggest_parser.add_argument("--orderbook-imbalance", type=float, required=True)
    suggest_parser.set_defaults(func=run_suggest_strategy)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
src/trading_bot/risk/risk_manager.py
New
+49
-0

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from trading_bot.config.settings import get_settings


@dataclass
class TradeSignal:
    symbol: str
    side: str  # "Buy" or "Sell"
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    size: float
    strategy_slug: str | None = None


@dataclass
class RiskAssessment:
    valid: bool
    reason: Optional[str] = None


class RiskManager:
    def __init__(self) -> None:
        self.settings = get_settings()

    def evaluate(self, signal: TradeSignal, account_equity: float) -> RiskAssessment:
        if signal.take_profit <= signal.entry_price and signal.side.lower() == "buy":
            return RiskAssessment(False, "Take profit must be above entry for long positions")
        if signal.take_profit >= signal.entry_price and signal.side.lower() == "sell":
            return RiskAssessment(False, "Take profit must be below entry for short positions")

        risk = abs(signal.entry_price - signal.stop_loss)
        reward = abs(signal.take_profit - signal.entry_price)
        if reward / max(risk, 1e-8) < self.settings.risk_reward_min_ratio:
            return RiskAssessment(False, "Risk/reward ratio below mandated minimum")

        leverage = signal.size * signal.entry_price / max(account_equity, 1e-8)
        if leverage < self.settings.leverage_min or leverage > self.settings.leverage_max:
            return RiskAssessment(False, "Leverage outside allowed range (10x-20x)")

        return RiskAssessment(True)


__all__ = ["RiskManager", "TradeSignal", "RiskAssessment"]
src/trading_bot/services/api.py
New
+37
-0

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

from trading_bot.config.settings import Settings, get_settings
from trading_bot.risk.risk_manager import TradeSignal
from trading_bot.services.trading_orchestrator import TradingOrchestrator

app = FastAPI(title="AI Trading Bot API")
_orchestrator = TradingOrchestrator()


@app.on_event("shutdown")
async def shutdown() -> None:
    await _orchestrator.stop()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/toggle-paper")
async def toggle_paper(settings: Settings = Depends(get_settings)) -> dict[str, bool]:
    settings.paper_trading = not settings.paper_trading
    return {"paperTrading": settings.paper_trading}


@app.post("/signals")
async def submit_signal(signal: TradeSignal) -> dict[str, str]:
    result = await _orchestrator.process_signal(signal)
    if not result.accepted:
        raise HTTPException(status_code=400, detail=result.reason or "order rejected")
    return {"status": "accepted", "orderId": result.order_id or "paper"}


__all__ = ["app"]
src/trading_bot/services/trading_orchestrator.py
New
+74
-0

from __future__ import annotations

import logging
from dataclasses import dataclass

from trading_bot.config.settings import get_settings
from trading_bot.data.ingestion_service import DataIngestionService
from trading_bot.data.paper_trading_repository import PaperTradeSample, PaperTradingRepository
from trading_bot.execution.execution_service import ExecutionService, ExecutionResult
from trading_bot.risk.risk_manager import TradeSignal
from trading_bot.strategy.selector import StrategyContext, StrategyDecision, StrategySelector

logger = logging.getLogger(__name__)


@dataclass
class AccountState:
    equity: float


class TradingOrchestrator:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.ingestion = DataIngestionService()
        self.execution = ExecutionService()
        self.paper_repository = PaperTradingRepository()
        self.strategy_selector = StrategySelector(repository=self.paper_repository)
        self._ingestion_started = False

    async def ensure_running(self) -> None:
        if not self._ingestion_started:
            await self.ingestion.start()
            self._ingestion_started = True

    async def process_signal(self, signal: TradeSignal) -> ExecutionResult:
        await self.ensure_running()
        account = await self._load_account_state()
        result = await self.execution.execute_signal(signal, account.equity)
        self._handle_execution_result(signal, result)
        return result

    async def stop(self) -> None:
        if self._ingestion_started:
            await self.ingestion.stop()
            self._ingestion_started = False
        await self.execution.close()

    async def _load_account_state(self) -> AccountState:
        # Placeholder for PostgreSQL fetch
        return AccountState(equity=10000.0)

    def _handle_execution_result(self, signal: TradeSignal, result: ExecutionResult) -> None:
        if result.accepted:
            logger.info(
                "Order accepted for %s (id=%s, strategy=%s)",
                signal.symbol,
                result.order_id,
                signal.strategy_slug or "unspecified",
            )
        else:
            logger.warning("Order rejected for %s: %s", signal.symbol, result.reason)

    def select_strategy(self, context: StrategyContext) -> StrategyDecision:
        """Return the recommended strategy for the provided market context."""

        return self.strategy_selector.select(context)

    def record_paper_trade(self, sample: PaperTradeSample) -> None:
        """Persist a completed paper trade for future learning."""

        self.paper_repository.add_sample(sample)


__all__ = ["TradingOrchestrator", "AccountState"]
src/trading_bot/services/training_pipeline.py
New
+45
-0

from __future__ import annotations

import pickle
from pathlib import Path

from trading_bot.config.settings import get_settings
from trading_bot.data.paper_trading_repository import PaperTradingRepository
from trading_bot.strategy.ai_model import MarketDataset, StrategyModel, TrainingConfig


def create_dataset(path: Path) -> MarketDataset:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    features = data["features"]
    targets = data["targets"]
    return MarketDataset(features, targets)


def train_model(data_path: Path, output_dir: Path, config: TrainingConfig) -> Path:
    dataset = create_dataset(data_path)
    settings = get_settings()
    paper_count = 0
    if settings.paper_trading:
        repository = PaperTradingRepository()
        extra_features, extra_targets = repository.to_dataset()
        paper_count = len(extra_targets)
        if extra_features:
            combined_features = dataset.features + extra_features
            combined_targets = dataset.targets + extra_targets
            dataset = MarketDataset(combined_features, combined_targets)
    model = StrategyModel(config)
    model.train(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "strategy_model.pt"
    model.save(model_path)
    metadata = {
        "training_samples": len(dataset),
        "feature_dim": config.feature_dim,
        "paper_trading_samples": paper_count,
    }
    model.save_metadata(output_dir / "strategy_model.json", metadata)
    return model_path


__all__ = ["train_model", "create_dataset"]
src/trading_bot/strategy/ai_model.py
New
+110
-0

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


class MarketDataset:
    def __init__(self, features: Sequence[Sequence[float]], targets: Sequence[float]) -> None:
        if len(features) != len(targets):
            raise ValueError("Features and targets must have the same length")
        self.features: List[List[float]] = [list(map(float, sample)) for sample in features]
        self.targets: List[float] = [float(label) for label in targets]
        expected_dim: int | None = None
        for sample in self.features:
            if expected_dim is None:
                expected_dim = len(sample)
            elif len(sample) != expected_dim:
                raise ValueError("All feature vectors must share the same dimension")
        self.feature_dim: int = expected_dim or 0

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Tuple[List[float], float]:
        return self.features[index], self.targets[index]


@dataclass
class TrainingConfig:
    feature_dim: int
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3


class StrategyModel:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.weights: List[float] = [0.0 for _ in range(config.feature_dim)]
        self.bias: float = 0.0

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)

    def _dot(self, features: Sequence[float]) -> float:
        return sum(w * x for w, x in zip(self.weights, features)) + self.bias

    @staticmethod
    def _binary_cross_entropy(pred: float, target: float) -> float:
        epsilon = 1e-12
        pred = min(max(pred, epsilon), 1.0 - epsilon)
        return -(target * math.log(pred) + (1.0 - target) * math.log(1.0 - pred))

    def _batch_indices(self, total: int) -> Iterable[Tuple[int, int]]:
        batch = self.config.batch_size
        for start in range(0, total, batch):
            yield start, min(start + batch, total)

    def train(self, dataset: MarketDataset) -> None:
        if dataset.feature_dim and dataset.feature_dim != self.config.feature_dim:
            raise ValueError(
                "Dataset feature dimension does not match training configuration"
            )
        for epoch in range(self.config.epochs):
            total_loss = 0.0
            for start, end in self._batch_indices(len(dataset)):
                batch_features = dataset.features[start:end]
                batch_targets = dataset.targets[start:end]
                gradients = [0.0 for _ in self.weights]
                bias_gradient = 0.0
                batch_loss = 0.0
                for features, target in zip(batch_features, batch_targets):
                    prediction = self._sigmoid(self._dot(features))
                    error = prediction - target
                    batch_loss += self._binary_cross_entropy(prediction, target)
                    for index, value in enumerate(features):
                        gradients[index] += error * value
                    bias_gradient += error
                batch_size = max(1, len(batch_features))
                lr = self.config.learning_rate / batch_size
                self.weights = [w - lr * g for w, g in zip(self.weights, gradients)]
                self.bias -= lr * bias_gradient
                total_loss += batch_loss
            avg_loss = total_loss / max(len(dataset), 1)
            print(f"Epoch {epoch + 1}/{self.config.epochs} - Loss: {avg_loss:.4f}")

    def predict(self, features: Sequence[float]) -> float:
        return self._sigmoid(self._dot(features))

    def save(self, path: Path) -> None:
        payload = {
            "weights": self.weights,
            "bias": self.bias,
            "config": self.config.__dict__,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")

    def save_metadata(self, path: Path, metadata: dict[str, float]) -> None:
        path.write_text(json.dumps(metadata, indent=2))


__all__ = ["StrategyModel", "TrainingConfig", "MarketDataset"]
src/trading_bot/strategy/presets.py
New
+240
-0

"""Predefined trading strategy configurations.

These presets capture the core execution rules, risk guidance, and
automation notes for a curated list of strategies confirmed by the user.
The intent is to make the strategies discoverable at runtime so that
CLI tools, dashboards, or future configuration UIs can expose them as
drop-in options without requiring hard coded copies in multiple places.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class StrategyPreset:
    """Immutable description of a trading strategy preset."""

    slug: str
    name: str
    category: str
    timeframe: str
    leverage: str
    description: str
    entry: str
    stop_loss: str
    take_profit: str
    exit: str
    automation: str
    references: List[str]


_PRESETS: Dict[str, StrategyPreset] = {
    "perp_basis_carry": StrategyPreset(
        slug="perp_basis_carry",
        name="Perpetual Funding/Basis Cash-and-Carry",
        category="Market-neutral carry",
        timeframe="Multi-day to multi-week",
        leverage="Low isolated leverage per leg; avoid cross",
        description=(
            "Delta-neutral program that pairs long spot exposure with a short "
            "perpetual swap whenever funding annualises above a configurable "
            "hurdle (8–12% APR typical)."
        ),
        entry=(
            "Monitor venue and aggregated sources (e.g., Coinglass) and open the "
            "hedge when funding exceeds the hurdle for three consecutive 8h "
            "periods while open interest trends higher without a large spot "
            "premium (>0.5%)."
        ),
        stop_loss=(
            "No directional stop; instead employ a kill switch if borrow fails, "
            "funding turns negative for multiple windows, or counterparty risk "
            "rises. Maintain isolated margin to avoid liquidation bleed."
        ),
        take_profit=(
            "Unwind when the 7-day rolling funding APR falls below ~6% or "
            "carry no longer covers borrow/execution costs."
        ),
        exit="Close the perp leg first, then release the spot hedge to lock carry.",
        automation=(
            "Schedule funding polling, delta hedging, and alerts around borrow "
            "events. Ensure API errors trigger an immediate flattening routine."
        ),
        references=[
            "BitMEX Blog funding studies",
            "BIS crypto carry analysis",
            "Coinglass funding datasets",
        ],
    ),
    "donchian_atr_breakout": StrategyPreset(
        slug="donchian_atr_breakout",
        name="Donchian/ATR Trend Breakout",
        category="Trend following",
        timeframe="4H to Daily",
        leverage="Max 2–3× isolated",
        description=(
            "Breakout system inspired by the classic Turtle rules, combining "
            "Donchian channel highs with a 200 EMA filter and ATR-based "
            "stops to capture sustained trends on BitMEX."
        ),
        entry=(
            "Enter when price closes above the 20-period Donchian high and the "
            "200 EMA, with rising open interest and stable funding."
        ),
        stop_loss="Place the stop 2×ATR(14) below entry or at the opposite channel edge.",
        take_profit="Trail exits using a 10-period Donchian channel or 1×ATR. Optional partials at 2R.",
        exit="Disable pyramiding by default to reduce whipsaw risk; exit on stop or trailing signal.",
        automation=(
            "Derive signals via TA library or TradingView alerts feeding the execution "
            "API. Include ADX>25 or similar volatility filter and full fee modelling."
        ),
        references=[
            "Richard Dennis Turtle methodology",
            "TradingView Donchian breakout scripts",
            "Community crypto backtests (TradeSearcher)",
        ],
    ),
    "liquidation_squeeze": StrategyPreset(
        slug="liquidation_squeeze",
        name="Liquidation/Open-Interest Squeeze",
        category="Flow-driven momentum",
        timeframe="1–15 minute scalps",
        leverage="Isolated 1–3× only",
        description=(
            "Trades forced liquidations by targeting dense liquidation pools "
            "highlighted by heatmap services when momentum and open interest "
            "accelerate into the levels."
        ),
        entry=(
            "Identify nearby liquidation clusters via Hyblock/Coinglass, confirm "
            "open interest expansion and cumulative volume delta alignment, then "
            "enter toward the pool."
        ),
        stop_loss="Protect behind the most recent micro-structure swing or imbalance.",
        take_profit="Scale out at the targeted liquidation band; trail with VWAP if extension continues.",
        exit=(
            "Abort if open interest stalls, funding flips in trade direction, or "
            "news volatility spikes."
        ),
        automation=(
            "Continuously fetch liquidation maps, throttle entries around events, "
            "and auto-cancel when pre-trade conditions fail."
        ),
        references=[
            "Hyblock Capital liquidation tools",
            "Coinglass open interest data",
            "Community execution tutorials",
        ],
    ),
    "vwap_reversion": StrategyPreset(
        slug="vwap_reversion",
        name="VWAP Reversion with Session Bands",
        category="Intraday mean reversion",
        timeframe="1–15 minute",
        leverage="1–2× isolated",
        description=(
            "Intraday fade that sells/buys extremes beyond VWAP bands when the "
            "market is range-bound, targeting a return to daily VWAP."
        ),
        entry=(
            "Enter against price when it tags ±k×VWAP bands with ADX < 20 and "
            "neutral funding, avoiding strong higher-timeframe trends."
        ),
        stop_loss="Place the stop just outside the band using an ATR buffer (~0.5×ATR).",
        take_profit="Target the VWAP re-test; optional partial at midpoint band.",
        exit=(
            "Stand aside after two consecutive losses and when trend filters signal "
            "a directional session."
        ),
        automation=(
            "Reset VWAP each session, maintain regime filters, and log outcomes "
            "for risk review."
        ),
        references=[
            "VWAP trading guides",
            "TradingOnramp session studies",
            "Practitioner walkthroughs (Learn with Sreenivas Doosa)",
        ],
    ),
    "funding_contrarian": StrategyPreset(
        slug="funding_contrarian",
        name="Funding-Rate Contrarian Swings",
        category="Sentiment extremes",
        timeframe="4H to Daily swings",
        leverage="1–2× isolated",
        description=(
            "Contrarian swing setup that fades extreme funding-rate z-scores when "
            "price momentum diverges, positioning for mean reversion."
        ),
        entry=(
            "Compute 24–72h funding z-scores; enter opposite the crowd when z-score "
            "exceeds ±2σ and price momentum disagrees while open interest diverges."
        ),
        stop_loss="Set a wide stop beyond the prior swing structure to allow for noise.",
        take_profit="Exit on funding mean reversion or when price momentum realigns.",
        exit="Skip signals around major news and respect low-frequency cadence.",
        automation=(
            "Aggregate venue funding data, calculate z-scores, and require OI/taker "
            "flow confirmation before dispatching orders."
        ),
        references=[
            "BitMEX funding research",
            "99Bitcoins market notes",
            "CryptoQuant funding divergence briefs",
        ],
    ),
    "orderbook_imbalance": StrategyPreset(
        slug="orderbook_imbalance",
        name="Order-Book Imbalance with CVD",
        category="Microstructure scalping",
        timeframe="Seconds to minutes",
        leverage="Small size, isolated only",
        description=(
            "High-frequency style approach that acts on queue imbalance and "
            "cumulative volume delta divergences to anticipate short-term "
            "moves."
        ),
        entry=(
            "Engage when price stalls while CVD diverges or when best-level depth "
            "imbalance breaches predefined thresholds in the trade direction."
        ),
        stop_loss="Place tight stops just beyond the observed imbalance or micro swing.",
        take_profit="Exit quickly on CVD flip or at the next micro liquidity pocket.",
        exit="If data quality degrades or imbalance normalises, flatten immediately.",
        automation=(
            "Stream L2 order book data, maintain rolling imbalance metrics, and "
            "disable during exchange instability."
        ),
        references=[
            "Bookmap order-flow education",
            "Microstructure research hubs",
            "Community CVD strategy repos",
        ],
    ),
}


def get_strategy_presets() -> Iterable[StrategyPreset]:
    """Return an iterable over all registered strategy presets."""

    return _PRESETS.values()


def get_strategy_preset(slug: str) -> StrategyPreset:
    """Fetch a preset by slug, raising KeyError if it does not exist."""

    try:
        return _PRESETS[slug]
    except KeyError as exc:  # pragma: no cover - defensive clause
        raise KeyError(f"Unknown strategy preset: {slug}") from exc


__all__ = [
    "StrategyPreset",
    "get_strategy_presets",
    "get_strategy_preset",
]

src/trading_bot/strategy/selector.py
New
+146
-0

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from trading_bot.data.paper_trading_repository import (
    PaperTradingRepository,
    StrategyPerformance,
)
from trading_bot.strategy.presets import StrategyPreset, get_strategy_preset


@dataclass(frozen=True)
class StrategyContext:
    """Summarises the current market regime for strategy selection."""

    funding_rate: float
    price_momentum: float
    volatility: float
    open_interest_change: float
    orderbook_imbalance: float

    def derive_regime(self) -> str:
        """Map raw features to a coarse market regime label."""

        if abs(self.orderbook_imbalance) > 0.6 and self.volatility >= 0.5:
            return "microstructure"
        if self.funding_rate > 0.01 and self.price_momentum <= 0.5:
            return "carry"
        if abs(self.price_momentum) > 0.8 and self.open_interest_change * self.price_momentum > 0:
            return "trend"
        if abs(self.funding_rate) > 0.01 and self.price_momentum * self.funding_rate < 0:
            return "funding_contrarian"
        if self.volatility < 0.4 and abs(self.price_momentum) < 0.4:
            return "mean_reversion"
        if self.volatility > 0.9 and self.open_interest_change > 0.5:
            return "squeeze"
        return "trend"


@dataclass(frozen=True)
class StrategyDecision:
    """Represents the selector's recommendation for a strategy."""

    strategy: StrategyPreset
    regime: str
    score: float
    win_rate: float
    expected_reward: float


class StrategySelector:
    """Heuristic selector that blends paper-trade stats with context rules."""

    _REGIME_DEFAULTS: Dict[str, str] = {
        "carry": "perp_basis_carry",
        "trend": "donchian_atr_breakout",
        "squeeze": "liquidation_squeeze",
        "mean_reversion": "vwap_reversion",
        "funding_contrarian": "funding_contrarian",
        "microstructure": "orderbook_imbalance",
    }

    def __init__(self, repository: PaperTradingRepository | None = None) -> None:
        self.repository = repository or PaperTradingRepository()

    def select(self, context: StrategyContext) -> StrategyDecision:
        regime = context.derive_regime()
        stats = self.repository.compute_statistics()
        base_scores = self._baseline_scores(context)
        adjusted_scores = self._adjust_scores(base_scores, stats, regime)
        best_slug, best_score = self._pick_best(regime, adjusted_scores)
        preset = get_strategy_preset(best_slug)
        performance = stats.get(best_slug, {}).get(regime)
        if performance is None:
            performance = stats.get(best_slug, {}).get("__overall__")
        win_rate = performance.win_rate if performance else 0.5
        expected_reward = performance.avg_reward if performance else 0.0
        return StrategyDecision(
            strategy=preset,
            regime=regime,
            score=best_score,
            win_rate=win_rate,
            expected_reward=expected_reward,
        )

    def _baseline_scores(self, context: StrategyContext) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        scores["perp_basis_carry"] = max(0.0, context.funding_rate) * 400
        scores["donchian_atr_breakout"] = max(
            0.0, abs(context.price_momentum) * 80 + context.open_interest_change * 40
        )
        scores["liquidation_squeeze"] = max(
            0.0, context.volatility * 60 + context.open_interest_change * 50
        )
        scores["vwap_reversion"] = max(0.0, (0.6 - abs(context.price_momentum)) * 70)
        scores["funding_contrarian"] = max(
            0.0, abs(context.funding_rate) * 90 - context.price_momentum * context.funding_rate * 50
        )
        scores["orderbook_imbalance"] = max(
            0.0, abs(context.orderbook_imbalance) * 120 + context.volatility * 20
        )
        return scores

    def _adjust_scores(
        self,
        scores: Dict[str, float],
        stats: Dict[str, Dict[str, StrategyPerformance]],
        regime: str,
    ) -> Dict[str, float]:
        adjusted: Dict[str, float] = {}
        preferred_slug = self._REGIME_DEFAULTS.get(regime)
        for slug, score in scores.items():
            performance = stats.get(slug, {}).get(regime)
            if performance is None:
                performance = stats.get(slug, {}).get("__overall__")
            if performance:
                multiplier = 0.75 + performance.win_rate
                score *= multiplier
                score += performance.avg_reward * 100
            if preferred_slug:
                if slug == preferred_slug:
                    score *= 1.5
                else:
                    score *= 0.6
            adjusted[slug] = score
        return adjusted

    def _pick_best(self, regime: str, scores: Dict[str, float]) -> Tuple[str, float]:
        candidate_slug = self._REGIME_DEFAULTS.get(regime, "donchian_atr_breakout")
        best_slug = candidate_slug
        best_score = scores.get(candidate_slug, 0.0)
        for slug, score in scores.items():
            if slug == candidate_slug:
                continue
            if score > best_score * 1.2:
                best_slug, best_score = slug, score
        return best_slug, best_score


__all__ = [
    "StrategyContext",
    "StrategyDecision",
    "StrategySelector",
]

tests/test_ai_model.py
New
+29
-0

import pytest

from trading_bot.strategy.ai_model import MarketDataset, StrategyModel, TrainingConfig


def test_market_dataset_requires_uniform_dimensions():
    with pytest.raises(ValueError):
        MarketDataset(features=[[1.0, 2.0], [1.0, 2.0, 3.0]], targets=[1.0, 0.0])


def test_strategy_model_validates_feature_dimension():
    dataset = MarketDataset(features=[[1.0, 2.0]], targets=[1.0])
    model = StrategyModel(TrainingConfig(feature_dim=3, epochs=1))
    with pytest.raises(ValueError):
        model.train(dataset)


def test_strategy_model_trains_with_matching_dimensions(capsys):
    dataset = MarketDataset(
        features=[[0.0, 0.0], [1.0, 1.0]],
        targets=[0.0, 1.0],
    )
    model = StrategyModel(TrainingConfig(feature_dim=2, epochs=1, learning_rate=0.1))
    model.train(dataset)
    captured = capsys.readouterr()
    assert "Epoch 1/1" in captured.out
    prediction_low = model.predict([0.0, 0.0])
    prediction_high = model.predict([1.0, 1.0])
    assert prediction_low < prediction_high
tests/test_paper_trading_repository.py
New
+55
-0

from __future__ import annotations

from pathlib import Path

from trading_bot.data.paper_trading_repository import (
    PaperTradeSample,
    PaperTradingRepository,
)


def test_repository_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "paper.jsonl"
    repository = PaperTradingRepository(path)
    repository.add_sample(
        PaperTradeSample(
            strategy_slug="donchian_atr_breakout",
            features=[0.1, 0.2, 0.3],
            outcome=1.0,
            market_regime="trend",
            reward=0.02,
        )
    )
    repository.add_sample(
        PaperTradeSample(
            strategy_slug="donchian_atr_breakout",
            features=[0.2, 0.1, 0.4],
            outcome=0.0,
            market_regime="trend",
            reward=-0.01,
        )
    )
    repository.add_sample(
        PaperTradeSample(
            strategy_slug="vwap_reversion",
            features=[0.05, -0.1, 0.2],
            outcome=1.0,
            market_regime="mean_reversion",
            reward=0.015,
        )
    )

    features, targets = repository.to_dataset()
    assert features == [[0.1, 0.2, 0.3], [0.2, 0.1, 0.4], [0.05, -0.1, 0.2]]
    assert targets == [1.0, 0.0, 1.0]

    stats = repository.compute_statistics()
    trend_stats = stats["donchian_atr_breakout"]["trend"]
    assert trend_stats.trades == 2
    assert trend_stats.win_rate == 0.5
    assert trend_stats.avg_reward == 0.005

    mean_rev_stats = stats["vwap_reversion"]["mean_reversion"]
    assert mean_rev_stats.trades == 1
    assert mean_rev_stats.win_rate == 1.0
    assert mean_rev_stats.avg_reward == 0.015
tests/test_risk_manager.py
New
+36
-0

from trading_bot.risk.risk_manager import RiskManager, TradeSignal


def test_risk_manager_rejects_low_ratio(monkeypatch):
    manager = RiskManager()

    signal = TradeSignal(
        symbol="XBTUSD",
        side="Buy",
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        confidence=0.8,
        size=1.0,
    )

    result = manager.evaluate(signal, account_equity=1000)
    assert not result.valid
    assert "Risk/reward" in result.reason


def test_risk_manager_accepts_valid_trade(monkeypatch):
    manager = RiskManager()

    signal = TradeSignal(
        symbol="XBTUSD",
        side="Buy",
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=115.0,
        confidence=0.9,
        size=100.0,
    )

    result = manager.evaluate(signal, account_equity=1000)
    assert result.valid
tests/test_strategy_presets.py
New
+29
-0

from trading_bot.strategy.presets import (
    StrategyPreset,
    get_strategy_preset,
    get_strategy_presets,
)


def test_all_presets_exposed() -> None:
    presets = list(get_strategy_presets())
    slugs = {preset.slug for preset in presets}
    expected = {
        "perp_basis_carry",
        "donchian_atr_breakout",
        "liquidation_squeeze",
        "vwap_reversion",
        "funding_contrarian",
        "orderbook_imbalance",
    }

    assert slugs == expected
    assert all(isinstance(preset, StrategyPreset) for preset in presets)


def test_lookup_returns_same_instance() -> None:
    preset = get_strategy_preset("perp_basis_carry")
    assert preset.name.startswith("Perpetual")

    again = get_strategy_preset("perp_basis_carry")
    assert preset is again
tests/test_strategy_selector.py
New
+66
-0

from __future__ import annotations

from pathlib import Path

from trading_bot.data.paper_trading_repository import (
    PaperTradeSample,
    PaperTradingRepository,
)
from trading_bot.strategy.selector import StrategyContext, StrategySelector


def test_selector_uses_regime_and_performance(tmp_path: Path) -> None:
    path = tmp_path / "paper.jsonl"
    repository = PaperTradingRepository(path)
    repository.add_sample(
        PaperTradeSample(
            strategy_slug="perp_basis_carry",
            features=[0.01, 0.0],
            outcome=1.0,
            market_regime="carry",
            reward=0.012,
        )
    )
    repository.add_sample(
        PaperTradeSample(
            strategy_slug="donchian_atr_breakout",
            features=[0.9, 0.5],
            outcome=1.0,
            market_regime="trend",
            reward=0.045,
        )
    )
    repository.add_sample(
        PaperTradeSample(
            strategy_slug="donchian_atr_breakout",
            features=[0.85, 0.55],
            outcome=0.0,
            market_regime="trend",
            reward=-0.01,
        )
    )

    selector = StrategySelector(repository=repository)

    carry_context = StrategyContext(
        funding_rate=0.015,
        price_momentum=0.2,
        volatility=0.3,
        open_interest_change=0.1,
        orderbook_imbalance=0.1,
    )
    carry_decision = selector.select(carry_context)
    assert carry_decision.strategy.slug == "perp_basis_carry"
    assert carry_decision.regime == "carry"

    trend_context = StrategyContext(
        funding_rate=0.005,
        price_momentum=0.95,
        volatility=1.2,
        open_interest_change=0.8,
        orderbook_imbalance=0.2,
    )
    trend_decision = selector.select(trend_context)
    assert trend_decision.strategy.slug == "donchian_atr_breakout"
    assert trend_decision.regime == "trend"
    assert trend_decision.win_rate > 0.0

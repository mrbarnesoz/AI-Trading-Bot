# Monitoring Setup

The trading bot exposes Prometheus metrics for both offline backtests and the
live controller. Metrics are published via the `prometheus-client` library and
can be scraped directly into Grafana or any other Prometheus-compatible
visualisation stack.

## Quick start

1. Run a backtest (or walk-forward) and expose metrics on port **9000**:

   ```powershell
   python scripts/backtest.py --metrics-port 9000
   ```

   or for walk-forward validation:

   ```powershell
   python scripts/backtest_walk_forward.py --train-days 120 --test-days 30 --metrics-port 9000
   ```

2. For live trading / paper trading:

   ```powershell
   python scripts/live_run.py --symbols XBTUSD --metrics-port 9001 --testnet --dry-run --initial-equity 10000
   ```

3. Point Prometheus to the exposed endpoints. Example `prometheus.yml`:

   ```yaml
   scrape_configs:
     - job_name: "ai-trading-bot-backtest"
       static_configs:
         - targets: ["localhost:9000"]
     - job_name: "ai-trading-bot-live"
       static_configs:
         - targets: ["localhost:9001"]
   ```

4. Import `monitoring/grafana_dashboard.json` into Grafana to get a starter
   dashboard with equity, drawdown, Sharpe, and per-symbol PnL panels.

## Metric reference

| Metric name                | Labels              | Description                                 |
|--------------------------- |-------------------- |---------------------------------------------|
| `trading_equity`           | `mode`              | Equity level (`backtest`, `walk_forward`, `live`). |
| `trading_drawdown`         | `mode`              | Absolute drawdown (positive value).          |
| `trading_sharpe`           | `mode`              | Sharpe ratio for the given mode.             |
| `trading_symbol_pnl`       | `mode`, `symbol`    | Mark-to-market PnL per symbol.               |

The live controller updates these gauges on every bar (and whenever explicit
account state updates are provided). Backtest scripts push the final summary of
each run.

## Notes

- The starter dashboard is intentionally minimal; customise it by layering
  additional PromQL expressions (e.g., rolling averages or histograms of trade
  frequency).
- When running multiple bot instances, provide unique ports (or run a reverse
  proxy) so Prometheus can scrape each instance independently.

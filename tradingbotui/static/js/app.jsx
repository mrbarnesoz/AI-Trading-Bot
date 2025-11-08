const { useState, useEffect, useMemo, useCallback } = React;

if (typeof dayjs !== "undefined" && typeof dayjs_plugin_relativeTime !== "undefined") {
  dayjs.extend(dayjs_plugin_relativeTime);
}

const POLL_INTERVAL_FAST = 8000;
const POLL_INTERVAL_SLOW = 15000;
const DEFAULT_TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"];
const DEFAULT_CAPITAL_OPTIONS = [5, 10, 15, 20, 25, 50, 75, 100];
const DEFAULT_SYMBOLS = ["XBTUSD", "ETHUSD", "SOLUSD", "XRPUSD", "ADAUSD", "DOGEUSD"];
const DEFAULT_STRATEGIES = ["trend_follow", "mean_reversion", "breakout", "scalper", "adaptive_meta"];
const DEFAULT_CONFIG_PLACEHOLDERS = [
  { name: "config.yaml", path: "config.yaml", warnings: [], updated_at: null },
];

function clsx(...tokens) {
  return tokens.filter(Boolean).join(" ");
}

function usePolling(url, interval = POLL_INTERVAL_SLOW, initialValue = null) {
  const [data, setData] = useState(initialValue);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchData = useCallback(async () => {
    try {
      const response = await fetch(url, { credentials: "same-origin" });
      if (!response.ok) {
        throw new Error(`Request failed (${response.status})`);
      }
      const payload = await response.json();
      setData(payload);
      setError(null);
    } catch (err) {
      console.error("Polling error for", url, err);
      const message = err && typeof err.message === "string" ? err.message : "";
      if (message.includes("(404)")) {
        setData(initialValue);
        setError(null);
      } else {
        setError(err);
      }
    } finally {
      setLoading(false);
    }
  }, [url]);

  useEffect(() => {
    let active = true;
    if (!active) {
      return undefined;
    }
    fetchData();
    if (interval <= 0) {
      return () => {
        active = false;
      };
    }
    const timer = setInterval(fetchData, interval);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [fetchData, interval]);

  return { data, error, loading, refresh: fetchData };
}

async function apiRequest(path, { method = "POST", body = undefined } = {}) {
  const options = {
    method,
    headers: {
      "Content-Type": "application/json",
    },
    credentials: "same-origin",
  };
  if (body !== undefined) {
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  const text = await response.text();
  const trimmed = text ? text.trim() : "";
  let payload = {};
  if (trimmed) {
    try {
      payload = JSON.parse(trimmed);
    } catch (err) {
      console.warn("Non-JSON response from", path, trimmed);
      payload = { raw: trimmed };
    }
  }
  if (!response.ok) {
    const message =
      (payload && payload.error) ||
      (payload && payload.message) ||
      (payload && payload.raw) ||
      `Request failed (${response.status})`;
    const details =
      payload && Array.isArray(payload.details) && payload.details.length
        ? payload.details.join("; ")
        : "";
    const fullMessage = details ? `${message}: ${details}` : message;
    throw new Error(fullMessage);
  }
  return payload;
}

function describeResultTimestamp(value) {
  if (!value) {
    return "";
  }
  if (typeof dayjs === "undefined") {
    return `Generated at ${value}`;
  }
  const parsed = dayjs(value);
  if (!parsed.isValid()) {
    return `Generated at ${value}`;
  }
  return `Generated ${parsed.fromNow()}`;
}

function NotificationBar({ alerts, messages, onDismiss }) {
  const items = [];
  if (Array.isArray(alerts)) {
    alerts.forEach((alert, idx) => {
      items.push({
        id: `alert-${idx}-${alert.message}`,
        level: alert.level || "info",
        text: alert.message || JSON.stringify(alert),
        sticky: true,
      });
    });
  }
  messages.forEach((msg) => {
    items.push(msg);
  });

  if (!items.length) {
    return null;
  }

  return (
    <div className="sticky top-0 z-30 shadow-lg">
      {items.map((item) => {
        const colour =
          item.level === "critical"
            ? "bg-red-900 border-red-500"
            : item.level === "warning"
            ? "bg-amber-900 border-amber-500"
            : item.level === "success"
            ? "bg-emerald-900 border-emerald-500"
            : "bg-slate-800 border-slate-600";
        return (
          <div
            key={item.id}
            className={clsx(
              "flex items-center justify-between px-4 py-2 border-b text-sm",
              colour
            )}
          >
            <span className="font-medium">{item.text}</span>
            {!item.sticky && (
              <button
                className="text-xs text-slate-200 hover:text-white"
                onClick={() => onDismiss(item.id)}
              >
                Dismiss
              </button>
            )}
          </div>
        );
      })}
    </div>
  );
}

function PortfolioSummary({ portfolio }) {
  if (!portfolio) {
    return null;
  }
  const metricCells = [
    { label: "PnL (30d)", value: portfolio.pnl_30d },
    { label: "Sharpe (30d)", value: portfolio.sharpe },
    { label: "Drawdown", value: portfolio.drawdown },
    { label: "Win Rate", value: portfolio.win_rate },
    { label: "Trades", value: portfolio.trades },
  ].filter((item) => item.value !== undefined && item.value !== null);

  if (!metricCells.length) {
    return null;
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
      {metricCells.map((metric) => (
        <div key={metric.label} className="rounded-lg bg-slate-800 px-3 py-2">
          <div className="text-xs uppercase tracking-wide text-slate-400">{metric.label}</div>
          <div className="text-lg font-semibold">
            {typeof metric.value === "number" ? metric.value.toFixed(3) : metric.value}
          </div>
        </div>
      ))}
    </div>
  );
}

function StrategyTable({
  strategies,
  filters,
  onFilterChange,
  onPromote,
  onSnapshot,
}) {
  const sorted = useMemo(() => {
    if (!Array.isArray(strategies)) {
      return [];
    }
    let rows = strategies.slice();
    if (filters.showUnderperforming) {
      rows = rows.filter((row) => row.underperforming);
    }
    if (filters.minTrades) {
      rows = rows.filter((row) => (row.trades || 0) >= filters.minTrades);
    }
    const sortKey = filters.sortKey || "sharpe";
    rows.sort((a, b) => {
      const av = a[sortKey] || 0;
      const bv = b[sortKey] || 0;
      return filters.sortDirection === "asc" ? av - bv : bv - av;
    });
    return rows;
  }, [strategies, filters]);

  if (!sorted.length) {
    return (
      <div className="rounded-lg border border-dashed border-slate-700 p-6 text-center text-slate-400">
        No strategy data available yet.
      </div>
    );
  }

  const bestCandidate = sorted[0];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3 justify-between">
        <div className="flex flex-wrap gap-3">
          <label className="flex items-center gap-2 text-sm text-slate-300">
            <input
              type="checkbox"
              className="rounded border-slate-500"
              checked={filters.showUnderperforming}
              onChange={(event) =>
                onFilterChange({ ...filters, showUnderperforming: event.target.checked })
              }
            />
            Underperforming only
          </label>
          <label className="flex items-center gap-2 text-sm text-slate-300">
            Min trades
            <input
              type="number"
              min="0"
              className="w-20 rounded bg-slate-900 px-2 py-1 text-right text-slate-200 border border-slate-700"
              value={filters.minTrades}
              onChange={(event) =>
                onFilterChange({ ...filters, minTrades: Number(event.target.value) || 0 })
              }
            />
          </label>
          <label className="flex items-center gap-2 text-sm text-slate-300">
            Sort
            <select
              className="rounded bg-slate-900 px-2 py-1 border border-slate-700"
              value={filters.sortKey}
              onChange={(event) =>
                onFilterChange({ ...filters, sortKey: event.target.value })
              }
            >
              <option value="sharpe">Sharpe</option>
              <option value="win_rate">Win Rate</option>
              <option value="drawdown">Drawdown</option>
              <option value="trades">Trades</option>
            </select>
            <button
              className="rounded bg-slate-800 px-2 py-1 text-xs border border-slate-700"
              onClick={() =>
                onFilterChange({
                  ...filters,
                  sortDirection: filters.sortDirection === "asc" ? "desc" : "asc",
                })
              }
            >
              {filters.sortDirection === "asc" ? "Asc" : "Desc"}
            </button>
          </label>
        </div>
        {bestCandidate && (
          <button
            className="rounded bg-emerald-600 px-3 py-2 text-sm font-semibold text-white hover:bg-emerald-500"
            onClick={() => onPromote(bestCandidate)}
          >
            Promote top candidate
          </button>
        )}
      </div>
      <div className="overflow-x-auto rounded-lg border border-slate-800">
        <table className="min-w-full divide-y divide-slate-800 text-sm">
          <thead className="bg-slate-900/80">
            <tr>
              <th className="px-3 py-2 text-left font-semibold text-slate-300">
                Strategy / Guidance
              </th>
              <th className="px-3 py-2 text-left font-semibold text-slate-300">Symbol</th>
              <th className="px-3 py-2 text-left font-semibold text-slate-300">Timeframe</th>
              <th className="px-3 py-2 text-right font-semibold text-slate-300">Sharpe</th>
              <th className="px-3 py-2 text-right font-semibold text-slate-300">Win %</th>
              <th className="px-3 py-2 text-right font-semibold text-slate-300">Drawdown</th>
              <th className="px-3 py-2 text-right font-semibold text-slate-300">Trades</th>
              <th className="px-3 py-2 text-left font-semibold text-slate-300">Warnings</th>
              <th className="px-3 py-2 text-right font-semibold text-slate-300">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {sorted.map((row) => {
              const rowKey = row.id || `${row.strategy}-${row.symbol}-${row.timeframe}`;
              const warning = row.underperforming;
              const label = row.strategy_label || row.strategy;
              const recommended = Array.isArray(row.recommended_timeframes)
                ? row.recommended_timeframes
                : [];
              const recommendedText = recommended.length ? recommended.join(", ") : "—";
              const defaultTimeframe = row.default_timeframe || (recommended[0] || "—");
              return (
                <tr key={rowKey} className={warning ? "bg-red-900/10" : ""}>
                  <td className="px-3 py-2 text-slate-200">
                    <div className="flex flex-col gap-1">
                      <span className="font-semibold text-slate-100">{label}</span>
                      <span className="text-[11px] uppercase tracking-wide text-slate-500">
                        ID: {row.strategy}
                      </span>
                      <div className="text-[11px] text-slate-400">
                        <span className="font-medium text-slate-300">Default:</span>{" "}
                        {defaultTimeframe}
                      </div>
                      <div className="text-[11px] text-slate-500">
                        <span className="font-medium text-slate-400">Recommended:</span>{" "}
                        {recommendedText}
                      </div>
                    </div>
                  </td>
                  <td className="px-3 py-2 text-slate-300">{row.symbol || "—"}</td>
                  <td
                    className="px-3 py-2 text-slate-300"
                    title={`Current: ${row.timeframe || "—"} • Default: ${defaultTimeframe} • Recommended: ${recommendedText}`}
                  >
                    {row.timeframe || "—"}
                  </td>
                  <td className="px-3 py-2 text-right text-slate-200">
                    {row.sharpe !== undefined && row.sharpe !== null ? row.sharpe.toFixed(3) : "—"}
                  </td>
                  <td className="px-3 py-2 text-right text-slate-200">
                    {row.win_rate !== undefined && row.win_rate !== null
                      ? (row.win_rate * 100).toFixed(1) + "%"
                      : "—"}
                  </td>
                  <td className="px-3 py-2 text-right text-slate-200">
                    {row.drawdown !== undefined && row.drawdown !== null
                      ? (row.drawdown * 100).toFixed(1) + "%"
                      : "—"}
                  </td>
                  <td className="px-3 py-2 text-right text-slate-200">{row.trades || 0}</td>
                  <td className="px-3 py-2 text-slate-300">
                    {Array.isArray(row.warnings) && row.warnings.length > 0 ? (
                      <div className="flex flex-wrap gap-1">
                        {row.warnings.map((warn, idx) => (
                          <span
                            key={idx}
                            className="rounded bg-amber-900/40 px-2 py-0.5 text-xs text-amber-200"
                          >
                            {warn}
                          </span>
                        ))}
                      </div>
                    ) : (
                      <span className="text-slate-500">—</span>
                    )}
                  </td>
                  <td className="px-3 py-2 text-right">
                    <div className="inline-flex items-center gap-2">
                      <button
                        className="rounded bg-indigo-600 px-3 py-1 text-xs font-semibold text-white hover:bg-indigo-500"
                        onClick={() => onPromote(row)}
                        disabled={!row.config}
                        title={row.config || "No config path available"}
                      >
                        Promote
                      </button>
                      <button
                        className="rounded bg-slate-800 px-3 py-1 text-xs text-slate-200 hover:bg-slate-700"
                        onClick={() => onSnapshot(row)}
                        disabled={!row.config}
                      >
                        Snapshot
                      </button>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
function BacktestResultModal({ preview, onClose }) {
  if (!preview) {
    return null;
  }
  const { job, results } = preview;
  const titleParts = [];
  if (job?.strategy) {
    titleParts.push(job.strategy);
  }
  if (job?.timeframe) {
    titleParts.push(job.timeframe);
  }
  if (Array.isArray(job?.pairs) && job.pairs.length) {
    titleParts.push(job.pairs.join(", "));
  } else if (job?.pair) {
    titleParts.push(job.pair);
  }
  const subtitle = titleParts.length ? titleParts.join(" • ") : job?.id;

  const copyPath = (path) => {
    if (!path || typeof navigator === "undefined" || !navigator.clipboard) {
      return;
    }
    navigator.clipboard.writeText(path).catch(() => {});
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 px-4 py-8 backdrop-blur-sm">
      <div className="max-h-[80vh] w-full max-w-4xl overflow-hidden rounded-lg border border-slate-800 bg-slate-950 shadow-2xl">
        <div className="flex items-start justify-between gap-3 border-b border-slate-800 px-5 py-4">
          <div className="space-y-1">
            <h3 className="text-lg font-semibold text-white">Backtest results</h3>
            {subtitle && <p className="text-xs text-slate-400">{subtitle}</p>}
          </div>
          <button
            className="rounded bg-slate-800 px-3 py-1.5 text-xs font-semibold text-slate-200 hover:bg-slate-700"
            onClick={onClose}
          >
            Close
          </button>
        </div>
        <div className="max-h-[65vh] overflow-y-auto px-5 py-4 space-y-4">
          {(!results || results.length === 0) && (
            <div className="rounded border border-slate-800 bg-slate-900/60 px-4 py-6 text-center text-sm text-slate-400">
              No result artifacts matched this job yet.
            </div>
          )}
          {results &&
            results.map((result) => {
              const meta = result.metadata || {};
              const summary = result.summary || {};
              const key = result.file || result.path || Math.random().toString(36).slice(2);
              const metrics = [
                { label: "Sharpe", value: result.sharpe },
                { label: "Trades", value: result.trades },
                { label: "Return", value: result.return },
                { label: "Capital %", value: result.capital_pct },
                { label: "Symbol", value: meta.symbol || meta.pair },
                { label: "Timeframe", value: meta.interval || meta.timeframe },
                { label: "Strategy", value: meta.strategy || meta.mode },
              ];
              const filteredMetrics = metrics.filter(
                (metric) => metric.value !== undefined && metric.value !== null && metric.value !== ""
              );
              return (
                <div key={key} className="space-y-2 rounded border border-slate-800 bg-slate-900/70 p-4">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                      <div className="text-sm font-semibold text-slate-100">{result.file || "Result"}</div>
                      <div className="text-xs text-slate-500">
                        {describeResultTimestamp(result.generated_at)}
                      </div>
                    </div>
                    <div className="flex flex-wrap items-center gap-2 text-xs text-slate-300">
                      {result.path && (
                        <button
                          className="rounded border border-slate-700 px-2 py-1 hover:bg-slate-800"
                          onClick={() => copyPath(result.path)}
                        >
                          Copy path
                        </button>
                      )}
                      {result.path && (
                        <a
                          className="rounded border border-slate-700 px-2 py-1 hover:bg-slate-800"
                          href={`/api/backtests/results/${encodeURIComponent(result.file)}`}
                          target="_blank"
                          rel="noreferrer"
                        >
                          Open JSON
                        </a>
                      )}
                    </div>
                  </div>
                  {filteredMetrics.length > 0 && (
                    <div className="grid gap-2 text-xs text-slate-300 md:grid-cols-3">
                      {filteredMetrics.map((metric) => (
                        <div key={metric.label} className="rounded bg-slate-950/60 px-3 py-2">
                          <div className="text-[10px] uppercase tracking-wide text-slate-500">{metric.label}</div>
                          <div className="text-sm font-semibold text-slate-100">
                            {typeof metric.value === "number" ? metric.value.toFixed(4) : metric.value}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                  {summary && Object.keys(summary).length > 0 && (
                    <details className="rounded border border-slate-800 bg-slate-950/50">
                      <summary className="cursor-pointer px-3 py-2 text-xs font-semibold text-slate-200">
                        Summary details
                      </summary>
                      <pre className="overflow-x-auto px-3 py-2 text-[11px] leading-relaxed text-slate-200">
                        {JSON.stringify(summary, null, 2)}
                      </pre>
                    </details>
                  )}
                </div>
              );
            })}
        </div>
      </div>
    </div>
  );
}

function BacktestPanel({
  data,
  onLaunch,
  launching,
  strategyOptions = [],
  strategyMetadata = {},
  symbolOptions = [],
  timeframeOptions = [],
  capitalOptions = [],
  onNotify = () => {},
  onRefresh = () => {},
  onClearJobs = () => {},
  clearingJobs = false,
  onClearGuardrails = () => {},
  clearingGuardrails = false,
}) {
  const [form, setForm] = useState({
    strategies: [],
    symbols: [],
    timeframes: {},
    capital_pct: capitalOptions[0] ?? DEFAULT_CAPITAL_OPTIONS[0] ?? 10,
    start_date: "",
    end_date: "",
  });
  const [resultPreview, setResultPreview] = useState(null);
  const [archivingJobId, setArchivingJobId] = useState(null);
  const [clearingResults, setClearingResults] = useState(false);
  const [openingResultId, setOpeningResultId] = useState(null);

  useEffect(() => {
    if (!form.strategies.length && strategyOptions.length) {
      setForm((prev) => ({ ...prev, strategies: [strategyOptions[0]] }));
    }
  }, [strategyOptions]);

  useEffect(() => {
    if (!form.symbols.length && symbolOptions.length) {
      setForm((prev) => ({ ...prev, symbols: [symbolOptions[0]] }));
    }
  }, [symbolOptions]);

  useEffect(() => {
    if (capitalOptions.length && !capitalOptions.includes(Number(form.capital_pct))) {
      setForm((prev) => ({ ...prev, capital_pct: capitalOptions[0] }));
    }
  }, [capitalOptions]);

  const labelForStrategy = useCallback(
    (strategy) => {
      if (!strategy) return "";
      const meta = strategyMetadata[strategy];
      return (meta && meta.label) || strategy;
    },
    [strategyMetadata]
  );

  const timeframeOptionsForStrategy = useCallback(
    (strategy) => {
      if (!strategy) {
        return [];
      }
      const meta = strategyMetadata[strategy] || {};
      const sourceOptions =
        Array.isArray(meta.recommended_timeframes) && meta.recommended_timeframes.length
          ? meta.recommended_timeframes
          : timeframeOptions;
      const normalized = Array.isArray(sourceOptions) ? sourceOptions : [];
      const seen = new Set();
      const ordered = [];
      normalized.forEach((value) => {
        const nextValue = String(value);
        if (!seen.has(nextValue)) {
          seen.add(nextValue);
          ordered.push(nextValue);
        }
      });
      return ordered;
    },
    [strategyMetadata, timeframeOptions]
  );

  useEffect(() => {
    const selectedStrategies = form.strategies;
    setForm((prev) => {
      const next = { ...(prev.timeframes || {}) };
      const keep = new Set(selectedStrategies);
      Object.keys(next).forEach((key) => {
        if (!keep.has(key)) {
          delete next[key];
        }
      });
      selectedStrategies.forEach((strategy) => {
        const options = timeframeOptionsForStrategy(strategy);
        if (!options.length) {
          delete next[strategy];
          return;
        }
        const meta = strategyMetadata[strategy] || {};
        const defaultCandidate =
          meta &&
          meta.default_timeframe &&
          options.includes(meta.default_timeframe)
            ? meta.default_timeframe
            : options[0];
        const current = prev.timeframes ? prev.timeframes[strategy] : undefined;
        if (current && options.includes(current)) {
          next[strategy] = current;
        } else {
          next[strategy] = defaultCandidate;
        }
      });
      const previous = prev.timeframes || {};
      const sameSize = Object.keys(next).length === Object.keys(previous).length;
      const unchanged =
        sameSize &&
        Object.entries(next).every(([key, value]) => previous[key] === value);
      if (unchanged) {
        return prev;
      }
      return { ...prev, timeframes: next };
    });
  }, [form.strategies, strategyMetadata, timeframeOptions, timeframeOptionsForStrategy]);

  const guardrailLogs = Array.isArray(data.guardrail_logs) ? data.guardrail_logs : [];
  const guardrailSnapshots = Array.isArray(data.guardrail_snapshots) ? data.guardrail_snapshots : [];
  const jobs = Array.isArray(data.jobs) ? data.jobs : [];
  const results = Array.isArray(data.results) ? data.results : [];
  const collectResultsForJob = useCallback(
    (job) => {
      if (!job) return [];
      const jobConfig = (job.config || "").toLowerCase();
      const jobStrategy = (job.strategy || "").toLowerCase();
      const jobTimeframe = (job.timeframe || "").toLowerCase();
      const planSummary = Array.isArray(job.plan_summary) ? job.plan_summary : [];
      const strategySet = new Set(
        planSummary
          .map((item) => String(item.strategy || "").toLowerCase())
          .filter((value) => value)
      );
      const timeframeSet = new Set(
        planSummary
          .map((item) => String(item.timeframe || "").toLowerCase())
          .filter((value) => value)
      );
      const symbolSet = new Set(
        planSummary
          .flatMap((item) => item.symbols || [])
          .map((symbol) => String(symbol).toUpperCase())
          .filter((value) => value)
      );
      const jobPairs = Array.isArray(job.pairs)
        ? job.pairs.map((symbol) => String(symbol).toUpperCase())
        : job.pair
        ? [String(job.pair).toUpperCase()]
        : [];
      jobPairs.forEach((symbol) => symbolSet.add(symbol));

      const matchesStrategy = (value) => {
        const lower = (value || "").toLowerCase();
        if (!lower) {
          return false;
        }
        if (strategySet.size) {
          return strategySet.has(lower);
        }
        return jobStrategy && (lower === jobStrategy || lower.includes(jobStrategy));
      };

      const matchesTimeframe = (value) => {
        const lower = (value || "").toLowerCase();
        if (timeframeSet.size) {
          if (!lower) {
            return false;
          }
          return timeframeSet.has(lower);
        }
        if (!jobTimeframe) {
          return true;
        }
        return !value || lower === jobTimeframe;
      };

      const matches = results.filter((result) => {
        if (!result || typeof result !== "object") {
          return false;
        }
        const resConfig = (result.config || "").toLowerCase();
        if (jobConfig && resConfig && resConfig === jobConfig) {
          return true;
        }
        const fileName = (result.file || "").toLowerCase();
        if (jobStrategy && fileName.includes(jobStrategy)) {
          return true;
        }
        const meta = result.metadata || {};
        const metaStrategy = (meta.strategy || meta.mode || "").toLowerCase();
        const metaTimeframe = (meta.interval || meta.timeframe || "").toLowerCase();
        const metaSymbol = (meta.symbol || meta.pair || "").toUpperCase();
        if (metaStrategy && matchesStrategy(metaStrategy)) {
          if (matchesTimeframe(metaTimeframe)) {
            return true;
          }
        }
        if (symbolSet.size && metaSymbol) {
          if (symbolSet.has(metaSymbol)) {
            if (matchesTimeframe(metaTimeframe)) {
              return true;
            }
          }
        }
        if (symbolSet.size && fileName) {
          const includesSymbol = Array.from(symbolSet).some((symbol) =>
            fileName.includes(symbol.toLowerCase())
          );
          if (includesSymbol) {
            return true;
          }
        }
        return false;
      });
      return matches;
    },
    [results]
  );

  const handleTimeframeChange = (strategy) => (event) => {
    const value = event.target.value;
    setForm((prev) => ({
      ...prev,
      timeframes: { ...(prev.timeframes || {}), [strategy]: value },
    }));
  };

  const handleMultiChange = (key) => (event) => {
    const selected = Array.from(event.target.selectedOptions).map((option) => option.value);
    setForm((prev) => ({ ...prev, [key]: selected }));
  };

  const handleInputChange = (key) => (event) => {
    const value = key === "capital_pct" ? Number(event.target.value) : event.target.value;
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const allTimeframesSelected =
    form.strategies.length > 0 &&
    form.strategies.every((strategy) => {
      const selection = form.timeframes ? form.timeframes[strategy] : "";
      if (!selection) {
        return false;
      }
      const options = timeframeOptionsForStrategy(strategy);
      return options.includes(selection);
    });
  const canSubmit = allTimeframesSelected && Number(form.capital_pct) > 0;

  const handleJobDoubleClick = (job) => {
    if (!job) {
      return;
    }
    const related = collectResultsForJob(job);
    if (!related.length) {
      const summaryLabels = Array.isArray(job.plan_summary)
        ? job.plan_summary.map((item) => item.label || item.strategy).filter(Boolean)
        : [];
      const descriptor =
        summaryLabels.length > 0
          ? summaryLabels.join(", ")
          : job.strategy || job.id || "selection";
      onNotify(`No matching backtest results found for ${descriptor}.`, "warning");
      return;
    }
    setResultPreview({ job, results: related });
  };

  const handleResultView = async (result) => {
    if (!result || !result.file) {
      onNotify("Result file reference missing.", "warning");
      return;
    }
    try {
      setOpeningResultId(result.file);
      const payload = await apiRequest(
        `/api/backtests/results/${encodeURIComponent(result.file)}`,
        { method: "GET" }
      );
      const detailMeta = payload && payload.metadata ? payload.metadata : {};
      const identity = result.identity || {};
      const jobStub = {
        strategy: identity.strategy || detailMeta.strategy || result.strategy,
        timeframe: detailMeta.interval || detailMeta.timeframe || result.timeframe,
        pairs: identity.symbol ? [identity.symbol] : undefined,
        pair: identity.symbol || detailMeta.symbol || detailMeta.pair,
        id: result.file,
      };
      setResultPreview({
        job: jobStub,
        results: [
          {
            ...payload,
            file: result.file,
            path: result.path || payload.path,
            generated_at: result.generated_at || payload.generated_at,
          },
        ],
      });
    } catch (err) {
      onNotify(`Failed to open result: ${err.message}`, "critical");
    } finally {
      setOpeningResultId(null);
    }
  };

  const handleClearResults = async () => {
    if (clearingResults) {
      return;
    }
    try {
      if (typeof window !== "undefined") {
        const confirmed = window.confirm("Clear all recent results from the panel?");
        if (!confirmed) {
          return;
        }
      }
      setClearingResults(true);
      await apiRequest("/api/backtests/results", { method: "DELETE" });
      onNotify("Cleared latest results.", "success");
      if (typeof onRefresh === "function") {
        onRefresh();
      }
    } catch (err) {
      onNotify(`Failed to clear results: ${err.message}`, "critical");
    } finally {
      setClearingResults(false);
    }
  };

  const handleArchiveClick = async (event, job) => {
    if (event) {
      event.stopPropagation();
      event.preventDefault();
    }
    if (!job || !job.id) {
      return;
    }
    const status = String(job.status || "").toLowerCase();
    const archivableStatuses = ["completed", "failed", "stopped", "cancelled"];
    if (!archivableStatuses.includes(status)) {
      onNotify("Only completed or failed jobs can be archived.", "warning");
      return;
    }
    let archiveResults = false;
    if (status === "completed") {
      archiveResults = typeof window !== "undefined" && window.confirm("Archive matching backtest result files as well?");
    }
    try {
      setArchivingJobId(job.id);
      const response = await apiRequest(`/api/jobs/${encodeURIComponent(job.id)}/archive`, {
        method: "POST",
        body: { archive_results: archiveResults },
      });
      const archivedCount = Array.isArray(response.archived_results) ? response.archived_results.length : 0;
      const message =
        archivedCount > 0
          ? `Archived job ${job.id} and moved ${archivedCount} result ${archivedCount === 1 ? "file" : "files"}.`
          : `Archived job ${job.id}.`;
      onNotify(message, "success");
      if (typeof onRefresh === "function") {
        onRefresh();
      }
    } catch (err) {
      onNotify(`Archive failed: ${err.message}`, "critical");
    } finally {
      setArchivingJobId(null);
    }
  };

  const submit = async (event) => {
    event.preventDefault();
    if (!canSubmit) {
      return;
    }
    const timeframesPayload = {};
    form.strategies.forEach((strategy) => {
      const choice = form.timeframes ? form.timeframes[strategy] : "";
      if (choice) {
        timeframesPayload[strategy] = choice;
      }
    });
    const payload = {
      strategies: form.strategies,
      symbols: form.symbols,
      capital_pct: Number(form.capital_pct) || 0,
    };
    if (Object.keys(timeframesPayload).length) {
      payload.timeframes = timeframesPayload;
      const primaryStrategy = form.strategies[0];
      if (primaryStrategy && timeframesPayload[primaryStrategy]) {
        payload.timeframe = timeframesPayload[primaryStrategy];
      }
    }
    if (form.start_date) {
      payload.start_date = form.start_date;
    }
    if (form.end_date) {
      payload.end_date = form.end_date;
    }
    await onLaunch(payload);
    setForm((prev) => ({ ...prev, start_date: "", end_date: "" }));
  };

  useEffect(() => {
    if (!resultPreview) {
      return;
    }
    const refreshed = collectResultsForJob(resultPreview.job);
    const currentKeys = Array.isArray(resultPreview.results)
      ? resultPreview.results.map((item) => item?.path || item?.file || "").join("|")
      : "";
    const refreshedKeys = refreshed.map((item) => item?.path || item?.file || "").join("|");
    const previousLength = Array.isArray(resultPreview.results) ? resultPreview.results.length : 0;
    if (previousLength === refreshed.length && currentKeys === refreshedKeys) {
      return;
    }
    setResultPreview((prev) => ({ ...prev, results: refreshed }));
  }, [results, resultPreview, collectResultsForJob]);

  return (
    <div className="space-y-6">
      <form
        onSubmit={submit}
        className="grid gap-4 rounded-lg border border-slate-800 bg-slate-900/40 p-4 md:grid-cols-3"
      >
        <div className="md:col-span-1 space-y-2">
          <label className="text-xs uppercase tracking-wide text-slate-400">Strategies</label>
          <select
            multiple
            className="h-32 w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-200"
            value={form.strategies}
            onChange={handleMultiChange("strategies")}
          >
            {strategyOptions.map((strategy) => (
              <option key={strategy} value={strategy}>
                {labelForStrategy(strategy)}
              </option>
            ))}
          </select>
          <p className="text-xs text-slate-500">
            Select one or more strategies (Ctrl/Cmd + click for multi-select).
          </p>
        </div>
        <div className="md:col-span-1 space-y-2">
          <label className="text-xs uppercase tracking-wide text-slate-400">Symbols</label>
          <select
            multiple
            className="h-32 w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-200"
            value={form.symbols}
            onChange={handleMultiChange("symbols")}
          >
            {symbolOptions.map((symbol) => (
              <option key={symbol} value={symbol}>
                {symbol}
              </option>
            ))}
          </select>
          <p className="text-xs text-slate-500">
            Select the BitMEX symbols to include. Leave empty to use the strategy defaults.
          </p>
        </div>
        <div className="md:col-span-1 grid grid-cols-2 gap-3">
          <div className="col-span-2 space-y-2">
            <label className="text-xs uppercase tracking-wide text-slate-400">Timeframes</label>
            {form.strategies.length === 0 ? (
              <p className="text-xs text-slate-500">
                Select at least one strategy to view recommended timeframes.
              </p>
            ) : (
              <div className="space-y-2">
                {form.strategies.map((strategy) => {
                  const options = timeframeOptionsForStrategy(strategy);
                  const value = (form.timeframes && form.timeframes[strategy]) || "";
                  const label = labelForStrategy(strategy);
                  return (
                    <div key={strategy} className="space-y-1">
                      <div className="text-[11px] uppercase tracking-wide text-slate-500">{label}</div>
                      {options.length ? (
                        <select
                          className="w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-200"
                          value={value}
                          onChange={handleTimeframeChange(strategy)}
                        >
                          {options.map((option) => (
                            <option key={option} value={option}>
                              {option}
                            </option>
                          ))}
                        </select>
                      ) : (
                        <div className="rounded border border-amber-800/60 bg-amber-900/20 px-3 py-2 text-xs text-amber-200">
                          No recommended timeframes available.
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
          <div className="space-y-2">
            <label className="text-xs uppercase tracking-wide text-slate-400">Capital %</label>
            <select
              className="w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-200"
              value={form.capital_pct}
              onChange={handleInputChange("capital_pct")}
            >
              {capitalOptions.map((option) => (
                <option key={option} value={option}>
                  {option}%
                </option>
              ))}
            </select>
          </div>
          <div className="space-y-2">
            <label className="text-xs uppercase tracking-wide text-slate-400">Start date</label>
            <input
              type="date"
              className="w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-200"
              value={form.start_date}
              onChange={handleInputChange("start_date")}
            />
          </div>
          <div className="space-y-2">
            <label className="text-xs uppercase tracking-wide text-slate-400">End date</label>
            <input
              type="date"
              className="w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-200"
              value={form.end_date}
              onChange={handleInputChange("end_date")}
            />
          </div>
        </div>
        <div className="md:col-span-3 flex justify-end">
          <button
            type="submit"
            className="inline-flex items-center gap-2 rounded bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-500 disabled:opacity-60"
            disabled={launching || !canSubmit}
          >
            {launching ? "Queueing..." : "Launch backtest"}
          </button>
        </div>
      </form>

      <div className="grid gap-5 md:grid-cols-2">
        <div className="space-y-2 rounded-lg border border-slate-800 bg-slate-900/50 p-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wide">
                Job queue
              </h3>
              <p className="text-xs text-slate-500">
                Double-click a job to open its recent backtest results.
              </p>
            </div>
            <button
              type="button"
              className="rounded bg-slate-800 px-3 py-1.5 text-xs font-semibold text-slate-200 hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-60"
              onClick={onClearJobs}
              disabled={clearingJobs}
            >
              {clearingJobs ? "Clearing…" : "Clear jobs"}
            </button>
          </div>
          <div className="space-y-3 max-h-72 overflow-y-auto pr-1">
            {jobs.length === 0 && <p className="text-sm text-slate-500">No queued jobs.</p>}
            {jobs.map((job) => {
              const planSummary = Array.isArray(job.plan_summary) ? job.plan_summary : [];
              const planLabels = planSummary.map((item) => item.label || item.strategy).filter(Boolean);
              const planTimeframes = planSummary
                .map((item) => item.timeframe)
                .filter((value, index, array) => value && array.indexOf(value) === index);
              const planSymbols = planSummary
                .flatMap((item) => item.symbols || [])
                .filter((value, index, array) => array.indexOf(value) === index);
              const planCount = planSummary.length || (Array.isArray(job.plan) ? job.plan.length : 0);
              return (
                <div
                  key={job.id}
                  className="rounded border border-slate-800 p-3"
                  onDoubleClick={() => handleJobDoubleClick(job)}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="font-semibold text-slate-100">
                        {job.type || "job"} • {job.status || "unknown"}
                      </span>
                      {planCount > 0 && (
                        <span className="ml-2 rounded bg-slate-800/70 px-2 py-0.5 text-[11px] font-semibold text-slate-300">
                          {planCount} strategy{planCount === 1 ? "" : " combos"}
                        </span>
                      )}
                    </div>
                    <span className="text-xs text-slate-400" title={job.submitted_at || ""}>
                      {job.submitted_at ? dayjs(job.submitted_at).fromNow() : ""}
                    </span>
                  </div>
                  <div className="mt-1 grid grid-cols-2 gap-1 text-xs text-slate-400">
                    {planLabels.length ? (
                      <div className="col-span-2">
                        Strategies: <span className="text-slate-200">{planLabels.join(", ")}</span>
                      </div>
                    ) : (
                      job.strategy && <div>Strategy: {job.strategy}</div>
                    )}
                    {planTimeframes.length ? (
                      <div className="col-span-2">
                        Timeframes: <span className="text-slate-200">{planTimeframes.join(", ")}</span>
                      </div>
                    ) : (
                      job.timeframe && <div>TF: {job.timeframe}</div>
                    )}
                    {planSymbols.length ? (
                      <div className="col-span-2">
                        Symbols: <span className="text-slate-200">{planSymbols.join(", ")}</span>
                      </div>
                    ) : (
                      job.pair && <div>Pair: {job.pair}</div>
                    )}
                    {job.capital_pct !== undefined && <div>Capital %: {job.capital_pct}</div>}
                  </div>
                  {planSummary.length > 0 && (
                    <div className="mt-3 space-y-3 rounded border border-slate-800/70 bg-slate-950/30 px-3 py-2 text-xs text-slate-200">
                      {planSummary.map((item, idx) => (
                        <div key={`${job.id}-${item.strategy}-${item.timeframe}-${idx}`} className="space-y-1">
                          <div className="font-semibold text-slate-100">
                            {item.label || item.strategy} • {item.timeframe || "—"}
                          </div>
                          <div className="text-slate-400">{item.entry_reason}</div>
                          <div className="text-slate-400">Exit plan: {item.exit_reason}</div>
                        </div>
                      ))}
                    </div>
                  )}
                  <div className="mt-2 flex flex-wrap gap-2 text-xs">
                    <button
                      className="rounded bg-slate-800 px-3 py-1 font-semibold text-slate-200 hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50"
                      onClick={(event) => handleArchiveClick(event, job)}
                      disabled={
                        archivingJobId === job.id ||
                        !["completed", "failed", "stopped", "cancelled"].includes(
                          String(job.status || "").toLowerCase()
                        )
                      }
                    >
                      {archivingJobId === job.id ? "Archiving..." : "Archive"}
                    </button>
                  </div>
                  {job.error && (
                    <div className="mt-2 rounded bg-red-900/40 px-3 py-2 text-xs text-red-200">
                      {job.error}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
        <div className="space-y-2 rounded-lg border border-slate-800 bg-slate-900/50 p-4">
          <div className="flex items-center justify-between gap-3">
            <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wide">
              Latest results
            </h3>
            <button
              type="button"
              className="rounded bg-slate-800 px-3 py-1.5 text-xs font-semibold text-slate-200 hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-60"
              onClick={handleClearResults}
              disabled={clearingResults || results.length === 0}
            >
              {clearingResults ? "Clearing…" : "Clear"}
            </button>
          </div>
          <div className="space-y-3 max-h-72 overflow-y-auto pr-1">
            {results.length === 0 && <p className="text-sm text-slate-500">No results yet.</p>}
            {results.map((result) => {
              const identity = result.identity || {};
              const summary = result.summary || {};
              const explanations = result.explanations || {};
              const title =
                result.strategy_label ||
                identity.label ||
                result.strategy ||
                identity.strategy ||
                result.config ||
                "Result";
              const subtitleParts = [];
              const symbol = identity.symbol || (result.metadata && (result.metadata.symbol || result.metadata.pair));
              const timeframe =
                identity.timeframe ||
                (result.metadata && (result.metadata.interval || result.metadata.timeframe));
              if (symbol) {
                subtitleParts.push(symbol);
              }
              if (timeframe) {
                subtitleParts.push(timeframe);
              }
              if (result.capital_pct) {
                subtitleParts.push(`${result.capital_pct}% capital`);
              }
              const metrics = [
                { label: "Sharpe", value: summary.calc_sharpe ?? result.sharpe },
                { label: "Trades", value: summary.trades_count ?? result.trades },
                { label: "Return", value: summary.total_return ?? result.return },
                { label: "Max DD", value: summary.max_drawdown },
              ].filter(
                (metric) =>
                  metric.value !== undefined &&
                  metric.value !== null &&
                  metric.value !== "" &&
                  !Number.isNaN(metric.value)
              );
              const entryReason = explanations.entry;
              const exitReason = explanations.exit;
              return (
                <div key={result.path || result.file || result.config} className="rounded border border-slate-800 p-3">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="font-semibold text-slate-100">{title}</div>
                      {subtitleParts.length > 0 && (
                        <div className="text-xs text-slate-400">{subtitleParts.join(" • ")}</div>
                      )}
                      {describeResultTimestamp(result.generated_at) && (
                        <div className="text-[11px] text-slate-500">
                          {describeResultTimestamp(result.generated_at)}
                        </div>
                      )}
                    </div>
                    <div className="flex flex-wrap gap-2 text-xs text-slate-200">
                      <button
                        className="rounded bg-slate-800 px-3 py-1 font-semibold hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-60"
                        onClick={() => handleResultView(result)}
                        disabled={openingResultId === result.file || !result.file}
                      >
                        {openingResultId === result.file ? "Opening…" : "View details"}
                      </button>
                      {result.file && (
                        <a
                          className="rounded border border-slate-700 px-3 py-1 hover:bg-slate-800"
                          href={`/api/backtests/results/${encodeURIComponent(result.file)}`}
                          target="_blank"
                          rel="noreferrer"
                        >
                          Open JSON
                        </a>
                      )}
                    </div>
                  </div>
                  {entryReason && (
                    <p className="mt-2 text-xs text-slate-300">
                      Purpose: <span className="text-slate-100">{entryReason}</span>
                    </p>
                  )}
                  {exitReason && (
                    <p className="text-xs text-slate-400">Exit plan: {exitReason}</p>
                  )}
                  {metrics.length > 0 && (
                    <div className="mt-2 grid gap-2 text-xs text-slate-300 sm:grid-cols-2">
                      {metrics.map((metric) => (
                        <div key={metric.label} className="rounded bg-slate-950/50 px-2 py-1.5">
                          <div className="text-[10px] uppercase tracking-wide text-slate-500">
                            {metric.label}
                          </div>
                          <div className="text-sm font-semibold text-slate-100">
                            {typeof metric.value === "number" ? metric.value.toFixed(4) : metric.value}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                  {Array.isArray(result.flags) && result.flags.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1">
                      {result.flags.map((flag) => (
                        <span
                          key={flag}
                          className="rounded bg-amber-900/30 px-2 py-0.5 text-xs text-amber-200"
                        >
                          {flag}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div className="grid gap-5 md:grid-cols-2">
        <div className="rounded-lg border border-red-900/40 bg-red-950/40 p-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold uppercase tracking-wide text-red-200">
              Guardrail violations (latest)
            </h3>
            <button
              type="button"
              className="rounded bg-red-900/60 px-3 py-1.5 text-xs font-semibold text-red-100 hover:bg-red-800 disabled:cursor-not-allowed disabled:opacity-60"
              onClick={onClearGuardrails}
              disabled={clearingGuardrails}
            >
              {clearingGuardrails ? "Clearing…" : "Clear logs"}
            </button>
          </div>
          <div className="mt-3 space-y-3 max-h-64 overflow-y-auto pr-1">
            {guardrailLogs.length === 0 && (
              <p className="text-sm text-red-200/70">No guardrail breaches recorded.</p>
            )}
            {guardrailLogs.map((log, idx) => (
              <div key={idx} className="rounded border border-red-800/50 bg-red-900/30 p-3 text-sm">
                <div className="flex items-center justify-between text-xs text-red-100/80">
                  <span>{log.rule || log.guardrail || "rule"}</span>
                  <span>{log.timestamp ? dayjs(log.timestamp).format("YYYY-MM-DD HH:mm") : ""}</span>
                </div>
                {log.message && <div className="mt-1 text-red-100">{log.message}</div>}
                {log.details && (
                  <pre className="mt-2 overflow-x-auto rounded bg-red-900/40 p-2 text-[11px] text-red-100">
                    {JSON.stringify(log.details, null, 2)}
                  </pre>
                )}
              </div>
            ))}
          </div>
        </div>
        <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-4">
          <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-200">
            Guardrail snapshots
          </h3>
          <div className="mt-3 space-y-3 max-h-64 overflow-y-auto pr-1 text-sm">
            {guardrailSnapshots.length === 0 && (
              <p className="text-slate-400">No snapshots captured yet.</p>
            )}
            {guardrailSnapshots.map((snap) => (
              <div key={snap.path} className="rounded border border-slate-800/80 p-3">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span>{snap.name}</span>
                  <span>{snap.created_at ? dayjs(snap.created_at).fromNow() : ""}</span>
                </div>
                <pre className="mt-2 overflow-x-auto rounded bg-slate-950/60 p-2 text-[11px] text-slate-200">
                  {JSON.stringify(snap.summary, null, 2)}
                </pre>
              </div>
            ))}
          </div>
        </div>
      </div>
      {resultPreview && (
        <BacktestResultModal preview={resultPreview} onClose={() => setResultPreview(null)} />
      )}
    </div>
  );
}


function TradingControls({
  status = {},
  strategyOptions = [],
  strategyMetadata = {},
  symbolOptions = [],
  timeframeOptions = [],
  capitalOptions = [],
  onStartPaper,
  onStartLive,
  onStop,
  onNotify = () => {},
  working = {},
}) {
  const [form, setForm] = useState({
    strategies: [],
    symbols: [],
    timeframes: {},
    capital_pct: capitalOptions[0] ?? DEFAULT_CAPITAL_OPTIONS[0] ?? 10,
  });

  useEffect(() => {
    if (!form.strategies.length && strategyOptions.length) {
      setForm((prev) => ({ ...prev, strategies: [strategyOptions[0]] }));
    }
  }, [strategyOptions]);

  useEffect(() => {
    if (!form.symbols.length && symbolOptions.length) {
      setForm((prev) => ({ ...prev, symbols: [symbolOptions[0]] }));
    }
  }, [symbolOptions]);

  useEffect(() => {
    if (capitalOptions.length && !capitalOptions.includes(Number(form.capital_pct))) {
      setForm((prev) => ({ ...prev, capital_pct: capitalOptions[0] }));
    }
  }, [capitalOptions]);

  const labelForStrategy = useCallback(
    (strategy) => {
      if (!strategy) return "";
      const meta = strategyMetadata[strategy];
      return (meta && meta.label) || strategy;
    },
    [strategyMetadata]
  );

  const timeframeOptionsForStrategy = useCallback(
    (strategy) => {
      if (!strategy) {
        return [];
      }
      const meta = strategyMetadata[strategy] || {};
      const base =
        Array.isArray(meta.recommended_timeframes) && meta.recommended_timeframes.length
          ? meta.recommended_timeframes
          : timeframeOptions;
      const unique = [];
      const seen = new Set();
      (Array.isArray(base) ? base : []).forEach((value) => {
        const tf = String(value);
        if (!seen.has(tf)) {
          seen.add(tf);
          unique.push(tf);
        }
      });
      return unique.length ? unique : DEFAULT_TIMEFRAMES;
    },
    [strategyMetadata, timeframeOptions]
  );

  useEffect(() => {
    const selectedStrategies = form.strategies;
    setForm((prev) => {
      const next = { ...(prev.timeframes || {}) };
      const keep = new Set(selectedStrategies);
      Object.keys(next).forEach((key) => {
        if (!keep.has(key)) {
          delete next[key];
        }
      });
      selectedStrategies.forEach((strategy) => {
        const options = timeframeOptionsForStrategy(strategy);
        if (!options.length) {
          delete next[strategy];
          return;
        }
        const meta = strategyMetadata[strategy] || {};
        const defaultCandidate =
          meta &&
          meta.default_timeframe &&
          options.includes(meta.default_timeframe)
            ? meta.default_timeframe
            : options[0];
        const current = prev.timeframes ? prev.timeframes[strategy] : undefined;
        if (current && options.includes(current)) {
          next[strategy] = current;
        } else {
          next[strategy] = defaultCandidate;
        }
      });
      const previous = prev.timeframes || {};
      const sameSize = Object.keys(next).length === Object.keys(previous).length;
      const unchanged =
        sameSize &&
        Object.entries(next).every(([key, value]) => previous[key] === value);
      if (unchanged) {
        return prev;
      }
      return { ...prev, timeframes: next };
    });
  }, [form.strategies, strategyMetadata, timeframeOptionsForStrategy]);

  const handleMultiChange = (key) => (event) => {
    const selected = Array.from(event.target.selectedOptions).map((option) => option.value);
    setForm((prev) => ({ ...prev, [key]: selected }));
  };

  const handleValueChange = (key) => (event) => {
    const value = key === "capital_pct" ? Number(event.target.value) : event.target.value;
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const paperStatus = status.paper || {};
  const liveStatus = status.live || {};
  const guardrails = status.guardrails || {};
  const openPositions = status.open_positions || {};
  const openPositionsNotional =
    typeof openPositions.notional_usd === "number" && Number.isFinite(openPositions.notional_usd)
      ? openPositions.notional_usd
      : null;
  const openPositionsPnl =
    typeof openPositions.pnl_usd === "number" && Number.isFinite(openPositions.pnl_usd)
      ? openPositions.pnl_usd
      : null;
  const openPnlPct =
    openPositionsNotional && openPositionsNotional !== 0 && openPositionsPnl !== null
      ? (openPositionsPnl / Math.abs(openPositionsNotional)) * 100
      : null;

  const formatRelative = (value) => {
    if (!value) return "Never";
    if (typeof dayjs === "undefined") {
      return value;
    }
    const parsed = dayjs(value);
    return parsed.isValid() ? parsed.fromNow() : value;
  };

  const formatUsd = (value, digits = 2) =>
    typeof value === "number" && Number.isFinite(value)
      ? value.toLocaleString(undefined, {
          style: "currency",
          currency: "USD",
          minimumFractionDigits: digits,
          maximumFractionDigits: digits,
        })
      : "—";

  const formatPercent = (value, digits = 2) =>
    typeof value === "number" && Number.isFinite(value)
      ? `${value.toFixed(digits)}%`
      : "—";

  const pnlToneClass = (value) =>
    typeof value === "number" && Number.isFinite(value)
      ? value > 0
        ? "text-emerald-400"
        : value < 0
        ? "text-rose-400"
        : "text-slate-200"
      : "text-slate-200";

  const allTimeframesSelected =
    form.strategies.length > 0 &&
    form.strategies.every((strategy) => {
      const selection = form.timeframes ? form.timeframes[strategy] : "";
      if (!selection) {
        return false;
      }
      const options = timeframeOptionsForStrategy(strategy);
      return options.includes(selection);
    });

  const canSubmit = allTimeframesSelected && Number(form.capital_pct) > 0;
  const hasSymbols = form.symbols && form.symbols.length > 0;
  const startPayload = {
    strategies: form.strategies,
    symbols: form.symbols.map((symbol) => String(symbol || "").toUpperCase()),
    timeframes: form.timeframes,
    capital_pct: Number(form.capital_pct) || 0,
  };

  const canStartPaper =
    canSubmit && hasSymbols && !paperStatus.running && !paperStatus.pending && !working.paper;
  const canStartLive =
    canSubmit && hasSymbols && guardrails.paper_is_recent && !liveStatus.running && !liveStatus.pending && !working.live;

  const handleStartPaper = () => {
    if (!canStartPaper || typeof onStartPaper !== "function") {
      if (!hasSymbols) {
        onNotify("Select at least one symbol to start paper trading.", "warning");
      }
      return;
    }
    onStartPaper(startPayload);
  };

  const handleStartLive = () => {
    if (!canStartLive || typeof onStartLive !== "function") {
      if (!guardrails.paper_is_recent) {
        onNotify("Run a recent paper session before going live.", "warning");
      } else if (!hasSymbols) {
        onNotify("Select at least one symbol before starting live trading.", "warning");
      }
      return;
    }
    onStartLive(startPayload);
  };

  const stopWorkingMode = working.stop || null;

  const handleStop = (mode) => () => {
    if (typeof onStop === "function") {
      onStop(mode);
    }
  };

  return (
    <div className="space-y-4 rounded-lg border border-slate-800 bg-slate-900/40 p-4">
      <div className="grid gap-6 md:grid-cols-[2fr,1fr]">
        <div className="space-y-4">
          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-2">
              <label className="text-xs uppercase tracking-wide text-slate-400">Strategies</label>
              <select
                multiple
                className="h-32 w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-200"
                value={form.strategies}
                onChange={handleMultiChange("strategies")}
              >
                {strategyOptions.map((strategy) => (
                  <option key={strategy} value={strategy}>
                    {labelForStrategy(strategy)}
                  </option>
                ))}
              </select>
              <p className="text-xs text-slate-500">Select one or more strategies to launch (Ctrl/Cmd + click).</p>
            </div>
            <div className="space-y-2">
              <label className="text-xs uppercase tracking-wide text-slate-400">Symbols</label>
              <select
                multiple
                className="h-32 w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-200"
                value={form.symbols}
                onChange={handleMultiChange("symbols")}
              >
                {symbolOptions.map((symbol) => (
                  <option key={symbol} value={symbol}>
                    {symbol}
                  </option>
                ))}
              </select>
              <p className="text-xs text-slate-500">Choose the trading pairs to launch sessions against.</p>
            </div>
            <div className="md:col-span-2 space-y-2">
              <label className="text-xs uppercase tracking-wide text-slate-400">Timeframes</label>
              {form.strategies.length === 0 ? (
                <p className="text-xs text-slate-500">Select at least one strategy to configure timeframes.</p>
              ) : (
                <div className="grid gap-2 md:grid-cols-2">
                  {form.strategies.map((strategy) => {
                    const options = timeframeOptionsForStrategy(strategy);
                    const value = (form.timeframes && form.timeframes[strategy]) || "";
                    const label = labelForStrategy(strategy);
                    return (
                      <div key={strategy} className="space-y-1">
                        <div className="text-[11px] uppercase tracking-wide text-slate-500">{label}</div>
                        {options.length ? (
                          <select
                            className="w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-200"
                            value={value}
                            onChange={(event) => {
                              const selected = event.target.value;
                              setForm((prev) => ({
                                ...prev,
                                timeframes: { ...(prev.timeframes || {}), [strategy]: selected },
                              }));
                            }}
                          >
                            {options.map((option) => (
                              <option key={option} value={option}>
                                {option}
                              </option>
                            ))}
                          </select>
                        ) : (
                          <div className="rounded border border-amber-800/60 bg-amber-900/20 px-3 py-2 text-xs text-amber-200">
                            No recommended timeframes available.
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
            <div className="space-y-2">
              <label className="text-xs uppercase tracking-wide text-slate-400">Capital %</label>
              <select
                className="w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-200"
                value={form.capital_pct}
                onChange={handleValueChange("capital_pct")}
              >
                {(capitalOptions.length ? capitalOptions : DEFAULT_CAPITAL_OPTIONS).map((option) => (
                  <option key={option} value={option}>
                    {option}%
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div className="flex flex-wrap gap-3">
            <button
              className="rounded bg-slate-200 px-4 py-2 text-sm font-semibold text-slate-900 hover:bg-white disabled:cursor-not-allowed disabled:opacity-60"
              onClick={handleStartPaper}
              disabled={!canStartPaper}
            >
              {working.paper ? "Starting paper..." : "Start paper trading"}
            </button>
            <button
              className="rounded bg-emerald-600 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-60"
              onClick={handleStartLive}
              disabled={!canStartLive}
            >
              {working.live ? "Starting live..." : "Start live trading"}
            </button>
            <button
              className="rounded bg-slate-800 px-4 py-2 text-sm font-semibold text-slate-200 hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-60"
              onClick={handleStop('paper')}
              disabled={(!paperStatus.running && !paperStatus.pending) || stopWorkingMode === 'paper'}
            >
              {stopWorkingMode === 'paper' ? "Stopping paper..." : "Stop paper"}
            </button>
            <button
              className="rounded bg-slate-800 px-4 py-2 text-sm font-semibold text-slate-200 hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-60"
              onClick={handleStop('live')}
              disabled={(!liveStatus.running && !liveStatus.pending) || stopWorkingMode === 'live'}
            >
              {stopWorkingMode === 'live' ? "Stopping live..." : "Stop live"}
            </button>
            <button
              className="rounded bg-red-700 px-4 py-2 text-sm font-semibold text-white hover:bg-red-600 disabled:cursor-not-allowed disabled:opacity-60"
              onClick={handleStop(undefined)}
              disabled={stopWorkingMode === 'all'}
            >
              {stopWorkingMode === 'all' ? "Stopping..." : "Stop all"}
            </button>
          </div>
          {!guardrails.paper_is_recent && (
            <div className="rounded border border-amber-800 bg-amber-900/30 px-3 py-2 text-xs text-amber-100">
              Run a paper session (or rerun) before attempting live trading. Last paper: {formatRelative(guardrails.last_paper)}
            </div>
          )}
        </div>
        <div className="space-y-3 text-sm">
          <div className="rounded border border-slate-800 bg-slate-950 px-3 py-3">
            <div className="flex items-center justify-between text-xs uppercase tracking-wide text-slate-400">
              <span>Paper status</span>
              <span>{paperStatus.running ? "running" : paperStatus.pending ? "starting" : "idle"}</span>
            </div>
            <div className="mt-1 text-xs text-slate-300">
              Started: {formatRelative(paperStatus.started_at)}
            </div>
            <div className="text-xs text-slate-300">
              Last stop: {formatRelative(paperStatus.ended_at)}
            </div>
          </div>
          <div className="rounded border border-slate-800 bg-slate-950 px-3 py-3">
            <div className="flex items-center justify-between text-xs uppercase tracking-wide text-slate-400">
              <span>Live status</span>
              <span>{liveStatus.running ? "running" : liveStatus.pending ? "starting" : "idle"}</span>
            </div>
            <div className="mt-1 text-xs text-slate-300">
              Started: {formatRelative(liveStatus.started_at)}
            </div>
            <div className="text-xs text-slate-300">
              Last stop: {formatRelative(liveStatus.ended_at)}
            </div>
          </div>
          <div className="rounded border border-slate-800 bg-slate-950 px-3 py-3">
            <div className="text-xs uppercase tracking-wide text-slate-400">Open positions</div>
            <div className="mt-2 space-y-1 text-xs text-slate-300">
              <div>Count: {openPositions.count ?? 0}</div>
              <div>Exposure: {formatUsd(openPositions.notional_usd, 0)}</div>
              <div>
                PnL:{" "}
                <span className={pnlToneClass(openPositions.pnl_usd)}>
                  {formatUsd(openPositions.pnl_usd, 2)}
                </span>
              </div>
              <div>
                Realised %:{" "}
                <span className={pnlToneClass(openPnlPct)}>
                  {formatPercent(openPnlPct, 2)}
                </span>
              </div>
            </div>
            {openPositions.symbols && Object.keys(openPositions.symbols).length > 0 && (
              <div className="mt-3 border-t border-slate-800 pt-2 text-xs text-slate-300">
                <div className="mb-1 text-[11px] uppercase tracking-wide text-slate-500">
                  By symbol
                </div>
                <div className="space-y-1">
                  {Object.entries(openPositions.symbols)
                    .slice(0, 6)
                    .map(([symbol, summary]) => {
                      const pnlValue =
                        summary && typeof summary.pnl_usd === "number" ? summary.pnl_usd : null;
                      const notionalValue =
                        summary && typeof summary.notional_usd === "number"
                          ? summary.notional_usd
                          : null;
                      const pct =
                        notionalValue && notionalValue !== 0 && pnlValue !== null
                          ? (pnlValue / Math.abs(notionalValue)) * 100
                          : null;
                      return (
                        <div
                          key={symbol}
                          className="flex items-center justify-between rounded bg-slate-900/70 px-2 py-1"
                        >
                          <div className="font-mono text-[11px] text-slate-200">
                            {symbol} · {summary && summary.count ? summary.count : 0} pos
                          </div>
                          <div className="flex items-center gap-2 text-[11px]">
                            <span className="text-slate-400">{formatUsd(notionalValue, 0)}</span>
                            <span className={pnlToneClass(pnlValue)}>
                              {formatUsd(pnlValue, 2)} ({formatPercent(pct, 2)})
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  {Object.keys(openPositions.symbols).length > 6 && (
                    <div className="text-[11px] text-slate-500">
                      Showing first 6 symbols. View the trade log for full detail.
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function ConfigManager({ data, onSave, onRevert, loading }) {
  let entries = Array.isArray(data.configs) ? data.configs.slice() : [];
  if (!entries.length) {
    entries = DEFAULT_CONFIG_PLACEHOLDERS;
  }
  const [selected, setSelected] = useState(null);
  const [content, setContent] = useState("");
  const [dirty, setDirty] = useState(false);

  useEffect(() => {
    if (!selected) {
      setContent("");
      setDirty(false);
    }
  }, [selected]);

  const loadConfig = async (entry) => {
    const payload = await apiRequest(`/api/configs/${encodeURIComponent(entry.name)}`, {
      method: "GET",
    });
    setSelected(entry);
    setContent(payload.content || "");
    setDirty(false);
  };

  const handleSave = async () => {
    if (!selected) return;
    await onSave(selected, content);
    setDirty(false);
  };

  const handleRevert = async () => {
    if (!selected) return;
    await onRevert(selected);
    setSelected(null);
    setContent("");
  };

  return (
    <div className="grid gap-4 md:grid-cols-[260px,1fr]">
      <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-3">
        <div className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-400">
          Configurations
        </div>
        <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
          {entries.map((entry) => (
            <button
              key={entry.path}
              className={clsx(
                "w-full rounded border px-3 py-2 text-left text-sm",
                selected && selected.path === entry.path
                  ? "border-indigo-500 bg-indigo-900/40 text-indigo-100"
                  : "border-slate-800 bg-slate-950 text-slate-200 hover:border-indigo-600"
              )}
              onClick={() => loadConfig(entry)}
            >
              <div className="font-semibold">{entry.name}</div>
              <div className="text-xs text-slate-400">
                Updated {entry.updated_at ? dayjs(entry.updated_at).fromNow() : "unknown"}
              </div>
              {Array.isArray(entry.warnings) && entry.warnings.length > 0 && (
                <div className="mt-1 flex flex-wrap gap-1">
                  {entry.warnings.map((warn, idx) => (
                    <span key={idx} className="rounded bg-amber-900/30 px-2 py-0.5 text-[11px] text-amber-200">
                      {warn}
                    </span>
                  ))}
                </div>
              )}
            </button>
          ))}
          {entries.length === 0 && <p className="text-sm text-slate-500">No config files discovered.</p>}
        </div>
      </div>
      <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-4">
        {!selected ? (
          <p className="text-sm text-slate-500">
            Select a configuration to view or edit its contents. Warnings from guardrails will appear once loaded.
          </p>
        ) : (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm font-semibold text-slate-100">{selected.name}</div>
                <div className="text-xs text-slate-400">{selected.path}</div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  className="rounded bg-emerald-600 px-3 py-1 text-xs font-semibold text-white hover:bg-emerald-500 disabled:opacity-50"
                  onClick={handleSave}
                  disabled={!dirty || loading}
                >
                  Save
                </button>
                <button
                  className="rounded bg-slate-800 px-3 py-1 text-xs text-slate-200 hover:bg-slate-700 disabled:opacity-50"
                  onClick={handleRevert}
                  disabled={loading}
                >
                  Revert
                </button>
              </div>
            </div>
            <textarea
              className="h-80 w-full rounded border border-slate-800 bg-slate-950 p-3 font-mono text-xs text-slate-100"
              value={content}
              onChange={(event) => {
                setContent(event.target.value);
                setDirty(true);
              }}
            />
          </div>
        )}
      </div>
    </div>
  );
}

function TradesPanel({ data, strategyMetadata = {}, onClear, clearing = false }) {
  const records = Array.isArray(data.records) ? data.records : [];
  const stats = data.stats || {};
  const recent = records.slice(-20).reverse();
  const strategyMap = strategyMetadata && typeof strategyMetadata === "object" ? strategyMetadata : {};

  const pnlToneClass = (value) =>
    typeof value === "number" && Number.isFinite(value)
      ? value > 0
        ? "text-emerald-400"
        : value < 0
        ? "text-rose-400"
        : "text-slate-200"
      : "text-slate-200";

  const formatNumber = (value, digits = 2) =>
    typeof value === "number" && Number.isFinite(value)
      ? value.toLocaleString(undefined, {
          minimumFractionDigits: digits,
          maximumFractionDigits: digits,
        })
      : "—";

  const formatPercent = (value, digits = 2) =>
    typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(digits)}%` : "—";

  const formatUsd = (value, digits = 2) =>
    typeof value === "number" && Number.isFinite(value)
      ? value.toLocaleString(undefined, {
          style: "currency",
          currency: "USD",
          minimumFractionDigits: digits,
          maximumFractionDigits: digits,
        })
      : "—";

  const formatAud = (value, digits = 2) =>
    typeof value === "number" && Number.isFinite(value)
      ? value.toLocaleString(undefined, {
          style: "currency",
          currency: "AUD",
          minimumFractionDigits: digits,
          maximumFractionDigits: digits,
        })
      : "—";

  const formatBaseValue = (value, baseCurrency, digits = 6) =>
    typeof value === "number" &&
    Number.isFinite(value) &&
    baseCurrency &&
    typeof baseCurrency === "string" &&
    baseCurrency.length
      ? `${value.toLocaleString(undefined, {
          minimumFractionDigits: digits,
          maximumFractionDigits: digits,
        })} ${baseCurrency}`
      : "—";

  const resolveStrategyLabel = (strategy) => {
    if (!strategy) return "—";
    const key = String(strategy);
    const normalized = key.replace(/\s+/g, "_").toLowerCase();
    const meta =
      strategyMap[key] ||
      strategyMap[key.toLowerCase()] ||
      strategyMap[normalized] ||
      null;
    if (meta && meta.label) {
      return meta.label;
    }
    return key.replace(/_/g, " ");
  };

  const resolveStatus = (value) => {
    if (value === undefined || value === null || value === "") {
      return { label: "Closed", tone: "closed" };
    }
    if (value === true) {
      return { label: "Open", tone: "open" };
    }
    if (value === false) {
      return { label: "Closed", tone: "closed" };
    }
    const text = String(value).trim().toLowerCase();
    if (!text) {
      return { label: "Closed", tone: "closed" };
    }
    if (["open", "opened", "opening", "active"].includes(text)) {
      return { label: "Open", tone: "open" };
    }
    if (["closed", "closing", "flat", "exit", "exited"].includes(text)) {
      return { label: "Closed", tone: "closed" };
    }
    const label = text.replace(/\b\w/g, (match) => match.toUpperCase());
    return { label, tone: "neutral" };
  };

  const statusToneClass = (tone) => {
    switch (tone) {
      case "open":
        return "border border-emerald-400/40 bg-emerald-500/15 text-emerald-300";
      case "closed":
        return "border border-slate-600/40 bg-slate-700/20 text-slate-200";
      default:
        return "border border-indigo-500/40 bg-indigo-500/15 text-indigo-200";
    }
  };

  const statEntries = Object.entries(stats).map(([key, value]) => {
    const label = key
      .replace(/_/g, " ")
      .replace(/\b\w/g, (match) => match.toUpperCase());
    let display = "—";
    let className = "text-slate-100";
    if (typeof value === "number" && Number.isFinite(value)) {
      switch (key) {
        case "pnl_pct":
          display = formatPercent(value, 2);
          className = pnlToneClass(value);
          break;
        case "pnl_usd":
          display = formatUsd(value, 2);
          className = pnlToneClass(value);
          break;
        case "drawdown":
          display = formatPercent(value * 100, 2);
          className = value > 0.3 ? "text-amber-300" : "text-slate-100";
          break;
        case "win_rate":
          display = formatPercent(value * 100, 1);
          break;
        case "notional_usd":
          display = formatUsd(value, 0);
          break;
        case "trades":
          display = Math.round(value).toLocaleString();
          break;
        case "sharpe":
          display = value.toFixed(3);
          break;
        default:
          display = value.toFixed(3);
      }
    }
    return { key, label, value: display, className };
  });

  const formatTimestamp = (value) => {
    if (!value) return "—";
    try {
      const parsed = dayjs(value);
      if (parsed.isValid()) {
        return parsed.format("YYYY-MM-DD HH:mm:ss");
      }
    } catch (err) {
      return value;
    }
    return value;
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {statEntries.map(({ key, label, value, className }) => (
            <div key={key} className="rounded bg-slate-900/60 px-3 py-2">
              <div className="text-xs uppercase tracking-wide text-slate-400">{label}</div>
              <div className={clsx("text-sm font-semibold", className)}>
                {value}
              </div>
            </div>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <a
            href="/api/trades/export"
            className="rounded bg-emerald-600 px-3 py-2 text-xs font-semibold text-white hover:bg-emerald-500"
          >
            Export CSV
          </a>
          <button
            type="button"
            className="rounded bg-slate-800 px-3 py-2 text-xs font-semibold text-slate-200 hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50"
            onClick={() => (typeof onClear === "function" ? onClear() : null)}
            disabled={clearing || typeof onClear !== "function"}
          >
            {clearing ? "Clearing…" : "Clear log"}
          </button>
        </div>
      </div>
      <div className="rounded-lg border border-slate-800 bg-slate-900/50">
        <table className="min-w-full divide-y divide-slate-800 text-xs">
          <thead className="bg-slate-900/80 text-slate-300">
            <tr>
              <th className="px-3 py-2 text-left">Timestamp</th>
              <th className="px-3 py-2 text-left">Strategy</th>
              <th className="px-3 py-2 text-left">Symbol</th>
              <th className="px-3 py-2 text-left">Timeframe</th>
              <th className="px-3 py-2 text-left">Status</th>
              <th className="px-3 py-2 text-right">PnL</th>
              <th className="px-3 py-2 text-right">Confidence</th>
              <th className="px-3 py-2 text-right">Volatility</th>
              <th className="px-3 py-2 text-right">Size</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800 text-slate-200">
            {recent.length === 0 && (
              <tr>
                <td colSpan="9" className="px-3 py-3 text-center text-slate-500">
                  No trades recorded yet.
                </td>
              </tr>
            )}
            {recent.map((trade, idx) => (
              <tr key={idx}>
                <td className="px-3 py-2">{formatTimestamp(trade.timestamp)}</td>
                <td className="px-3 py-2">
                  {resolveStrategyLabel(trade.strategy || (trade.meta && trade.meta.strategy))}
                </td>
                <td className="px-3 py-2">{trade.symbol || "—"}</td>
                <td className="px-3 py-2">{trade.timeframe || (trade.meta && trade.meta.timeframe) || "—"}</td>
                <td className="px-3 py-2">
                  {(() => {
                    const { label, tone } = resolveStatus(trade.status);
                    return (
                      <span
                        className={clsx(
                          "inline-flex items-center rounded px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide",
                          statusToneClass(tone)
                        )}
                      >
                        {label}
                      </span>
                    );
                  })()}
                </td>
                <td className="px-3 py-2 text-right">
                  {(() => {
                    const pnlPct =
                      typeof trade.pnl_pct === "number" && Number.isFinite(trade.pnl_pct)
                        ? trade.pnl_pct
                        : null;
                    const pnlUsd =
                      typeof trade.pnl_value_usd === "number" && Number.isFinite(trade.pnl_value_usd)
                        ? trade.pnl_value_usd
                        : typeof trade.pnl === "number"
                        ? trade.pnl
                        : null;
                    const pnlBase =
                      typeof trade.pnl_value_base === "number" && Number.isFinite(trade.pnl_value_base)
                        ? trade.pnl_value_base
                        : null;
                    const pnlAud =
                      typeof trade.pnl_value_aud === "number" && Number.isFinite(trade.pnl_value_aud)
                        ? trade.pnl_value_aud
                        : null;
                    const symbolText =
                      typeof trade.symbol === "string" ? trade.symbol : "";
                    const baseCurrency =
                      trade.base_currency ||
                      (symbolText
                        ? symbolText.replace(/(USDT|USDC|USD|PERP)$/i, "")
                        : "");
                    const toneClass = pnlToneClass(pnlPct !== null ? pnlPct : pnlUsd);
                    return (
                      <div className="space-y-0.5">
                        <div className={clsx("font-semibold", toneClass)}>
                          {pnlPct !== null ? formatPercent(pnlPct, 2) : formatUsd(pnlUsd, 2)}
                        </div>
                        <div className={clsx("text-[11px]", toneClass, "opacity-80")}>
                          {formatUsd(pnlUsd, 2)}
                        </div>
                        <div className={clsx("text-[11px]", toneClass, "opacity-80")}>
                          {formatBaseValue(pnlBase, baseCurrency)}
                        </div>
                        <div className={clsx("text-[11px]", toneClass, "opacity-80")}>
                          {formatAud(pnlAud, 2)}
                        </div>
                      </div>
                    );
                  })()}
                </td>
                <td className="px-3 py-2 text-right">
                  {trade.confidence !== undefined && trade.confidence !== null
                    ? Number(trade.confidence).toFixed(3)
                    : "—"}
                </td>
                <td className="px-3 py-2 text-right">
                  {trade.volatility !== undefined && trade.volatility !== null
                    ? Number(trade.volatility).toFixed(3)
                    : "—"}
                </td>
                <td className="px-3 py-2 text-right">
                  {(() => {
                    const positionSize =
                      typeof trade.position_size === "number" && Number.isFinite(trade.position_size)
                        ? trade.position_size
                        : null;
                    const notionalUsd =
                      typeof trade.notional_usd === "number" && Number.isFinite(trade.notional_usd)
                        ? trade.notional_usd
                        : positionSize;
                    const notionalBase =
                      typeof trade.notional_base === "number" && Number.isFinite(trade.notional_base)
                        ? trade.notional_base
                        : null;
                    const notionalAud =
                      typeof trade.notional_aud === "number" && Number.isFinite(trade.notional_aud)
                        ? trade.notional_aud
                        : null;
                    const symbolText =
                      typeof trade.symbol === "string" ? trade.symbol : "";
                    const baseCurrency =
                      trade.base_currency ||
                      (symbolText
                        ? symbolText.replace(/(USDT|USDC|USD|PERP)$/i, "")
                        : "");
                    return (
                      <div className="space-y-0.5">
                        <div className="font-mono text-sm text-slate-200">
                          {positionSize !== null ? formatNumber(positionSize, 2) : "—"}
                        </div>
                        <div className="text-[11px] text-slate-400">
                          {notionalUsd !== null ? formatUsd(notionalUsd, 2) : "—"}
                        </div>
                        <div className="text-[11px] text-slate-400">
                          {formatBaseValue(notionalBase, baseCurrency)}
                        </div>
                        <div className="text-[11px] text-slate-400">
                          {formatAud(notionalAud, 2)}
                        </div>
                      </div>
                    );
                  })()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function DecisionTelemetry({ data }) {
  const metrics = (data && data.metrics) || {};
  const events = Array.isArray(data && data.events) ? data.events : [];
  const status = (data && data.status) || "idle";
  const heartbeat = data && data.heartbeat;
  const activeModes = Array.isArray(data && data.active_modes) ? data.active_modes : [];
  const lastEventAt =
    data && data.last_event_at ? data.last_event_at : metrics.last_event_at || null;

  const stageEntries = Object.entries(metrics.by_stage || {}).map(([stage, count]) => ({
    stage,
    count,
  }));

  const strategyEntries = Object.entries(metrics.by_strategy || {}).map(([strategy, info]) => ({
    strategy,
    total: info.total || 0,
    byStage: info.by_stage || {},
  }));

  const formatStage = (stage) =>
    String(stage || "unknown")
      .replace(/_/g, " ")
      .replace(/\b\w/g, (match) => match.toUpperCase());

  const statusLabel =
    status === "scanning"
      ? `Scanning${activeModes.length ? ` (${activeModes.join(", ")})` : ""}`
      : "Idle";
  const statusColour =
    status === "scanning"
      ? "bg-emerald-900/40 text-emerald-200 border border-emerald-800/60"
      : "bg-slate-800 text-slate-200 border border-slate-700";
  const heartbeatLabel = heartbeat ? dayjs(heartbeat).fromNow() : "No heartbeat yet";
  const lastEventLabel = lastEventAt ? dayjs(lastEventAt).fromNow() : "Never";

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-200">
          Decision telemetry
        </h3>
        <div className="flex items-center gap-2">
          <span className="rounded-full bg-slate-800 px-2 py-0.5 text-xs font-semibold text-slate-200">
            {metrics.total_events || 0} events
          </span>
          <span className={clsx("rounded-full px-2 py-0.5 text-[11px] font-semibold", statusColour)}>
            {statusLabel}
          </span>
        </div>
      </div>
      <div className="grid gap-3 md:grid-cols-2">
        <div className="space-y-2">
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">
            By stage
          </div>
          {stageEntries.length ? (
            stageEntries.map(({ stage, count }) => (
              <div
                key={stage}
                className="flex items-center justify-between rounded border border-slate-800 px-3 py-1 text-xs text-slate-200"
              >
                <span>{formatStage(stage)}</span>
                <span className="font-mono text-slate-100">{count}</span>
              </div>
            ))
          ) : (
            <div className="rounded border border-slate-800 px-3 py-2 text-xs text-slate-500">
              No decision events recorded yet.
            </div>
          )}
        </div>
        <div className="space-y-2">
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">
            By strategy
          </div>
          {strategyEntries.length ? (
            strategyEntries.map(({ strategy, total, byStage }) => (
              <div
                key={strategy}
                className="rounded border border-slate-800 px-3 py-2 text-xs text-slate-200 space-y-1"
              >
                <div className="flex items-center justify-between font-semibold">
                  <span>{strategy}</span>
                  <span className="font-mono text-slate-100">{total}</span>
                </div>
                {Object.keys(byStage).length > 0 && (
                  <div className="grid grid-cols-2 gap-1 text-[11px] text-slate-400">
                    {Object.entries(byStage).map(([stage, count]) => (
                      <div key={`${strategy}-${stage}`} className="flex items-center justify-between">
                        <span>{formatStage(stage)}</span>
                        <span className="font-mono text-slate-200">{count}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))
          ) : (
            <div className="rounded border border-slate-800 px-3 py-2 text-xs text-slate-500">
              No strategy-level decisions observed.
            </div>
          )}
        </div>
      </div>
      <div className="space-y-2">
        <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">
          Recent events
        </div>
        <div className="max-h-56 overflow-y-auto divide-y divide-slate-800 rounded border border-slate-800 text-xs text-slate-200">
          {events.length ? (
            events.slice(0, 20).map((event, idx) => (
              <div key={`${event.timestamp}-${idx}`} className="px-3 py-2 space-y-0.5">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[11px] text-slate-400">
                    {event.timestamp ? dayjs(event.timestamp).format("YYYY-MM-DD HH:mm:ss") : "—"}
                  </span>
                  <span className="rounded-full bg-slate-800 px-2 py-0.5 text-[11px] font-semibold text-slate-200">
                    {formatStage(event.stage)}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Strategy: {event.strategy || "—"}</span>
                  <span className="text-slate-400">Symbol: {event.symbol || "—"}</span>
                </div>
                {event.details && Object.keys(event.details).length > 0 && (
                  <div className="text-[11px] text-slate-400">
                    Details: {JSON.stringify(event.details)}
                  </div>
                )}
              </div>
            ))
          ) : (
            <div className="px-3 py-4 space-y-1 text-slate-500">
              <div>No events recorded yet.</div>
              <div className="text-[11px] text-slate-400">
                Scanner status: {statusLabel}. Heartbeat {heartbeatLabel}. Last event {lastEventLabel}.
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function LearningPanel({ state, onToggle }) {
  const online = !!(state && state.online_learning);
  const periodic = !!(state && state.periodic_retraining);

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-4 space-y-4">
      <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-200">
        Learning & Adaptation
      </h3>
      <div className="flex flex-col gap-3 text-sm text-slate-200">
        <label className="flex items-center justify-between gap-4">
          <span>
            <span className="font-semibold">Online learning</span>
            <span className="ml-2 text-xs text-slate-400">
              Let the agent adjust parameters continuously while trading.
            </span>
          </span>
          <button
            className={clsx(
              "rounded px-3 py-1 text-xs font-semibold",
              online ? "bg-emerald-600 text-white" : "bg-slate-800 text-slate-200"
            )}
            onClick={() => onToggle({ online_learning: !online })}
          >
            {online ? "Enabled" : "Disabled"}
          </button>
        </label>
        <label className="flex items-center justify-between gap-4">
          <span>
            <span className="font-semibold">Periodic retraining</span>
            <span className="ml-2 text-xs text-slate-400">
              Schedule offline retraining cycles using the latest data.
            </span>
          </span>
          <button
            className={clsx(
              "rounded px-3 py-1 text-xs font-semibold",
              periodic ? "bg-emerald-600 text-white" : "bg-slate-800 text-slate-200"
            )}
            onClick={() => onToggle({ periodic_retraining: !periodic })}
          >
            {periodic ? "Enabled" : "Disabled"}
          </button>
        </label>
      </div>
      <div className="rounded border border-slate-800 bg-slate-950/60 p-3 text-xs text-slate-400">
        TradingView webhook endpoint ready at <code>/api/feeds/alerts</code>. Configure your TradingView alert
        payload with the shared secret from <code>TRADINGVIEW_SHARED_SECRET</code>. Alerts are forwarded into Kafka
        and DuckDB for downstream learners.
      </div>
    </div>
  );
}

function SystemControls({ state, onKillSwitch, onRestart, working }) {
  const killArmed = !!(state && state.kill_switch);
  const drawdown = state && state.drawdown_threshold ? state.drawdown_threshold : null;
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-4 space-y-4">
      <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-200">
        System controls
      </h3>
      <div className="flex flex-wrap items-center gap-3">
        <button
          className={clsx(
            "rounded px-4 py-2 text-sm font-semibold",
            killArmed ? "bg-red-700 text-white" : "bg-emerald-600 text-white"
          )}
          onClick={() => onKillSwitch(!killArmed)}
          disabled={working}
        >
          {killArmed ? "Disable kill switch" : "Enable kill switch"}
        </button>
        <button
          className="rounded bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-500 disabled:opacity-60"
          onClick={onRestart}
          disabled={working}
        >
          Restart orchestrator
        </button>
        {state && state.last_restart && (
          <span className="text-xs text-slate-400">
            Last restart {dayjs(state.last_restart).fromNow()}
          </span>
        )}
      </div>
      {drawdown !== null && (
        <div className="text-xs text-slate-400">
          Kill switch threshold: {(drawdown * 100).toFixed(2)}% drawdown
        </div>
      )}
    </div>
  );
}

function KafkaControls({ status, onAction, working }) {
  const manageEnabled = status ? status.manage_enabled !== false : true;
  const running = !!(status && status.running);
  const error = status && status.error;
  const services = Array.isArray(status && status.services) ? status.services : [];
  const hasServices = services.length > 0;
  const availability = status ? status.available !== false : false;

  const pillClass = running
    ? "bg-emerald-800/60 text-emerald-100 border border-emerald-500/60"
    : "bg-red-900/50 text-red-200 border border-red-500/60";

  const serviceRows = services.map((svc, idx) => {
    const name = svc.Service || svc.Name || svc.container || `service-${idx}`;
    const state = svc.State || svc.status || svc.Status || "";
    const health = svc.Health || svc.health || "";
    return (
      <div key={name} className="flex items-center justify-between rounded border border-slate-800/60 px-3 py-1 text-xs">
        <span className="font-medium text-slate-200">{name}</span>
        <span className="text-slate-400">
          {state}
          {health ? ` (${health})` : ""}
        </span>
      </div>
    );
  });

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-200">Kafka &amp; Feeds</h3>
        <span className={clsx("rounded-full px-2 py-0.5 text-xs font-semibold", pillClass)}>
          {running ? "Running" : "Stopped"}
        </span>
      </div>
      {!manageEnabled && (
        <div className="rounded border border-slate-700 bg-slate-950/80 px-3 py-2 text-xs text-slate-300">
          RunBot is configured with <code>RUNBOT_SKIP_KAFKA=1</code>, so automatic management is disabled.
        </div>
      )}
      {error && (
        <div className="rounded border border-red-800 bg-red-900/40 px-3 py-2 text-xs text-red-100">
          {error}
        </div>
      )}
      {!availability && !error && (
        <div className="rounded border border-slate-700 bg-slate-950/80 px-3 py-2 text-xs text-slate-300">
          Compose file not detected yet. Start RunBot or configure the stack to enable management.
        </div>
      )}
      <div className="flex flex-wrap gap-2">
        <button
          className="rounded bg-emerald-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-emerald-500 disabled:opacity-50"
          onClick={() => onAction("start")}
          disabled={working || running || !manageEnabled}
        >
          Start
        </button>
        <button
          className="rounded bg-red-700 px-3 py-1.5 text-xs font-semibold text-white hover:bg-red-600 disabled:opacity-50"
          onClick={() => onAction("stop")}
          disabled={working || !running || !manageEnabled}
        >
          Stop
        </button>
        <button
          className="rounded bg-slate-800 px-3 py-1.5 text-xs font-semibold text-slate-200 hover:bg-slate-700 disabled:opacity-50"
          onClick={() => onAction("restart")}
          disabled={working || !manageEnabled}
        >
          Restart
        </button>
      </div>
      <div className="space-y-2 text-xs text-slate-300">
        {hasServices ? (
          serviceRows
        ) : (
          <p className="text-slate-500">No container status reported yet.</p>
        )}
      </div>
    </div>
  );
}

function App() {
  const strategiesPoll = usePolling("/api/strategies", POLL_INTERVAL_FAST, {
    strategies: [],
    portfolio: null,
    alerts: [],
  });
  const systemPoll = usePolling("/api/system/state", POLL_INTERVAL_FAST, {
    state: {},
    alerts: [],
  });
  const kafkaPoll = usePolling("/api/system/kafka", POLL_INTERVAL_FAST, {});
  const tradingPoll = usePolling("/api/trading/status", POLL_INTERVAL_FAST, {
    paper: {},
    live: {},
    guardrails: {},
    open_positions: { count: 0, notional_usd: 0, pnl_usd: 0, symbols: {} },
  });
  const backtestsPoll = usePolling("/api/backtests", POLL_INTERVAL_SLOW, {
    jobs: [],
    results: [],
    guardrail_logs: [],
    guardrail_snapshots: [],
  });
  const configsPoll = usePolling("/api/configs", POLL_INTERVAL_SLOW, { configs: [] });
  const tradesPoll = usePolling("/api/trades?limit=400", POLL_INTERVAL_SLOW, { records: [], stats: {} });
  const tradeMetricsPoll = usePolling("/api/trades/metrics", POLL_INTERVAL_SLOW, {
    metrics: { total_events: 0, by_stage: {}, by_strategy: {} },
    events: [],
  });

  const strategyRows = useMemo(
    () =>
      Array.isArray(strategiesPoll.data && strategiesPoll.data.strategies)
        ? strategiesPoll.data.strategies
        : [],
    [strategiesPoll.data]
  );

  const manifestData = useMemo(() => {
    const manifest = strategiesPoll.data && strategiesPoll.data.manifest;
    return manifest && typeof manifest === "object" ? manifest : {};
  }, [strategiesPoll.data]);

  const strategyMetadata = useMemo(() => {
    const meta = {};
    Object.entries(manifestData).forEach(([key, entry]) => {
      if (!entry || typeof entry !== "object") {
        return;
      }
      const recommended = Array.isArray(entry.timeframes) ? entry.timeframes.map(String) : [];
      meta[key] = {
        label: entry.label || key,
        description: entry.description || "",
        recommended_timeframes: recommended,
        default_timeframe: entry.default || (recommended[0] || ""),
      };
    });
    strategyRows.forEach((row) => {
      if (!row || !row.strategy) {
        return;
      }
      const key = row.strategy;
      const existing = meta[key] || {
        label: row.strategy_label || key,
        description: row.strategy_description || "",
        recommended_timeframes: [],
        default_timeframe: "",
      };
      const recommended = Array.isArray(row.recommended_timeframes)
        ? row.recommended_timeframes.map(String)
        : [];
      const merged = Array.isArray(existing.recommended_timeframes)
        ? existing.recommended_timeframes.slice()
        : [];
      recommended.forEach((tf) => {
        if (!merged.includes(tf)) {
          merged.push(tf);
        }
      });
      existing.recommended_timeframes = merged;
      if (!existing.default_timeframe) {
        const defaultCandidate = row.default_timeframe || merged[0];
        if (defaultCandidate) {
          existing.default_timeframe = defaultCandidate;
        }
      }
      if (!existing.label && row.strategy_label) {
        existing.label = row.strategy_label;
      }
      meta[key] = existing;
    });
    return meta;
  }, [manifestData, strategyRows]);

  const strategyOptions = useMemo(() => {
    const set = new Set(Object.keys(strategyMetadata));
    if (!set.size) {
      strategyRows.forEach((row) => {
        if (row && row.strategy) {
          set.add(row.strategy);
        }
      });
    }
    if (!set.size && configsPoll.data && Array.isArray(configsPoll.data.configs)) {
      configsPoll.data.configs.forEach((entry) => {
        if (entry && entry.name) {
          const base = entry.name.replace(/\.[^.]+$/, "");
          set.add(base);
        }
      });
    }
    if (!set.size) {
      DEFAULT_STRATEGIES.forEach((strategy) => set.add(strategy));
    }
    return Array.from(set).sort((a, b) => {
      const labelA = (strategyMetadata[a] && strategyMetadata[a].label) || a;
      const labelB = (strategyMetadata[b] && strategyMetadata[b].label) || b;
      return labelA.localeCompare(labelB);
    });
  }, [strategyMetadata, strategyRows, configsPoll.data]);

  const symbolOptions = useMemo(() => {
    const set = new Set();
    strategyRows.forEach((row) => {
      if (row && row.symbol) {
        set.add(String(row.symbol).toUpperCase());
      }
    });
    if (!set.size) {
      DEFAULT_SYMBOLS.forEach((symbol) => set.add(symbol));
    }
    return Array.from(set).sort();
  }, [strategyRows]);

  const timeframeOptions = useMemo(() => {
    const set = new Set(DEFAULT_TIMEFRAMES);
    strategyRows.forEach((row) => {
      if (row && row.timeframe) {
        set.add(String(row.timeframe));
      }
      if (row && Array.isArray(row.recommended_timeframes)) {
        row.recommended_timeframes.forEach((tf) => set.add(String(tf)));
      }
    });
    Object.values(strategyMetadata).forEach((entry) => {
      if (entry && Array.isArray(entry.recommended_timeframes)) {
        entry.recommended_timeframes.forEach((tf) => set.add(String(tf)));
      }
    });
    const ordering = (value) => {
      const index = DEFAULT_TIMEFRAMES.indexOf(value);
      return index === -1 ? DEFAULT_TIMEFRAMES.length + value.charCodeAt(0) : index;
    };
    return Array.from(set).sort((a, b) => {
      const aRank = ordering(a);
      const bRank = ordering(b);
      if (aRank === bRank) {
        return a.localeCompare(b);
      }
      return aRank - bRank;
    });
  }, [strategyRows, strategyMetadata]);

const capitalOptions = useMemo(() => DEFAULT_CAPITAL_OPTIONS, []);

const [learningState, setLearningState] = useState(
  (systemPoll.data && systemPoll.data.state) || {}
);
useEffect(() => {
  if (systemPoll.data && systemPoll.data.state) {
    setLearningState(systemPoll.data.state);
  }
}, [systemPoll.data]);

const [messages, setMessages] = useState([]);
const [filters, setFilters] = useState({
  showUnderperforming: false,
  minTrades: 0,
  sortKey: "sharpe",
  sortDirection: "desc",
});
const [working, setWorking] = useState(false);
const [launching, setLaunching] = useState(false);
const [kafkaWorking, setKafkaWorking] = useState(false);
const [tradingWorking, setTradingWorking] = useState({ paper: false, live: false, stop: null });
const [clearingTrades, setClearingTrades] = useState(false);
const [clearingJobs, setClearingJobs] = useState(false);
const [clearingGuardrails, setClearingGuardrails] = useState(false);

const pushMessage = useCallback((text, level = "info") => {
  setMessages((prev) => [
    ...prev,
    {
      id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      level,
      text,
    },
  ]);
}, []);

const dismissMessage = (id) => {
  setMessages((prev) => prev.filter((msg) => msg.id !== id));
};

const [bitmexCreds, setBitmexCreds] = useState({ configured: false });
const [bitmexForm, setBitmexForm] = useState({ api_key: "", api_secret: "" });
const [savingBitmexCreds, setSavingBitmexCreds] = useState(false);

const fetchBitmexCreds = useCallback(async () => {
  try {
    const data = await apiRequest("/api/exchanges/bitmex/credentials", { method: "GET" });
    if (data) {
      setBitmexCreds(data);
    }
  } catch (err) {
    pushMessage(`Unable to load BitMEX credentials: ${err.message}`, "warning");
  }
}, [pushMessage]);

useEffect(() => {
  fetchBitmexCreds();
}, [fetchBitmexCreds]);

const handleClearTrades = async () => {
  if (clearingTrades) {
    return;
  }
  try {
    setClearingTrades(true);
    await apiRequest("/api/trades/clear", { method: "POST" });
    pushMessage("Trade log cleared.", "success");
    tradesPoll.refresh();
    tradeMetricsPoll.refresh();
  } catch (err) {
    pushMessage(`Clear failed: ${err.message}`, "critical");
  } finally {
    setClearingTrades(false);
  }
};

const handleClearJobs = async () => {
  if (clearingJobs) {
    return;
  }
  try {
    setClearingJobs(true);
    await apiRequest("/api/backtests/jobs/clear", { method: "POST" });
    pushMessage("Job queue cleared.", "success");
    backtestsPoll.refresh();
  } catch (err) {
    pushMessage(`Clear job queue failed: ${err.message}`, "critical");
  } finally {
    setClearingJobs(false);
  }
};

const handleClearGuardrails = async () => {
  if (clearingGuardrails) {
    return;
  }
  try {
    setClearingGuardrails(true);
    await apiRequest("/api/guardrails/clear", { method: "POST" });
    pushMessage("Guardrail logs cleared.", "success");
    backtestsPoll.refresh();
  } catch (err) {
    pushMessage(`Clear guardrail logs failed: ${err.message}`, "critical");
  } finally {
    setClearingGuardrails(false);
  }
};

const handlePromote = async (row) => {
  if (!row || !row.config) {
    pushMessage("No config available for promotion.", "warning");
    return;
  }
  try {
    const payload = await apiRequest("/api/strategies/promote", {
      body: { config: row.config },
    });
    pushMessage("Strategy promoted successfully.", "success");
    strategiesPoll.refresh();
    systemPoll.refresh();
    backtestsPoll.refresh();
    if (payload && payload.summary && payload.summary.destination) {
      pushMessage(`Active config updated: ${payload.summary.destination}`, "info");
    }
  } catch (err) {
    pushMessage(`Promotion failed: ${err.message}`, "critical");
  }
};

const handleSnapshot = (row) => {
  if (!row || !row.config) {
    pushMessage("No config available to snapshot.", "warning");
    return;
  }
  pushMessage(`Snapshot stored for ${row.config}`, "info");
};



const handleStartTrading = async (mode, payload) => {
  const endpoint = mode === "paper" ? "/api/trading/paper" : "/api/trading/live";
  try {
    setTradingWorking((prev) => ({ ...prev, [mode]: true }));
    const response = await apiRequest(endpoint, { body: payload });
    const message = response && response.message ? response.message : `${mode === "paper" ? "Paper" : "Live"} trading request submitted.`;
    const level = response && response.status === "ok" ? "success" : "info";
    pushMessage(message, level);
    if (response && Array.isArray(response.launch_plan) && response.launch_plan.length) {
      const summary = response.launch_plan
        .map((entry) => {
          const strat = entry.strategy || "?";
          const tf = entry.timeframe || "?";
          const sym = entry.symbol || "(default)";
          return `${strat} @ ${tf} (${sym})`;
        })
        .join(", ");
      pushMessage(`Plan: ${summary}`, "info");
    }
    if (response && Array.isArray(response.details) && response.details.length) {
      response.details.forEach((detail) => pushMessage(detail, "warning"));
    }
    tradingPoll.refresh();
  } catch (err) {
    pushMessage(`${mode === "paper" ? "Paper" : "Live"} trading failed: ${err.message}`, "critical");
  } finally {
    setTradingWorking((prev) => ({ ...prev, [mode]: false }));
  }
};

const handleStopTrading = async (mode) => {
  try {
    setTradingWorking((prev) => ({ ...prev, stop: mode || "all" }));
    const body = mode ? { mode } : {};
    const response = await apiRequest("/api/trading/stop", { body });
    const statusText = response && response.status ? response.status : "stopped";
    const level = statusText === "stopped" ? "success" : "info";
    const message = statusText === "stopped" ? "Trading processes stopped." : `Stop response: ${statusText}`;
    pushMessage(message, level);
    tradingPoll.refresh();
  } catch (err) {
    pushMessage(`Stop request failed: ${err.message}`, "critical");
  } finally {
    setTradingWorking((prev) => ({ ...prev, stop: null }));
  }
};

const handleBacktest = async (payload) => {
  try {
    setLaunching(true);
    const response = await apiRequest("/api/backtests", { body: payload });
    const ids = response.job_ids || [];
    if (ids.length) {
      pushMessage(`Queued backtests: ${ids.join(", ")}`, "success");
    } else {
      pushMessage("Backtest request acknowledged.", "info");
    }
    const plan = Array.isArray(response.launch_plan) ? response.launch_plan : [];
    if (plan.length) {
      const summary = plan
        .map((entry) => {
          const strat = entry.strategy || "?";
          const tf = entry.timeframe || "?";
          const symbols = Array.isArray(entry.symbols) && entry.symbols.length ? entry.symbols.join("/") : "default";
          return `${strat} @ ${tf} (${symbols})`;
        })
        .join(", ");
      pushMessage(`Launch plan: ${summary}`, "info");
    }
    if (Array.isArray(response.warnings) && response.warnings.length) {
      response.warnings.forEach((warning) => pushMessage(warning, "warning"));
    }
    backtestsPoll.refresh();
    strategiesPoll.refresh();
  } catch (err) {
    pushMessage(`Backtest launch failed: ${err.message}`, "critical");
  } finally {
    setLaunching(false);
  }
};

const handleConfigSave = async (entry, content) => {
  try {
    await apiRequest(`/api/configs/${encodeURIComponent(entry.name)}`, {
      method: "PUT",
      body: { content },
    });
    pushMessage(`Saved ${entry.name}`, "success");
    configsPoll.refresh();
  } catch (err) {
    pushMessage(`Save failed: ${err.message}`, "critical");
    throw err;
  }
};

const handleConfigRevert = async (entry) => {
  try {
    await apiRequest(`/api/configs/${encodeURIComponent(entry.name)}/revert`, {
      method: "POST",
    });
    pushMessage(`Reverted ${entry.name}`, "info");
    configsPoll.refresh();
  } catch (err) {
    pushMessage(`Revert failed: ${err.message}`, "critical");
    throw err;
  }
};

const handleLearningToggle = async (payload) => {
  try {
    const response = await apiRequest("/learning/toggle", { body: payload });
    if (response && response.state) {
      setLearningState(response.state);
    } else {
      setLearningState((prev) => ({ ...prev, ...payload }));
    }
    pushMessage("Learning settings updated.", "success");
    systemPoll.refresh();
  } catch (err) {
    pushMessage(`Update failed: ${err.message}`, "critical");
  }
};

const handleKillSwitch = async (armed) => {
  try {
    setWorking(true);
    await apiRequest("/api/system/kill-switch", { body: { armed } });
    pushMessage(`Kill switch ${armed ? "armed" : "disarmed"}.`, "success");
    systemPoll.refresh();
    strategiesPoll.refresh();
  } catch (err) {
    pushMessage(`Kill switch failed: ${err.message}`, "critical");
  } finally {
    setWorking(false);
  }
};

const handleRestart = async () => {
  try {
    setWorking(true);
    const payload = await apiRequest("/api/system/restart", { body: {} });
    pushMessage(
      `Restarted orchestrator. Cancelled jobs: ${payload.cancelled_jobs || 0}`,
      "success"
    );
    systemPoll.refresh();
    backtestsPoll.refresh();
    strategiesPoll.refresh();
  } catch (err) {
    pushMessage(`Restart failed: ${err.message}`, "critical");
  } finally {
    setWorking(false);
  }
};

const handleKafkaAction = async (action) => {
  const label = action ? action.charAt(0).toUpperCase() + action.slice(1) : "Status";
  if (kafkaWorking) {
    return;
  }
  try {
    setKafkaWorking(true);
    const response = await apiRequest("/api/system/kafka", { body: { action } });
    const success = response && response.success !== false;
    const errorDetail =
      response?.error ||
      response?.start?.error ||
      response?.stop?.error ||
      response?.stderr ||
      "";
    const message = success
      ? `Kafka ${label.toLowerCase()} request completed.`
      : `Kafka ${label.toLowerCase()} request encountered issues${errorDetail ? `: ${errorDetail}` : ""}.`;
    pushMessage(message, success ? "success" : "critical");
    kafkaPoll.refresh();
    systemPoll.refresh();
  } catch (err) {
    pushMessage(`Kafka ${label.toLowerCase()} failed: ${err.message}`, "critical");
  } finally {
    setKafkaWorking(false);
  }
};

const handleBitmexInputChange = (field) => (event) => {
  const value = event.target.value;
  setBitmexForm((prev) => ({ ...prev, [field]: value }));
};

const handleBitmexSubmit = async (event) => {
  event.preventDefault();
  if (savingBitmexCreds) {
    return;
  }
  const key = bitmexForm.api_key.trim();
  const secret = bitmexForm.api_secret.trim();
  if (!key || !secret) {
    pushMessage("API key and secret are required.", "warning");
    return;
  }
  try {
    setSavingBitmexCreds(true);
    const payload = await apiRequest("/api/exchanges/bitmex/credentials", {
      method: "POST",
      body: { api_key: key, api_secret: secret },
    });
    if (payload) {
      setBitmexCreds(payload);
      setBitmexForm({ api_key: "", api_secret: "" });
    }
    pushMessage("BitMEX credentials updated.", "success");
  } catch (err) {
    pushMessage(`Update failed: ${err.message}`, "critical");
  } finally {
    setSavingBitmexCreds(false);
  }
};

const alertsCombined = [
  ...(strategiesPoll.data && strategiesPoll.data.alerts ? strategiesPoll.data.alerts : []),
  ...(systemPoll.data && systemPoll.data.alerts ? systemPoll.data.alerts : []),
];
const tradingStatus = tradingPoll.data || {};
if (tradingStatus.guardrails && tradingStatus.guardrails.paper_is_recent === false) {
  alertsCombined.push({ level: "warning", message: "Paper trading session is stale. Run paper before going live." });
}
if (tradingPoll.error) {
  const message = tradingPoll.error.message || String(tradingPoll.error);
  alertsCombined.push({ level: "critical", message: `Trading status error: ${message}` });
}
if (kafkaPoll.error) {
  const message = kafkaPoll.error.message || String(kafkaPoll.error);
  alertsCombined.push({ level: "critical", message: `Kafka status error: ${message}` });
}
const kafkaStatus = kafkaPoll.data || {};

return (
  <div className="flex min-h-screen flex-col bg-slate-950 text-slate-100">
    <NotificationBar alerts={alertsCombined} messages={messages} onDismiss={dismissMessage} />
    <main className="mx-auto w-full max-w-7xl flex-1 space-y-10 px-4 py-8">
      <header className="space-y-3">
        <h1 className="text-2xl font-semibold text-white">AI Trading Bot Control Center</h1>
        <p className="text-sm text-slate-400">
          Monitor live strategies, launch research jobs, and steer the adaptive learning loop across BitMEX symbols.
        </p>
        <PortfolioSummary portfolio={strategiesPoll.data && strategiesPoll.data.portfolio} />
      </header>

      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-white">BitMEX API access</h2>
        <div className="space-y-4 rounded-lg border border-slate-800 bg-slate-900/60 p-4">
          <p className="text-sm text-slate-300">
            Provide a BitMEX API key so the bot can download market data and place simulated or live orders.
            Use an API key with <span className="font-semibold text-white">Order/Trade</span> permissions only
            (no withdrawal rights). If you rotate keys or switch accounts, submitting the form overwrites the existing credentials.
          </p>
          <ul className="list-disc space-y-1 pl-5 text-sm text-slate-300">
            <li>Create the key under <span className="font-semibold">API Key Restrictions → \"Order\"</span>.</li>
            <li>Disable withdrawal permissions and keep the key scoped to the trading sub-account you intend to automate.</li>
            <li>You can remove or replace the key at any time by saving new values below.</li>
          </ul>
          <div className="text-xs text-slate-400">
            Status:{" "}
            {bitmexCreds.configured ? (
              <span className="text-emerald-300">
                Configured {bitmexCreds.api_key_preview ? `(key ${bitmexCreds.api_key_preview})` : ""}
                {bitmexCreds.updated_at ? ` · Updated ${dayjs(bitmexCreds.updated_at).fromNow()}` : ""}
              </span>
            ) : (
              <span className="text-amber-300">Not configured</span>
            )}
          </div>
          <form className="space-y-3" onSubmit={handleBitmexSubmit}>
            <div>
              <label className="text-xs uppercase tracking-wide text-slate-400">API key</label>
              <input
                type="text"
                className="mt-1 w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none"
                value={bitmexForm.api_key}
                onChange={handleBitmexInputChange("api_key")}
                placeholder="BMEX123..."
              />
            </div>
            <div>
              <label className="text-xs uppercase tracking-wide text-slate-400">API secret</label>
              <input
                type="password"
                className="mt-1 w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none"
                value={bitmexForm.api_secret}
                onChange={handleBitmexInputChange("api_secret")}
                placeholder="••••••"
              />
            </div>
            <div className="flex flex-wrap items-center gap-3 text-xs text-slate-400">
              <button
                type="submit"
                className="rounded bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-60"
                disabled={savingBitmexCreds}
              >
                {bitmexCreds.configured ? "Overwrite credentials" : "Save credentials"}
              </button>
              <span>Saving replaces any previously stored key.</span>
            </div>
          </form>
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-white">Strategy monitoring</h2>
        <StrategyTable
          strategies={strategiesPoll.data && strategiesPoll.data.strategies}
          filters={filters}
          onFilterChange={setFilters}
          onPromote={handlePromote}
          onSnapshot={handleSnapshot}
        />
      </section>

      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-white">Backtest management</h2>
        <BacktestPanel
          data={backtestsPoll.data || {}}
          onLaunch={handleBacktest}
          launching={launching}
          strategyOptions={strategyOptions}
          strategyMetadata={strategyMetadata}
          symbolOptions={symbolOptions}
          timeframeOptions={timeframeOptions}
          capitalOptions={capitalOptions}
          onNotify={pushMessage}
          onRefresh={backtestsPoll.refresh}
          onClearJobs={handleClearJobs}
          clearingJobs={clearingJobs}
          onClearGuardrails={handleClearGuardrails}
          clearingGuardrails={clearingGuardrails}
        />
      </section>

<section className="space-y-4">
  <h2 className="text-lg font-semibold text-white">Execution control</h2>
  <TradingControls
    status={tradingPoll.data || {}}
    strategyOptions={strategyOptions}
    strategyMetadata={strategyMetadata}
    symbolOptions={symbolOptions}
    timeframeOptions={timeframeOptions}
    capitalOptions={capitalOptions}
    onStartPaper={(payload) => handleStartTrading("paper", payload)}
    onStartLive={(payload) => handleStartTrading("live", payload)}
    onStop={handleStopTrading}
    onNotify={pushMessage}
    working={tradingWorking}
  />
</section>


      <section className="grid gap-6 md:grid-cols-[2fr,1fr]">
        <div className="space-y-4">
          <h2 className="text-lg font-semibold text-white">Trade log & rolling stats</h2>
          <TradesPanel
            data={tradesPoll.data || {}}
            strategyMetadata={strategyMetadata}
            onClear={handleClearTrades}
            clearing={clearingTrades}
          />
          <DecisionTelemetry data={tradeMetricsPoll.data || {}} />
        </div>
        <div className="space-y-4">
          <KafkaControls status={kafkaStatus} onAction={handleKafkaAction} working={kafkaWorking} />
          <SystemControls
            state={systemPoll.data && systemPoll.data.state}
            onKillSwitch={handleKillSwitch}
            onRestart={handleRestart}
            working={working}
          />
          <LearningPanel state={learningState} onToggle={handleLearningToggle} />
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-white">Configuration management</h2>
        <ConfigManager
          data={configsPoll.data || { configs: [] }}
          onSave={handleConfigSave}
          onRevert={handleConfigRevert}
          loading={configsPoll.loading}
        />
      </section>
    </main>
  </div>
);
}

const rootElement = document.getElementById("app-root");
if (rootElement) {
  const root = ReactDOM.createRoot(rootElement);
  root.render(<App />);
}

"""Rule-based meta strategy selector with funding-aware sizing."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import yaml

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - pyarrow is a hard dependency for caching
    raise ImportError("pyarrow is required for meta selector caching") from exc


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "meta_selector.yaml"
DEFAULT_EXPORT_DIR = PROJECT_ROOT / "results" / "meta_decisions"


def _safe_read_parquet(path: Path) -> Optional[pd.DataFrame]:
    """Read a parquet file if valid, otherwise quarantine the bad file."""
    try:
        if not path.exists() or path.stat().st_size < 16:
            return None
        pq.ParquetFile(path)  # validates footer and schema
        return pd.read_parquet(path, engine="pyarrow")
    except Exception as exc:  # pragma: no cover - safety net
        try:
            bad_suffix = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            quarantined = path.with_suffix(path.suffix + f".bad.{bad_suffix}")
            shutil.move(str(path), quarantined)
            logger.warning("Meta cache invalid; moved to %s: %s", quarantined, exc)
        except Exception:
            logger.warning("Meta cache invalid and could not be moved: %s", exc)
        return None


def _safe_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Atomically write dataframe to parquet to avoid partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".parquet", delete=False
    ) as tmp_handle:
        tmp_path = Path(tmp_handle.name)
    try:
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


@dataclass
class Context:
    timestamp: pd.Timestamp
    symbol: str
    regime: str
    features: Dict[str, float]
    model_outputs: Dict[str, float]
    market_state: Dict[str, float]
    risk_state: Dict[str, float]
    health: Dict[str, float] = field(default_factory=dict)
    mode: str = "live"


@dataclass
class Decision:
    timestamp: pd.Timestamp
    symbol: str
    regime: str
    strategy_id: str
    weight: float
    direction: str
    size_frac: float
    confidence: float
    execution: str
    rationale: str

    def to_record(self) -> Dict[str, object]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class MetaStrategySelector:
    """Rule-based selector that combines strategy scores, funding bias and risk gates."""

    def __init__(
        self,
        config_path: Path | str = DEFAULT_CONFIG_PATH,
        export_dir: Path | str = DEFAULT_EXPORT_DIR,
        flush_threshold: int = 100,
        enable_cache: bool = True,
    ) -> None:
        self.config_path = Path(config_path)
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.flush_threshold = flush_threshold
        self._buffer: Dict[str, List[Dict[str, object]]] = {}
        self._stats: Counter[str] = Counter()
        self.mode: str = "live"
        self.enable_cache = bool(enable_cache)
        self._load_config()

    def _load_config(self) -> None:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Meta selector config not found at {self.config_path}")
        with self.config_path.open("r", encoding="utf-8") as fh:
            self.config = yaml.safe_load(fh)
        self.weights: Dict[str, Dict[str, float]] = self.config["weights"]
        self.thresholds = self.config["thresholds"]
        self.limits = self.config["limits"]
        self.sizing = self.config["sizing"]
        self.policy = self.config.get("policy", "rules")
        self.strategy_ids = list(self.weights.keys())
        self.feature_keys = sorted({key for mapping in self.weights.values() for key in mapping.keys()})

    def reload(self) -> None:
        self._load_config()

    def set_mode(self, mode: str) -> None:
        """Set operating mode (e.g., 'live' or 'backtest')."""
        self.mode = mode.lower()

    def set_cache_enabled(self, enabled: bool) -> None:
        """Toggle on-disk cache persistence."""
        self.enable_cache = bool(enabled)

    def create_context_from_row(
        self,
        timestamp: pd.Timestamp,
        symbol: str,
        regime: str,
        row: pd.Series,
        probability: float,
        health: Optional[Dict[str, float]] = None,
    ) -> Context:
        if not isinstance(row, pd.Series):
            row = pd.Series()
        features = {k: float(row.get(k, 0.0)) for k in self.feature_keys}

        vol_regime = str(row.get("vol_regime", "")).lower()
        features.setdefault("vol_regime_high", 1.0 if vol_regime == "high" else 0.0)
        features.setdefault("vol_regime_low", 1.0 if vol_regime == "low" else 0.0)
        features.setdefault("latency_ok", float(row.get("latency_ok", 1.0)))

        funding_z = float(row.get("funding_z", row.get("funding_z_score", 0.0)))
        features.setdefault("funding_z", funding_z)
        adx = float(row.get("adx", row.get("trend_strength", 0.0)))
        features.setdefault("adx", adx)

        model_outputs = {
            "p_long": float(probability),
            "p_short": float(max(0.0, 1.0 - probability)),
            "p_hold": 1.0 - float(probability),
        }

        market_state = {
            "vol_regime": vol_regime,
            "spread_z": float(row.get("spread_z", 0.0)),
            "latency_watchdog": float(row.get("latency_watchdog", 0.0)),
            "funding_z": funding_z,
        }

        liquidity_ratio = float(row.get("liquidity_buffer_to_atr", row.get("liquidity_buffer_ratio", 999.0)))
        risk_state = {
            "liquidity_buffer_to_atr": liquidity_ratio,
            "daily_pnl_pct": float(row.get("daily_pnl_pct", 0.0)),
            "current_leverage": float(row.get("current_leverage", 0.0)),
            "atr": float(row.get("atr", 0.0)),
        }

        health_state = dict(health or {})
        health_state.setdefault("latency_ok", float(row.get("latency_ok", features.get("latency_ok", 1.0))))
        health_state.setdefault("liq_buffer_atr", liquidity_ratio)
        health_state.setdefault("simulated", 0.0)

        if self.mode != "live":
            features["latency_ok"] = 1.0
            market_state["latency_watchdog"] = 0.0
            risk_state["liquidity_buffer_to_atr"] = max(liquidity_ratio, 999.0)
            health_state["latency_ok"] = 1.0
            health_state["liq_buffer_atr"] = 999.0
            health_state["simulated"] = max(health_state.get("simulated", 1.0), 1.0)

        return Context(
            timestamp=timestamp,
            symbol=symbol,
            regime=regime,
            features=features,
            model_outputs=model_outputs,
            market_state=market_state,
            risk_state=risk_state,
            health=health_state,
            mode=self.mode,
        )

    def select_strategy(self, context: Context) -> Decision:
        if self.policy != "rules":
            raise ValueError("Only 'rules' policy is supported in this build.")

        regime_key = self._normalise_regime(context.regime)
        reasoning: List[str] = []

        if self._risk_force_flat(context, reasoning):
            decision = self._flat_decision(context, reasoning)
            self._record(decision)
            return decision

        scores = self._compute_scores(context)
        candidate = self._select_candidate_strategy(context, scores, reasoning)

        direction, confidence = self._determine_direction(context, regime_key, candidate, reasoning)
        if direction == "flat":
            decision = self._flat_decision(context, reasoning, strategy_id="S5")
            self._record(decision)
            return decision

        size_frac = self._determine_size(context, candidate, regime_key, direction, reasoning)
        if size_frac <= 0.0:
            reasoning.append('Size fraction reduced to zero; staying flat.')
            self._stats["blocked_zero_size"] += 1
            decision = self._flat_decision(context, reasoning, strategy_id='S5')
            self._record(decision)
            return decision

        execution = self._execution_mode(regime_key, confidence)
        decision = Decision(
            timestamp=context.timestamp,
            symbol=context.symbol,
            regime=regime_key,
            strategy_id=candidate,
            weight=scores.get(candidate, 0.0),
            direction=direction,
            size_frac=size_frac,
            confidence=confidence,
            execution=execution,
            rationale=" | ".join(reasoning),
        )
        self._record(decision)
        return decision

    def _risk_force_flat(self, context: Context, reasoning: List[str]) -> bool:
        ratio = context.risk_state.get("liquidity_buffer_to_atr", 999.0)
        watchdog = context.market_state.get("latency_watchdog", 0.0)
        daily_loss = context.risk_state.get("daily_pnl_pct", 0.0)

        if self.mode == "live":
            if ratio < 5.0:
                reasoning.append(f"Flat: liquidity buffer {ratio:.2f} < 5xATR")
                self._stats["blocked_liquidity_buffer"] += 1
                return True
            if watchdog > 0:
                reasoning.append("Flat: latency watchdog triggered")
                self._stats["blocked_latency_watchdog"] += 1
                return True
        if daily_loss <= -0.10:
            reasoning.append(f"Flat: daily drawdown {daily_loss:.2%} exceeds limit")
            self._stats["blocked_daily_drawdown"] += 1
            return True
        return False

    def _flat_decision(self, context: Context, reasoning: List[str], strategy_id: str = "S5") -> Decision:
        if not reasoning:
            reasoning.append("Flat fallback selected")
        return Decision(
            timestamp=context.timestamp,
            symbol=context.symbol,
            regime=self._normalise_regime(context.regime),
            strategy_id=strategy_id,
            weight=0.0,
            direction="flat",
            size_frac=0.0,
            confidence=context.model_outputs.get("p_hold", 0.0),
            execution="maker",
            rationale=" | ".join(reasoning),
        )

    def _compute_scores(self, context: Context) -> Dict[str, float]:
        def value_for(feature: str) -> float:
            if feature in context.features:
                return float(context.features[feature])
            if feature in context.market_state:
                return float(context.market_state[feature])
            if feature in context.risk_state:
                return float(context.risk_state[feature])
            return 0.0

        scores: Dict[str, float] = {}
        for strategy_id, mapping in self.weights.items():
            score = 0.0
            for feature, weight in mapping.items():
                score += weight * value_for(feature)
            scores[strategy_id] = score
        return scores

    def _select_candidate_strategy(self, context: Context, scores: Dict[str, float], reasoning: List[str]) -> str:
        spread_limit = self.limits.get("spread_z_max_for_S4", 1.0)
        latency_ok = context.features.get("latency_ok", 1.0) >= 0.5
        spread_z = context.market_state.get("spread_z", 0.0)
        candidates = dict(scores)

        if spread_z > spread_limit or not latency_ok:
            candidates.pop("S4", None)
            reasoning.append(f"S4 disabled (spread_z={spread_z:.2f}, latency_ok={latency_ok})")
            if spread_z > spread_limit:
                self._stats["blocked_spread_guard"] += 1
            if not latency_ok:
                self._stats["blocked_latency_flag"] += 1

        if not candidates:
            return "S5"
        best = max(candidates.items(), key=lambda kv: kv[1])[0]
        reasoning.append(f"Selected best scoring strategy {best} (score={candidates[best]:.3f})")
        return best

    def _determine_direction(self, context: Context, regime_key: str, candidate: str, reasoning: List[str]) -> tuple[str, float]:
        candidate_key = (candidate or "").upper()
        if candidate_key == "S2":
            return self._determine_mean_reversion_direction(context, reasoning)
        return self._determine_trend_direction(context, regime_key, reasoning)

    def _determine_trend_direction(self, context: Context, regime_key: str, reasoning: List[str]) -> tuple[str, float]:
        entry_thresholds = self.thresholds["theta_entry"]
        theta = entry_thresholds.get(regime_key, 0.6)

        p_long = context.model_outputs.get("p_long", 0.0)
        p_short = context.model_outputs.get("p_short", 0.0)

        adx = context.features.get("adx", 0.0)
        funding_z = context.features.get("funding_z", context.market_state.get("funding_z", 0.0))
        adx_thresh = self.limits.get("adx_weak_trend", 15)

        block_long = funding_z >= self.limits.get("funding_z_block_long", 1.5) and adx >= adx_thresh
        block_short = funding_z <= self.limits.get("funding_z_block_short", -2.5) and adx >= adx_thresh

        if block_long:
            reasoning.append(f"Longs blocked by funding_z={funding_z:.2f}, ADX={adx:.1f}")
            self._stats["blocked_funding_long"] += 1
        if block_short:
            reasoning.append(f"Shorts blocked by funding_z={funding_z:.2f}, ADX={adx:.1f}")
            self._stats["blocked_funding_short"] += 1

        if p_long >= p_short and p_long >= theta and not block_long:
            reasoning.append(f"Direction long (p_long={p_long:.2f} >= theta={theta:.2f})")
            return "long", p_long
        if p_short >= theta and not block_short:
            reasoning.append(f"Direction short (p_short={p_short:.2f} >= theta={theta:.2f})")
            return "short", p_short

        eps = self.limits.get("trend_override_epsilon", 0.03)
        override_adx = self.limits.get("trend_override_adx", 30)
        prob_gap = abs(p_long - p_short)
        if adx >= override_adx and prob_gap >= eps:
            direction = "long" if p_long >= p_short else "short"
            if direction == "long" and not block_long:
                reasoning.append(
                    f"Trend override enabled (ADX={adx:.1f}, gap={prob_gap:.3f}); entering long below theta."
                )
                self._stats["trend_override"] += 1
                return "long", p_long
            if direction == "short" and not block_short:
                reasoning.append(
                    f"Trend override enabled (ADX={adx:.1f}, gap={prob_gap:.3f}); entering short below theta."
                )
                self._stats["trend_override"] += 1
                return "short", p_short

        reasoning.append("No side meets entry confidence; remaining flat.")
        self._stats["blocked_low_confidence"] += 1
        return "flat", max(p_long, p_short)

    def _determine_mean_reversion_direction(self, context: Context, reasoning: List[str]) -> tuple[str, float]:
        signal = context.features.get("signal_mean_reversion", context.market_state.get("signal_mean_reversion", 0.0))
        prob = context.features.get("prob_mean_reversion", context.market_state.get("prob_mean_reversion", 0.5))
        prob = float(prob)

        if abs(signal) <= 1e-9:
            reasoning.append("Mean reversion signal neutral; staying flat.")
            self._stats["mr_neutral"] += 1
            return "flat", 0.5

        direction = "long" if signal > 0 else "short"
        confidence = prob if direction == "long" else (1.0 - prob)
        confidence = float(max(min(confidence, 1.0), 0.0))

        reasoning.append(
            f"Mean reversion direction {direction} (signal={signal:.2f}, prob={prob:.2f}, confidence={confidence:.2f})"
        )
        self._stats["mr_selected"] += 1
        return direction, confidence

    def _determine_size(self, context: Context, strategy_id: str, regime_key: str, direction: str, reasoning: List[str]) -> float:
        sizing_cfg = self.sizing.get(regime_key, {})
        base_size = sizing_cfg.get("base_size_frac", 0.05)
        max_size = sizing_cfg.get("max_size_frac", base_size)
        size = base_size

        adx = context.features.get("adx", 0.0)
        vol_high = context.features.get("vol_regime_high", 0.0) >= 0.5

        if strategy_id == "S2" and adx > 25 and vol_high:
            cap = self.limits.get("S2_cap_high_trend", max_size)
            size = min(size, cap)
            reasoning.append(f"S2 capped under high trend (ADX={adx:.1f}, vol_high={vol_high}) -> {size:.3f}")
            self._stats["s2_capped_high_trend"] += 1

        funding_z = context.features.get("funding_z", 0.0)
        bias = 1 - min(abs(funding_z) / 2.5, 1)
        size *= max(bias, 0.25)
        reasoning.append(f"Funding scaling applied (funding_z={funding_z:.2f}, bias={bias:.2f}) -> {size:.3f}")

        cap = self._leverage_cap(regime_key)
        current_lev = context.risk_state.get("current_leverage", 0.0)
        if current_lev >= cap:
            reasoning.append(f"Leverage {current_lev:.2f} >= cap {cap}, flattening.")
            self._stats["blocked_leverage_cap"] += 1
            return 0.0

        size = min(size, max_size)
        reasoning.append(f"Final size fraction {size:.3f} (max {max_size:.3f}) direction {direction}")
        return size

    def _execution_mode(self, regime_key: str, confidence: float) -> str:
        theta_high = self.thresholds["theta_high"].get(regime_key, 0.75)
        return "cross" if confidence >= theta_high else "maker"

    def _record(self, decision: Decision) -> None:
        record = decision.to_record()
        self._try_log_mlflow(record)
        date_key = decision.timestamp.strftime("%Y-%m-%d")
        self._buffer.setdefault(date_key, []).append(record)
        if len(self._buffer[date_key]) >= self.flush_threshold:
            self.flush(date_key)

    def _try_log_mlflow(self, record: Dict[str, object]) -> None:
        try:
            import mlflow

            mlflow.log_dict(record, f"meta_selector/{record['symbol']}_{record['timestamp']}.json")
        except Exception:
            logger.debug("Skipping MLflow logging for meta selector decision.")

    def flush(self, date_key: Optional[str] = None) -> None:
        keys = [date_key] if date_key else list(self._buffer.keys())
        for key in keys:
            records = self._buffer.get(key)
            if not records:
                continue
            if not self.enable_cache:
                self._buffer[key] = []
                continue

            path = self.export_dir / f"{key}.parquet"
            df = pd.DataFrame(records)
            existing = _safe_read_parquet(path)
            if existing is not None and not existing.empty:
                df = (
                    pd.concat([existing, df], ignore_index=True)
                    .drop_duplicates(subset=["timestamp", "strategy_id"], keep="last")
                )
            _safe_write_parquet(df, path)
            self._buffer[key] = []

    def __del__(self) -> None:
        try:
            self.flush()
        except Exception:
            pass

    def _normalise_regime(self, regime: str) -> str:
        mapping = {
            "scalping": "HFT",
            "hft": "HFT",
            "micro": "HFT",
            "intraday": "intraday",
            "swing": "swing",
        }
        return mapping.get(regime.lower(), regime)

    def _leverage_cap(self, regime_key: str) -> float:
        caps = {"HFT": 5.0, "intraday": 3.0, "swing": 2.0}
        return caps.get(regime_key, 3.0)

    def snapshot_stats(self, reset: bool = False) -> Dict[str, int]:
        """Return current blocker/decision counters."""
        stats = dict(self._stats)
        if reset:
            self._stats.clear()
        return stats

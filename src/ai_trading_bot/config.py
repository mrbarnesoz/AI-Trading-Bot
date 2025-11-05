"""Configuration utilities for the AI Trading Bot project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _clean_probability_list(values: Optional[List[float]], *, ascending: bool = True) -> List[float]:
    if not values:
        return []
    cleaned = sorted({float(v) for v in values})
    return cleaned if ascending else list(reversed(cleaned))


@dataclass
class DataConfig:
    symbol: str = "AAPL"
    start_date: str = "2018-01-01"
    end_date: Optional[str] = None
    interval: str = "1d"
    source: str = "yfinance"
    cache_dir: Path = Path("data") / "raw"

    def __post_init__(self) -> None:
        if not isinstance(self.cache_dir, Path):
            self.cache_dir = Path(self.cache_dir)


@dataclass
class FeatureConfig:
    indicators: List[str] = field(default_factory=lambda: ["sma", "ema", "rsi", "macd"])
    sma_window: int = 10
    ema_window: int = 21
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9


@dataclass
class ModelConfig:
    test_size: float = 0.2
    random_state: int = 42
    model_type: str = "random_forest"
    n_estimators: int = 200
    max_depth: Optional[int] = 6
    n_jobs: int = -1
    model_dir: Path = Path("models")

    def __post_init__(self) -> None:
        if not isinstance(self.model_dir, Path):
            self.model_dir = Path(self.model_dir)


@dataclass
class BacktestConfig:
    initial_capital: float = 10000.0
    transaction_cost: float = 0.001  # 0.1%
    risk_free_rate: float = 0.01
    position_capital_fraction: float = 0.2
    max_total_capital_fraction: float = 0.6
    max_position_units: int = 3
    maker_fee_bps: float = 0.0
    taker_fee_bps: float = 0.0
    per_side_fee_bps: float = 0.0
    funding_rate_column: str = "fundingRate"
    funding_accrual_hours: int = 8
    signal_persistence_bars: int = 4
    flip_cooldown_bars: int = 2
    fallback_spread_bps: float = 5.0
    backtest_min_lock: Optional[float] = 0.2
    entry_mode: str = "auto"
    max_entry_wait_bars: int = 6
    cancel_spread_bps: float = 15.0
    reprice_on: List[str] = field(default_factory=lambda: ["new_bar"])
    reprice_tick_offset: float = 1.0
    partial_fill_min_qty_frac: float = 1.0
    allow_cross_after_timeout: bool = False
    min_hold_bars: int = 4
    cooldown_bars_after_exit: int = 6
    max_trades_per_day_side: int = 4
    invert_side: bool = False
    enforce_gates: bool = False
    maker_ratio_target: float = 0.8
    slippage_bps_target: float = 4.0
    trades_per_day_target: float = 2.0
    max_drawdown_floor: float = -0.25
    min_gate_trades: int = 100
    min_slippage_bps: float = 0.5
    slippage_bps: float = 0.0
    simulate_funding: bool = True
    max_daily_loss_pct: float = 0.05
    max_drawdown_pct: float = 0.2

    def __post_init__(self) -> None:
        self.signal_persistence_bars = max(1, int(self.signal_persistence_bars))
        self.flip_cooldown_bars = max(0, int(self.flip_cooldown_bars))
        self.fallback_spread_bps = max(0.0, float(self.fallback_spread_bps))
        if self.backtest_min_lock is not None:
            self.backtest_min_lock = max(0.0, float(self.backtest_min_lock))
        self.entry_mode = str(self.entry_mode or "auto").lower()
        self.max_entry_wait_bars = max(1, int(self.max_entry_wait_bars))
        self.cancel_spread_bps = max(0.0, float(self.cancel_spread_bps))
        self.reprice_on = [str(x).lower() for x in (self.reprice_on or ["new_bar"])]
        self.reprice_tick_offset = max(0.0, float(self.reprice_tick_offset))
        self.partial_fill_min_qty_frac = min(1.0, max(0.0, float(self.partial_fill_min_qty_frac)))
        self.allow_cross_after_timeout = bool(self.allow_cross_after_timeout)
        self.min_hold_bars = max(0, int(self.min_hold_bars))
        self.cooldown_bars_after_exit = max(0, int(self.cooldown_bars_after_exit))
        self.max_trades_per_day_side = max(1, int(self.max_trades_per_day_side))
        self.invert_side = bool(self.invert_side)
        self.enforce_gates = bool(self.enforce_gates)
        self.maker_ratio_target = float(self.maker_ratio_target)
        self.slippage_bps_target = max(0.0, float(self.slippage_bps_target))
        self.trades_per_day_target = max(0.1, float(self.trades_per_day_target))
        self.max_drawdown_floor = float(self.max_drawdown_floor)
        self.min_gate_trades = max(1, int(self.min_gate_trades))
        self.min_slippage_bps = max(0.0, float(self.min_slippage_bps))
        self.per_side_fee_bps = max(0.0, float(self.per_side_fee_bps))
        self.slippage_bps = max(0.0, float(self.slippage_bps))
        self.simulate_funding = bool(self.simulate_funding)


@dataclass
class FilterConfig:
    allow_long: bool = True
    allow_short: bool = True
    min_atr_frac: Optional[float] = None
    trend_slope_lookback: int = 48
    min_trend_slope: Optional[float] = None
    hysteresis_k_atr: Optional[float] = None
    min_confidence: Optional[float] = None
    min_adx: Optional[float] = None


@dataclass
class SignalsPostConfig:
    min_hold_bars: int = 0
    hysteresis_k_atr: float = 0.0


@dataclass
class SignalsConfig:
    post: SignalsPostConfig = field(default_factory=SignalsPostConfig)


@dataclass
class TrailingConfig:
    enabled: bool = False
    update: Dict[str, Dict[str, float]] = field(default_factory=dict)
    k_atr: Dict[str, Dict[str, float]] = field(default_factory=lambda: {"stop": {}, "take": {}})
    min_lock: Dict[str, float] = field(default_factory=lambda: {"R_multiple": 1.0})
    floor_ceiling: Dict[str, Dict[str, int]] = field(default_factory=dict)
    slippage_guard_bps: float = 5.0
    max_updates_per_min: int = 60

    def stop_multiplier(self, regime: str, default: float = 3.0) -> float:
        stop_map = self.k_atr.get("stop", {})
        return float(
            stop_map.get(regime)
            or stop_map.get(regime.lower())
            or stop_map.get(regime.upper())
            or default
        )

    def take_multiplier(self, regime: str, default: float = 1.5) -> float:
        take_map = self.k_atr.get("take", {})
        return float(
            take_map.get(regime)
            or take_map.get(regime.lower())
            or take_map.get(regime.upper())
            or default
        )

    def cadence_for(self, regime: str) -> Dict[str, float]:
        return (
            self.update.get(regime)
            or self.update.get(regime.lower(), {})
            or self.update.get(regime.upper(), {})
            or {}
        )

    def min_ticks(self, side: str) -> int:
        side_map = self.floor_ceiling.get(side) or self.floor_ceiling.get(side.lower()) or self.floor_ceiling.get(side.upper()) or {}
        return int(side_map.get("min_px_distance_ticks", 1))


@dataclass
class RiskConfig:
    trailing: TrailingConfig = field(default_factory=TrailingConfig)
    latency_arb: "LatencyArbConfig" = field(default_factory=lambda: LatencyArbConfig())


@dataclass
class LatencyArbConfig:
    enabled: bool = False
    min_average_bps: float = 3.0
    min_trades: int = 50
    summary_path: Path = Path("results") / "latency_arbitrage" / "latency_arbitrage_summary.json"

    def __post_init__(self) -> None:
        if not isinstance(self.summary_path, Path):
            self.summary_path = Path(self.summary_path)
        self.min_average_bps = float(self.min_average_bps)
        self.min_trades = max(0, int(self.min_trades))


@dataclass
class PipelineConfig:
    lookahead: int = 1
    target_column: str = "target_return"
    long_threshold: float = 0.55
    short_threshold: float = 0.45
    long_bands: List[float] = field(default_factory=list)
    short_bands: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.long_bands = _clean_probability_list(self.long_bands, ascending=True)
        self.short_bands = _clean_probability_list(self.short_bands, ascending=True)

        if self.long_bands:
            if self.long_bands[0] <= 0 or self.long_bands[-1] > 1:
                raise ValueError("Long probability bands must lie in the interval (0, 1].")
            if self.long_threshold is None or self.long_threshold < self.long_bands[0]:
                self.long_threshold = self.long_bands[0]

        if self.short_bands:
            if self.short_bands[0] < 0 or self.short_bands[-1] >= 1:
                raise ValueError("Short probability bands must lie in the interval [0, 1).")
            if self.short_threshold is None or self.short_threshold > self.short_bands[-1]:
                self.short_threshold = self.short_bands[-1]

        if self.short_threshold is None:
            self.short_threshold = max(0.0, 1 - self.long_threshold)

        if not 0 <= self.short_threshold <= 1 or not 0 <= self.long_threshold <= 1:
            raise ValueError("Thresholds must be between 0 and 1.")
        if self.short_threshold >= self.long_threshold:
            raise ValueError("Short threshold must be below long threshold.")

        if self.long_bands and self.short_bands and self.short_bands[-1] >= self.long_bands[0]:
            raise ValueError("Short probability bands must be strictly below long probability bands.")


@dataclass
class StrategyModeConfig:
    name: str
    interval: str
    lookback_days: int = 5
    description: str = ""
    long_threshold: Optional[float] = None
    short_threshold: Optional[float] = None
    long_bands: List[float] = field(default_factory=list)
    short_bands: List[float] = field(default_factory=list)
    trend_weight: float = 1.0
    volatility_weight: float = 1.0
    position_fraction: Optional[float] = None
    max_total_fraction: Optional[float] = None
    max_position_units: Optional[int] = None

    def __post_init__(self) -> None:
        self.long_bands = _clean_probability_list(self.long_bands, ascending=True)
        self.short_bands = _clean_probability_list(self.short_bands, ascending=True)
        if self.long_threshold is not None and not 0 < self.long_threshold <= 1:
            raise ValueError("Mode long_threshold must lie in (0, 1].")
        if self.short_threshold is not None and not 0 <= self.short_threshold < 1:
            raise ValueError("Mode short_threshold must lie in [0, 1).")
        if (
            self.long_threshold is not None
            and self.short_threshold is not None
            and self.short_threshold >= self.long_threshold
        ):
            raise ValueError("Mode short_threshold must be below long_threshold.")


@dataclass
class StrategyConfig:
    name: str = "ml_trend"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    signals: SignalsConfig = field(default_factory=SignalsConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    sweep_mode: bool = False
    modes: List[StrategyModeConfig] = field(
        default_factory=lambda: [
            StrategyModeConfig(
                name="scalping",
                interval="1m",
                lookback_days=2,
                description="Sub-minute scalping on high-volatility conditions.",
                long_bands=[0.6, 0.7, 0.85],
                short_bands=[0.4, 0.3, 0.15],
                volatility_weight=1.5,
                trend_weight=0.7,
                position_fraction=0.05,
                max_total_fraction=0.2,
                max_position_units=4,
            ),
            StrategyModeConfig(
                name="intraday",
                interval="5m",
                lookback_days=7,
                description="Intraday trading on 1–15 minute bars.",
                long_bands=[0.55, 0.65, 0.8],
                short_bands=[0.45, 0.35, 0.2],
                volatility_weight=1.0,
                trend_weight=1.0,
                position_fraction=0.1,
                max_total_fraction=0.4,
                max_position_units=3,
            ),
            StrategyModeConfig(
                name="swing",
                interval="1h",
                lookback_days=60,
                description="Swing trading on hourly/daily bars.",
                long_bands=[0.6, 0.7],
                short_bands=[0.4, 0.3],
                volatility_weight=0.8,
                trend_weight=1.4,
                position_fraction=0.2,
                max_total_fraction=0.6,
                max_position_units=2,
            ),
        ]
    )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AppConfig":
        """Create an `AppConfig` object from a nested dictionary."""
        raw_modes = config_dict.get("modes", [])
        modes = [StrategyModeConfig(**raw_mode) for raw_mode in raw_modes] if raw_modes else None
        return cls(
            data=DataConfig(**config_dict.get("data", {})),
            features=FeatureConfig(**config_dict.get("features", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            backtest=BacktestConfig(**config_dict.get("backtest", {})),
            risk=RiskConfig(
                trailing=TrailingConfig(
                    **config_dict.get("risk", {}).get("trailing", {})
                ),
                latency_arb=LatencyArbConfig(
                    **config_dict.get("risk", {}).get("latency_arb", {})
                ),
            ),
            filters=FilterConfig(**config_dict.get("filters", {})),
            signals=SignalsConfig(
                post=SignalsPostConfig(**config_dict.get("signals", {}).get("post", {}))
            ),
            pipeline=PipelineConfig(**config_dict.get("pipeline", {})),
            strategy=StrategyConfig(**config_dict.get("strategy", {})),
            sweep_mode=bool(config_dict.get("sweep_mode", False)),
            modes=modes if modes is not None else cls().modes,
        )

    def to_nested_dict(self) -> Dict[str, Any]:
        """Return the configuration as a nested dictionary."""
        data_dict = vars(self.data).copy()
        data_dict["cache_dir"] = str(data_dict["cache_dir"])

        model_dict = vars(self.model).copy()
        model_dict["model_dir"] = str(model_dict["model_dir"])

        modes = [vars(mode).copy() for mode in self.modes]

        return {
            "data": data_dict,
            "features": vars(self.features),
            "model": model_dict,
            "backtest": vars(self.backtest),
            "filters": vars(self.filters),
            "signals": {
                "post": vars(self.signals.post),
            },
            "risk": {
                "trailing": {
                    "enabled": self.risk.trailing.enabled,
                    "update": self.risk.trailing.update,
                    "k_atr": self.risk.trailing.k_atr,
                    "min_lock": self.risk.trailing.min_lock,
                    "floor_ceiling": self.risk.trailing.floor_ceiling,
                    "slippage_guard_bps": self.risk.trailing.slippage_guard_bps,
                    "max_updates_per_min": self.risk.trailing.max_updates_per_min,
                },
                "latency_arb": {
                    "enabled": self.risk.latency_arb.enabled,
                    "min_average_bps": self.risk.latency_arb.min_average_bps,
                    "min_trades": self.risk.latency_arb.min_trades,
                    "summary_path": str(self.risk.latency_arb.summary_path),
                },
            },
            "pipeline": vars(self.pipeline),
            "strategy": {
                "name": self.strategy.name,
                "params": dict(self.strategy.params),
            },
            "sweep_mode": bool(self.sweep_mode),
            "modes": modes,
        }


def load_config(path: Path | str = "config.yaml") -> AppConfig:
    """Load configuration from a YAML file or return defaults if it does not exist."""
    config_path = Path(path)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            raw_config: Dict[str, Any] = yaml.safe_load(f) or {}
        return AppConfig.from_dict(raw_config)
    return AppConfig()


def save_config(config: AppConfig, path: Path | str = "config.yaml") -> None:
    """Persist the configuration to disk as YAML."""
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_nested_dict(), f, sort_keys=False)



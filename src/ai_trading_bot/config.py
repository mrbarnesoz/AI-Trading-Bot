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
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
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
            pipeline=PipelineConfig(**config_dict.get("pipeline", {})),
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
            "pipeline": vars(self.pipeline),
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

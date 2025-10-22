"""Configuration utilities for the AI Trading Bot project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


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


@dataclass
class PipelineConfig:
    lookahead: int = 1
    target_column: str = "target_return"


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AppConfig":
        """Create an `AppConfig` object from a nested dictionary."""
        return cls(
            data=DataConfig(**config_dict.get("data", {})),
            features=FeatureConfig(**config_dict.get("features", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            backtest=BacktestConfig(**config_dict.get("backtest", {})),
            pipeline=PipelineConfig(**config_dict.get("pipeline", {})),
        )

    def to_nested_dict(self) -> Dict[str, Any]:
        """Return the configuration as a nested dictionary."""
        data_dict = vars(self.data).copy()
        data_dict["cache_dir"] = str(data_dict["cache_dir"])

        model_dict = vars(self.model).copy()
        model_dict["model_dir"] = str(model_dict["model_dir"])

        return {
            "data": data_dict,
            "features": vars(self.features),
            "model": model_dict,
            "backtest": vars(self.backtest),
            "pipeline": vars(self.pipeline),
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

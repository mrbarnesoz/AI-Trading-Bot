from __future__ import annotations

from pathlib import Path

import pandas as pd

from ai_trading_bot.meta.select import Context, MetaStrategySelector


def _write_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "meta_selector.yaml"
    cfg.write_text(
        """policy: rules
weights:
  S1:
    trend_slope: 1.0
    adx: 0.6
    vol_regime_high: 0.4
    spread_z: -0.4
    signal_trend: 0.5
  S2:
    vwap_dev_z: 1.0
    adx: -0.6
    vol_regime_low: 0.5
    signal_mean_reversion: 0.6
    prob_mean_reversion: 0.4
  S3:
    funding_z: -1.0
    oi_mom: 0.6
    price_mom: -0.3
  S4:
    imbalance_dOB: 1.0
    spread_z: -0.7
    latency_ok: 0.5
  S5: {}
thresholds:
  theta_entry:
    HFT: 0.55
    intraday: 0.60
    swing: 0.65
  theta_high:
    HFT: 0.70
    intraday: 0.75
    swing: 0.80
limits:
  spread_z_max_for_S4: 1.0
  S2_cap_high_trend: 0.2
  funding_z_block_long: 1.5
  funding_z_block_short: -2.5
  adx_weak_trend: 15
sizing:
  HFT:
    base_size_frac: 0.02
    max_size_frac: 0.05
  intraday:
    base_size_frac: 0.05
    max_size_frac: 0.10
  swing:
    base_size_frac: 0.10
    max_size_frac: 0.20
""",
        encoding="utf-8",
    )
    return cfg


def _selector(tmp_path: Path) -> MetaStrategySelector:
    cfg = _write_config(tmp_path)
    export_dir = tmp_path / "export"
    return MetaStrategySelector(config_path=cfg, export_dir=export_dir, flush_threshold=10)


def _base_context(**overrides):
    ctx = Context(
        timestamp=pd.Timestamp("2025-01-01T00:00:00Z"),
        symbol="XBTUSD",
        regime="intraday",
        features={
            "trend_slope": 0.0,
            "adx": 10.0,
            "vol_regime_high": 0.0,
            "vol_regime_low": 1.0,
            "spread_z": 0.2,
            "funding_z": 0.0,
            "vwap_dev_z": 0.0,
            "latency_ok": 1.0,
            "signal_trend": 0.0,
            "signal_mean_reversion": 0.0,
            "prob_mean_reversion": 0.5,
        },
        model_outputs={"p_long": 0.6, "p_short": 0.3, "p_hold": 0.1},
        market_state={
            "spread_z": 0.2,
            "latency_watchdog": 0.0,
            "funding_z": 0.0,
            "vol_regime": "low",
            "signal_mean_reversion": 0.0,
            "prob_mean_reversion": 0.5,
        },
        risk_state={"liquidity_buffer_to_atr": 10.0, "daily_pnl_pct": 0.0, "current_leverage": 0.0},
    )
    for key, value in overrides.items():
        setattr(ctx, key, value)
    return ctx


def test_force_flat_on_liquidity(tmp_path: Path) -> None:
    selector = _selector(tmp_path)
    ctx = _base_context(risk_state={"liquidity_buffer_to_atr": 3.0, "daily_pnl_pct": 0.0, "current_leverage": 0.0})
    decision = selector.select_strategy(ctx)
    assert decision.strategy_id == "S5"
    assert decision.direction == "flat"


def test_block_long_by_funding(tmp_path: Path) -> None:
    selector = _selector(tmp_path)
    ctx = _base_context(
        features={
            "trend_slope": 0.0,
            "adx": 18.0,
            "vol_regime_high": 1.0,
            "vol_regime_low": 0.0,
            "spread_z": 0.1,
            "funding_z": 2.0,
            "latency_ok": 1.0,
        },
        model_outputs={"p_long": 0.9, "p_short": 0.05, "p_hold": 0.05},
    )
    decision = selector.select_strategy(ctx)
    assert decision.direction == "flat"
    assert decision.strategy_id == "S5"


def test_size_scaled_by_funding(tmp_path: Path) -> None:
    selector = _selector(tmp_path)
    ctx = _base_context(
        features={
            "trend_slope": 1.0,
            "adx": 12.0,
            "vol_regime_high": 0.0,
            "vol_regime_low": 1.0,
            "spread_z": 0.1,
            "funding_z": 2.5,
            "latency_ok": 1.0,
        },
        model_outputs={"p_long": 0.8, "p_short": 0.1, "p_hold": 0.1},
    )
    decision = selector.select_strategy(ctx)
    assert 0.0124 <= decision.size_frac <= 0.0126  # 0.05 * 0.25


def test_cap_s2_under_high_trend(tmp_path: Path) -> None:
    selector = _selector(tmp_path)
    ctx = _base_context(
        regime="swing",
        features={
            "vwap_dev_z": 3.0,
            "adx": 30.0,
            "vol_regime_high": 1.0,
            "vol_regime_low": 0.0,
            "spread_z": 0.1,
            "funding_z": 0.0,
            "latency_ok": 1.0,
        },
        model_outputs={"p_short": 0.7, "p_long": 0.2, "p_hold": 0.1},
        market_state={"spread_z": 0.1, "latency_watchdog": 0.0, "funding_z": 0.0, "vol_regime": "high"},
    )
    decision = selector.select_strategy(ctx)
    if decision.strategy_id == "S2":
        assert decision.size_frac <= 0.2


def test_disable_s4_when_spread_wide(tmp_path: Path) -> None:
    selector = _selector(tmp_path)
    ctx = _base_context(
        features={
            "imbalance_dOB": 5.0,
            "spread_z": 2.0,
            "latency_ok": 1.0,
            "adx": 5.0,
            "vol_regime_high": 0.0,
            "vol_regime_low": 1.0,
            "funding_z": 0.0,
        },
        model_outputs={"p_long": 0.7, "p_short": 0.2, "p_hold": 0.1},
        market_state={"spread_z": 2.0, "latency_watchdog": 0.0, "funding_z": 0.0, "vol_regime": "low"},
    )
    decision = selector.select_strategy(ctx)
    assert decision.strategy_id != "S4"

def test_flat_when_leverage_cap(tmp_path: Path) -> None:
    selector = _selector(tmp_path)
    ctx = _base_context(
        risk_state={"liquidity_buffer_to_atr": 10.0, "daily_pnl_pct": 0.0, "current_leverage": 3.5},
        model_outputs={"p_long": 0.8, "p_short": 0.1, "p_hold": 0.1},
    )
    decision = selector.select_strategy(ctx)
    assert decision.direction == "flat"
    assert decision.strategy_id == "S5"

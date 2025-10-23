"""Stage 3 training entry point for intraday, swing, and HFT regimes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import mlflow

from models.gbm_classifier import GBMConfig
from training.datasets import DatasetSpec, load_dataset
from training.rolling_hft import RollingConfig, rolling_hft_training
from training.walkforward import WalkForwardConfig, run_walkforward_training


DEFAULT_REGIMES = ("hft", "intraday", "swing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 3 model training.")
    parser.add_argument("--gold-root", type=Path, default=Path("data/gold"), help="Root directory of gold datasets.")
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=list(DEFAULT_REGIMES),
        help="Regimes to train (default: hft intraday swing).",
    )
    parser.add_argument("--symbols", nargs="+", help="Optional subset of symbols to include.")
    parser.add_argument("--limit", type=int, help="Optional row cap for smoke tests.")
    parser.add_argument("--tracking-uri", help="MLflow tracking URI (defaults to environment).")
    parser.add_argument("--label-column", default="label_sign", help="Column containing classification labels.")
    parser.add_argument("--weight-column", default=None, help="Optional column containing sample weights.")
    parser.add_argument("--folds", type=int, default=8, help="Number of walk-forward folds for non-HFT regimes.")
    parser.add_argument("--embargo", type=int, default=3, help="Embargo length (in bars) for walk-forward splits.")
    return parser.parse_args()


def _lightgbm_params() -> Dict[str, float]:
    return {
        "objective": "multiclass",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 200,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "num_class": 3,
    }


def train_intraday_or_swing(
    regime: str,
    spec: DatasetSpec,
    symbols: Iterable[str] | None,
    limit: int | None,
    folds: int,
    embargo: int,
) -> List[dict]:
    features, labels, weights = load_dataset(spec, symbols=symbols, limit=limit)
    gbm_config = GBMConfig(params=_lightgbm_params(), num_boost_round=500, early_stopping_rounds=50)
    wf_config = WalkForwardConfig(folds=folds, embargo=embargo, params=gbm_config, experiment=f"{regime}-walkforward")
    return [result.__dict__ for result in run_walkforward_training(features, labels, weights, wf_config)]


def train_hft(regime: str, spec: DatasetSpec, symbols: Iterable[str] | None, limit: int | None) -> List[dict]:
    features, labels, _ = load_dataset(spec, symbols=symbols, limit=limit)
    rolling_config = RollingConfig(window_days=7, step_days=2, experiment=f"{regime}-rolling")
    return rolling_hft_training(features, labels, rolling_config)


def main() -> None:
    args = parse_args()
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    outcomes: Dict[str, List[dict]] = {}
    for regime in args.regimes:
        regime = regime.lower()
        if regime not in DEFAULT_REGIMES:
            raise ValueError(f"Unknown regime '{regime}'. Expected one of {DEFAULT_REGIMES}.")
        spec = DatasetSpec(
            root=args.gold_root,
            regime=regime,
            label_column=args.label_column,
            weight_column=args.weight_column,
        )
        if regime == "hft":
            outcomes[regime] = train_hft(regime, spec, args.symbols, args.limit)
        else:
            outcomes[regime] = train_intraday_or_swing(regime, spec, args.symbols, args.limit, args.folds, args.embargo)

    print(json.dumps(outcomes, default=str, indent=2))


if __name__ == "__main__":
    main()

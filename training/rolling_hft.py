"""Rolling training loop for HFT regime."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import mlflow
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, Dataset, random_split

from models.tcn_classifier import LightningTCN, TCNConfig
from training import metrics


@dataclass
class RollingConfig:
    window_days: int
    step_days: int
    sequence_length: int = 40
    batch_size: int = 128
    max_epochs: int = 15
    patience: int = 5
    learning_rate: float = 1e-3
    hidden_channels: int = 64
    num_layers: int = 4
    kernel_size: int = 3
    experiment: str = "hft-rolling"


class SequenceDataset(Dataset):
    """Sliding-window dataset feeding TCN with fixed-length sequences."""

    def __init__(self, features: np.ndarray, labels: np.ndarray, sequence_length: int) -> None:
        if features.shape[0] != labels.shape[0]:
            raise ValueError("Features and labels must have matching number of rows.")
        if sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return max(0, self.features.shape[0] - self.sequence_length + 1)

    def __getitem__(self, idx: int):
        window = self.features[idx : idx + self.sequence_length]
        target = self.labels[idx + self.sequence_length - 1]
        # TCN expects (N, C, L)
        tensor = torch.from_numpy(window.transpose(1, 0))
        return tensor, torch.tensor(target, dtype=torch.long)


def rolling_hft_training(
    features: pd.DataFrame,
    labels: pd.Series,
    config: RollingConfig,
) -> List[dict]:
    """Rolling TCN training loop for the HFT regime."""
    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("features index must be a DatetimeIndex for rolling training.")

    mlflow.set_experiment(config.experiment)
    label_values = sorted(labels.unique().tolist())
    label_to_index = {value: idx for idx, value in enumerate(label_values)}
    encoded_labels = labels.map(label_to_index)

    idx = features.index
    start_date = idx.min().floor("D")
    end_date = idx.max().ceil("D")
    current = start_date
    results: List[dict] = []

    while current + timedelta(days=config.window_days) <= end_date:
        window_end = current + timedelta(days=config.window_days)
        window_mask = (idx >= current) & (idx < window_end)
        if window_mask.sum() < config.sequence_length + 10:
            current += timedelta(days=config.step_days)
            continue

        window_features = features.loc[window_mask]
        window_encoded = encoded_labels.loc[window_mask]
        window_numeric = labels.loc[window_mask].astype(float)

        feature_matrix = window_features.to_numpy(dtype=np.float32)
        encoded_array = window_encoded.to_numpy(dtype=np.int64)
        dataset = SequenceDataset(feature_matrix, encoded_array, config.sequence_length)
        numeric_labels = window_numeric.to_numpy()[config.sequence_length - 1 :]
        if len(dataset) < config.batch_size:
            current += timedelta(days=config.step_days)
            continue

        train_size = max(int(len(dataset) * 0.8), 1)
        val_size = len(dataset) - train_size
        if val_size == 0:
            val_size = 1
            train_size -= 1
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

        tcn_config = TCNConfig(
            input_channels=window_features.shape[1],
            hidden_channels=config.hidden_channels,
            kernel_size=config.kernel_size,
            num_layers=config.num_layers,
            lr=config.learning_rate,
        )
        model = LightningTCN(tcn_config)
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=config.patience, mode="min"),
        ]
        trainer = pl.Trainer(
            max_epochs=config.max_epochs,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            callbacks=callbacks,
        )
        trainer.fit(model, train_loader, val_loader)

        model.eval()
        all_logits: list[np.ndarray] = []
        with torch.no_grad():
            for batch in DataLoader(dataset, batch_size=config.batch_size, shuffle=False):
                inputs, _ = batch
                logits = model(inputs)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_logits.append(probs)
        probabilities = np.concatenate(all_logits, axis=0)

        long_idx = label_to_index.get(1, label_values[-1])
        short_idx = label_to_index.get(-1, label_values[0])
        long_probs = probabilities[:, long_idx]
        short_probs = probabilities[:, short_idx]

        long_sharpe = metrics.compute_sharpe(long_probs, numeric_labels)
        short_sharpe = metrics.compute_sharpe(short_probs, -numeric_labels)
        calmar = metrics.compute_calmar(long_probs, numeric_labels)

        metrics_payload = {
            "long_sharpe": long_sharpe,
            "short_sharpe": short_sharpe,
            "calmar": calmar,
            "samples": len(dataset),
        }

        with mlflow.start_run(run_name=f"hft_window_{current:%Y%m%d}") as run:
            mlflow.log_metrics(metrics_payload)
            mlflow.log_params(
                {
                    "sequence_length": config.sequence_length,
                    "batch_size": config.batch_size,
                    "max_epochs": config.max_epochs,
                    "learning_rate": config.learning_rate,
                    "hidden_channels": config.hidden_channels,
                    "num_layers": config.num_layers,
                }
            )
            with TemporaryDirectory() as tmp:
                ckpt_path = Path(tmp) / "tcn.ckpt"
                trainer.save_checkpoint(str(ckpt_path))
                mlflow.log_artifact(str(ckpt_path), artifact_path="model")

        results.append(
            {
                "window_start": current,
                "window_end": window_end,
                "metrics": metrics_payload,
            }
        )
        current += timedelta(days=config.step_days)

    return results

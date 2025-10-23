"""Cost and slippage model utilities."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(slots=True)
class CostModel:
    """Represents execution costs expressed in basis points (bps)."""

    taker_fee_bps: float
    maker_fee_bps: float = 0.0
    slippage_bps: float = 0.0
    maker_fill_probability: float = 0.5

    def effective_bps(self, crossing: bool) -> float:
        """Return the expected bps cost for a trade."""
        if crossing:
            return self.taker_fee_bps + self.slippage_bps
        maker_component = self.maker_fee_bps
        taker_component = self.taker_fee_bps + self.slippage_bps
        return self.maker_fill_probability * maker_component + (1.0 - self.maker_fill_probability) * taker_component


def compute_trade_cost(
    turnover: pl.Expr,
    crossing: pl.Expr,
    cost_model: CostModel,
) -> pl.Expr:
    """Return an expression computing per-bar cost given turnover and whether we cross the spread."""

    taker_cost = (cost_model.taker_fee_bps + cost_model.slippage_bps) / 10_000.0
    maker_cost = (
        cost_model.maker_fill_probability * (cost_model.maker_fee_bps / 10_000.0)
        + (1.0 - cost_model.maker_fill_probability) * ((cost_model.taker_fee_bps + cost_model.slippage_bps) / 10_000.0)
    )
    return pl.when(crossing).then(turnover * taker_cost).otherwise(turnover * maker_cost)


__all__ = ["CostModel", "compute_trade_cost"]

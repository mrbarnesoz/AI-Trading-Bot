# Data Contracts Overview

This document tracks the canonical schemas and expectations for BitMEX ETL datasets.

## Bronze Layer

- `trades`: raw trade prints normalized to the schema in `schemas/trades_schema.yaml`. Primary key `(ts, symbol, tick_id)`; timestamps are UTC nanoseconds.
- `l2_updates`: level-2 order book updates with actions `insert`, `update`, `delete`. Schema defined in `schemas/l2_updates_schema.yaml`.
- `funding`, `open_interest`: derivatives context series, aligned to BitMEX publication timestamps.

## Silver Layer

- `l2_snapshots`: top-of-book snapshots emitted at the cadence configured in `conf/cadence.yaml`. Derived fields such as spread and cumulative depth must satisfy QC checks (`spread >= 0`).
- `ohlcv`: OHLCV bars resampled from bronze trades. Base timeframe is `1m`; longer intervals are aggregated from that base.

## Quality Gates

1. **Timestamp monotonicity** – all feeds must be sorted by `ts`.
2. **Coverage** – trade minutes per symbol/day ≥ 95%.
3. **Spread sanity** – no negative spreads in snapshots.
4. **Checksum manifests** – every raw archive emits an MD5 JSON manifest alongside the source file.

Additional validation logic lives in `etl/qc_validations.py`. Update this document whenever schemas or QC rules change.

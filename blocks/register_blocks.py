"""Example script to register Prefect blocks for BitMEX orchestration."""

from __future__ import annotations

from prefect.blocks.system import JSON, Secret


def register_storage_block() -> None:
    JSON(value={
        "raw_root": "s3://bitmex/raw",
        "bronze_root": "s3://bitmex/bronze",
        "silver_root": "s3://bitmex/silver",
        "gold_root": "s3://bitmex/gold",
        "duckdb_path": "bitmex_warehouse.duckdb",
        "auto_ingest_l2": True,
        "l2_archive_prefix": "data/orderBookL2_25",
    }).save("bitmex-storage-config", overwrite=True)


def register_alert_block() -> None:
    Secret(value="slack://YOUR/WEBHOOK,mailto://ops@example.com").save(
        "alert-webhooks", overwrite=True
    )


def main() -> None:
    register_storage_block()
    register_alert_block()
    print("Prefect blocks registered: bitmex-storage-config, alert-webhooks")


if __name__ == "__main__":
    main()

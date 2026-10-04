"""
db/migration.py
---------------
Lightweight, idempotent schema migration utility for DRISHTI-SSS.
Ensures that an existing database schema is upgraded to the current SQLAlchemy models
without relying on Base.metadata.create_all() alone.
"""

from __future__ import annotations

import logging

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection, Engine

from app.db.models import Base
from app.db.session import engine as default_engine

logger = logging.getLogger(__name__)


def upgrade_db_schema(bind: Engine | Connection | None = None) -> None:
    """
    Ensure all tables, missing columns, foreign keys, and indexes exist.

    Safe and idempotent to run repeatedly:
    1. Ensures all tables declared in Base.metadata exist (e.g. investigation_batches).
    2. Inspects the 'scans' table:
       - If 'batch_id' column is missing, adds 'batch_id VARCHAR(64)' with foreign key
         referencing investigation_batches(batch_id) ON DELETE SET NULL.
       - Ensures index 'ix_scans_batch_id' on scans(batch_id) exists.
       - On PostgreSQL, ensures foreign key constraint exists if column was present without FK.
    """
    target_bind = bind if bind is not None else default_engine

    # 1. Ensure all model tables exist (e.g. investigation_batches, scans, detections)
    Base.metadata.create_all(bind=target_bind)

    # 2. Inspect 'scans' table
    insp = inspect(target_bind)
    if not insp.has_table("scans"):
        return

    existing_columns = {col["name"] for col in insp.get_columns("scans")}

    dialect_name = "sqlite"
    if hasattr(target_bind, "dialect"):
        dialect_name = target_bind.dialect.name
    elif hasattr(target_bind, "engine"):
        dialect_name = target_bind.engine.dialect.name

    def _apply_migrations(conn: Connection) -> None:
        # Step A: Add missing scans.batch_id column if needed
        if "batch_id" not in existing_columns:
            logger.info("Migrating scans table: adding batch_id column...")
            if dialect_name == "postgresql":
                conn.execute(
                    text(
                        "ALTER TABLE scans "
                        "ADD COLUMN IF NOT EXISTS batch_id VARCHAR(64) "
                        "REFERENCES investigation_batches(batch_id) ON DELETE SET NULL;"
                    )
                )
            else:
                conn.execute(
                    text(
                        "ALTER TABLE scans "
                        "ADD COLUMN batch_id VARCHAR(64) "
                        "REFERENCES investigation_batches(batch_id) ON DELETE SET NULL;"
                    )
                )
        else:
            # Step B: If column already exists on PostgreSQL, ensure foreign key constraint exists
            if dialect_name == "postgresql":
                try:
                    fks = insp.get_foreign_keys("scans")
                    has_batch_fk = any(
                        fk.get("referred_table") == "investigation_batches"
                        and "batch_id" in fk.get("constrained_columns", [])
                        for fk in fks
                    )
                    if not has_batch_fk:
                        conn.execute(
                            text(
                                "DO $$ "
                                "BEGIN "
                                "    IF NOT EXISTS ("
                                "        SELECT 1 FROM pg_constraint WHERE conname = 'scans_batch_id_fkey'"
                                "    ) THEN "
                                "        ALTER TABLE scans "
                                "        ADD CONSTRAINT scans_batch_id_fkey "
                                "        FOREIGN KEY (batch_id) "
                                "        REFERENCES investigation_batches(batch_id) ON DELETE SET NULL; "
                                "    END IF; "
                                "END $$;"
                            )
                        )
                except Exception as ex:
                    logger.warning("Could not verify or add foreign key constraint on scans.batch_id: %s", ex)

        # Step C: Ensure index on scans(batch_id) exists
        try:
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_scans_batch_id ON scans(batch_id);"))
        except Exception as ex:
            logger.warning("Could not create index ix_scans_batch_id: %s", ex)

    if isinstance(target_bind, Engine):
        with target_bind.begin() as conn:
            _apply_migrations(conn)
    else:
        _apply_migrations(target_bind)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Running database schema upgrade on default engine...")
    upgrade_db_schema()
    print("Schema upgrade completed successfully.")

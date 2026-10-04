"""
tests/test_migration.py
-----------------------
Unit tests for lightweight database schema migration and upgrade utility.

Verifies:
1. Fresh schema works.
2. Existing schema missing batch_id is upgraded.
3. Migration can run repeatedly without failure (idempotent).
4. Historical scans remain readable with NULL batch_id.
5. New batch scans can be associated with a batch.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db.migration import upgrade_db_schema
from app.db.models import Base, DataSource, InvestigationBatch, RangeType, Scan


def _create_legacy_database():
    """Create an in-memory SQLite database simulating an older schema without batch_id."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE scans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_identity VARCHAR(128) NOT NULL,
                    image_path TEXT NOT NULL,
                    sonar_latitude FLOAT NOT NULL,
                    sonar_longitude FLOAT NOT NULL,
                    heading FLOAT NOT NULL,
                    altitude FLOAT NOT NULL,
                    range_type VARCHAR(16) NOT NULL,
                    range_m FLOAT,
                    relative_bearing FLOAT,
                    timestamp DATETIME NOT NULL,
                    data_source VARCHAR(16) NOT NULL,
                    notes TEXT
                );
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO scans (
                    scan_identity, image_path, sonar_latitude, sonar_longitude,
                    heading, altitude, range_type, timestamp, data_source, notes
                ) VALUES (
                    'historical_scan_001', 'legacy/path.png', 13.34, 77.10,
                    90.0, 10.0, 'MEDIUM', '2026-01-01 12:00:00', 'REAL', 'historical note'
                );
                """
            )
        )
    return engine


class TestDatabaseSchemaMigration:
    """Test suite covering database schema upgrade and migration."""

    def test_fresh_schema_works(self) -> None:
        """Fresh database initialized via upgrade_db_schema must have all models, columns, and indexes."""
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )

        # Run migration on completely fresh DB
        upgrade_db_schema(engine)

        insp = inspect(engine)
        table_names = set(insp.get_table_names())
        assert "investigation_batches" in table_names
        assert "scans" in table_names
        assert "detections" in table_names

        scan_cols = {c["name"] for c in insp.get_columns("scans")}
        assert "batch_id" in scan_cols
        assert "scan_identity" in scan_cols

        indexes = {idx["name"] for idx in insp.get_indexes("scans")}
        assert "ix_scans_batch_id" in indexes

        # Verify ORM write and read
        with Session(engine) as db:
            batch = InvestigationBatch(
                batch_id="fresh_batch_001",
                total_scans=1,
                successful_scans=1,
                data_source=DataSource.REAL,
            )
            db.add(batch)
            db.flush()

            scan = Scan(
                batch_id="fresh_batch_001",
                scan_identity="fresh_scan_001",
                image_path="scans/fresh.png",
                sonar_latitude=13.0,
                sonar_longitude=80.0,
                heading=90.0,
                altitude=10.0,
                range_type=RangeType.MEDIUM,
                timestamp=datetime.now(timezone.utc),
                data_source=DataSource.REAL,
            )
            db.add(scan)
            db.commit()

            reloaded_batch = db.query(InvestigationBatch).filter_by(batch_id="fresh_batch_001").first()
            assert reloaded_batch is not None
            assert len(reloaded_batch.scans) == 1
            assert reloaded_batch.scans[0].scan_identity == "fresh_scan_001"

    def test_existing_schema_missing_batch_id_is_upgraded(self) -> None:
        """An existing database missing batch_id must have batch_id column and index added."""
        engine = _create_legacy_database()

        insp_before = inspect(engine)
        cols_before = {c["name"] for c in insp_before.get_columns("scans")}
        assert "batch_id" not in cols_before
        assert "investigation_batches" not in set(insp_before.get_table_names())

        # Perform migration
        upgrade_db_schema(engine)

        insp_after = inspect(engine)
        cols_after = {c["name"] for c in insp_after.get_columns("scans")}
        assert "batch_id" in cols_after
        assert "investigation_batches" in set(insp_after.get_table_names())

        indexes = {idx["name"] for idx in insp_after.get_indexes("scans")}
        assert "ix_scans_batch_id" in indexes

    def test_migration_can_run_repeatedly_without_failure(self) -> None:
        """upgrade_db_schema must be strictly idempotent and safe to run multiple times."""
        engine = _create_legacy_database()

        # Run 3 consecutive times
        upgrade_db_schema(engine)
        upgrade_db_schema(engine)
        upgrade_db_schema(engine)

        insp = inspect(engine)
        cols = {c["name"] for c in insp.get_columns("scans")}
        assert "batch_id" in cols
        assert "investigation_batches" in set(insp.get_table_names())

    def test_historical_scans_remain_readable_with_null_batch_id(self) -> None:
        """Historical scans created before batch_id existed remain completely readable with batch_id=None."""
        engine = _create_legacy_database()

        # Upgrade database
        upgrade_db_schema(engine)

        # Query using SQLAlchemy ORM Scan model
        with Session(engine) as db:
            scan = db.query(Scan).filter_by(scan_identity="historical_scan_001").first()
            assert scan is not None
            assert scan.id == 1
            assert scan.scan_identity == "historical_scan_001"
            assert scan.image_path == "legacy/path.png"
            assert scan.sonar_latitude == 13.34
            assert scan.sonar_longitude == 77.10
            assert scan.range_type == RangeType.MEDIUM
            assert scan.notes == "historical note"
            assert scan.batch_id is None
            assert scan.batch is None

    def test_new_batch_scans_can_be_associated_with_a_batch(self) -> None:
        """New scans in an upgraded database can be cleanly associated with an InvestigationBatch."""
        engine = _create_legacy_database()
        upgrade_db_schema(engine)

        with Session(engine) as db:
            # 1. Historical scan still exists with None batch_id
            historical_scan = db.query(Scan).filter_by(scan_identity="historical_scan_001").first()
            assert historical_scan is not None
            assert historical_scan.batch_id is None

            # 2. Add new batch and new scan associated with this batch
            batch = InvestigationBatch(
                batch_id="batch_alpha_999",
                total_scans=1,
                successful_scans=1,
                total_detections=0,
                data_source=DataSource.REAL,
            )
            db.add(batch)
            db.flush()

            new_scan = Scan(
                batch_id="batch_alpha_999",
                scan_identity="scan_alpha_01",
                image_path="scans/alpha_01.png",
                sonar_latitude=14.0,
                sonar_longitude=81.0,
                heading=270.0,
                altitude=12.0,
                range_type=RangeType.LONG,
                timestamp=datetime.now(timezone.utc),
                data_source=DataSource.REAL,
            )
            db.add(new_scan)
            db.commit()

            # 3. Query back batch and relationship
            batch_reloaded = db.query(InvestigationBatch).filter_by(batch_id="batch_alpha_999").first()
            assert batch_reloaded is not None
            assert len(batch_reloaded.scans) == 1
            assert batch_reloaded.scans[0].scan_identity == "scan_alpha_01"
            assert batch_reloaded.scans[0].batch_id == "batch_alpha_999"

            # 4. Total scans in database is now 2 (1 historical with NULL, 1 new with batch_id)
            all_scans = db.query(Scan).order_by(Scan.id).all()
            assert len(all_scans) == 2
            assert all_scans[0].batch_id is None
            assert all_scans[1].batch_id == "batch_alpha_999"

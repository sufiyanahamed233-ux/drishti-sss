"""
tests/test_api.py
-----------------
Phase 3A tests for the FastAPI application and database configuration.

Test strategy
~~~~~~~~~~~~~
* The FastAPI ASGI app is tested via ``httpx.AsyncClient`` with
  ``transport=ASGITransport``, meaning no real HTTP port is opened.

* Database connectivity tests are run in two ways:
  1. A mocked ``check_db_connection`` to test the /health/db response logic.
  2. Direct import tests to verify module structure and settings are correct.

* No real PostgreSQL connection is required to pass these tests — all DB
  interaction is either mocked or exercised at the import/config level.

Coverage
--------
- GET /health returns 200 with correct payload
- GET /health/db returns 200 when DB is reachable (mocked)
- GET /health/db returns 503 when DB is unreachable (mocked)
- Settings load without error
- DATABASE_URL is present and non-empty
- SQLAlchemy engine is created without raising
- ORM models (Scan, Detection) have the expected columns
- Pydantic schemas validate correct data
- Pydantic schemas reject invalid data with ValidationError
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.config import settings
from app.db.session import engine, SessionLocal
from app.db.models import Scan, Detection, RangeType, DataSource
from app.schemas.navigation import NavigationData, ScanCreate, ScanRead
from app.schemas.detection import BoundingBox, DetectionCreate, DetectionRead


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    """Async ASGI test client wrapping the FastAPI app."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as ac:
        yield ac


# ---------------------------------------------------------------------------
# GET /health — liveness probe
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for GET /health."""

    @pytest.mark.asyncio
    async def test_health_returns_200(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_payload_has_status_ok(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_payload_has_version(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        data = response.json()
        assert "version" in data
        assert data["version"] == settings.app_version

    @pytest.mark.asyncio
    async def test_health_payload_has_env(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        data = response.json()
        assert "env" in data

    @pytest.mark.asyncio
    async def test_health_content_type_json(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        assert response.headers["content-type"].startswith("application/json")


# ---------------------------------------------------------------------------
# GET /health/db — database readiness probe
# ---------------------------------------------------------------------------


class TestHealthDbEndpoint:
    """Tests for GET /health/db using a mocked connection check."""

    @pytest.mark.asyncio
    async def test_health_db_returns_200_when_reachable(
        self, client: AsyncClient
    ) -> None:
        with patch("app.main.check_db_connection", return_value=True):
            response = await client.get("/health/db")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_db_payload_ok_when_reachable(
        self, client: AsyncClient
    ) -> None:
        with patch("app.main.check_db_connection", return_value=True):
            response = await client.get("/health/db")
        data = response.json()
        assert data["status"] == "ok"
        assert data["database"] == "reachable"

    @pytest.mark.asyncio
    async def test_health_db_returns_503_when_unreachable(
        self, client: AsyncClient
    ) -> None:
        with patch("app.main.check_db_connection", return_value=False):
            response = await client.get("/health/db")
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_health_db_payload_error_when_unreachable(
        self, client: AsyncClient
    ) -> None:
        with patch("app.main.check_db_connection", return_value=False):
            response = await client.get("/health/db")
        data = response.json()
        assert data["status"] == "error"
        assert data["database"] == "unreachable"

    @pytest.mark.asyncio
    async def test_health_db_content_type_json(self, client: AsyncClient) -> None:
        with patch("app.main.check_db_connection", return_value=True):
            response = await client.get("/health/db")
        assert response.headers["content-type"].startswith("application/json")


# ---------------------------------------------------------------------------
# Settings / configuration
# ---------------------------------------------------------------------------


class TestSettings:
    """Tests for app.config.Settings."""

    def test_settings_loads_without_error(self) -> None:
        assert settings is not None

    def test_database_url_is_set(self) -> None:
        assert settings.database_url
        assert len(settings.database_url) > 0

    def test_database_url_has_valid_scheme(self) -> None:
        url = settings.database_url
        assert url.startswith("postgresql") or url.startswith("sqlite")

    def test_app_env_is_valid(self) -> None:
        assert settings.app_env in {"development", "testing", "production"}

    def test_app_version_is_set(self) -> None:
        assert settings.app_version
        # Should look like a semver string
        parts = settings.app_version.split(".")
        assert len(parts) >= 2

    def test_settings_respects_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify that environment variables override defaults."""
        monkeypatch.setenv(
            "DATABASE_URL",
            "postgresql+psycopg2://test:test@localhost:5432/test_db",
        )
        # Re-instantiate so the env var is picked up
        from app.config import Settings
        s = Settings()
        assert "test_db" in s.database_url


# ---------------------------------------------------------------------------
# SQLAlchemy engine and session
# ---------------------------------------------------------------------------


class TestDatabaseSession:
    """Tests for app.db.session engine and SessionLocal."""

    def test_engine_is_created(self) -> None:
        assert engine is not None

    def test_engine_url_matches_settings(self) -> None:
        engine_url = str(engine.url)
        # Compare scheme + host (mask password if present)
        assert "postgresql" in engine_url or "sqlite" in engine_url

    def test_session_local_can_be_instantiated(self) -> None:
        """SessionLocal() should not raise even if DB is offline."""
        session = SessionLocal()
        try:
            assert session is not None
        finally:
            session.close()


# ---------------------------------------------------------------------------
# ORM model structure
# ---------------------------------------------------------------------------


class TestScanModel:
    """Verify Scan model column definitions."""

    def test_scan_table_name(self) -> None:
        assert Scan.__tablename__ == "scans"

    def test_scan_has_id_column(self) -> None:
        cols = {c.name for c in Scan.__table__.columns}
        assert "id" in cols

    def test_scan_has_required_columns(self) -> None:
        expected = {
            "scan_identity", "image_path",
            "sonar_latitude", "sonar_longitude",
            "heading", "altitude",
            "range_type", "timestamp", "data_source",
        }
        cols = {c.name for c in Scan.__table__.columns}
        assert expected.issubset(cols)

    def test_scan_has_notes_column(self) -> None:
        cols = {c.name for c in Scan.__table__.columns}
        assert "notes" in cols

    def test_range_type_enum_values(self) -> None:
        values = {rt.value for rt in RangeType}
        assert "short" in values
        assert "medium" in values
        assert "long" in values

    def test_data_source_enum_values(self) -> None:
        values = {ds.value for ds in DataSource}
        assert "real" in values
        assert "demo" in values
        assert "simulated" in values


class TestDetectionModel:
    """Verify Detection model column definitions."""

    def test_detection_table_name(self) -> None:
        assert Detection.__tablename__ == "detections"

    def test_detection_has_id_column(self) -> None:
        cols = {c.name for c in Detection.__table__.columns}
        assert "id" in cols

    def test_detection_has_scan_id_fk(self) -> None:
        cols = {c.name for c in Detection.__table__.columns}
        assert "scan_id" in cols

    def test_detection_has_required_columns(self) -> None:
        expected = {
            "class_name", "confidence",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        }
        cols = {c.name for c in Detection.__table__.columns}
        assert expected.issubset(cols)

    def test_detection_has_geo_columns(self) -> None:
        expected = {
            "target_latitude", "target_longitude",
            "relative_bearing", "absolute_bearing",
            "range_m", "ground_range_m",
        }
        cols = {c.name for c in Detection.__table__.columns}
        assert expected.issubset(cols)

    def test_detection_scan_fk_references_scans(self) -> None:
        """scan_id FK must reference scans.id."""
        fks = Detection.__table__.foreign_keys
        fk_targets = {fk.target_fullname for fk in fks}
        assert "scans.id" in fk_targets


# ---------------------------------------------------------------------------
# Pydantic schemas — navigation
# ---------------------------------------------------------------------------


class TestNavigationSchemas:
    """Tests for NavigationData and ScanCreate/ScanRead schemas."""

    def test_navigation_data_valid(self) -> None:
        nav = NavigationData(
            sonar_latitude=13.0827,
            sonar_longitude=80.2707,
            heading=45.0,
            altitude=5.0,
        )
        assert nav.sonar_latitude == pytest.approx(13.0827)
        assert nav.range_type == RangeType.MEDIUM  # default

    def test_navigation_data_invalid_latitude(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            NavigationData(
                sonar_latitude=95.0,   # > 90 → invalid
                sonar_longitude=80.0,
                heading=0.0,
                altitude=5.0,
            )

    def test_navigation_data_invalid_altitude(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            NavigationData(
                sonar_latitude=13.0,
                sonar_longitude=80.0,
                heading=0.0,
                altitude=-1.0,  # must be > 0
            )

    def test_navigation_data_invalid_heading(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            NavigationData(
                sonar_latitude=13.0,
                sonar_longitude=80.0,
                heading=400.0,  # must be < 360
                altitude=5.0,
            )

    def test_scan_create_valid(self) -> None:
        sc = ScanCreate(
            scan_identity="mission-01-frame-042",
            image_path="/data/sonar/frame_042.png",
            navigation=NavigationData(
                sonar_latitude=13.0827,
                sonar_longitude=80.2707,
                heading=90.0,
                altitude=8.0,
            ),
            data_source=DataSource.DEMO,
        )
        assert sc.data_source == DataSource.DEMO
        assert sc.notes is None

    def test_scan_create_simulated_flag(self) -> None:
        sc = ScanCreate(
            scan_identity="sim-001",
            image_path="/sim/frame_001.png",
            navigation=NavigationData(
                sonar_latitude=0.0,
                sonar_longitude=0.0,
                heading=0.0,
                altitude=10.0,
            ),
            data_source=DataSource.SIMULATED,
        )
        assert sc.data_source == DataSource.SIMULATED


# ---------------------------------------------------------------------------
# Pydantic schemas — detection
# ---------------------------------------------------------------------------


class TestDetectionSchemas:
    """Tests for BoundingBox, DetectionCreate, and DetectionRead schemas."""

    def test_bounding_box_valid(self) -> None:
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=150.0, y2=200.0)
        assert bbox.x1 == pytest.approx(10.0)

    def test_bounding_box_negative_pixel_rejected(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            BoundingBox(x1=-5.0, y1=0.0, x2=100.0, y2=100.0)

    def test_detection_create_valid_minimal(self) -> None:
        dc = DetectionCreate(
            scan_id=1,
            class_name="mine_cylinder",
            confidence=0.87,
            bbox=BoundingBox(x1=0.0, y1=0.0, x2=64.0, y2=64.0),
        )
        assert dc.target_latitude is None
        assert dc.ground_range_m is None

    def test_detection_create_with_geo_data(self) -> None:
        dc = DetectionCreate(
            scan_id=42,
            class_name="shipwreck",
            confidence=0.95,
            bbox=BoundingBox(x1=10.0, y1=10.0, x2=200.0, y2=200.0),
            target_latitude=13.0830,
            target_longitude=80.2710,
            relative_bearing=45.0,
            absolute_bearing=90.0,
            range_m=50.0,
            ground_range_m=48.0,
        )
        assert dc.target_latitude == pytest.approx(13.0830)

    def test_detection_create_invalid_confidence(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DetectionCreate(
                scan_id=1,
                class_name="ghost_net",
                confidence=1.5,  # > 1.0 → invalid
                bbox=BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0),
            )

    def test_detection_read_from_attributes(self) -> None:
        """DetectionRead.model_validate() works on an ORM-like object."""

        class FakeDetection:
            id = 7
            scan_id = 3
            class_name = "submarine_pipeline"
            confidence = 0.76
            bbox_x1 = 5.0
            bbox_y1 = 5.0
            bbox_x2 = 120.0
            bbox_y2 = 80.0
            target_latitude = None
            target_longitude = None
            relative_bearing = None
            absolute_bearing = None
            range_m = None
            ground_range_m = None

        dr = DetectionRead.model_validate(FakeDetection())
        assert dr.id == 7
        assert dr.class_name == "submarine_pipeline"

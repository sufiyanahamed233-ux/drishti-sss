"""
test_investigation_report.py
----------------------------
Automated test suite for Phase 3 Investigation Report generation:
- GET /api/scans/{scan_id}/report/json
- GET /api/scans/{scan_id}/report/geojson

Verifies:
- Valid JSON report structure and provenance distinction (Req 3, 4)
- Valid GeoJSON FeatureCollection (RFC 7946) (Req 5, 13)
- Correct [longitude, latitude] coordinate order (Req 5, 12)
- GeoJSON properties completeness (Req 6)
- Multiple detections handling (Req 12)
- Zero detections handling (Req 8, 12)
- Null / unmapped coordinates handling (Req 7, 12)
- Non-existent scan returns 404 (Req 9, 12)
"""

from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timezone
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base, DataSource, Detection, RangeType, Scan
from app.db.session import get_db
from app.main import app


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_db() -> Generator[Session, None, None]:
    """Provide an isolated in-memory SQLite database and override get_db."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionTest = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    def _override_get_db():
        db = SessionTest()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override_get_db
    test_session = SessionTest()
    try:
        yield test_session
    finally:
        app.dependency_overrides.pop(get_db, None)
        test_session.close()


@pytest_asyncio.fixture
async def client(mock_db: Session) -> AsyncClient:
    """Async HTTP client with overridden db dependency."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as ac:
        yield ac


def _seed_scan_with_detections(db: Session) -> Scan:
    """Seed a scan with multiple georeferenced detections."""
    scan = Scan(
        scan_identity="SURVEY-MISSION-ALPHA-01",
        image_path="/data/sonar_alpha_01.png",
        sonar_latitude=13.0827,
        sonar_longitude=80.2707,
        heading=45.0,
        altitude=12.5,
        range_type=RangeType.SLANT,
        range_m=50.0,
        relative_bearing=15.0,
        timestamp=datetime(2026, 10, 4, 7, 0, 0, tzinfo=timezone.utc),
        data_source=DataSource.REAL,
        notes="High-priority seabed survey",
    )
    db.add(scan)
    db.flush()

    det1 = Detection(
        scan_id=scan.id,
        class_name="mine_cylinder",
        confidence=0.94,
        bbox_x1=100.0,
        bbox_y1=150.0,
        bbox_x2=200.0,
        bbox_y2=250.0,
        target_latitude=13.0831,
        target_longitude=80.2712,
        relative_bearing=15.0,
        absolute_bearing=60.0,
        range_m=48.5,
        ground_range_m=46.86,
    )
    det2 = Detection(
        scan_id=scan.id,
        class_name="shipwreck",
        confidence=0.88,
        bbox_x1=400.0,
        bbox_y1=350.0,
        bbox_x2=600.0,
        bbox_y2=550.0,
        target_latitude=13.0835,
        target_longitude=80.2718,
        relative_bearing=30.0,
        absolute_bearing=75.0,
        range_m=62.0,
        ground_range_m=60.72,
    )
    db.add_all([det1, det2])
    db.commit()
    db.refresh(scan)
    return scan


def _seed_scan_zero_detections(db: Session) -> Scan:
    """Seed a scan with zero detections."""
    scan = Scan(
        scan_identity="SURVEY-CLEAR-BETA-02",
        image_path="/data/sonar_clear.png",
        sonar_latitude=12.5000,
        sonar_longitude=80.1000,
        heading=90.0,
        altitude=15.0,
        range_type=RangeType.GROUND,
        range_m=75.0,
        relative_bearing=0.0,
        timestamp=datetime(2026, 10, 4, 8, 30, 0, tzinfo=timezone.utc),
        data_source=DataSource.DEMO,
        notes="Clear water scan",
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)
    return scan


def _seed_scan_with_null_coords(db: Session) -> Scan:
    """Seed a scan with detections having null / unmapped coordinates."""
    scan = Scan(
        scan_identity="SURVEY-PARTIAL-GAMMA-03",
        image_path="/data/sonar_partial.png",
        sonar_latitude=14.0000,
        sonar_longitude=81.0000,
        heading=180.0,
        altitude=20.0,
        range_type=RangeType.SLANT,
        range_m=60.0,
        relative_bearing=0.0,
        timestamp=datetime(2026, 10, 4, 9, 15, 0, tzinfo=timezone.utc),
        data_source=DataSource.SIMULATED,
    )
    db.add(scan)
    db.flush()

    # One georeferenced, one without georeferencing
    det_georef = Detection(
        scan_id=scan.id,
        class_name="ghost_net",
        confidence=0.85,
        bbox_x1=50.0,
        bbox_y1=50.0,
        bbox_x2=120.0,
        bbox_y2=120.0,
        target_latitude=13.9995,
        target_longitude=81.0005,
        relative_bearing=-10.0,
        absolute_bearing=170.0,
        range_m=35.0,
        ground_range_m=28.72,
    )
    det_unmapped = Detection(
        scan_id=scan.id,
        class_name="debris",
        confidence=0.72,
        bbox_x1=300.0,
        bbox_y1=300.0,
        bbox_x2=350.0,
        bbox_y2=350.0,
        target_latitude=None,
        target_longitude=None,
        relative_bearing=None,
        absolute_bearing=None,
        range_m=None,
        ground_range_m=None,
    )
    db.add_all([det_georef, det_unmapped])
    db.commit()
    db.refresh(scan)
    return scan


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


class TestJSONInvestigationReport:
    """Tests for GET /api/scans/{scan_id}/report/json."""

    @pytest.mark.asyncio
    async def test_json_report_structure_and_fields(
        self, client: AsyncClient, mock_db: Session
    ) -> None:
        """JSON report contains all required sections and fields per Req 3."""
        scan = _seed_scan_with_detections(mock_db)
        res = await client.get(f"/api/scans/{scan.id}/report/json")
        assert res.status_code == 200

        data = res.json()

        # 1. Report metadata
        assert "report_metadata" in data
        assert data["report_metadata"]["report_type"] == "INVESTIGATION_REPORT"
        assert "provenance" in data["report_metadata"]

        # 2. Scan identity & ID
        assert data["scan_identity"] == "SURVEY-MISSION-ALPHA-01"
        assert data["scan_id"] == scan.id

        # 3. Timestamp & data source
        assert data["timestamp"] is not None
        assert data["data_source"] == "real"

        # 4. Sonar navigation metadata
        assert "sonar_navigation_metadata" in data
        nav = data["sonar_navigation_metadata"]
        assert nav["sonar_latitude"] == pytest.approx(13.0827)
        assert nav["sonar_longitude"] == pytest.approx(80.2707)
        assert nav["heading"] == pytest.approx(45.0)
        assert nav["altitude"] == pytest.approx(12.5)
        assert nav["range_m"] == pytest.approx(50.0)
        assert nav["range_type"] == "slant"
        assert nav["relative_bearing"] == pytest.approx(15.0)

        # 5. Detection count & detections
        assert data["detection_count"] == 2
        assert len(data["detections"]) == 2

        # 6. Detection item fields
        det1 = data["detections"][0]
        assert det1["class_name"] == "mine_cylinder"
        assert det1["detection_class"] == "mine_cylinder"
        assert det1["confidence"] == pytest.approx(0.94)
        assert det1["bbox"] == {"x1": 100.0, "y1": 150.0, "x2": 200.0, "y2": 250.0}
        assert det1["range"] == pytest.approx(48.5)
        assert det1["range_type"] == "slant"
        assert det1["ground_range"] == pytest.approx(46.86)
        assert det1["relative_bearing"] == pytest.approx(15.0)
        assert det1["absolute_bearing"] == pytest.approx(60.0)
        assert det1["target_latitude"] == pytest.approx(13.0831)
        assert det1["target_longitude"] == pytest.approx(80.2712)

    @pytest.mark.asyncio
    async def test_json_report_provenance_distinction(
        self, client: AsyncClient, mock_db: Session
    ) -> None:
        """Preserves distinct provenance: AI detection vs deterministic geolocation (Req 4)."""
        scan = _seed_scan_with_detections(mock_db)
        res = await client.get(f"/api/scans/{scan.id}/report/json")
        assert res.status_code == 200

        data = res.json()
        prov = data["report_metadata"]["provenance"]

        # Platform navigation must indicate sensor/measured
        assert "sensor" in prov["navigation_data"].lower() or "measured" in prov["navigation_data"].lower()

        # Detections must indicate AI/YOLO
        assert "ai" in prov["detection_data"].lower() or "yolo" in prov["detection_data"].lower()

        # Geolocation must state deterministic calculation and NOT predicted by AI
        assert "deterministic" in prov["geolocation_data"].lower()
        assert "not predicted by ai" in prov["geolocation_data"].lower()

        # Detections themselves must carry explicit provenance tags
        for det in data["detections"]:
            assert det["detection_source"] == "AI_YOLO"
            assert det["geolocation_method"] == "DETERMINISTIC_GEOMETRY"

    @pytest.mark.asyncio
    async def test_json_report_zero_detections(
        self, client: AsyncClient, mock_db: Session
    ) -> None:
        """Scan with zero detections returns report with detection_count = 0 (Req 8)."""
        scan = _seed_scan_zero_detections(mock_db)
        res = await client.get(f"/api/scans/{scan.id}/report/json")
        assert res.status_code == 200

        data = res.json()
        assert data["detection_count"] == 0
        assert data["detections"] == []
        assert data["scan_identity"] == "SURVEY-CLEAR-BETA-02"

    @pytest.mark.asyncio
    async def test_json_report_includes_null_coordinates(
        self, client: AsyncClient, mock_db: Session
    ) -> None:
        """JSON report still includes detections with null coordinates (Req 7)."""
        scan = _seed_scan_with_null_coords(mock_db)
        res = await client.get(f"/api/scans/{scan.id}/report/json")
        assert res.status_code == 200

        data = res.json()
        assert data["detection_count"] == 2
        classes = [d["class_name"] for d in data["detections"]]
        assert "ghost_net" in classes
        assert "debris" in classes

        debris_det = next(d for d in data["detections"] if d["class_name"] == "debris")
        assert debris_det["target_latitude"] is None
        assert debris_det["target_longitude"] is None

    @pytest.mark.asyncio
    async def test_json_report_missing_scan_returns_404(
        self, client: AsyncClient, mock_db: Session
    ) -> None:
        """Request for non-existent scan ID returns 404 (Req 9)."""
        res = await client.get("/api/scans/999999/report/json")
        assert res.status_code == 404
        assert "not found" in res.json()["detail"].lower()


class TestGeoJSONInvestigationReport:
    """Tests for GET /api/scans/{scan_id}/report/geojson."""

    @pytest.mark.asyncio
    async def test_geojson_valid_feature_collection(
        self, client: AsyncClient, mock_db: Session
    ) -> None:
        """Returns valid RFC 7946 GeoJSON FeatureCollection (Req 5, 13)."""
        scan = _seed_scan_with_detections(mock_db)
        res = await client.get(f"/api/scans/{scan.id}/report/geojson")
        assert res.status_code == 200

        data = res.json()
        assert data["type"] == "FeatureCollection"
        assert isinstance(data["features"], list)
        assert len(data["features"]) == 2

        for feat in data["features"]:
            assert feat["type"] == "Feature"
            assert feat["geometry"]["type"] == "Point"
            coords = feat["geometry"]["coordinates"]
            assert len(coords) == 2
            assert isinstance(coords[0], float)
            assert isinstance(coords[1], float)

    @pytest.mark.asyncio
    async def test_geojson_coordinate_order_strictly_lon_lat(
        self, client: AsyncClient, mock_db: Session
    ) -> None:
        """GeoJSON Point coordinates must strictly be [longitude, latitude] (Req 5, 12)."""
        scan = _seed_scan_with_detections(mock_db)
        res = await client.get(f"/api/scans/{scan.id}/report/geojson")
        assert res.status_code == 200

        data = res.json()
        feat1 = data["features"][0]
        coords1 = feat1["geometry"]["coordinates"]

        # Longitude is ~80.27, Latitude is ~13.08
        lon, lat = coords1[0], coords1[1]
        assert lon == pytest.approx(80.2712)
        assert lat == pytest.approx(13.0831)

        feat2 = data["features"][1]
        coords2 = feat2["geometry"]["coordinates"]
        assert coords2[0] == pytest.approx(80.2718)
        assert coords2[1] == pytest.approx(13.0835)

    @pytest.mark.asyncio
    async def test_geojson_properties_completeness(
        self, client: AsyncClient, mock_db: Session
    ) -> None:
        """GeoJSON Feature properties include all required fields per Req 6."""
        scan = _seed_scan_with_detections(mock_db)
        res = await client.get(f"/api/scans/{scan.id}/report/geojson")
        assert res.status_code == 200

        feat = res.json()["features"][0]
        props = feat["properties"]

        assert props["scan_id"] == scan.id
        assert props["scan_identity"] == "SURVEY-MISSION-ALPHA-01"
        assert "detection_id" in props
        assert props["detection_index"] == 0
        assert props["class_name"] == "mine_cylinder"
        assert props["confidence"] == pytest.approx(0.94)
        assert props["range_m"] == pytest.approx(48.5)
        assert props["ground_range_m"] == pytest.approx(46.86)
        assert props["relative_bearing"] == pytest.approx(15.0)
        assert props["absolute_bearing"] == pytest.approx(60.0)
        assert props["timestamp"] is not None
        assert props["data_source"] == "real"

    @pytest.mark.asyncio
    async def test_geojson_zero_detections_returns_empty_feature_collection(
        self, client: AsyncClient, mock_db: Session
    ) -> None:
        """Scan with zero detections returns valid empty FeatureCollection (Req 8)."""
        scan = _seed_scan_zero_detections(mock_db)
        res = await client.get(f"/api/scans/{scan.id}/report/geojson")
        assert res.status_code == 200

        data = res.json()
        assert data["type"] == "FeatureCollection"
        assert data["features"] == []

    @pytest.mark.asyncio
    async def test_geojson_omits_null_coordinates(
        self, client: AsyncClient, mock_db: Session
    ) -> None:
        """Detection with null coordinates is omitted from GeoJSON features (Req 7)."""
        scan = _seed_scan_with_null_coords(mock_db)
        res = await client.get(f"/api/scans/{scan.id}/report/geojson")
        assert res.status_code == 200

        data = res.json()
        assert data["type"] == "FeatureCollection"
        # Only 1 of 2 detections was georeferenced; the unmapped one must NOT have a Point feature
        assert len(data["features"]) == 1
        assert data["features"][0]["properties"]["class_name"] == "ghost_net"

    @pytest.mark.asyncio
    async def test_geojson_missing_scan_returns_404(
        self, client: AsyncClient, mock_db: Session
    ) -> None:
        """Request for non-existent scan ID returns 404 (Req 9)."""
        res = await client.get("/api/scans/999999/report/geojson")
        assert res.status_code == 404
        assert "not found" in res.json()["detail"].lower()


class TestPDFInvestigationReport:
    """Tests for GET /api/scans/{scan_id}/report/pdf."""

    @pytest.mark.asyncio
    async def test_pdf_report_valid_scan_with_detections(
        self, client: AsyncClient, mock_db: Session
    ) -> None:
        """Valid scan with detections returns 200 application/pdf with proper attachment header."""
        scan = _seed_scan_with_detections(mock_db)
        res = await client.get(f"/api/scans/{scan.id}/report/pdf")
        assert res.status_code == 200

        # Check content type
        assert res.headers["content-type"] == "application/pdf"

        # Check content-disposition and filename
        cd = res.headers.get("content-disposition", "")
        assert "attachment" in cd
        assert f"drishti_sss_scan_{scan.id}_report.pdf" in cd

        # Check non-empty valid PDF binary payload
        assert len(res.content) > 1000
        assert res.content.startswith(b"%PDF-")

    @pytest.mark.asyncio
    async def test_pdf_report_zero_detections(
        self, client: AsyncClient, mock_db: Session
    ) -> None:
        """Zero-detection scan successfully generates PDF without crashing."""
        scan = _seed_scan_zero_detections(mock_db)
        res = await client.get(f"/api/scans/{scan.id}/report/pdf")
        assert res.status_code == 200
        assert res.headers["content-type"] == "application/pdf"
        assert res.content.startswith(b"%PDF-")
        assert len(res.content) > 1000

    @pytest.mark.asyncio
    async def test_pdf_report_null_unmapped_coordinates(
        self, client: AsyncClient, mock_db: Session
    ) -> None:
        """Scan with unmapped / null coordinates generates PDF with 'Not available' labels."""
        scan = _seed_scan_with_null_coords(mock_db)
        res = await client.get(f"/api/scans/{scan.id}/report/pdf")
        assert res.status_code == 200
        assert res.headers["content-type"] == "application/pdf"
        assert res.content.startswith(b"%PDF-")
        assert len(res.content) > 1000

    @pytest.mark.asyncio
    async def test_pdf_report_missing_scan_returns_404(
        self, client: AsyncClient, mock_db: Session
    ) -> None:
        """Non-existent scan returns 404 Not Found."""
        res = await client.get("/api/scans/999999/report/pdf")
        assert res.status_code == 404
        assert "not found" in res.json()["detail"].lower()


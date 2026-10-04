"""
test_batches_api.py
-------------------
Automated test suite for batch upload and validation:
POST /api/batches/analyze

Tests the LOCKED navigation.csv design (multi-row per scan_file):

Existing tests preserved:
- Valid multi-scan batch processing
- All files in single form field
- Zero-detection scans preserved
- Missing navigation record rejected
- Navigation record referencing missing image rejected
- Missing CSV column rejected
- Invalid latitude/longitude/heading/altitude rejected
- Malformed timestamp rejected
- Unsupported image extension rejected
- Empty batch upload rejected
- Empty CSV rows rejected
- Path traversal rejected
- Invalid data_source rejected

New tests for locked design:
- One scan + one detection geometry row
- One scan + two detection geometry rows
- Two scans with independent geometries
- Zero-detection scan (empty detection fields)
- Invalid range_type rejected (strict, no silent default)
- Duplicate detection_index within a scan rejected
- Partial detection geometry rejected
- Inconsistent repeated scan-level metadata rejected
- Mixed zero-detection + detection rows rejected
- target_geometries reaches ScanInput (verified via mock call inspection)
"""

from __future__ import annotations

import io
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.analysis import get_detector
from app.config import settings
from app.db.models import Base
from app.db.session import get_db
from app.detection.yolo_detector import YOLODetector
from app.main import app
from app.schemas.report import TargetGeometryInput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_dummy_image(format: str = "JPEG") -> bytes:
    """Generate dummy image bytes."""
    img = Image.new("RGB", (32, 32), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def _nav_header() -> str:
    return (
        "scan_file,latitude,longitude,heading,altitude,timestamp,data_source,"
        "detection_index,target_range_m,range_type,relative_bearing\n"
    )


def _nav_row(
    scan_file: str,
    lat: float = 13.34,
    lon: float = 77.10,
    hdg: float = 90.0,
    alt: float = 10.0,
    ts: str = "2026-10-04T11:00:00Z",
    ds: str = "SIMULATED",
    det_idx: str = "",
    range_m: str = "",
    range_type: str = "",
    bearing: str = "",
) -> str:
    return f"{scan_file},{lat},{lon},{hdg},{alt},{ts},{ds},{det_idx},{range_m},{range_type},{bearing}\n"


@pytest.fixture
def mock_db() -> Generator[Session, None, None]:
    """Provide an isolated SQLite database and override get_db."""
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


@pytest.fixture
def mock_detector() -> MagicMock:
    """Provide a mock YOLODetector returning zero detections by default."""
    detector = MagicMock(spec=YOLODetector)
    detector.detect.return_value = []
    app.dependency_overrides[get_detector] = lambda: detector
    yield detector
    app.dependency_overrides.pop(get_detector, None)


@pytest_asyncio.fixture
async def client(mock_db: Session, mock_detector: MagicMock) -> AsyncClient:
    """Async client with overridden db and detector."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as ac:
        yield ac


# ---------------------------------------------------------------------------
# Helper: build multipart files list
# ---------------------------------------------------------------------------


def _build_files(
    csv_text: str,
    images: dict[str, bytes],
) -> list:
    """Build httpx multipart files list."""
    files = [
        ("navigation", ("navigation.csv", csv_text.encode("utf-8"), "text/csv"))
    ]
    for name, data in images.items():
        files.append(("scans", (name, data, "image/jpeg")))
    return files


# ---------------------------------------------------------------------------
# Original Tests (preserved)
# ---------------------------------------------------------------------------


class TestBatchUploadEndpoint:
    """Test suite for POST /api/batches/analyze."""

    @pytest.mark.asyncio
    async def test_valid_multi_scan_batch(
        self, client: AsyncClient, mock_detector: MagicMock
    ) -> None:
        """Upload valid batch with 2 scans and matching navigation.csv (no detection rows)."""
        csv_content = (
            _nav_header()
            + _nav_row("scan_001.jpg", lat=13.340900, lon=77.101000, hdg=90.0, alt=10.0)
            + _nav_row("scan_002.jpg", lat=13.341200, lon=77.101500, hdg=92.0, alt=11.0)
        )
        img1 = _make_dummy_image("JPEG")
        img2 = _make_dummy_image("JPEG")

        files = _build_files(csv_content, {"scan_001.jpg": img1, "scan_002.jpg": img2})
        response = await client.post("/api/batches/analyze", files=files)
        assert response.status_code == 200

        data = response.json()
        assert data["total_scans"] == 2
        assert data["successful_scans"] == 2
        assert data["total_detections"] == 0
        assert len(data["scans"]) == 2

        s1 = data["scans"][0]
        assert s1["sonar_latitude"] == pytest.approx(13.340900)
        assert s1["sonar_longitude"] == pytest.approx(77.101000)
        assert s1["heading"] == pytest.approx(90.0)
        assert s1["altitude"] == pytest.approx(10.0)
        assert s1["data_source"] == "simulated"
        assert s1["detection_count"] == 0

    @pytest.mark.asyncio
    async def test_all_files_in_single_form_field(
        self, client: AsyncClient
    ) -> None:
        """Batch uploaded with all files (including CSV) in single 'files' field."""
        csv_content = (
            _nav_header()
            + _nav_row("scan_alpha.png", lat=12.0, lon=80.0, hdg=45.0, alt=15.0, ds="REAL")
        )
        img = _make_dummy_image("PNG")

        files = [
            ("files", ("navigation.csv", csv_content.encode("utf-8"), "text/csv")),
            ("files", ("scan_alpha.png", img, "image/png")),
        ]

        response = await client.post("/api/batches/analyze", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["total_scans"] == 1
        assert data["scans"][0]["sonar_latitude"] == pytest.approx(12.0)

    @pytest.mark.asyncio
    async def test_zero_detection_scans_preserved(
        self, client: AsyncClient, mock_detector: MagicMock
    ) -> None:
        """Zero-detection scans are preserved in the DB and returned in response."""
        mock_detector.detect.return_value = []
        csv_content = (
            _nav_header()
            + _nav_row("clear_sea.jpg", lat=10.0, lon=75.0, hdg=0.0, alt=20.0, ds="DEMO")
        )
        img = _make_dummy_image("JPEG")
        files = _build_files(csv_content, {"clear_sea.jpg": img})

        response = await client.post("/api/batches/analyze", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["total_scans"] == 1
        assert data["scans"][0]["detection_count"] == 0
        assert data["scans"][0]["detections"] == []

    @pytest.mark.asyncio
    async def test_missing_navigation_record_rejected(
        self, client: AsyncClient
    ) -> None:
        """Uploaded image lacks corresponding record in navigation.csv -> 422."""
        csv_content = (
            _nav_header()
            + _nav_row("scan_001.jpg")
        )
        img1 = _make_dummy_image("JPEG")
        img2 = _make_dummy_image("JPEG")

        files = _build_files(csv_content, {"scan_001.jpg": img1, "scan_unreferenced.jpg": img2})
        response = await client.post("/api/batches/analyze", files=files)
        assert response.status_code == 422
        detail = response.json()["detail"].lower()
        assert "lack corresponding navigation metadata" in detail
        assert "scan_unreferenced.jpg" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_navigation_record_referencing_missing_image_rejected(
        self, client: AsyncClient
    ) -> None:
        """navigation.csv references image that was not uploaded -> 422."""
        csv_content = (
            _nav_header()
            + _nav_row("scan_001.jpg")
            + _nav_row("scan_missing.jpg", lat=13.5, lon=77.5, hdg=10.0, alt=12.0)
        )
        img1 = _make_dummy_image("JPEG")

        files = _build_files(csv_content, {"scan_001.jpg": img1})
        response = await client.post("/api/batches/analyze", files=files)
        assert response.status_code == 422
        assert "references missing image file" in response.json()["detail"].lower()
        assert "scan_missing.jpg" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_missing_csv_column_rejected(
        self, client: AsyncClient
    ) -> None:
        """navigation.csv missing required column -> 422."""
        csv_content = (
            "scan_file,latitude,longitude,heading,timestamp,data_source\n"
            "scan_001.jpg,13.0,77.0,0.0,2026-10-04T11:00:00Z,SIMULATED\n"
        )
        img = _make_dummy_image("JPEG")
        files = _build_files(csv_content, {"scan_001.jpg": img})

        response = await client.post("/api/batches/analyze", files=files)
        assert response.status_code == 422
        assert "missing required column" in response.json()["detail"].lower()
        assert "altitude" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_invalid_latitude_rejected(
        self, client: AsyncClient
    ) -> None:
        """Latitude out of range [-90, 90] -> 422."""
        csv_content = _nav_header() + _nav_row("scan_001.jpg", lat=95.5)
        img = _make_dummy_image("JPEG")
        files = _build_files(csv_content, {"scan_001.jpg": img})

        response = await client.post("/api/batches/analyze", files=files)
        assert response.status_code == 422
        assert "invalid latitude" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_invalid_longitude_rejected(
        self, client: AsyncClient
    ) -> None:
        """Longitude out of range [-180, 180] -> 422."""
        csv_content = _nav_header() + _nav_row("scan_001.jpg", lon=195.0)
        img = _make_dummy_image("JPEG")
        files = _build_files(csv_content, {"scan_001.jpg": img})

        response = await client.post("/api/batches/analyze", files=files)
        assert response.status_code == 422
        assert "invalid longitude" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_invalid_heading_rejected(
        self, client: AsyncClient
    ) -> None:
        """Heading >= 360.0 -> 422."""
        csv_content = _nav_header() + _nav_row("scan_001.jpg", hdg=360.0)
        img = _make_dummy_image("JPEG")
        files = _build_files(csv_content, {"scan_001.jpg": img})

        response = await client.post("/api/batches/analyze", files=files)
        assert response.status_code == 422
        assert "invalid heading" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_invalid_altitude_rejected(
        self, client: AsyncClient
    ) -> None:
        """Altitude <= 0 -> 422."""
        csv_content = _nav_header() + _nav_row("scan_001.jpg", alt=0.0)
        img = _make_dummy_image("JPEG")
        files = _build_files(csv_content, {"scan_001.jpg": img})

        response = await client.post("/api/batches/analyze", files=files)
        assert response.status_code == 422
        assert "invalid altitude" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_malformed_timestamp_rejected(
        self, client: AsyncClient
    ) -> None:
        """Malformed timestamp string -> 422."""
        csv_content = _nav_header() + _nav_row("scan_001.jpg", ts="not-a-timestamp")
        img = _make_dummy_image("JPEG")
        files = _build_files(csv_content, {"scan_001.jpg": img})

        response = await client.post("/api/batches/analyze", files=files)
        assert response.status_code == 422
        assert "invalid timestamp" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_unsupported_image_extension_rejected(
        self, client: AsyncClient
    ) -> None:
        """File with unsupported extension (.bmp) -> 422."""
        csv_content = (
            "scan_file,latitude,longitude,heading,altitude,timestamp,data_source,"
            "detection_index,target_range_m,range_type,relative_bearing\n"
            "scan_001.bmp,13.0,77.0,0.0,10.0,2026-10-04T11:00:00Z,SIMULATED,,,,\n"
        )
        files = [
            ("scans", ("scan_001.bmp", b"dummy_bmp_bytes", "image/bmp")),
            ("navigation", ("navigation.csv", csv_content.encode("utf-8"), "text/csv")),
        ]

        response = await client.post("/api/batches/analyze", files=files)
        assert response.status_code == 422
        assert "unsupported" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_empty_batch_rejected(
        self, client: AsyncClient
    ) -> None:
        """Empty request with no files -> 422."""
        response = await client.post("/api/batches/analyze")
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_empty_csv_rows_rejected(
        self, client: AsyncClient
    ) -> None:
        """navigation.csv with headers but zero records -> 422."""
        csv_content = _nav_header()
        img = _make_dummy_image("JPEG")
        files = _build_files(csv_content, {"scan_001.jpg": img})

        response = await client.post("/api/batches/analyze", files=files)
        assert response.status_code == 422
        detail = response.json()["detail"].lower()
        assert "empty batch" in detail or "no scan records" in detail

    @pytest.mark.asyncio
    async def test_path_traversal_rejected(
        self, client: AsyncClient
    ) -> None:
        """Attempted path traversal in scan_file -> 422."""
        csv_content = _nav_header() + _nav_row("../scan_001.jpg")
        img = _make_dummy_image("JPEG")
        files = _build_files(csv_content, {"scan_001.jpg": img})

        response = await client.post("/api/batches/analyze", files=files)
        assert response.status_code == 422
        detail = response.json()["detail"].lower()
        assert "path traversal" in detail or "invalid scan_file" in detail

    @pytest.mark.asyncio
    async def test_invalid_data_source_rejected(
        self, client: AsyncClient
    ) -> None:
        """Unknown data_source value -> 422."""
        csv_content = _nav_header() + _nav_row("scan_001.jpg", ds="UNSUPPORTED_DATA_SOURCE")
        img = _make_dummy_image("JPEG")
        files = _build_files(csv_content, {"scan_001.jpg": img})

        response = await client.post("/api/batches/analyze", files=files)
        assert response.status_code == 422
        assert "invalid data_source" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# New Tests: Locked Multi-Row Design
# ---------------------------------------------------------------------------


class TestLockedNavigationCsvDesign:
    """Tests for the locked multi-row-per-scan navigation.csv design."""

    @pytest.mark.asyncio
    async def test_one_scan_one_detection_geometry(
        self, client: AsyncClient, mock_detector: MagicMock
    ) -> None:
        """One scan with one detection row -> target_geometries has 1 entry -> 200."""
        from app.detection.yolo_detector import Detection as YOLODetection
        # Mock detector returns 1 detection
        mock_det = MagicMock()
        mock_det.class_name = "mine"
        mock_det.confidence = 0.92
        mock_det.x1 = 10.0
        mock_det.y1 = 10.0
        mock_det.x2 = 50.0
        mock_det.y2 = 50.0
        mock_detector.detect.return_value = [mock_det]

        csv_content = (
            _nav_header()
            + _nav_row("scan_001.jpg", det_idx="0", range_m="35.0", range_type="SLANT", bearing="315.0")
        )
        img = _make_dummy_image("JPEG")
        files = _build_files(csv_content, {"scan_001.jpg": img})

        response = await client.post("/api/batches/analyze", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["total_scans"] == 1
        assert data["total_detections"] == 1
        assert data["scans"][0]["detection_count"] == 1

    @pytest.mark.asyncio
    async def test_one_scan_two_detection_geometry_rows(
        self, client: AsyncClient, mock_detector: MagicMock
    ) -> None:
        """One scan with two detection rows -> both TargetGeometryInput objects passed -> 200."""
        # Mock detector returns 2 detections
        def _make_det(cls: str):
            d = MagicMock()
            d.class_name = cls
            d.confidence = 0.85
            d.x1 = 5.0
            d.y1 = 5.0
            d.x2 = 25.0
            d.y2 = 25.0
            return d

        mock_detector.detect.return_value = [_make_det("mine"), _make_det("debris")]

        csv_content = (
            _nav_header()
            + _nav_row("scan_001.jpg", det_idx="0", range_m="35.0", range_type="SLANT", bearing="315.0")
            + _nav_row("scan_001.jpg", det_idx="1", range_m="42.0", range_type="SLANT", bearing="330.0")
        )
        img = _make_dummy_image("JPEG")
        files = _build_files(csv_content, {"scan_001.jpg": img})

        response = await client.post("/api/batches/analyze", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["total_scans"] == 1
        assert data["total_detections"] == 2
        assert data["scans"][0]["detection_count"] == 2

    @pytest.mark.asyncio
    async def test_two_scans_independent_geometries(
        self, client: AsyncClient, mock_detector: MagicMock
    ) -> None:
        """Two scans each with one detection row -> 2 ScanInputs each with 1 geometry -> 200."""
        mock_det = MagicMock()
        mock_det.class_name = "target"
        mock_det.confidence = 0.9
        mock_det.x1 = 5.0
        mock_det.y1 = 5.0
        mock_det.x2 = 20.0
        mock_det.y2 = 20.0
        mock_detector.detect.return_value = [mock_det]

        csv_content = (
            _nav_header()
            + _nav_row("scan_001.jpg", det_idx="0", range_m="30.0", range_type="SLANT", bearing="90.0")
            + _nav_row("scan_002.jpg", lat=13.342, lon=77.102, hdg=180.0, det_idx="0", range_m="50.0", range_type="GROUND", bearing="270.0")
        )
        img1 = _make_dummy_image("JPEG")
        img2 = _make_dummy_image("JPEG")
        files = _build_files(csv_content, {"scan_001.jpg": img1, "scan_002.jpg": img2})

        response = await client.post("/api/batches/analyze", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["total_scans"] == 2
        assert data["total_detections"] == 2

    @pytest.mark.asyncio
    async def test_zero_detection_scan_empty_detection_fields(
        self, client: AsyncClient, mock_detector: MagicMock
    ) -> None:
        """Zero-detection scan (all four detection fields empty) -> target_geometries=[] -> 200."""
        mock_detector.detect.return_value = []

        csv_content = (
            _nav_header()
            + _nav_row("clear_sea.jpg")  # all detection fields empty -> zero detection
        )
        img = _make_dummy_image("JPEG")
        files = _build_files(csv_content, {"clear_sea.jpg": img})

        response = await client.post("/api/batches/analyze", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["scans"][0]["detection_count"] == 0
        assert data["scans"][0]["detections"] == []

    @pytest.mark.asyncio
    async def test_invalid_range_type_rejected_strict(
        self, client: AsyncClient
    ) -> None:
        """Invalid range_type value -> 422 (never silently defaulted)."""
        csv_content = (
            _nav_header()
            + _nav_row("scan_001.jpg", det_idx="0", range_m="35.0", range_type="OBLIQUE", bearing="90.0")
        )
        img = _make_dummy_image("JPEG")
        files = _build_files(csv_content, {"scan_001.jpg": img})

        response = await client.post("/api/batches/analyze", files=files)
        assert response.status_code == 422
        detail = response.json()["detail"].lower()
        assert "invalid range_type" in detail or "oblique" in detail

    @pytest.mark.asyncio
    async def test_duplicate_detection_index_within_scan_rejected(
        self, client: AsyncClient
    ) -> None:
        """Duplicate detection_index within the same scan -> 422."""
        csv_content = (
            _nav_header()
            + _nav_row("scan_001.jpg", det_idx="0", range_m="35.0", range_type="SLANT", bearing="315.0")
            + _nav_row("scan_001.jpg", det_idx="0", range_m="42.0", range_type="SLANT", bearing="330.0")
        )
        img = _make_dummy_image("JPEG")
        files = _build_files(csv_content, {"scan_001.jpg": img})

        response = await client.post("/api/batches/analyze", files=files)
        assert response.status_code == 422
        detail = response.json()["detail"].lower()
        assert "duplicate detection_index" in detail or "duplicate" in detail

    @pytest.mark.asyncio
    async def test_partial_detection_geometry_rejected(
        self, client: AsyncClient
    ) -> None:
        """Only some detection-level fields populated -> 422 (partial geometry)."""
        csv_content = (
            _nav_header()
            # detection_index set, but range_m, range_type, relative_bearing missing
            + _nav_row("scan_001.jpg", det_idx="0", range_m="", range_type="", bearing="")
        )
        img = _make_dummy_image("JPEG")
        files = _build_files(csv_content, {"scan_001.jpg": img})

        response = await client.post("/api/batches/analyze", files=files)
        assert response.status_code == 422
        detail = response.json()["detail"].lower()
        assert "partial" in detail or "missing field" in detail

    @pytest.mark.asyncio
    async def test_inconsistent_scan_level_metadata_rejected(
        self, client: AsyncClient
    ) -> None:
        """Two rows with the same scan_file but different latitude -> 422."""
        csv_content = (
            _nav_header()
            # Second row has a DIFFERENT but still-valid latitude for the same scan_file
            + _nav_row("scan_001.jpg", lat=13.340, det_idx="0", range_m="35.0", range_type="SLANT", bearing="315.0")
            + _nav_row("scan_001.jpg", lat=14.999, det_idx="1", range_m="42.0", range_type="SLANT", bearing="330.0")
        )
        img = _make_dummy_image("JPEG")
        files = _build_files(csv_content, {"scan_001.jpg": img})

        response = await client.post("/api/batches/analyze", files=files)
        assert response.status_code == 422
        detail = response.json()["detail"].lower()
        assert "inconsistent" in detail or "differs" in detail

    @pytest.mark.asyncio
    async def test_mixed_zero_detection_and_detection_rows_rejected(
        self, client: AsyncClient
    ) -> None:
        """One row with empty detection fields + one with detection fields for same scan -> 422."""
        csv_content = (
            _nav_header()
            + _nav_row("scan_001.jpg")  # zero-detection marker
            + _nav_row("scan_001.jpg", det_idx="0", range_m="35.0", range_type="SLANT", bearing="315.0")
        )
        img = _make_dummy_image("JPEG")
        files = _build_files(csv_content, {"scan_001.jpg": img})

        response = await client.post("/api/batches/analyze", files=files)
        assert response.status_code == 422
        detail = response.json()["detail"].lower()
        assert "mix" in detail or "zero-detection" in detail

    @pytest.mark.asyncio
    async def test_target_geometries_passed_to_scan_input(
        self, client: AsyncClient, mock_detector: MagicMock
    ) -> None:
        """
        Verify that TargetGeometryInput objects are actually passed into
        ScanInput.target_geometries and reach the analysis pipeline.
        Inspects the captured ScanInput via a mock on run_batch_analysis.
        """
        mock_det = MagicMock()
        mock_det.class_name = "mine"
        mock_det.confidence = 0.88
        mock_det.x1 = 5.0
        mock_det.y1 = 5.0
        mock_det.x2 = 30.0
        mock_det.y2 = 30.0
        mock_detector.detect.return_value = [mock_det, mock_det]

        csv_content = (
            _nav_header()
            + _nav_row("scan_001.jpg", det_idx="0", range_m="35.0", range_type="SLANT", bearing="315.0")
            + _nav_row("scan_001.jpg", det_idx="1", range_m="42.0", range_type="GROUND", bearing="330.0")
        )
        img = _make_dummy_image("JPEG")
        files = _build_files(csv_content, {"scan_001.jpg": img})

        captured_inputs: list = []

        original_run = __import__(
            "app.api.analysis", fromlist=["run_batch_analysis"]
        ).run_batch_analysis

        def _capturing_run(scans, db, detector):
            captured_inputs.extend(scans)
            return original_run(scans, db, detector)

        with patch("app.services.batch_input.run_batch_analysis", side_effect=_capturing_run):
            response = await client.post("/api/batches/analyze", files=files)

        assert response.status_code == 200
        assert len(captured_inputs) == 1

        scan_in = captured_inputs[0]
        geoms = scan_in.target_geometries
        assert len(geoms) == 2

        # Verify TargetGeometryInput values
        geom_by_idx = {g.detection_index: g for g in geoms}
        assert 0 in geom_by_idx
        assert 1 in geom_by_idx
        assert geom_by_idx[0].range == pytest.approx(35.0)
        assert geom_by_idx[0].range_type == "SLANT"
        assert geom_by_idx[0].relative_bearing == pytest.approx(315.0)
        assert geom_by_idx[1].range == pytest.approx(42.0)
        assert geom_by_idx[1].range_type == "GROUND"
        assert geom_by_idx[1].relative_bearing == pytest.approx(330.0)


# ---------------------------------------------------------------------------
# Batch Image Persistence Tests
# ---------------------------------------------------------------------------


class TestBatchImagePersistence:
    """Test suite proving persistent storage of batch-uploaded scan images."""

    @pytest.fixture(autouse=True)
    def setup_isolated_storage(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        storage = tmp_path / "data" / "scans"
        storage.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(settings, "scans_dir", storage)
        return storage

    def test_default_storage_dir_configuration(self) -> None:
        """Verify default scans_dir points to project data/scans directory."""
        from app.config import BASE_DIR, Settings

        fresh_settings = Settings()
        assert fresh_settings.data_dir == BASE_DIR / "data"
        assert fresh_settings.scans_dir == BASE_DIR / "data" / "scans"

    @pytest.mark.asyncio
    async def test_uploaded_batch_images_are_persisted(
        self, client: AsyncClient, mock_detector: MagicMock, setup_isolated_storage: Path
    ) -> None:
        """Uploaded batch images are copied to persistent storage and exist on disk."""
        mock_detector.detect.return_value = []
        csv_content = (
            _nav_header()
            + _nav_row("scan_001.jpg", lat=13.34, lon=77.10)
            + _nav_row("scan_002.jpg", lat=13.35, lon=77.11)
        )
        img1_bytes = _make_dummy_image("JPEG")
        img2_bytes = _make_dummy_image("JPEG")

        files = _build_files(
            csv_content, {"scan_001.jpg": img1_bytes, "scan_002.jpg": img2_bytes}
        )
        response = await client.post("/api/batches/analyze", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["total_scans"] == 2

        for scan in data["scans"]:
            img_path = scan["image_path"]
            assert os.path.isfile(img_path), f"Persisted image not found on disk: {img_path}"
            assert Path(img_path).resolve().parent == setup_isolated_storage.resolve()
            with open(img_path, "rb") as f:
                content = f.read()
            assert len(content) > 0

    @pytest.mark.asyncio
    async def test_stored_scan_input_image_path_points_to_existing_persistent_file(
        self, client: AsyncClient, mock_detector: MagicMock, setup_isolated_storage: Path
    ) -> None:
        """ScanInput constructed in process_investigation_batch uses persistent file paths."""
        mock_detector.detect.return_value = []
        csv_content = _nav_header() + _nav_row("scan_persistent.jpg", lat=13.34, lon=77.10)
        img_bytes = _make_dummy_image("JPEG")
        files = _build_files(csv_content, {"scan_persistent.jpg": img_bytes})

        captured_inputs: list = []
        original_run = __import__(
            "app.api.analysis", fromlist=["run_batch_analysis"]
        ).run_batch_analysis

        def _capturing_run(scans, db, detector):
            captured_inputs.extend(scans)
            return original_run(scans, db, detector)

        with patch("app.services.batch_input.run_batch_analysis", side_effect=_capturing_run):
            response = await client.post("/api/batches/analyze", files=files)

        assert response.status_code == 200
        assert len(captured_inputs) == 1
        scan_in = captured_inputs[0]

        # ScanInput image_path must exist on disk in persistent storage
        assert os.path.isfile(scan_in.image_path)
        assert "investigation_batch_" not in scan_in.image_path
        assert Path(scan_in.image_path).resolve().parent == setup_isolated_storage.resolve()

    @pytest.mark.asyncio
    async def test_temporary_directory_deleted_without_breaking_scan(
        self, client: AsyncClient, mock_detector: MagicMock
    ) -> None:
        """Temporary staging directory is cleaned up, but the persistent scan image remains intact."""
        mock_detector.detect.return_value = []
        csv_content = _nav_header() + _nav_row("scan_clean_temp.jpg", lat=13.34, lon=77.10)
        img_bytes = _make_dummy_image("JPEG")
        files = _build_files(csv_content, {"scan_clean_temp.jpg": img_bytes})

        response = await client.post("/api/batches/analyze", files=files)
        assert response.status_code == 200
        data = response.json()
        scan = data["scans"][0]
        img_path = scan["image_path"]

        # The image path is persistent and unaffected by tempdir cleanup
        assert os.path.isfile(img_path)
        with open(img_path, "rb") as f:
            persisted_bytes = f.read()
        assert persisted_bytes == img_bytes

        # GET /api/scans/{id}/image also works
        img_res = await client.get(f"/api/scans/{scan['id']}/image")
        assert img_res.status_code == 200
        assert img_res.content == img_bytes

    @pytest.mark.asyncio
    async def test_get_scan_image_works_after_batch_ingestion(
        self, client: AsyncClient, mock_detector: MagicMock
    ) -> None:
        """GET /api/scans/{id}/image serves the uploaded image for batch-created scans."""
        mock_detector.detect.return_value = []
        csv_content = _nav_header() + _nav_row("batch_get_test.jpg", lat=13.34, lon=77.10)
        img_bytes = _make_dummy_image("JPEG")
        files = _build_files(csv_content, {"batch_get_test.jpg": img_bytes})

        response = await client.post("/api/batches/analyze", files=files)
        assert response.status_code == 200
        scan_id = response.json()["scans"][0]["id"]

        image_response = await client.get(f"/api/scans/{scan_id}/image")
        assert image_response.status_code == 200
        assert image_response.content == img_bytes

    @pytest.mark.asyncio
    async def test_zero_detection_scans_still_retain_their_image(
        self, client: AsyncClient, mock_detector: MagicMock
    ) -> None:
        """Zero-detection batch scans still retain their image file and GET endpoint serves it."""
        mock_detector.detect.return_value = []
        csv_content = _nav_header() + _nav_row("zero_det_scan.png", lat=12.5, lon=78.2)
        img_bytes = _make_dummy_image("PNG")
        files = _build_files(csv_content, {"zero_det_scan.png": img_bytes})

        response = await client.post("/api/batches/analyze", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["total_detections"] == 0
        scan = data["scans"][0]
        assert scan["detection_count"] == 0
        assert os.path.isfile(scan["image_path"])

        image_response = await client.get(f"/api/scans/{scan['id']}/image")
        assert image_response.status_code == 200
        assert image_response.content == img_bytes

    @pytest.mark.asyncio
    async def test_multiple_batch_scans_retain_separate_images(
        self, client: AsyncClient, mock_detector: MagicMock
    ) -> None:
        """
        Prove:
        1. Multiple scans in a single batch retain separate files and images.
        2. Consecutive batches uploading the same filename avoid collision and retain separate files.
        """
        mock_detector.detect.return_value = []
        img_red = Image.new("RGB", (32, 32), color=(255, 0, 0))
        buf_red = io.BytesIO()
        img_red.save(buf_red, format="JPEG")
        bytes_red = buf_red.getvalue()

        img_blue = Image.new("RGB", (32, 32), color=(0, 0, 255))
        buf_blue = io.BytesIO()
        img_blue.save(buf_blue, format="JPEG")
        bytes_blue = buf_blue.getvalue()

        # Part 1: Single batch with multiple scans
        csv_single = (
            _nav_header()
            + _nav_row("scan_a.jpg", lat=10.0, lon=70.0)
            + _nav_row("scan_b.jpg", lat=10.1, lon=70.1)
        )
        files_single = _build_files(
            csv_single, {"scan_a.jpg": bytes_red, "scan_b.jpg": bytes_blue}
        )
        res_single = await client.post("/api/batches/analyze", files=files_single)
        assert res_single.status_code == 200
        scans_single = res_single.json()["scans"]
        assert len(scans_single) == 2
        id_a, path_a = scans_single[0]["id"], scans_single[0]["image_path"]
        id_b, path_b = scans_single[1]["id"], scans_single[1]["image_path"]
        assert path_a != path_b
        assert os.path.isfile(path_a)
        assert os.path.isfile(path_b)
        get_a = await client.get(f"/api/scans/{id_a}/image")
        get_b = await client.get(f"/api/scans/{id_b}/image")
        assert get_a.status_code == 200
        assert get_b.status_code == 200
        assert get_a.content == bytes_red
        assert get_b.content == bytes_blue

        # Part 2: Consecutive batches uploading identical filename (scan_001.jpg)
        header_with_id = (
            "scan_file,latitude,longitude,heading,altitude,timestamp,data_source,"
            "scan_identity,detection_index,target_range_m,range_type,relative_bearing\n"
        )
        csv1 = header_with_id + "scan_001.jpg,10.0,70.0,90.0,10.0,2026-10-04T11:00:00Z,SIMULATED,mission_batch_1,,,,\n"
        files1 = _build_files(csv1, {"scan_001.jpg": bytes_red})
        res1 = await client.post("/api/batches/analyze", files=files1)
        assert res1.status_code == 200
        scan1_id = res1.json()["scans"][0]["id"]
        scan1_path = res1.json()["scans"][0]["image_path"]

        csv2 = header_with_id + "scan_001.jpg,11.0,71.0,90.0,10.0,2026-10-04T11:05:00Z,SIMULATED,mission_batch_2,,,,\n"
        files2 = _build_files(csv2, {"scan_001.jpg": bytes_blue})
        res2 = await client.post("/api/batches/analyze", files=files2)
        assert res2.status_code == 200
        scan2_id = res2.json()["scans"][0]["id"]
        scan2_path = res2.json()["scans"][0]["image_path"]

        # Collision avoidance: paths must be distinct and both files must exist
        assert scan1_id != scan2_id
        assert scan1_path != scan2_path
        assert os.path.isfile(scan1_path)
        assert os.path.isfile(scan2_path)

        # GET images must return the respective separate content
        get_res1 = await client.get(f"/api/scans/{scan1_id}/image")
        get_res2 = await client.get(f"/api/scans/{scan2_id}/image")
        assert get_res1.status_code == 200
        assert get_res2.status_code == 200
        assert get_res1.content == bytes_red
        assert get_res2.content == bytes_blue
        assert get_res1.content != get_res2.content

    def test_persist_batch_scan_image_path_traversal_prevention(self, tmp_path: Path) -> None:
        """persist_batch_scan_image blocks directory traversal attacks."""
        from app.services.batch_input import persist_batch_scan_image
        from fastapi import HTTPException

        src_file = tmp_path / "src.jpg"
        src_file.write_bytes(b"data")
        storage_dir = tmp_path / "storage"
        storage_dir.mkdir()

        with pytest.raises(HTTPException) as exc_info:
            persist_batch_scan_image(src_file, "../evil.jpg", storage_dir)
        assert exc_info.value.status_code == 422


# ---------------------------------------------------------------------------
# Investigation Batch Results and Reports Tests
# ---------------------------------------------------------------------------


class TestInvestigationBatchResultsAndReports:
    """Test suite verifying batch persistence, GET /api/batches/{batch_id}, and reports."""

    @pytest.mark.asyncio
    async def test_batch_id_generated_and_assigned_to_all_scans(
        self, client: AsyncClient, mock_detector: MagicMock
    ) -> None:
        """Batch upload generates a batch_id and associates all scans with it."""
        mock_detector.detect.return_value = []
        csv_content = (
            _nav_header()
            + _nav_row("scan_a.jpg", lat=12.5, lon=75.5)
            + _nav_row("scan_b.jpg", lat=12.6, lon=75.6)
        )
        files = _build_files(
            csv_content,
            {"scan_a.jpg": _make_dummy_image(), "scan_b.jpg": _make_dummy_image()},
        )
        response = await client.post("/api/batches/analyze", files=files)
        assert response.status_code == 200
        data = response.json()

        batch_id = data.get("batch_id")
        assert batch_id is not None
        assert batch_id.startswith("batch_")
        assert len(data["scans"]) == 2

        for scan in data["scans"]:
            assert scan.get("batch_id") == batch_id

    @pytest.mark.asyncio
    async def test_get_batch_results_including_zero_detection_scans(
        self, client: AsyncClient, mock_detector: MagicMock
    ) -> None:
        """GET /api/batches/{batch_id} returns all scans, detections, and class statistics."""
        from app.detection.yolo_detector import Detection as YOLODetection

        def _mock_detect(path: str):
            if "scan_det" in os.path.basename(path):
                return [
                    YOLODetection(
                        class_id=0,
                        class_name="pipeline",
                        confidence=0.92,
                        x1=10.0,
                        y1=20.0,
                        x2=50.0,
                        y2=60.0,
                    ),
                    YOLODetection(
                        class_id=2,
                        class_name="debris",
                        confidence=0.88,
                        x1=70.0,
                        y1=80.0,
                        x2=110.0,
                        y2=120.0,
                    ),
                ]
            return []

        mock_detector.detect.side_effect = _mock_detect

        # 2 scans: scan_det has 2 detections, scan_zero has 0 detections
        csv_content = (
            _nav_header()
            + _nav_row(
                "scan_det.jpg",
                lat=13.0,
                lon=80.0,
                det_idx="0",
                range_m="30.0",
                range_type="SLANT",
                bearing="45.0",
            )
            + _nav_row(
                "scan_det.jpg",
                lat=13.0,
                lon=80.0,
                det_idx="1",
                range_m="40.0",
                range_type="GROUND",
                bearing="90.0",
            )
            + _nav_row("scan_zero.jpg", lat=13.1, lon=80.1)
        )
        files = _build_files(
            csv_content,
            {"scan_det.jpg": _make_dummy_image(), "scan_zero.jpg": _make_dummy_image()},
        )
        upload_res = await client.post("/api/batches/analyze", files=files)
        assert upload_res.status_code == 200, upload_res.json()
        batch_id = upload_res.json()["batch_id"]

        # Fetch batch by ID
        get_res = await client.get(f"/api/batches/{batch_id}")
        assert get_res.status_code == 200
        batch_data = get_res.json()

        assert batch_data["batch_id"] == batch_id
        assert batch_data["total_scans"] == 2
        assert batch_data["successful_scans"] == 2
        assert batch_data["total_detections"] == 2
        assert batch_data["class_counts"] == {"pipeline": 1, "debris": 1}
        assert batch_data["data_source"].upper() == "SIMULATED"

        # Verify scans inside batch
        scans = batch_data["scans"]
        assert len(scans) == 2

        # Verify scan with detections
        det_scan = next(s for s in scans if s["scan_identity"] == "scan_det")
        assert det_scan["detection_count"] == 2
        assert len(det_scan["detections"]) == 2
        for d in det_scan["detections"]:
            assert d["target_latitude"] is not None
            assert d["target_longitude"] is not None

        # Verify zero-detection scan is preserved
        zero_scan = next(s for s in scans if s["scan_identity"] == "scan_zero")
        assert zero_scan["detection_count"] == 0
        assert len(zero_scan["detections"]) == 0

    @pytest.mark.asyncio
    async def test_get_batch_404_for_unknown_id(self, client: AsyncClient) -> None:
        """GET /api/batches/{batch_id} returns 404 for unknown batch ID."""
        res = await client.get("/api/batches/batch_nonexistent_000000")
        assert res.status_code == 404
        assert "not found" in res.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_batch_json_report(
        self, client: AsyncClient, mock_detector: MagicMock
    ) -> None:
        """GET /api/batches/{batch_id}/report/json returns valid batch report JSON."""
        from app.detection.yolo_detector import Detection as YOLODetection

        mock_detector.detect.return_value = [
            YOLODetection(
                class_id=1,
                class_name="wreck",
                confidence=0.95,
                x1=5.0,
                y1=5.0,
                x2=25.0,
                y2=25.0,
            )
        ]
        csv_content = _nav_header() + _nav_row(
            "scan_wreck.jpg",
            lat=15.0,
            lon=73.0,
            det_idx="0",
            range_m="50.0",
            range_type="SLANT",
            bearing="30.0",
        )
        files = _build_files(csv_content, {"scan_wreck.jpg": _make_dummy_image()})
        upload_res = await client.post("/api/batches/analyze", files=files)
        batch_id = upload_res.json()["batch_id"]

        report_res = await client.get(f"/api/batches/{batch_id}/report/json")
        assert report_res.status_code == 200
        report = report_res.json()

        assert report["batch_id"] == batch_id
        assert report["report_metadata"]["report_title"] == "DRISHTI-SSS Sonar Investigation Report"
        assert "provenance" in report
        assert "Deterministic geometry" in report["provenance"]["geolocation_data"]
        assert "NOT predicted by AI" in report["provenance"]["geolocation_data"]

        stats = report["survey_statistics"]
        assert stats["total_scans"] == 1
        assert stats["successful_scans"] == 1
        assert stats["total_detections"] == 1
        assert stats["mapped_detections"] == 1
        assert stats["class_counts"] == {"wreck": 1}

        assert len(report["scans"]) == 1
        assert len(report["all_detections"]) == 1
        det = report["all_detections"][0]
        assert det["class_name"] == "wreck"
        assert det["target_latitude"] is not None
        assert det["target_longitude"] is not None
        assert det["batch_id"] == batch_id

    @pytest.mark.asyncio
    async def test_batch_geojson_report(
        self, client: AsyncClient, mock_detector: MagicMock
    ) -> None:
        """GET /api/batches/{batch_id}/report/geojson returns RFC 7946 FeatureCollection."""
        from app.detection.yolo_detector import Detection as YOLODetection

        mock_detector.detect.return_value = [
            YOLODetection(
                class_id=0,
                class_name="pipeline",
                confidence=0.87,
                x1=10.0,
                y1=10.0,
                x2=30.0,
                y2=30.0,
            )
        ]
        csv_content = _nav_header() + _nav_row(
            "scan_pipe.jpg",
            lat=18.0,
            lon=72.0,
            det_idx="0",
            range_m="45.0",
            range_type="GROUND",
            bearing="120.0",
        )
        files = _build_files(csv_content, {"scan_pipe.jpg": _make_dummy_image()})
        upload_res = await client.post("/api/batches/analyze", files=files)
        batch_id = upload_res.json()["batch_id"]

        geojson_res = await client.get(f"/api/batches/{batch_id}/report/geojson")
        assert geojson_res.status_code == 200
        geojson = geojson_res.json()

        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) == 1

        feature = geojson["features"][0]
        assert feature["type"] == "Feature"
        assert feature["geometry"]["type"] == "Point"

        # Coordinates must be [longitude, latitude]
        coords = feature["geometry"]["coordinates"]
        assert len(coords) == 2
        lon, lat = coords
        assert pytest.approx(lon, abs=0.1) == 72.0
        assert pytest.approx(lat, abs=0.1) == 18.0

        # Properties
        props = feature["properties"]
        assert props["batch_id"] == batch_id
        assert props["class_name"] == "pipeline"
        assert props["confidence"] == pytest.approx(0.87)
        assert props["detection_source"] == "AI_YOLO"
        assert props["geolocation_method"] == "DETERMINISTIC_GEOMETRY"

    @pytest.mark.asyncio
    async def test_batch_pdf_report(
        self, client: AsyncClient, mock_detector: MagicMock
    ) -> None:
        """GET /api/batches/{batch_id}/report/pdf returns valid PDF binary document."""
        mock_detector.detect.return_value = []
        csv_content = (
            _nav_header()
            + _nav_row("scan_pdf1.jpg", lat=14.0, lon=74.0)
            + _nav_row("scan_pdf2.jpg", lat=14.1, lon=74.1)
        )
        files = _build_files(
            csv_content,
            {"scan_pdf1.jpg": _make_dummy_image(), "scan_pdf2.jpg": _make_dummy_image()},
        )
        upload_res = await client.post("/api/batches/analyze", files=files)
        batch_id = upload_res.json()["batch_id"]

        pdf_res = await client.get(f"/api/batches/{batch_id}/report/pdf")
        assert pdf_res.status_code == 200
        assert pdf_res.headers["content-type"] == "application/pdf"
        assert f"drishti_sss_batch_{batch_id}_report.pdf" in pdf_res.headers["content-disposition"]
        assert pdf_res.content.startswith(b"%PDF")
        assert len(pdf_res.content) > 1000

    @pytest.mark.asyncio
    async def test_batch_reports_404_for_unknown_id(self, client: AsyncClient) -> None:
        """Report endpoints return 404 for unknown batch ID."""
        for endpoint in ["report/json", "report/geojson", "report/pdf"]:
            res = await client.get(f"/api/batches/batch_nonexistent_123/{endpoint}")
            assert res.status_code == 404
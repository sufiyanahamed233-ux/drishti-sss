"""
test_analysis.py
----------------
Unit and integration tests for the Phase 3C batch scan analysis pipeline
with per-detection sonar geometry.

Coverage
--------
- Single scan with zero detections (persists scan, empty detections)
- Single scan with one detection and its own geometry
- Multiple detections with different ranges/bearings producing distinct georeferencing results
- Validation: Missing geometry for a detection returns HTTP 422
- Validation: Duplicate detection_index is rejected with HTTP 422
- Validation: Out-of-range detection_index is rejected with HTTP 422
- Validation: Zero detections with non-empty target_geometries rejected with HTTP 422
- Deterministic georeferencing integration (lat/lon, bearings, ground range)
- SLANT range vs GROUND range handling
- Batch analysis with multiple scans processed independently
- Duplicate scan prevention (replaces scan with same identity)
- Data source provenance flags (REAL, DEMO, SIMULATED)
"""

from __future__ import annotations

import math
from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import Session, sessionmaker

from app.api.analysis import run_batch_analysis, run_scan_analysis
from app.db.models import Base, DataSource, Detection as DB_Detection, RangeType, Scan as DB_Scan
from app.detection.yolo_detector import Detection, YOLODetector
from app.georeferencing.engine import georeference
from app.schemas.report import ScanInput, TargetGeometryInput


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """Provide an isolated, in-memory SQLite database session."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionTest = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = SessionTest()
    try:
        yield session
    finally:
        session.close()


def _make_sample_input(
    scan_identity: str = "test-scan-001",
    image_path: str = "/data/sonar_001.png",
    range_val: float | None = 50.0,
    range_type: str = "SLANT",
    altitude: float = 10.0,
    data_source: str = "REAL",
    target_geometries: list[TargetGeometryInput] | None = None,
) -> ScanInput:
    return ScanInput(
        scan_identity=scan_identity,
        image_path=image_path,
        sonar_latitude=13.0827,
        sonar_longitude=80.2707,
        heading=45.0,
        altitude=altitude,
        range=range_val,
        range_type=range_type,
        relative_bearing=15.0,
        data_source=data_source,
        target_geometries=target_geometries or [],
    )


# ---------------------------------------------------------------------------
# Tests: Zero Detections
# ---------------------------------------------------------------------------


class TestZeroDetections:
    """Verifies that scans with 0 detections are preserved and stored correctly."""

    def test_zero_detections_persists_scan(self, db_session: Session) -> None:
        mock_detector = MagicMock(spec=YOLODetector)
        mock_detector.detect.return_value = []

        scan_in = _make_sample_input(
            scan_identity="scan-zero-01",
            target_geometries=[],
        )
        result = run_scan_analysis(scan_in, db_session, mock_detector)

        assert result.scan_identity == "scan-zero-01"
        assert result.detection_count == 0
        assert len(result.detections) == 0

        # Verify database state
        db_scan = db_session.query(DB_Scan).filter(DB_Scan.scan_identity == "scan-zero-01").first()
        assert db_scan is not None
        assert db_scan.sonar_latitude == pytest.approx(13.0827)
        assert len(db_scan.detections) == 0

    def test_zero_detections_with_unexpected_geometries_rejected(self, db_session: Session) -> None:
        mock_detector = MagicMock(spec=YOLODetector)
        mock_detector.detect.return_value = []

        scan_in = _make_sample_input(
            scan_identity="scan-zero-err",
            target_geometries=[
                TargetGeometryInput(detection_index=0, range=50.0, range_type="SLANT", relative_bearing=10.0)
            ],
        )
        with pytest.raises(HTTPException) as exc_info:
            run_scan_analysis(scan_in, db_session, mock_detector)
        assert exc_info.value.status_code == 422


# ---------------------------------------------------------------------------
# Tests: Per-Detection Geometry & Multiple Detections (Phase 3C)
# ---------------------------------------------------------------------------


class TestPerDetectionGeometry:
    """Verifies per-detection geometry inputs, validation, and distinct calculations."""

    def test_single_detection_with_own_geometry(self, db_session: Session) -> None:
        mock_detector = MagicMock(spec=YOLODetector)
        mock_detector.detect.return_value = [
            Detection(class_id=0, class_name="submarine_pipeline", confidence=0.88, x1=5, y1=5, x2=50, y2=50)
        ]

        scan_in = _make_sample_input(
            scan_identity="scan-single-01",
            altitude=10.0,
            target_geometries=[
                TargetGeometryInput(detection_index=0, range=60.0, range_type="SLANT", relative_bearing=25.0)
            ],
        )
        result = run_scan_analysis(scan_in, db_session, mock_detector)

        assert result.detection_count == 1
        det = result.detections[0]
        assert det.class_name == "submarine_pipeline"
        assert det.range_m == pytest.approx(60.0)
        assert det.relative_bearing == pytest.approx(25.0)
        # Ground range = sqrt(60^2 - 10^2) = sqrt(3500) ≈ 59.16
        assert det.ground_range_m == pytest.approx(math.sqrt(60.0**2 - 10.0**2), abs=1e-3)

    def test_multiple_detections_produce_different_georeferencing(self, db_session: Session) -> None:
        """Each detection uses its own geometry and computes distinct lat/lon."""
        mock_detector = MagicMock(spec=YOLODetector)
        mock_detector.detect.return_value = [
            Detection(class_id=1, class_name="shipwreck", confidence=0.92, x1=10, y1=20, x2=80, y2=90),
            Detection(class_id=3, class_name="mine_cylinder", confidence=0.85, x1=150, y1=160, x2=200, y2=210),
        ]

        scan_in = _make_sample_input(
            scan_identity="scan-multi-diff",
            altitude=10.0,
            target_geometries=[
                TargetGeometryInput(detection_index=0, range=40.0, range_type="SLANT", relative_bearing=-30.0),
                TargetGeometryInput(detection_index=1, range=100.0, range_type="GROUND", relative_bearing=60.0),
            ],
        )
        result = run_scan_analysis(scan_in, db_session, mock_detector)

        assert result.detection_count == 2
        det0 = result.detections[0]
        det1 = result.detections[1]

        # Geometry values must be specific to each detection
        assert det0.range_m == pytest.approx(40.0)
        assert det0.relative_bearing == pytest.approx(-30.0)
        assert det0.ground_range_m == pytest.approx(math.sqrt(40.0**2 - 10.0**2), abs=1e-3)

        assert det1.range_m == pytest.approx(100.0)
        assert det1.relative_bearing == pytest.approx(60.0)
        assert det1.ground_range_m == pytest.approx(100.0)  # GROUND range

        # Latitudes and longitudes must be distinct
        assert det0.target_latitude != det1.target_latitude
        assert det0.target_longitude != det1.target_longitude
        assert det0.absolute_bearing != det1.absolute_bearing

        # Verify DB persistence
        db_scan = db_session.query(DB_Scan).filter(DB_Scan.scan_identity == "scan-multi-diff").first()
        assert db_scan is not None
        assert len(db_scan.detections) == 2
        assert db_scan.detections[0].range_m == pytest.approx(40.0)
        assert db_scan.detections[1].range_m == pytest.approx(100.0)

    def test_missing_geometry_for_detection_rejected(self, db_session: Session) -> None:
        """Scan has 2 detections but only geometry for index 0 is provided."""
        mock_detector = MagicMock(spec=YOLODetector)
        mock_detector.detect.return_value = [
            Detection(class_id=1, class_name="shipwreck", confidence=0.9, x1=0, y1=0, x2=20, y2=20),
            Detection(class_id=2, class_name="ghost_net", confidence=0.8, x1=30, y1=30, x2=50, y2=50),
        ]

        scan_in = _make_sample_input(
            scan_identity="missing-geom",
            target_geometries=[
                TargetGeometryInput(detection_index=0, range=50.0, range_type="SLANT", relative_bearing=0.0)
            ],
        )
        with pytest.raises(HTTPException) as exc_info:
            run_scan_analysis(scan_in, db_session, mock_detector)
        assert exc_info.value.status_code == 422
        assert "missing" in exc_info.value.detail.lower()

    def test_no_geometry_when_detections_exist_rejected(self, db_session: Session) -> None:
        """Scan has 1 detection but target_geometries is completely empty."""
        mock_detector = MagicMock(spec=YOLODetector)
        mock_detector.detect.return_value = [
            Detection(class_id=1, class_name="shipwreck", confidence=0.9, x1=0, y1=0, x2=20, y2=20)
        ]

        scan_in = _make_sample_input(
            scan_identity="no-geom",
            target_geometries=[],
        )
        with pytest.raises(HTTPException) as exc_info:
            run_scan_analysis(scan_in, db_session, mock_detector)
        assert exc_info.value.status_code == 422

    def test_duplicate_detection_index_rejected(self, db_session: Session) -> None:
        """Two geometries with the same detection_index=0."""
        mock_detector = MagicMock(spec=YOLODetector)
        mock_detector.detect.return_value = [
            Detection(class_id=1, class_name="shipwreck", confidence=0.9, x1=0, y1=0, x2=20, y2=20),
            Detection(class_id=2, class_name="ghost_net", confidence=0.8, x1=30, y1=30, x2=50, y2=50),
        ]

        scan_in = _make_sample_input(
            scan_identity="dup-index",
            target_geometries=[
                TargetGeometryInput(detection_index=0, range=40.0, range_type="SLANT", relative_bearing=0.0),
                TargetGeometryInput(detection_index=0, range=50.0, range_type="SLANT", relative_bearing=10.0),
            ],
        )
        with pytest.raises(HTTPException) as exc_info:
            run_scan_analysis(scan_in, db_session, mock_detector)
        assert exc_info.value.status_code == 422
        assert "duplicate" in exc_info.value.detail.lower()

    def test_out_of_range_detection_index_rejected(self, db_session: Session) -> None:
        """Scan has 1 detection, but target_geometry has detection_index=5."""
        mock_detector = MagicMock(spec=YOLODetector)
        mock_detector.detect.return_value = [
            Detection(class_id=1, class_name="shipwreck", confidence=0.9, x1=0, y1=0, x2=20, y2=20)
        ]

        scan_in = _make_sample_input(
            scan_identity="out-of-range",
            target_geometries=[
                TargetGeometryInput(detection_index=5, range=50.0, range_type="SLANT", relative_bearing=0.0)
            ],
        )
        with pytest.raises(HTTPException) as exc_info:
            run_scan_analysis(scan_in, db_session, mock_detector)
        assert exc_info.value.status_code == 422
        assert "outside the valid range" in exc_info.value.detail.lower()


# ---------------------------------------------------------------------------
# Tests: Georeferencing Integration
# ---------------------------------------------------------------------------


class TestGeoreferencingIntegration:
    """Verifies deterministic math in the pipeline matches the georeferencing engine."""

    def test_pipeline_georeferencing_matches_engine(self, db_session: Session) -> None:
        mock_detector = MagicMock(spec=YOLODetector)
        mock_detector.detect.return_value = [
            Detection(class_id=0, class_name="submarine_pipeline", confidence=0.88, x1=5, y1=5, x2=50, y2=50)
        ]

        scan_in = _make_sample_input(
            scan_identity="geo-test-01",
            altitude=10.0,
            target_geometries=[
                TargetGeometryInput(detection_index=0, range=50.0, range_type="SLANT", relative_bearing=15.0)
            ],
        )
        result = run_scan_analysis(scan_in, db_session, mock_detector)
        det_result = result.detections[0]

        expected_geo = georeference(
            sonar_lat=13.0827,
            sonar_lon=80.2707,
            sonar_heading=45.0,
            sonar_altitude=10.0,
            target_slant_range=50.0,
            target_relative_bearing=15.0,
            range_type="SLANT",
        )

        assert det_result.target_latitude == pytest.approx(expected_geo.latitude, abs=1e-6)
        assert det_result.target_longitude == pytest.approx(expected_geo.longitude, abs=1e-6)
        assert det_result.absolute_bearing == pytest.approx(expected_geo.absolute_bearing_deg, abs=1e-3)
        assert det_result.ground_range_m == pytest.approx(expected_geo.ground_range_m, abs=1e-3)

    def test_slant_vs_ground_range_math(self, db_session: Session) -> None:
        mock_detector = MagicMock(spec=YOLODetector)
        mock_detector.detect.return_value = [
            Detection(class_id=0, class_name="submarine_pipeline", confidence=0.8, x1=0, y1=0, x2=10, y2=10)
        ]

        # 1. Slant range: 50m slant, 30m altitude -> ground range = sqrt(50^2 - 30^2) = 40m
        scan_slant = _make_sample_input(
            scan_identity="slant-calc",
            altitude=30.0,
            target_geometries=[
                TargetGeometryInput(detection_index=0, range=50.0, range_type="SLANT", relative_bearing=0.0)
            ],
        )
        res_slant = run_scan_analysis(scan_slant, db_session, mock_detector)
        assert res_slant.detections[0].ground_range_m == pytest.approx(40.0, abs=1e-4)

        # 2. Ground range: 50m ground range directly
        scan_ground = _make_sample_input(
            scan_identity="ground-calc",
            altitude=30.0,
            target_geometries=[
                TargetGeometryInput(detection_index=0, range=50.0, range_type="GROUND", relative_bearing=0.0)
            ],
        )
        res_ground = run_scan_analysis(scan_ground, db_session, mock_detector)
        assert res_ground.detections[0].ground_range_m == pytest.approx(50.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Tests: Batch Analysis
# ---------------------------------------------------------------------------


class TestBatchAnalysis:
    """Verifies batch execution with multiple scans processed independently."""

    def test_batch_processes_all_scans_independently(self, db_session: Session) -> None:
        mock_detector = MagicMock(spec=YOLODetector)

        def side_effect(img: str):
            if "scan1" in str(img):
                return []  # zero detections
            elif "scan2" in str(img):
                return [Detection(class_id=1, class_name="shipwreck", confidence=0.9, x1=0, y1=0, x2=20, y2=20)]
            else:
                return [
                    Detection(class_id=2, class_name="ghost_net", confidence=0.75, x1=10, y1=10, x2=30, y2=30),
                    Detection(class_id=3, class_name="mine_cylinder", confidence=0.82, x1=40, y1=40, x2=60, y2=60),
                ]

        mock_detector.detect.side_effect = side_effect

        batch = [
            _make_sample_input(scan_identity="scan1", image_path="/data/scan1.png", target_geometries=[]),
            _make_sample_input(
                scan_identity="scan2",
                image_path="/data/scan2.png",
                target_geometries=[
                    TargetGeometryInput(detection_index=0, range=50.0, range_type="SLANT", relative_bearing=10.0)
                ],
            ),
            _make_sample_input(
                scan_identity="scan3",
                image_path="/data/scan3.png",
                target_geometries=[
                    TargetGeometryInput(detection_index=0, range=40.0, range_type="SLANT", relative_bearing=-15.0),
                    TargetGeometryInput(detection_index=1, range=70.0, range_type="GROUND", relative_bearing=35.0),
                ],
            ),
        ]

        batch_result = run_batch_analysis(batch, db_session, mock_detector)

        assert batch_result.total_scans == 3
        assert batch_result.successful_scans == 3
        assert batch_result.total_detections == 3
        assert len(batch_result.scans) == 3

        assert batch_result.scans[0].detection_count == 0
        assert batch_result.scans[1].detection_count == 1
        assert batch_result.scans[2].detection_count == 2


# ---------------------------------------------------------------------------
# Tests: Duplicate and Data Source Handling
# ---------------------------------------------------------------------------


class TestMetadataAndDuplicates:
    """Verifies duplicate prevention and data source flags."""

    def test_duplicate_scan_identity_replaces_record(self, db_session: Session) -> None:
        mock_detector = MagicMock(spec=YOLODetector)
        mock_detector.detect.return_value = []

        scan_in = _make_sample_input(scan_identity="unique-id-123", target_geometries=[])
        run_scan_analysis(scan_in, db_session, mock_detector)
        assert db_session.query(DB_Scan).count() == 1

        # Re-run same identity
        run_scan_analysis(scan_in, db_session, mock_detector)
        assert db_session.query(DB_Scan).count() == 1

    def test_data_source_provenance_flags(self, db_session: Session) -> None:
        mock_detector = MagicMock(spec=YOLODetector)
        mock_detector.detect.return_value = []

        for src in ["REAL", "DEMO", "SIMULATED"]:
            scan_in = _make_sample_input(scan_identity=f"scan-{src}", data_source=src, target_geometries=[])
            result = run_scan_analysis(scan_in, db_session, mock_detector)
            assert result.data_source.lower() == src.lower()


# ---------------------------------------------------------------------------
# Tests: range_type Literal validation (Phase 3C strict constraint)
# ---------------------------------------------------------------------------


class TestRangeTypeValidation:
    """Verifies that range_type is strictly constrained to 'SLANT' or 'GROUND'."""

    # --- TargetGeometryInput ---

    def test_target_geometry_slant_accepted(self) -> None:
        """SLANT must be accepted by TargetGeometryInput."""
        tg = TargetGeometryInput(detection_index=0, range=50.0, range_type="SLANT", relative_bearing=0.0)
        assert tg.range_type == "SLANT"

    def test_target_geometry_ground_accepted(self) -> None:
        """GROUND must be accepted by TargetGeometryInput."""
        tg = TargetGeometryInput(detection_index=0, range=50.0, range_type="GROUND", relative_bearing=0.0)
        assert tg.range_type == "GROUND"

    def test_target_geometry_invalid_range_type_rejected(self) -> None:
        """Any value other than SLANT/GROUND must raise a Pydantic ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TargetGeometryInput(detection_index=0, range=50.0, range_type="OBLIQUE", relative_bearing=0.0)

    def test_target_geometry_lowercase_slant_rejected(self) -> None:
        """Lowercase 'slant' is not a valid Literal value and must be rejected."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TargetGeometryInput(detection_index=0, range=50.0, range_type="slant", relative_bearing=0.0)

    def test_target_geometry_empty_string_rejected(self) -> None:
        """An empty string must be rejected."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TargetGeometryInput(detection_index=0, range=50.0, range_type="", relative_bearing=0.0)

    # --- ScanInput ---

    def test_scan_input_slant_accepted(self) -> None:
        """SLANT must be accepted by ScanInput."""
        scan = ScanInput(
            image_path="/tmp/img.png",
            sonar_latitude=0.0,
            sonar_longitude=0.0,
            heading=0.0,
            altitude=10.0,
            range_type="SLANT",
        )
        assert scan.range_type == "SLANT"

    def test_scan_input_ground_accepted(self) -> None:
        """GROUND must be accepted by ScanInput."""
        scan = ScanInput(
            image_path="/tmp/img.png",
            sonar_latitude=0.0,
            sonar_longitude=0.0,
            heading=0.0,
            altitude=10.0,
            range_type="GROUND",
        )
        assert scan.range_type == "GROUND"

    def test_scan_input_invalid_range_type_rejected(self) -> None:
        """Any value other than SLANT/GROUND must raise a Pydantic ValidationError for ScanInput."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ScanInput(
                image_path="/tmp/img.png",
                sonar_latitude=0.0,
                sonar_longitude=0.0,
                heading=0.0,
                altitude=10.0,
                range_type="INVALID_TYPE",
            )

    def test_scan_input_lowercase_ground_rejected(self) -> None:
        """Lowercase 'ground' is not a valid Literal value and must be rejected for ScanInput."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ScanInput(
                image_path="/tmp/img.png",
                sonar_latitude=0.0,
                sonar_longitude=0.0,
                heading=0.0,
                altitude=10.0,
                range_type="ground",
            )

    def test_scan_input_default_range_type_is_slant(self) -> None:
        """When range_type is omitted, it must default to 'SLANT'."""
        scan = ScanInput(
            image_path="/tmp/img.png",
            sonar_latitude=0.0,
            sonar_longitude=0.0,
            heading=0.0,
            altitude=10.0,
        )
        assert scan.range_type == "SLANT"

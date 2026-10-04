"""
test_analysis.py
----------------
Unit and integration tests for the Phase 3B batch scan analysis pipeline.

Coverage
--------
- Single scan with zero detections (persists scan, empty detections)
- Single scan with multiple detections (persists scan and all linked detections)
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
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import Session, sessionmaker

from app.api.analysis import run_batch_analysis, run_scan_analysis
from app.db.models import Base, DataSource, Detection as DB_Detection, RangeType, Scan as DB_Scan
from app.detection.yolo_detector import Detection, YOLODetector
from app.georeferencing.engine import georeference
from app.schemas.report import ScanInput


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
    range_val: float = 50.0,
    range_type: str = "SLANT",
    altitude: float = 10.0,
    data_source: str = "REAL",
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
    )


# ---------------------------------------------------------------------------
# Tests: Zero Detections
# ---------------------------------------------------------------------------


class TestZeroDetections:
    """Verifies that scans with 0 detections are preserved and stored correctly."""

    def test_zero_detections_persists_scan(self, db_session: Session) -> None:
        mock_detector = MagicMock(spec=YOLODetector)
        mock_detector.detect.return_value = []

        scan_in = _make_sample_input(scan_identity="scan-zero-01")
        result = run_scan_analysis(scan_in, db_session, mock_detector)

        assert result.scan_identity == "scan-zero-01"
        assert result.detection_count == 0
        assert len(result.detections) == 0

        # Verify database state
        db_scan = db_session.query(DB_Scan).filter(DB_Scan.scan_identity == "scan-zero-01").first()
        assert db_scan is not None
        assert db_scan.sonar_latitude == pytest.approx(13.0827)
        assert len(db_scan.detections) == 0


# ---------------------------------------------------------------------------
# Tests: Multiple Detections
# ---------------------------------------------------------------------------


class TestMultipleDetections:
    """Verifies that multiple detections are properly georeferenced and persisted."""

    def test_multiple_detections_linked_to_scan(self, db_session: Session) -> None:
        mock_detector = MagicMock(spec=YOLODetector)
        mock_detector.detect.return_value = [
            Detection(
                class_id=1,
                class_name="shipwreck",
                confidence=0.92,
                x1=10.0,
                y1=20.0,
                x2=80.0,
                y2=90.0,
            ),
            Detection(
                class_id=3,
                class_name="mine_cylinder",
                confidence=0.85,
                x1=150.0,
                y1=160.0,
                x2=200.0,
                y2=210.0,
            ),
        ]

        scan_in = _make_sample_input(scan_identity="scan-multi-01")
        result = run_scan_analysis(scan_in, db_session, mock_detector)

        assert result.detection_count == 2
        assert len(result.detections) == 2
        classes = {d.class_name for d in result.detections}
        assert classes == {"shipwreck", "mine_cylinder"}

        # Check DB persistence
        db_scan = db_session.query(DB_Scan).filter(DB_Scan.scan_identity == "scan-multi-01").first()
        assert db_scan is not None
        assert len(db_scan.detections) == 2
        for db_det in db_scan.detections:
            assert db_det.scan_id == db_scan.id
            assert db_det.target_latitude is not None
            assert db_det.target_longitude is not None


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
            range_val=50.0,
            range_type="SLANT",
            altitude=10.0,
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
            range_val=50.0,
            range_type="SLANT",
            altitude=30.0,
        )
        res_slant = run_scan_analysis(scan_slant, db_session, mock_detector)
        assert res_slant.detections[0].ground_range_m == pytest.approx(40.0, abs=1e-4)

        # 2. Ground range: 50m ground range directly
        scan_ground = _make_sample_input(
            scan_identity="ground-calc",
            range_val=50.0,
            range_type="GROUND",
            altitude=30.0,
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
            _make_sample_input(scan_identity="scan1", image_path="/data/scan1.png"),
            _make_sample_input(scan_identity="scan2", image_path="/data/scan2.png"),
            _make_sample_input(scan_identity="scan3", image_path="/data/scan3.png"),
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

        scan_in = _make_sample_input(scan_identity="unique-id-123")
        run_scan_analysis(scan_in, db_session, mock_detector)
        assert db_session.query(DB_Scan).count() == 1

        # Re-run same identity
        run_scan_analysis(scan_in, db_session, mock_detector)
        assert db_session.query(DB_Scan).count() == 1

    def test_data_source_provenance_flags(self, db_session: Session) -> None:
        mock_detector = MagicMock(spec=YOLODetector)
        mock_detector.detect.return_value = []

        for src in ["REAL", "DEMO", "SIMULATED"]:
            scan_in = _make_sample_input(scan_identity=f"scan-{src}", data_source=src)
            result = run_scan_analysis(scan_in, db_session, mock_detector)
            assert result.data_source.lower() == src.lower()

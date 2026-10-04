"""
test_scans_api.py
-----------------
API tests for Phase 3B endpoints:
- POST /api/scans/analyze
- GET  /api/scans
- GET  /api/scans/{scan_id}

Uses httpx.AsyncClient with ASGITransport, with DB session and YOLO detector
isolated via FastAPI app.dependency_overrides.
"""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.api.analysis import get_detector
from sqlalchemy.pool import StaticPool
from app.db.models import Base
from app.db.session import get_db
from app.detection.yolo_detector import Detection, YOLODetector
from app.main import app


# ---------------------------------------------------------------------------
# Test Fixtures & Dependency Overrides
# ---------------------------------------------------------------------------


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
    """Provide a mock YOLODetector and override get_detector."""
    detector = MagicMock(spec=YOLODetector)
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
# POST /api/scans/analyze
# ---------------------------------------------------------------------------


class TestAnalyzeEndpoint:
    """Tests for POST /api/scans/analyze."""

    @pytest.mark.asyncio
    async def test_analyze_batch_multiple_scans(
        self, client: AsyncClient, mock_detector: MagicMock
    ) -> None:
        """Batch with multiple scans: one zero-detection, one with detections."""
        def detect_side_effect(img: str):
            if "scan_zero" in str(img):
                return []
            return [
                Detection(class_id=1, class_name="shipwreck", confidence=0.91, x1=10, y1=10, x2=50, y2=50)
            ]

        mock_detector.detect.side_effect = detect_side_effect

        payload = {
            "scans": [
                {
                    "scan_identity": "scan_zero",
                    "image_path": "/data/scan_zero.png",
                    "sonar_latitude": 13.08,
                    "sonar_longitude": 80.27,
                    "heading": 45.0,
                    "altitude": 10.0,
                    "range": 50.0,
                    "range_type": "SLANT",
                    "relative_bearing": 0.0,
                    "data_source": "REAL",
                    "target_geometries": [],
                },
                {
                    "scan_identity": "scan_with_det",
                    "image_path": "/data/scan_with_det.png",
                    "sonar_latitude": 13.08,
                    "sonar_longitude": 80.27,
                    "heading": 90.0,
                    "altitude": 15.0,
                    "range": 60.0,
                    "range_type": "SLANT",
                    "relative_bearing": 30.0,
                    "data_source": "DEMO",
                    "target_geometries": [
                        {
                            "detection_index": 0,
                            "range": 60.0,
                            "range_type": "SLANT",
                            "relative_bearing": 30.0,
                        }
                    ],
                },
            ]
        }

        response = await client.post("/api/scans/analyze", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["total_scans"] == 2
        assert data["successful_scans"] == 2
        assert data["total_detections"] == 1
        assert len(data["scans"]) == 2

        # Check scan with zero detections
        scan0 = data["scans"][0]
        assert scan0["scan_identity"] == "scan_zero"
        assert scan0["detection_count"] == 0
        assert scan0["detections"] == []

        # Check scan with detection
        scan1 = data["scans"][1]
        assert scan1["scan_identity"] == "scan_with_det"
        assert scan1["detection_count"] == 1
        assert len(scan1["detections"]) == 1
        det = scan1["detections"][0]
        assert det["class_name"] == "shipwreck"
        assert det["target_latitude"] is not None
        assert det["target_longitude"] is not None

    @pytest.mark.asyncio
    async def test_analyze_accepts_list_payload(
        self, client: AsyncClient, mock_detector: MagicMock
    ) -> None:
        """Endpoint accepts a JSON list directly as well as an object wrapper."""
        mock_detector.detect.return_value = []
        payload = [
            {
                "scan_identity": "list-input-01",
                "image_path": "/data/list_01.png",
                "sonar_latitude": 12.0,
                "sonar_longitude": 80.0,
                "heading": 0.0,
                "altitude": 10.0,
                "range": 30.0,
            }
        ]
        response = await client.post("/api/scans/analyze", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total_scans"] == 1
        assert data["scans"][0]["scan_identity"] == "list-input-01"


# ---------------------------------------------------------------------------
# GET /api/scans & GET /api/scans/{scan_id}
# ---------------------------------------------------------------------------


class TestScansCrudEndpoints:
    """Tests for GET /api/scans and GET /api/scans/{scan_id}."""

    @pytest.mark.asyncio
    async def test_get_all_scans_includes_zero_detections(
        self, client: AsyncClient, mock_detector: MagicMock
    ) -> None:
        mock_detector.detect.return_value = []

        # Ingest one scan
        await client.post(
            "/api/scans/analyze",
            json={
                "scans": [
                    {
                        "scan_identity": "scan-list-test",
                        "image_path": "/data/img.png",
                        "sonar_latitude": 13.0,
                        "sonar_longitude": 80.0,
                        "heading": 0.0,
                        "altitude": 10.0,
                        "range": 40.0,
                    }
                ]
            },
        )

        response = await client.get("/api/scans")
        assert response.status_code == 200
        scans = response.json()
        assert len(scans) >= 1
        assert any(s["scan_identity"] == "scan-list-test" for s in scans)

    @pytest.mark.asyncio
    async def test_get_scan_detail_by_id(
        self, client: AsyncClient, mock_detector: MagicMock
    ) -> None:
        mock_detector.detect.return_value = [
            Detection(class_id=3, class_name="mine_cylinder", confidence=0.89, x1=5, y1=5, x2=25, y2=25)
        ]

        post_res = await client.post(
            "/api/scans/analyze",
            json={
                "scans": [
                    {
                        "scan_identity": "detail-test",
                        "image_path": "/data/detail.png",
                        "sonar_latitude": 13.5,
                        "sonar_longitude": 80.5,
                        "heading": 180.0,
                        "altitude": 20.0,
                        "range": 50.0,
                        "range_type": "SLANT",
                        "relative_bearing": 45.0,
                        "target_geometries": [
                            {
                                "detection_index": 0,
                                "range": 50.0,
                                "range_type": "SLANT",
                                "relative_bearing": 45.0,
                            }
                        ],
                    }
                ]
            },
        )
        scan_id = post_res.json()["scans"][0]["id"]

        # Fetch detail
        response = await client.get(f"/api/scans/{scan_id}")
        assert response.status_code == 200
        detail = response.json()

        assert detail["id"] == scan_id
        assert detail["scan_identity"] == "detail-test"
        # Check metadata
        assert detail["metadata"]["scan_identity"] == "detail-test"
        # Check navigation
        assert detail["navigation"]["sonar_latitude"] == pytest.approx(13.5)
        # Check detections & georeferencing
        assert detail["detection_count"] == 1
        det = detail["detections"][0]
        assert det["class_name"] == "mine_cylinder"
        assert det["target_latitude"] is not None
        assert det["absolute_bearing"] == pytest.approx(225.0)

    @pytest.mark.asyncio
    async def test_get_scan_not_found(self, client: AsyncClient) -> None:
        response = await client.get("/api/scans/999999")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_analyze_missing_target_geometries_returns_422(
        self, client: AsyncClient, mock_detector: MagicMock
    ) -> None:
        """API returns 422 when YOLO finds detections but target_geometries is omitted."""
        mock_detector.detect.return_value = [
            Detection(class_id=1, class_name="shipwreck", confidence=0.9, x1=0, y1=0, x2=20, y2=20)
        ]
        payload = [
            {
                "scan_identity": "missing-geom-api",
                "image_path": "/data/test.png",
                "sonar_latitude": 13.0,
                "sonar_longitude": 80.0,
                "heading": 0.0,
                "altitude": 10.0,
            }
        ]
        response = await client.post("/api/scans/analyze", json=payload)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_analyze_duplicate_detection_index_returns_422(
        self, client: AsyncClient, mock_detector: MagicMock
    ) -> None:
        """API returns 422 when duplicate detection_index is provided."""
        mock_detector.detect.return_value = [
            Detection(class_id=1, class_name="shipwreck", confidence=0.9, x1=0, y1=0, x2=20, y2=20),
            Detection(class_id=2, class_name="ghost_net", confidence=0.8, x1=10, y1=10, x2=30, y2=30),
        ]
        payload = [
            {
                "scan_identity": "dup-geom-api",
                "image_path": "/data/test.png",
                "sonar_latitude": 13.0,
                "sonar_longitude": 80.0,
                "heading": 0.0,
                "altitude": 10.0,
                "target_geometries": [
                    {"detection_index": 0, "range": 40.0, "relative_bearing": 0.0},
                    {"detection_index": 0, "range": 50.0, "relative_bearing": 10.0},
                ],
            }
        ]
        response = await client.post("/api/scans/analyze", json=payload)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_analyze_out_of_range_detection_index_returns_422(
        self, client: AsyncClient, mock_detector: MagicMock
    ) -> None:
        """API returns 422 when detection_index exceeds number of detections."""
        mock_detector.detect.return_value = [
            Detection(class_id=1, class_name="shipwreck", confidence=0.9, x1=0, y1=0, x2=20, y2=20)
        ]
        payload = [
            {
                "scan_identity": "out-of-range-api",
                "image_path": "/data/test.png",
                "sonar_latitude": 13.0,
                "sonar_longitude": 80.0,
                "heading": 0.0,
                "altitude": 10.0,
                "target_geometries": [
                    {"detection_index": 3, "range": 40.0, "relative_bearing": 0.0}
                ],
            }
        ]
        response = await client.post("/api/scans/analyze", json=payload)
        assert response.status_code == 422

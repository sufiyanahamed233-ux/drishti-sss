"""
api/analysis.py
---------------
Batch scan analysis pipeline and API endpoint for Drishti SSS (Phase 3B).

Provides:
- Core batch analysis orchestration: YOLO detection + deterministic georeferencing.
- PostgreSQL persistence for Scan and Detection records.
- POST /api/scans/analyze endpoint accepting multiple sonar scans.
"""

from __future__ import annotations

import uuid
from typing import Union

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.models import DataSource, Detection as DB_Detection, RangeType, Scan as DB_Scan
from app.db.session import get_db
from app.detection.yolo_detector import YOLODetector
from app.georeferencing.engine import georeference
from app.schemas.report import (
    BatchAnalysisRequest,
    BatchAnalysisResult,
    ScanInput,
    ScanResult,
)


router = APIRouter(prefix="/scans", tags=["analysis"])


# ---------------------------------------------------------------------------
# Detector dependency
# ---------------------------------------------------------------------------

_detector: YOLODetector | None = None


def get_detector() -> YOLODetector:
    """Dependency that returns a shared YOLODetector instance."""
    global _detector
    if _detector is None:
        _detector = YOLODetector()
    return _detector


# ---------------------------------------------------------------------------
# Core analysis pipeline
# ---------------------------------------------------------------------------


def run_scan_analysis(
    scan_input: ScanInput,
    db: Session,
    detector: YOLODetector,
) -> ScanResult:
    """
    Process a single sonar scan through the analysis pipeline:
    1. Run YOLO object detection on the sonar image.
    2. Georeference each detected target deterministically using platform observables.
    3. Persist the Scan and Detection records in the database.
    4. Return the complete ScanResult.
    """
    # 1. Resolve scan identity and prevent duplicate records
    scan_identity = (
        scan_input.scan_identity.strip()
        if scan_input.scan_identity and scan_input.scan_identity.strip()
        else f"scan_{uuid.uuid4().hex[:10]}"
    )

    existing_scan = (
        db.query(DB_Scan)
        .filter(DB_Scan.scan_identity == scan_identity)
        .first()
    )
    if existing_scan:
        db.delete(existing_scan)
        db.flush()

    # 2. Resolve enums
    data_source_enum = DataSource(scan_input.data_source)
    range_type_enum = RangeType(scan_input.range_type)

    # 3. Run YOLO inference
    detections = detector.detect(scan_input.image_path)

    # 4. Georeference detections deterministically
    db_detections: list[DB_Detection] = []
    for det in detections:
        det_range = float(scan_input.range)
        det_bearing = float(scan_input.relative_bearing)

        geo = georeference(
            sonar_lat=float(scan_input.sonar_latitude),
            sonar_lon=float(scan_input.sonar_longitude),
            sonar_heading=float(scan_input.heading),
            sonar_altitude=float(scan_input.altitude),
            target_slant_range=det_range,
            target_relative_bearing=det_bearing,
            range_type=scan_input.range_type,
        )

        db_det = DB_Detection(
            class_name=det.class_name,
            confidence=float(det.confidence),
            bbox_x1=float(det.x1),
            bbox_y1=float(det.y1),
            bbox_x2=float(det.x2),
            bbox_y2=float(det.y2),
            target_latitude=geo.latitude,
            target_longitude=geo.longitude,
            relative_bearing=det_bearing,
            absolute_bearing=geo.absolute_bearing_deg,
            range_m=det_range,
            ground_range_m=geo.ground_range_m,
        )
        db_detections.append(db_det)

    # 5. Persist Scan record (including zero-detection scans)
    db_scan = DB_Scan(
        scan_identity=scan_identity,
        image_path=scan_input.image_path,
        sonar_latitude=float(scan_input.sonar_latitude),
        sonar_longitude=float(scan_input.sonar_longitude),
        heading=float(scan_input.heading),
        altitude=float(scan_input.altitude),
        range_type=range_type_enum,
        range_m=float(scan_input.range),
        relative_bearing=float(scan_input.relative_bearing),
        timestamp=scan_input.timestamp,
        data_source=data_source_enum,
        notes=scan_input.notes,
        detections=db_detections,
    )

    db.add(db_scan)
    db.commit()
    db.refresh(db_scan)

    return ScanResult.model_validate(db_scan)


def run_batch_analysis(
    scans: list[ScanInput],
    db: Session,
    detector: YOLODetector,
) -> BatchAnalysisResult:
    """Run analysis on a batch of scans, processing each independently."""
    results: list[ScanResult] = []
    total_detections = 0

    for scan_input in scans:
        result = run_scan_analysis(scan_input, db, detector)
        results.append(result)
        total_detections += result.detection_count

    return BatchAnalysisResult(
        total_scans=len(scans),
        successful_scans=len(results),
        total_detections=total_detections,
        scans=results,
    )


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/analyze",
    response_model=BatchAnalysisResult,
    status_code=status.HTTP_200_OK,
    summary="Analyze a batch of sonar scans",
    response_description="Batch analysis results with georeferenced detections",
)
def analyze_scans(
    request: Union[BatchAnalysisRequest, list[ScanInput]],
    db: Session = Depends(get_db),
    detector: YOLODetector = Depends(get_detector),
) -> BatchAnalysisResult:
    """
    Ingest and analyze a batch of sonar scans.

    For each scan:
    - Runs YOLO11s detection on the image.
    - Computes ground range and geographic position using deterministic georeferencing.
    - Stores the scan and all detection records in PostgreSQL.
    - Preserves scans even if zero targets are detected.
    """
    if isinstance(request, BatchAnalysisRequest):
        scan_inputs = request.scans
    else:
        scan_inputs = request

    return run_batch_analysis(scan_inputs, db, detector)

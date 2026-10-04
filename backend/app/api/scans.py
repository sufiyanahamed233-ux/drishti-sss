"""
api/scans.py
------------
API endpoints for listing and retrieving persisted sonar scans and detections.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db.models import Scan as DB_Scan
from app.db.session import get_db
from app.schemas.report import ScanResult
from app.schemas.scan_report import (
    GeoJSONFeatureCollection,
    InvestigationReportJSON,
    build_geojson_investigation_report,
    build_json_investigation_report,
)


router = APIRouter(prefix="/scans", tags=["scans"])



@router.get(
    "",
    response_model=list[ScanResult],
    summary="List all sonar scans",
    response_description="All scan records, including zero-detection scans",
)
@router.get(
    "/",
    response_model=list[ScanResult],
    include_in_schema=False,
)
def list_scans(db: Session = Depends(get_db)) -> list[ScanResult]:
    """Retrieve all processed scans with their detections and georeferencing."""
    scans = db.query(DB_Scan).order_by(DB_Scan.id.asc()).all()
    return [ScanResult.model_validate(scan) for scan in scans]


@router.get(
    "/{scan_id}",
    response_model=ScanResult,
    summary="Get scan by ID",
    response_description="Complete scan record with navigation, detections, and georeferencing",
)
def get_scan(scan_id: int, db: Session = Depends(get_db)) -> ScanResult:
    """Retrieve a single scan by its primary key."""
    scan = db.query(DB_Scan).filter(DB_Scan.id == scan_id).first()
    if scan is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scan with ID {scan_id} not found",
        )
    return ScanResult.model_validate(scan)


@router.get(
    "/{scan_id}/image",
    summary="Get original sonar image for a scan",
    response_description="The original sonar image file",
    responses={
        200: {"content": {"image/*": {}}},
        404: {"description": "Scan or image file not found"},
    },
)
def get_scan_image(scan_id: int, db: Session = Depends(get_db)) -> FileResponse:
    """
    Return the original sonar image stored for a given scan.

    The image path is read exclusively from the database record for this scan —
    the client never supplies a filesystem path.
    """
    scan = db.query(DB_Scan).filter(DB_Scan.id == scan_id).first()
    if scan is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scan with ID {scan_id} not found",
        )

    image_path: str = scan.image_path
    if not os.path.isfile(image_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image file not found on disk: {image_path}",
        )

    return FileResponse(
        path=image_path,
        media_type="image/*",
        filename=os.path.basename(image_path),
    )


@router.get(
    "/{scan_id}/report/json",
    response_model=InvestigationReportJSON,
    summary="Get JSON investigation report for a scan",
    response_description="Structured investigation report containing metadata, navigation, detections, and deterministic georeferencing",
)
def get_scan_report_json(scan_id: int, db: Session = Depends(get_db)) -> InvestigationReportJSON:
    """
    Generate and return a structured JSON investigation report for a sonar scan.

    Includes platform navigation metadata, AI-derived YOLO detections, and
    deterministic geolocation coordinates.
    """
    scan = db.query(DB_Scan).filter(DB_Scan.id == scan_id).first()
    if scan is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scan with ID {scan_id} not found",
        )
    return build_json_investigation_report(scan)


@router.get(
    "/{scan_id}/report/geojson",
    response_model=GeoJSONFeatureCollection,
    summary="Get GeoJSON investigation report for a scan",
    response_description="Valid RFC 7946 GeoJSON FeatureCollection of georeferenced detections",
)
def get_scan_report_geojson(scan_id: int, db: Session = Depends(get_db)) -> GeoJSONFeatureCollection:
    """
    Generate and return a valid RFC 7946 GeoJSON FeatureCollection for a sonar scan.

    Each georeferenced detection is exported as a Point Feature with [longitude, latitude]
    coordinates. Detections with null coordinates are omitted from features.
    Scans with zero detections return an empty FeatureCollection.
    """
    scan = db.query(DB_Scan).filter(DB_Scan.id == scan_id).first()
    if scan is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scan with ID {scan_id} not found",
        )
    return build_geojson_investigation_report(scan)


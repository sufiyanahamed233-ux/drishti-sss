"""
api/scans.py
------------
API endpoints for listing and retrieving persisted sonar scans and detections.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.models import Scan as DB_Scan
from app.db.session import get_db
from app.schemas.report import ScanResult


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

"""
api/batches.py
--------------
REST API endpoint for uploading and analyzing structured investigation batches:
POST /api/batches/analyze

Accepts:
- Sonar images (.jpg, .jpeg, .png)
- One navigation.csv file linking each image to platform navigation metadata.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from sqlalchemy.orm import Session

from app.api.analysis import get_detector
from app.db.session import get_db
from app.detection.yolo_detector import YOLODetector
from app.schemas.report import BatchAnalysisResult
from app.services.batch_input import process_investigation_batch

router = APIRouter(prefix="/batches", tags=["batches"])


@router.post(
    "/analyze",
    response_model=BatchAnalysisResult,
    status_code=status.HTTP_200_OK,
    summary="Upload and analyze an investigation batch",
    response_description="Batch analysis results for all scans in the investigation batch",
)
async def analyze_investigation_batch(
    request: Request,
    scans: list[UploadFile] = File(
        default=[],
        description="Sonar image files (scan_*.jpg, .jpeg, .png)",
    ),
    navigation: UploadFile | None = File(
        default=None,
        description="navigation.csv file with metadata for each scan_file",
    ),
    db: Session = Depends(get_db),
    detector: YOLODetector = Depends(get_detector),
) -> BatchAnalysisResult:
    """
    Ingest and analyze a multi-scan investigation batch.

    Accepts multipart/form-data containing:
    - Multiple sonar scan image files (.jpg, .jpeg, .png)
    - One navigation.csv containing scan-level navigation records linked by scan_file

    Validates:
    - Complete CSV schema and data types.
    - Bidirectional 1-to-1 match between uploaded images and navigation.csv rows.
    - Sequentially processes each scan and persists records in PostgreSQL.
    - Preserves zero-detection scans in the final results.
    """
    all_uploads: list[UploadFile] = []

    if scans:
        for s in scans:
            if getattr(s, "filename", None):
                all_uploads.append(s)

    if navigation is not None and getattr(navigation, "filename", None):
        all_uploads.append(navigation)

    # Fallback to inspecting request form for custom field names (e.g. 'files')
    if not all_uploads:
        form = await request.form()
        for _, value in form.multi_items():
            if hasattr(value, "filename") and getattr(value, "filename", None):
                all_uploads.append(value)

    if not all_uploads:
        raise HTTPException(
            status_code=422,
            detail="Empty batch upload: no files provided.",
        )


    return await process_investigation_batch(all_uploads, db, detector)


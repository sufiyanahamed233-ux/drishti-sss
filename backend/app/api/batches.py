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

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile, status
from sqlalchemy.orm import Session, joinedload

from app.api.analysis import get_detector
from app.db.models import InvestigationBatch, Scan as DB_Scan
from app.db.session import get_db
from app.detection.yolo_detector import YOLODetector
from app.reports.pdf_generator import generate_batch_pdf_report
from app.schemas.report import BatchAnalysisResult, InvestigationBatchResponse, ScanResult
from app.schemas.scan_report import (
    BatchInvestigationReportJSON,
    GeoJSONFeatureCollection,
    build_batch_geojson_investigation_report,
    build_batch_json_investigation_report,
)
from app.services.batch_input import process_investigation_batch

router = APIRouter(prefix="/batches", tags=["batches"])


def get_batch_or_404(batch_id: str, db: Session) -> InvestigationBatch:
    """Helper to fetch an InvestigationBatch or raise HTTP 404."""
    batch = (
        db.query(InvestigationBatch)
        .options(joinedload(InvestigationBatch.scans).joinedload(DB_Scan.detections))
        .filter(InvestigationBatch.batch_id == batch_id)
        .first()
    )
    if batch is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Investigation batch with ID '{batch_id}' not found.",
        )
    return batch


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
    - Creates a persistent InvestigationBatch identity.
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


@router.get(
    "/{batch_id}",
    response_model=InvestigationBatchResponse,
    summary="Get investigation batch results",
    response_description="Complete persisted investigation batch with all scans, detections, and class statistics",
)
def get_investigation_batch(
    batch_id: str,
    db: Session = Depends(get_db),
) -> InvestigationBatchResponse:
    """Retrieve complete results and scan metadata for a specific investigation batch."""
    batch = get_batch_or_404(batch_id, db)

    # Deterministic scan ordering
    sorted_scans = sorted(batch.scans or [], key=lambda s: s.id)
    scan_results = [ScanResult.model_validate(s) for s in sorted_scans]

    # Calculate class counts across all scans
    class_counts: dict[str, int] = {}
    total_dets = 0
    for s in sorted_scans:
        for d in (s.detections or []):
            total_dets += 1
            class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1

    ds_str = (
        batch.data_source.value
        if hasattr(batch.data_source, "value")
        else str(batch.data_source)
    )

    return InvestigationBatchResponse(
        batch_id=batch.batch_id,
        created_at=batch.created_at,
        total_scans=batch.total_scans,
        successful_scans=len(scan_results),
        total_detections=total_dets,
        class_counts=class_counts,
        data_source=ds_str,
        scans=scan_results,
    )


@router.get(
    "/{batch_id}/report/json",
    response_model=BatchInvestigationReportJSON,
    summary="Get JSON investigation report for a batch",
    response_description="Structured batch investigation report containing provenance, statistics, scan summaries, and detections",
)
def get_batch_report_json(
    batch_id: str,
    db: Session = Depends(get_db),
) -> BatchInvestigationReportJSON:
    """Generate and return a structured JSON investigation report for a batch."""
    batch = get_batch_or_404(batch_id, db)
    return build_batch_json_investigation_report(batch)


@router.get(
    "/{batch_id}/report/geojson",
    response_model=GeoJSONFeatureCollection,
    summary="Get GeoJSON investigation report for a batch",
    response_description="Valid RFC 7946 GeoJSON FeatureCollection of all georeferenced detections across the batch",
)
def get_batch_report_geojson(
    batch_id: str,
    db: Session = Depends(get_db),
) -> GeoJSONFeatureCollection:
    """Generate and return an RFC 7946 GeoJSON FeatureCollection for a batch."""
    batch = get_batch_or_404(batch_id, db)
    return build_batch_geojson_investigation_report(batch)


@router.get(
    "/{batch_id}/report/pdf",
    summary="Get PDF investigation report for a batch",
    response_description="Binary downloadable PDF investigation report for the batch",
    responses={
        200: {
            "content": {"application/pdf": {}},
            "description": "Downloadable PDF investigation report",
        },
        404: {"description": "Batch not found"},
    },
)
def get_batch_report_pdf(
    batch_id: str,
    db: Session = Depends(get_db),
) -> Response:
    """Generate and return a downloadable PDF investigation report for a batch."""
    batch = get_batch_or_404(batch_id, db)
    pdf_bytes = generate_batch_pdf_report(batch)
    filename = f"drishti_sss_batch_{batch_id}_report.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


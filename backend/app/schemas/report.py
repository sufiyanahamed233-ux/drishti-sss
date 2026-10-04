"""
schemas/report.py
-----------------
Pydantic schemas for batch analysis requests, scan reports, and detection results.

Used by the batch scan analysis pipeline (Phase 3B) and REST API endpoints.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Sequence, Union

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Bounding box & Detection result schemas
# ---------------------------------------------------------------------------


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    model_config = {"from_attributes": True}


class DetectionResult(BaseModel):
    """Report schema for a single georeferenced detection."""

    id: int | None = None
    scan_id: int | None = None
    class_name: str
    confidence: float
    bbox: BoundingBox
    target_latitude: float | None = None
    target_longitude: float | None = None
    relative_bearing: float | None = None
    absolute_bearing: float | None = None
    range_m: float | None = None
    ground_range_m: float | None = None

    model_config = {"from_attributes": True}

    @model_validator(mode="before")
    @classmethod
    def extract_from_orm(cls, data: Any) -> Any:
        """Extract attributes from SQLAlchemy ORM Detection object."""
        if hasattr(data, "bbox_x1"):
            return {
                "id": getattr(data, "id", None),
                "scan_id": getattr(data, "scan_id", None),
                "class_name": getattr(data, "class_name", ""),
                "confidence": getattr(data, "confidence", 0.0),
                "bbox": {
                    "x1": getattr(data, "bbox_x1", 0.0),
                    "y1": getattr(data, "bbox_y1", 0.0),
                    "x2": getattr(data, "bbox_x2", 0.0),
                    "y2": getattr(data, "bbox_y2", 0.0),
                },
                "target_latitude": getattr(data, "target_latitude", None),
                "target_longitude": getattr(data, "target_longitude", None),
                "relative_bearing": getattr(data, "relative_bearing", None),
                "absolute_bearing": getattr(data, "absolute_bearing", None),
                "range_m": getattr(data, "range_m", None),
                "ground_range_m": getattr(data, "ground_range_m", None),
            }
        return data


# ---------------------------------------------------------------------------
# Navigation & Metadata sub-schemas
# ---------------------------------------------------------------------------


class NavigationMetadata(BaseModel):
    """Navigation metadata of the sonar platform at scan time."""

    sonar_latitude: float
    sonar_longitude: float
    heading: float
    altitude: float
    range: float | None = None
    range_type: str
    relative_bearing: float | None = None

    model_config = {"from_attributes": True}


class ScanMetadata(BaseModel):
    """Administrative and provenance metadata for a scan."""

    id: int
    scan_identity: str
    image_path: str
    timestamp: datetime
    data_source: str
    notes: str | None = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Scan result schema
# ---------------------------------------------------------------------------


class ScanResult(BaseModel):
    """Complete scan details, including navigation, detections, and georeferencing."""

    id: int
    batch_id: str | None = None
    scan_identity: str
    image_path: str
    sonar_latitude: float
    sonar_longitude: float
    heading: float
    altitude: float
    range: float | None = None
    range_type: str
    relative_bearing: float | None = None
    timestamp: datetime
    data_source: str
    notes: str | None = None
    detection_count: int
    detections: list[DetectionResult]

    metadata: ScanMetadata | None = None
    navigation: NavigationMetadata | None = None

    model_config = {"from_attributes": True}

    @model_validator(mode="before")
    @classmethod
    def populate_from_orm(cls, data: Any) -> Any:
        """Convert ORM Scan model into ScanResult response dict."""
        if hasattr(data, "sonar_latitude"):
            dets = getattr(data, "detections", []) or []
            data_source_val = getattr(data, "data_source")
            data_source_str = (
                data_source_val.value
                if hasattr(data_source_val, "value")
                else str(data_source_val)
            )

            range_type_val = getattr(data, "range_type")
            range_type_str = (
                range_type_val.value
                if hasattr(range_type_val, "value")
                else str(range_type_val)
            )

            range_val = getattr(data, "range_m", None)
            rel_bearing_val = getattr(data, "relative_bearing", None)

            scan_id = getattr(data, "id", 0)
            batch_id_val = getattr(data, "batch_id", None)
            identity = getattr(data, "scan_identity", "")
            img_path = getattr(data, "image_path", "")
            ts = getattr(data, "timestamp", None)
            notes_val = getattr(data, "notes", None)
            lat = getattr(data, "sonar_latitude", 0.0)
            lon = getattr(data, "sonar_longitude", 0.0)
            hdg = getattr(data, "heading", 0.0)
            alt = getattr(data, "altitude", 0.0)

            return {
                "id": scan_id,
                "batch_id": batch_id_val,
                "scan_identity": identity,
                "image_path": img_path,
                "sonar_latitude": lat,
                "sonar_longitude": lon,
                "heading": hdg,
                "altitude": alt,
                "range": range_val,
                "range_type": range_type_str,
                "relative_bearing": rel_bearing_val,
                "timestamp": ts,
                "data_source": data_source_str,
                "notes": notes_val,
                "detection_count": len(dets),
                "detections": dets,
                "metadata": {
                    "id": scan_id,
                    "scan_identity": identity,
                    "image_path": img_path,
                    "timestamp": ts,
                    "data_source": data_source_str,
                    "notes": notes_val,
                },
                "navigation": {
                    "sonar_latitude": lat,
                    "sonar_longitude": lon,
                    "heading": hdg,
                    "altitude": alt,
                    "range": range_val,
                    "range_type": range_type_str,
                    "relative_bearing": rel_bearing_val,
                },
            }
        return data


# ---------------------------------------------------------------------------
# Batch scan input & response schemas
# ---------------------------------------------------------------------------


class TargetGeometryInput(BaseModel):
    """Per-detection sonar geometry observables for deterministic georeferencing."""

    detection_index: int = Field(..., ge=0, description="Zero-based index of the YOLO detection")
    range: float = Field(..., ge=0.0, description="Range to the target in metres")
    range_type: Literal["SLANT", "GROUND"] = Field(default="SLANT", description="SLANT or GROUND range type")
    relative_bearing: float = Field(default=0.0, description="Relative bearing to the target in degrees")

    @model_validator(mode="before")
    @classmethod
    def handle_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "range" not in data and "range_m" in data:
                data["range"] = data["range_m"]
        return data


class ScanInput(BaseModel):
    """Input payload for a single sonar scan in a batch analysis request."""

    image_path: str = Field(..., description="Path to the sonar image file")
    sonar_latitude: float = Field(..., ge=-90.0, le=90.0, description="WGS-84 latitude (deg)")
    sonar_longitude: float = Field(..., ge=-180.0, le=180.0, description="WGS-84 longitude (deg)")
    heading: float = Field(..., description="Platform heading in degrees [0, 360)")
    altitude: float = Field(..., gt=0.0, description="Platform altitude above seabed in metres")
    range: float | None = Field(default=None, ge=0.0, description="Optional scan-level observable range in metres")
    range_type: Literal["SLANT", "GROUND"] = Field(default="SLANT", description="SLANT or GROUND")
    relative_bearing: float | None = Field(default=None, description="Optional scan-level relative bearing in degrees")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data_source: str = Field(default="REAL", description="REAL, DEMO, or SIMULATED")
    scan_identity: str | None = Field(default=None, description="Optional scan identifier")
    notes: str | None = Field(default=None, description="Optional operator notes")
    target_geometries: list[TargetGeometryInput] = Field(
        default_factory=list,
        description="Per-detection geometry inputs matching YOLO detections",
    )

    @model_validator(mode="before")
    @classmethod
    def handle_aliases(cls, data: Any) -> Any:
        """Allow range_m as an alias for range."""
        if isinstance(data, dict):
            if "range" not in data and "range_m" in data:
                data["range"] = data["range_m"]
        return data


class BatchAnalysisRequest(BaseModel):
    """Batch analysis request containing multiple sonar scan inputs."""

    scans: list[ScanInput]


class BatchAnalysisResult(BaseModel):
    """Response containing batch execution metrics and individual scan reports."""

    batch_id: str | None = None
    total_scans: int
    successful_scans: int
    total_detections: int
    scans: list[ScanResult]


class InvestigationBatchResponse(BaseModel):
    """Complete persisted investigation batch details including class breakdown."""

    batch_id: str
    created_at: datetime
    total_scans: int
    successful_scans: int
    total_detections: int
    class_counts: dict[str, int] = Field(default_factory=dict)
    data_source: str
    scans: list[ScanResult] = Field(default_factory=list)

    model_config = {"from_attributes": True}

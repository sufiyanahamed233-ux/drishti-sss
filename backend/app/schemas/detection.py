"""
schemas/detection.py
--------------------
Pydantic schemas for detection results stored in the database.

These schemas mirror the :class:`~app.db.models.Detection` ORM model and are
used for API response serialisation.  Georeferenced fields are optional
because the Phase 3A foundation does not integrate YOLO or georeferencing yet.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared field definitions
# ---------------------------------------------------------------------------

_Confidence = Annotated[
    float, Field(ge=0.0, le=1.0, description="Detection confidence [0.0, 1.0]")
]
_Pixel = Annotated[float, Field(ge=0.0, description="Pixel coordinate (non-negative)")]
_OptLatitude = Annotated[
    float | None,
    Field(default=None, ge=-90.0, le=90.0, description="Decimal degrees (WGS-84)"),
]
_OptLongitude = Annotated[
    float | None,
    Field(default=None, ge=-180.0, le=180.0, description="Decimal degrees (WGS-84)"),
]
_OptBearing = Annotated[
    float | None,
    Field(default=None, description="Bearing in degrees"),
]
_OptRange = Annotated[
    float | None,
    Field(default=None, ge=0.0, description="Range in metres"),
]


# ---------------------------------------------------------------------------
# BBox helper
# ---------------------------------------------------------------------------


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates."""

    x1: _Pixel
    y1: _Pixel
    x2: _Pixel
    y2: _Pixel

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Detection schemas
# ---------------------------------------------------------------------------


class DetectionCreate(BaseModel):
    """
    Input schema for persisting a detection result linked to a scan.

    Georeferenced fields are optional at creation time (they will be filled
    in by the georeferencing pipeline in a later phase).
    """

    scan_id: int = Field(..., description="Primary key of the parent Scan record")
    class_name: str = Field(..., max_length=64, description="YOLO class label")
    confidence: _Confidence
    bbox: BoundingBox

    # Optional georeferenced output (Phase 3B+)
    target_latitude: _OptLatitude = None
    target_longitude: _OptLongitude = None
    relative_bearing: _OptBearing = None
    absolute_bearing: _OptBearing = None
    range_m: _OptRange = None
    ground_range_m: _OptRange = None


class DetectionRead(BaseModel):
    """Response schema for a persisted detection record."""

    id: int
    scan_id: int
    class_name: str
    confidence: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float

    target_latitude: float | None
    target_longitude: float | None
    relative_bearing: float | None
    absolute_bearing: float | None
    range_m: float | None
    ground_range_m: float | None

    model_config = {"from_attributes": True}

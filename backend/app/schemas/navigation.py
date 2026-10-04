"""
schemas/navigation.py
---------------------
Pydantic schemas for sonar platform navigation data carried in a scan.

These schemas are used for both API request validation and response
serialisation.  All field names match the :class:`~app.db.models.Scan`
ORM columns so they can be converted with ``model_validate(orm_obj)``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from pydantic import BaseModel, Field, field_validator

from app.db.models import DataSource, RangeType


# ---------------------------------------------------------------------------
# Shared field definitions
# ---------------------------------------------------------------------------

_Latitude = Annotated[float, Field(ge=-90.0, le=90.0, description="Decimal degrees (WGS-84)")]
_Longitude = Annotated[float, Field(ge=-180.0, le=180.0, description="Decimal degrees (WGS-84)")]
_Heading = Annotated[float, Field(ge=0.0, lt=360.0, description="True-North heading in degrees [0, 360)")]
_Altitude = Annotated[float, Field(gt=0.0, description="Height above seabed/target plane in metres")]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class NavigationData(BaseModel):
    """
    Navigation state of the sonar platform at the moment of image capture.

    This is embedded inside :class:`ScanCreate` and returned inside
    :class:`ScanRead`.
    """

    sonar_latitude: _Latitude
    sonar_longitude: _Longitude
    heading: _Heading
    altitude: _Altitude
    range_type: RangeType = RangeType.MEDIUM

    model_config = {"from_attributes": True}


class ScanCreate(BaseModel):
    """Request body for creating a new scan record."""

    scan_identity: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Operator-assigned identifier for this scan",
    )
    image_path: str = Field(..., description="Path to the sonar image file")
    navigation: NavigationData
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC acquisition time",
    )
    data_source: DataSource = Field(
        DataSource.REAL,
        description="Data provenance flag: REAL | DEMO | SIMULATED",
    )
    notes: str | None = Field(None, description="Free-text operator notes")

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        """Make sure the stored timestamp is always timezone-aware (UTC)."""
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class ScanRead(BaseModel):
    """Response schema for a persisted scan record."""

    id: int
    scan_identity: str
    image_path: str
    sonar_latitude: float
    sonar_longitude: float
    heading: float
    altitude: float
    range_type: RangeType
    timestamp: datetime
    data_source: DataSource
    notes: str | None

    model_config = {"from_attributes": True}

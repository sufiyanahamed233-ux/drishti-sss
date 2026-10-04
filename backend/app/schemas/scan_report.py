"""
schemas/scan_report.py
----------------------
Pydantic schemas and deterministic builders for JSON and GeoJSON investigation
reports for DRISHTI-SSS sonar scans.

Preserves the critical scientific and operational provenance distinction:
- Platform navigation: measured / sensor input
- Target detections: AI-derived (YOLO neural network)
- Target geolocations: deterministic geometry-derived calculation (NOT predicted by AI)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, Field

from app.db.models import Scan as DB_Scan


# ---------------------------------------------------------------------------
# Shared Sub-schemas
# ---------------------------------------------------------------------------


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    model_config = {"from_attributes": True}


class ProvenanceMetadata(BaseModel):
    """Explicit provenance disclosure for investigation reports."""

    navigation_data: str = Field(
        default="Measured/input platform sensors (GNSS, gyrocompass, altimeter)",
        description="Source of sonar platform position and attitude",
    )
    detection_data: str = Field(
        default="AI-derived object detection (YOLO neural network)",
        description="Source of bounding boxes, classification, and confidence",
    )
    geolocation_data: str = Field(
        default="Deterministic geometry-derived geolocation (NOT predicted by AI)",
        description="Mathematical projection from sonar nav observables; never AI-predicted",
    )


class InvestigationReportMetadata(BaseModel):
    """Metadata describing the investigation report itself."""

    report_title: str = "DRISHTI-SSS Sonar Investigation Report"
    report_type: str = "INVESTIGATION_REPORT"
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of report generation",
    )
    provenance: ProvenanceMetadata = Field(default_factory=ProvenanceMetadata)
    data_classification: str = "UNCLASSIFIED"


class SonarNavigationMetadata(BaseModel):
    """Navigation metadata of the sonar platform at scan time."""

    sonar_latitude: float = Field(..., description="WGS-84 latitude of the sonar platform (deg)")
    sonar_longitude: float = Field(..., description="WGS-84 longitude of the sonar platform (deg)")
    heading: float = Field(..., description="Platform True-North heading (deg)")
    altitude: float = Field(..., description="Platform altitude above seabed (m)")
    range_m: float | None = Field(default=None, description="Sonar range observable in metres")
    range: float | None = Field(default=None, description="Alias for range_m")
    range_type: str = Field(..., description="Sonar range mode (e.g. SLANT, GROUND, MEDIUM)")
    relative_bearing: float | None = Field(default=None, description="Scan-level relative bearing (deg)")

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# JSON Report Detections & Root Schema
# ---------------------------------------------------------------------------


class InvestigationDetection(BaseModel):
    """
    Single detection item in the investigation report.

    Clearly documents AI-predicted features vs deterministic geolocation features.
    """

    detection_id: int | None = Field(default=None, description="Primary key of Detection record")
    id: int | None = Field(default=None, description="Alias for detection_id")
    detection_index: int = Field(default=0, description="Zero-based index of detection in scan")

    # AI-derived observables
    class_name: str = Field(..., description="Object class label predicted by YOLO")
    detection_class: str = Field(..., description="Alias for class_name")
    confidence: float = Field(..., description="Detection confidence score [0.0, 1.0]")
    bbox: BoundingBox = Field(..., description="Pixel bounding box [x1, y1, x2, y2]")
    bounding_box: BoundingBox = Field(..., description="Alias for bbox")

    # Geometry observables & deterministic geolocation
    range: float | None = Field(default=None, description="Range observable in metres")
    range_m: float | None = Field(default=None, description="Range observable in metres")
    range_type: str | None = Field(default=None, description="Range type (e.g. SLANT or GROUND)")
    ground_range: float | None = Field(default=None, description="Horizontal ground range in metres")
    ground_range_m: float | None = Field(default=None, description="Horizontal ground range in metres")
    relative_bearing: float | None = Field(default=None, description="Relative bearing to target (deg)")
    absolute_bearing: float | None = Field(default=None, description="True-North bearing to target (deg)")
    target_latitude: float | None = Field(default=None, description="WGS-84 target latitude (deg)")
    target_longitude: float | None = Field(default=None, description="WGS-84 target longitude (deg)")

    # Provenance tags
    detection_source: str = "AI_YOLO"
    geolocation_method: str = "DETERMINISTIC_GEOMETRY"

    model_config = {"from_attributes": True}


class InvestigationReportJSON(BaseModel):
    """
    Complete JSON Investigation Report schema for a single scan.
    """

    report_metadata: InvestigationReportMetadata = Field(default_factory=InvestigationReportMetadata)
    metadata: InvestigationReportMetadata = Field(default_factory=InvestigationReportMetadata)
    scan_id: int = Field(..., description="Database ID of the scan")
    id: int = Field(..., description="Alias for scan_id")
    scan_identity: str = Field(..., description="Operator/mission scan identity")
    timestamp: datetime = Field(..., description="Acquisition timestamp")
    data_source: str = Field(..., description="Data provenance (REAL, DEMO, SIMULATED)")
    sonar_navigation_metadata: SonarNavigationMetadata = Field(..., description="Platform navigation state")
    navigation: SonarNavigationMetadata = Field(..., description="Alias for sonar_navigation_metadata")
    detection_count: int = Field(..., description="Total detections in this scan")
    detections: list[InvestigationDetection] = Field(default_factory=list, description="All detected objects")

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# GeoJSON FeatureCollection Schemas (RFC 7946)
# ---------------------------------------------------------------------------


class GeoJSONPointGeometry(BaseModel):
    """RFC 7946 Point Geometry with [longitude, latitude] coordinates."""

    type: Literal["Point"] = "Point"
    coordinates: list[float] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Coordinates in [longitude, latitude] order (WGS-84)",
    )


class GeoJSONFeatureProperties(BaseModel):
    """Properties for a georeferenced detection Point Feature."""

    scan_id: int
    scan_identity: str
    detection_id: int | None = None
    detection_index: int = 0
    class_name: str
    confidence: float
    range_m: float | None = None
    ground_range_m: float | None = None
    relative_bearing: float | None = None
    absolute_bearing: float | None = None
    timestamp: datetime
    data_source: str
    detection_source: str = "AI_YOLO"
    geolocation_method: str = "DETERMINISTIC_GEOMETRY"

    model_config = {"from_attributes": True}


class GeoJSONFeature(BaseModel):
    """RFC 7946 Feature object."""

    type: Literal["Feature"] = "Feature"
    id: int | None = None
    geometry: GeoJSONPointGeometry
    properties: GeoJSONFeatureProperties

    model_config = {"from_attributes": True}


class GeoJSONFeatureCollection(BaseModel):
    """RFC 7946 FeatureCollection object."""

    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[GeoJSONFeature] = Field(default_factory=list)

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Deterministic Builder Functions
# ---------------------------------------------------------------------------


def build_json_investigation_report(scan: DB_Scan) -> InvestigationReportJSON:
    """
    Construct a deterministic JSON investigation report from a persisted DB Scan.
    """
    data_source_str = (
        scan.data_source.value
        if hasattr(scan.data_source, "value")
        else str(scan.data_source)
    )
    range_type_str = (
        scan.range_type.value
        if hasattr(scan.range_type, "value")
        else str(scan.range_type)
    )

    nav = SonarNavigationMetadata(
        sonar_latitude=scan.sonar_latitude,
        sonar_longitude=scan.sonar_longitude,
        heading=scan.heading,
        altitude=scan.altitude,
        range_m=scan.range_m,
        range=scan.range_m,
        range_type=range_type_str,
        relative_bearing=scan.relative_bearing,
    )

    detections_list = sorted(
        scan.detections or [],
        key=lambda d: getattr(d, "id", 0) or 0,
    )

    report_detections: list[InvestigationDetection] = []
    for idx, d in enumerate(detections_list):
        bbox = BoundingBox(
            x1=float(d.bbox_x1),
            y1=float(d.bbox_y1),
            x2=float(d.bbox_x2),
            y2=float(d.bbox_y2),
        )
        report_detections.append(
            InvestigationDetection(
                detection_id=d.id,
                id=d.id,
                detection_index=idx,
                class_name=d.class_name,
                detection_class=d.class_name,
                confidence=float(d.confidence),
                bbox=bbox,
                bounding_box=bbox,
                range=float(d.range_m) if d.range_m is not None else None,
                range_m=float(d.range_m) if d.range_m is not None else None,
                range_type=range_type_str,
                ground_range=float(d.ground_range_m) if d.ground_range_m is not None else None,
                ground_range_m=float(d.ground_range_m) if d.ground_range_m is not None else None,
                relative_bearing=float(d.relative_bearing) if d.relative_bearing is not None else None,
                absolute_bearing=float(d.absolute_bearing) if d.absolute_bearing is not None else None,
                target_latitude=float(d.target_latitude) if d.target_latitude is not None else None,
                target_longitude=float(d.target_longitude) if d.target_longitude is not None else None,
                detection_source="AI_YOLO",
                geolocation_method="DETERMINISTIC_GEOMETRY",
            )
        )

    meta = InvestigationReportMetadata(
        generated_at=scan.timestamp if scan.timestamp else datetime.now(timezone.utc)
    )

    return InvestigationReportJSON(
        report_metadata=meta,
        metadata=meta,
        scan_id=scan.id,
        id=scan.id,
        scan_identity=scan.scan_identity,
        timestamp=scan.timestamp,
        data_source=data_source_str,
        sonar_navigation_metadata=nav,
        navigation=nav,
        detection_count=len(report_detections),
        detections=report_detections,
    )


def build_geojson_investigation_report(scan: DB_Scan) -> GeoJSONFeatureCollection:
    """
    Construct a deterministic GeoJSON FeatureCollection from a persisted DB Scan.

    Detections with null target_latitude or target_longitude are excluded from the
    features list (per requirements).
    """
    data_source_str = (
        scan.data_source.value
        if hasattr(scan.data_source, "value")
        else str(scan.data_source)
    )

    detections_list = sorted(
        scan.detections or [],
        key=lambda d: getattr(d, "id", 0) or 0,
    )

    features: list[GeoJSONFeature] = []
    for idx, d in enumerate(detections_list):
        # Exclude detection if coordinates are missing / null
        if d.target_latitude is None or d.target_longitude is None:
            continue

        geom = GeoJSONPointGeometry(
            type="Point",
            coordinates=[float(d.target_longitude), float(d.target_latitude)],
        )
        props = GeoJSONFeatureProperties(
            scan_id=scan.id,
            scan_identity=scan.scan_identity,
            detection_id=d.id,
            detection_index=idx,
            class_name=d.class_name,
            confidence=float(d.confidence),
            range_m=float(d.range_m) if d.range_m is not None else None,
            ground_range_m=float(d.ground_range_m) if d.ground_range_m is not None else None,
            relative_bearing=float(d.relative_bearing) if d.relative_bearing is not None else None,
            absolute_bearing=float(d.absolute_bearing) if d.absolute_bearing is not None else None,
            timestamp=scan.timestamp,
            data_source=data_source_str,
            detection_source="AI_YOLO",
            geolocation_method="DETERMINISTIC_GEOMETRY",
        )
        features.append(
            GeoJSONFeature(
                type="Feature",
                id=d.id,
                geometry=geom,
                properties=props,
            )
        )

    return GeoJSONFeatureCollection(
        type="FeatureCollection",
        features=features,
    )

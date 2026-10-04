"""
db/models.py
------------
SQLAlchemy ORM models for the Drishti Phase 3A database schema.

Models
~~~~~~
:class:`Scan`
    One record per sonar image processed.  Stores sonar platform state
    (position, heading, altitude) and metadata flags.

:class:`Detection`
    One record per detected object within a :class:`Scan`.  Stores class,
    confidence, bounding box, georeferenced target position, and bearing/range.

Relationships
~~~~~~~~~~~~~
``Scan`` ← one-to-many → ``Detection``

    Access via ``scan.detections`` (list) or ``detection.scan`` (back-ref).
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class RangeType(str, enum.Enum):
    """Sonar range mode recorded at scan time."""

    SHORT = "short"      # e.g. 0–25 m
    MEDIUM = "medium"    # e.g. 25–75 m
    LONG = "long"        # e.g. 75–150 m
    EXTENDED = "extended"


class DataSource(str, enum.Enum):
    """Indicates whether the data is from a real deployment or simulated."""

    REAL = "real"
    DEMO = "demo"
    SIMULATED = "simulated"


# ---------------------------------------------------------------------------
# Scan model
# ---------------------------------------------------------------------------


class Scan(Base):
    """
    One sonar scan / image acquisition event.

    Columns
    ~~~~~~~
    id                : surrogate primary key
    scan_identity     : operator-assigned identifier (e.g. mission leg + frame)
    image_path        : absolute or relative path to the stored sonar image
    sonar_latitude    : WGS-84 latitude of the sonar at capture time (deg)
    sonar_longitude   : WGS-84 longitude of the sonar at capture time (deg)
    heading           : True-North heading of the platform at capture time (deg)
    altitude          : height of the sonar above seabed / target plane (m)
    range_type        : sonar range mode (short / medium / long / extended)
    timestamp         : UTC datetime of the acquisition event
    data_source       : REAL, DEMO, or SIMULATED – provenance flag
    notes             : free-text operator notes (optional)
    """

    __tablename__ = "scans"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    scan_identity: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    image_path: Mapped[str] = mapped_column(Text, nullable=False)

    # ── Sonar platform state ──────────────────────────────────────────────────
    sonar_latitude: Mapped[float] = mapped_column(Float, nullable=False)
    sonar_longitude: Mapped[float] = mapped_column(Float, nullable=False)
    heading: Mapped[float] = mapped_column(Float, nullable=False)
    altitude: Mapped[float] = mapped_column(Float, nullable=False)

    # ── Operational metadata ──────────────────────────────────────────────────
    range_type: Mapped[RangeType] = mapped_column(
        Enum(RangeType, name="range_type_enum"),
        nullable=False,
        default=RangeType.MEDIUM,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    data_source: Mapped[DataSource] = mapped_column(
        Enum(DataSource, name="data_source_enum"),
        nullable=False,
        default=DataSource.REAL,
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Relationship ──────────────────────────────────────────────────────────
    detections: Mapped[list[Detection]] = relationship(
        "Detection",
        back_populates="scan",
        cascade="all, delete-orphan",
        lazy="select",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Scan id={self.id!r} identity={self.scan_identity!r} "
            f"source={self.data_source.value!r}>"
        )


# ---------------------------------------------------------------------------
# Detection model
# ---------------------------------------------------------------------------


class Detection(Base):
    """
    One detected object within a :class:`Scan`.

    Columns
    ~~~~~~~
    id                  : surrogate primary key
    scan_id             : FK → scans.id
    class_name          : YOLO class label (e.g. "mine_cylinder")
    confidence          : detection confidence score [0.0, 1.0]
    bbox_x1 … bbox_y2   : bounding box pixel corners (top-left, bottom-right)
    target_latitude     : georeferenced WGS-84 latitude of the target (deg)
    target_longitude    : georeferenced WGS-84 longitude of the target (deg)
    relative_bearing    : bearing to target relative to sonar bow (deg)
    absolute_bearing    : True-North bearing to the target (deg)
    range_m             : slant range to the target (m)
    ground_range_m      : horizontal ground range to the target (m)
    """

    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    scan_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("scans.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # ── Detection result ──────────────────────────────────────────────────────
    class_name: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    # Bounding box (pixel coordinates)
    bbox_x1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x2: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y2: Mapped[float] = mapped_column(Float, nullable=False)

    # ── Georeferenced target position ─────────────────────────────────────────
    target_latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_longitude: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Bearing and range ─────────────────────────────────────────────────────
    relative_bearing: Mapped[float | None] = mapped_column(Float, nullable=True)
    absolute_bearing: Mapped[float | None] = mapped_column(Float, nullable=True)
    range_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    ground_range_m: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Relationship ──────────────────────────────────────────────────────────
    scan: Mapped[Scan] = relationship("Scan", back_populates="detections")

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Detection id={self.id!r} class={self.class_name!r} "
            f"conf={self.confidence:.2f} scan_id={self.scan_id!r}>"
        )

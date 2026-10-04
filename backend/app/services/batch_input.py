"""
services/batch_input.py
-----------------------
Backend ingestion service for the frozen investigation_batch structure:
investigation_batch/
â”œâ”€â”€ scans/
â”‚   â”œâ”€â”€ scan_001.jpg
â”‚   â””â”€â”€ ...
â””â”€â”€ navigation.csv

Locked navigation.csv format (one row per detection geometry):
    scan_file,latitude,longitude,heading,altitude,timestamp,data_source,
    detection_index,target_range_m,range_type,relative_bearing

Design rules:
- Multiple rows may share the same scan_file (one row per target detection).
- Scan-level metadata (latitude, longitude, heading, altitude, timestamp,
  data_source) must be identical across all rows for the same scan_file.
- detection_index maps directly to the zero-based YOLO detection index.
- A zero-detection scan is represented by exactly ONE row where
  detection_index, target_range_m, range_type, and relative_bearing are
  all empty.
- A scan cannot mix zero-detection rows and detection rows.
- range_type must be exactly "SLANT" or "GROUND" â€” never silently defaulted.
"""

from __future__ import annotations

import csv
import io
import math
import os
import shutil
import tempfile
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.analysis import run_batch_analysis
from app.config import settings
from app.detection.yolo_detector import YOLODetector
from app.schemas.report import BatchAnalysisResult, ScanInput, TargetGeometryInput
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png"}

REQUIRED_CSV_COLUMNS: set[str] = {
    "scan_file",
    "latitude",
    "longitude",
    "heading",
    "altitude",
    "timestamp",
    "data_source",
}

# Scan-level fields that must be identical across all repeated rows for the
# same scan_file.
SCAN_LEVEL_FIELDS: tuple[str, ...] = (
    "latitude",
    "longitude",
    "heading",
    "altitude",
    "timestamp",
    "data_source",
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_row_scan_level(row: dict[str, str], row_idx: int) -> dict[str, Any]:
    """
    Parse and validate the scan-level fields from a single normalised CSV row.
    Returns a dict with typed values for scan-level keys.
    Raises HTTPException(422) on any validation failure.
    """
    # 1. latitude
    lat_str = row.get("latitude", "")
    try:
        latitude = float(lat_str)
        if math.isnan(latitude) or not (-90.0 <= latitude <= 90.0):
            raise ValueError()
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Row {row_idx}: invalid latitude '{lat_str}'. Must be a float between -90.0 and 90.0.",
        )

    # 2. longitude
    lon_str = row.get("longitude", "")
    try:
        longitude = float(lon_str)
        if math.isnan(longitude) or not (-180.0 <= longitude <= 180.0):
            raise ValueError()
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Row {row_idx}: invalid longitude '{lon_str}'. Must be a float between -180.0 and 180.0.",
        )

    # 3. heading [0.0, 360.0)
    hdg_str = row.get("heading", "")
    try:
        heading = float(hdg_str)
        if math.isnan(heading) or not (0.0 <= heading < 360.0):
            raise ValueError()
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Row {row_idx}: invalid heading '{hdg_str}'. Must be a float in [0.0, 360.0).",
        )

    # 4. altitude > 0.0
    alt_str = row.get("altitude", "")
    try:
        altitude = float(alt_str)
        if math.isnan(altitude) or altitude <= 0.0:
            raise ValueError()
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Row {row_idx}: invalid altitude '{alt_str}'. Must be a positive float (> 0.0).",
        )

    # 5. timestamp (ISO-8601)
    ts_str = row.get("timestamp", "")
    try:
        clean_ts = ts_str
        if clean_ts.endswith("Z"):
            clean_ts = clean_ts[:-1] + "+00:00"
        timestamp = datetime.fromisoformat(clean_ts)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
    except Exception:
        raise HTTPException(
            status_code=422,
            detail=f"Row {row_idx}: invalid timestamp format '{ts_str}'. Must be valid ISO-8601.",
        )

    # 6. data_source
    ds_str = row.get("data_source", "").strip().upper()
    if ds_str not in {"REAL", "DEMO", "SIMULATED"}:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Row {row_idx}: invalid data_source '{row.get('data_source')}'. "
                f"Supported values: REAL, DEMO, SIMULATED."
            ),
        )

    return {
        "latitude": latitude,
        "longitude": longitude,
        "heading": heading,
        "altitude": altitude,
        "timestamp": timestamp,
        "data_source": ds_str,
    }


def _parse_detection_row(
    row: dict[str, str],
    row_idx: int,
) -> "TargetGeometryInput | None":
    """
    Parse detection-level fields from a normalised CSV row.

    Returns:
    - None                   -- zero-detection row (all four detection fields empty)
    - TargetGeometryInput    -- fully populated detection geometry

    Raises HTTPException(422) on partial or invalid detection data.
    range_type is NEVER silently defaulted -- it must be exactly SLANT or GROUND.
    """
    det_idx_raw = row.get("detection_index", "").strip()
    range_raw = row.get("target_range_m", "").strip()
    range_type_raw = row.get("range_type", "").strip()
    bearing_raw = row.get("relative_bearing", "").strip()

    all_empty = not det_idx_raw and not range_raw and not range_type_raw and not bearing_raw

    if all_empty:
        return None  # zero-detection marker row

    # At least one field is non-empty -- all four are required together.
    partial_fields = {
        "detection_index": det_idx_raw,
        "target_range_m": range_raw,
        "range_type": range_type_raw,
        "relative_bearing": bearing_raw,
    }
    missing = [k for k, v in partial_fields.items() if not v]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Row {row_idx}: partial detection geometry -- "
                f"missing field(s): {', '.join(sorted(missing))}. "
                f"All four detection fields must be provided together."
            ),
        )

    # Parse detection_index (non-negative integer)
    try:
        det_idx = int(det_idx_raw)
        if det_idx < 0:
            raise ValueError()
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Row {row_idx}: invalid detection_index '{det_idx_raw}'. Must be a non-negative integer.",
        )

    # Parse target_range_m (non-negative float)
    try:
        target_range = float(range_raw)
        if math.isnan(target_range) or target_range < 0.0:
            raise ValueError()
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Row {row_idx}: invalid target_range_m '{range_raw}'. Must be a non-negative float.",
        )

    # Parse range_type -- strict, never silently defaulted
    range_type_upper = range_type_raw.upper()
    if range_type_upper not in {"SLANT", "GROUND"}:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Row {row_idx}: invalid range_type '{range_type_raw}'. "
                f"Must be exactly 'SLANT' or 'GROUND'."
            ),
        )

    # Parse relative_bearing (numeric float)
    try:
        rel_bearing = float(bearing_raw)
        if math.isnan(rel_bearing):
            raise ValueError()
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Row {row_idx}: invalid relative_bearing '{bearing_raw}'. Must be a numeric float.",
        )

    return TargetGeometryInput(
        detection_index=det_idx,
        range=target_range,
        range_type=range_type_upper,
        relative_bearing=rel_bearing,
    )


# ---------------------------------------------------------------------------
# CSV Parsing & Validation
# ---------------------------------------------------------------------------


def validate_and_parse_navigation_csv(csv_content: "str | bytes") -> "list[dict[str, Any]]":
    """
    Validate and parse the locked navigation.csv format.

    Multiple rows may share the same scan_file (one row per detection geometry).
    Rows are grouped by scan_file; scan-level metadata must be consistent across
    all rows for a given scan; detection rows are collected into TargetGeometryInput
    objects and placed into target_geometries.

    Returns a list of per-scan summary dicts, each containing:
        scan_file, latitude, longitude, heading, altitude, timestamp,
        data_source, scan_identity, notes, target_geometries

    Raises HTTPException(422) on any validation failure.
    """
    if isinstance(csv_content, bytes):
        try:
            text = csv_content.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                text = csv_content.decode("latin-1")
            except Exception as err:
                raise HTTPException(
                    status_code=422,
                    detail=f"Unable to decode navigation.csv text encoding: {err}",
                ) from err
    else:
        text = csv_content

    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="navigation.csv is empty.",
        )

    stream = io.StringIO(text)
    reader = csv.DictReader(stream)

    if not reader.fieldnames:
        raise HTTPException(
            status_code=422,
            detail="navigation.csv has no header row.",
        )

    # Normalize column names: strip whitespace and lowercase
    raw_to_norm: dict[str, str] = {
        col: col.strip().lower() for col in reader.fieldnames if col is not None
    }
    norm_columns = set(raw_to_norm.values())

    missing_cols = REQUIRED_CSV_COLUMNS - norm_columns
    if missing_cols:
        raise HTTPException(
            status_code=422,
            detail=f"navigation.csv missing required column(s): {', '.join(sorted(missing_cols))}",
        )

    # -----------------------------------------------------------------------
    # Pass 1: parse every raw row and group by scan_file (preserving insertion
    # order for deterministic output).
    # -----------------------------------------------------------------------
    grouped_rows: dict[str, list] = {}
    row_idx_map: dict[str, list] = defaultdict(list)

    for row_idx, raw_row in enumerate(reader, start=2):  # line 1 is header
        row: dict[str, str] = {}
        for raw_k, raw_v in raw_row.items():
            if raw_k in raw_to_norm:
                row[raw_to_norm[raw_k]] = (raw_v or "").strip()

        # --- scan_file validation ---
        scan_file = row.get("scan_file", "").strip()
        if not scan_file:
            raise HTTPException(
                status_code=422,
                detail=f"Row {row_idx}: 'scan_file' cannot be empty.",
            )

        # Path traversal security check
        base_name = os.path.basename(scan_file.replace("\\", "/"))
        if base_name != scan_file or ".." in scan_file or "/" in scan_file or "\\" in scan_file:
            raise HTTPException(
                status_code=422,
                detail=f"Row {row_idx}: invalid scan_file path '{scan_file}'. Path traversal is not permitted.",
            )

        # Extension check
        ext = os.path.splitext(scan_file)[1].lower()
        if ext not in SUPPORTED_IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Row {row_idx}: unsupported image format '{ext}' for file '{scan_file}'. "
                    f"Supported formats: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
                ),
            )

        # Parse scan-level metadata
        scan_meta = _parse_row_scan_level(row, row_idx)
        scan_meta["scan_file"] = scan_file
        scan_meta["notes"] = row.get("notes") or None
        scan_meta["scan_identity"] = row.get("scan_identity") or Path(scan_file).stem

        # Parse detection-level columns (None = zero-detection marker)
        detection_geom = _parse_detection_row(row, row_idx)
        scan_meta["_detection_geom"] = detection_geom

        if scan_file not in grouped_rows:
            grouped_rows[scan_file] = []
        grouped_rows[scan_file].append(scan_meta)
        row_idx_map[scan_file].append(row_idx)

    if not grouped_rows:
        raise HTTPException(
            status_code=422,
            detail="navigation.csv contains no scan records (empty batch).",
        )

    # -----------------------------------------------------------------------
    # Pass 2: validate per-scan consistency and build final scan summaries.
    # -----------------------------------------------------------------------
    scan_summaries: list[dict[str, Any]] = []

    for scan_file, rows in grouped_rows.items():
        row_numbers = row_idx_map[scan_file]

        # Verify scan-level metadata is consistent across all rows for this scan
        first = rows[0]
        for i, subsequent in enumerate(rows[1:], start=1):
            for field in SCAN_LEVEL_FIELDS:
                if first[field] != subsequent[field]:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"Inconsistent scan-level metadata for '{scan_file}': "
                            f"field '{field}' differs between row {row_numbers[0]} "
                            f"('{first[field]}') and row {row_numbers[i]} "
                            f"('{subsequent[field]}')."
                        ),
                    )

        # Collect detection geometries
        geom_entries = [r["_detection_geom"] for r in rows]

        has_zero_det_row = any(g is None for g in geom_entries)
        has_det_rows = any(g is not None for g in geom_entries)

        # A scan cannot mix zero-detection rows and detection rows
        if has_zero_det_row and has_det_rows:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"'{scan_file}': cannot mix a zero-detection row with "
                    f"detection rows for the same scan."
                ),
            )

        if has_zero_det_row:
            if len(rows) > 1:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"'{scan_file}': zero-detection scan must have exactly one "
                        f"row in navigation.csv, but {len(rows)} rows were found."
                    ),
                )
            target_geometries: list = []
        else:
            target_geometries = [g for g in geom_entries if g is not None]
            indices = [tg.detection_index for tg in target_geometries]
            if len(indices) != len(set(indices)):
                seen_set: set = set()
                for idx, row_no in zip(indices, row_numbers):
                    if idx in seen_set:
                        raise HTTPException(
                            status_code=422,
                            detail=(
                                f"'{scan_file}': duplicate detection_index {idx} "
                                f"at row {row_no}."
                            ),
                        )
                    seen_set.add(idx)

        scan_summaries.append(
            {
                "scan_file": scan_file,
                "latitude": first["latitude"],
                "longitude": first["longitude"],
                "heading": first["heading"],
                "altitude": first["altitude"],
                "timestamp": first["timestamp"],
                "data_source": first["data_source"],
                "scan_identity": first["scan_identity"],
                "notes": first["notes"],
                "target_geometries": target_geometries,
            }
        )

    return scan_summaries


# ---------------------------------------------------------------------------
# Batch File Relationship Validation
# ---------------------------------------------------------------------------


def validate_batch_files(
    scan_summaries: "list[dict[str, Any]]",
    uploaded_image_names: "set[str]",
) -> None:
    """
    Validate that navigation.csv records and uploaded image files have an exact
    1-to-1 match (by unique scan_file).
    """
    csv_files = {s["scan_file"] for s in scan_summaries}

    missing_images = csv_files - uploaded_image_names
    if missing_images:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Navigation metadata references missing image file(s): "
                f"{', '.join(sorted(missing_images))}"
            ),
        )

    unreferenced_images = uploaded_image_names - csv_files
    if unreferenced_images:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Uploaded image file(s) lack corresponding navigation metadata: "
                f"{', '.join(sorted(unreferenced_images))}"
            ),
        )


# ---------------------------------------------------------------------------
# Investigation Batch Processor
# ---------------------------------------------------------------------------


def persist_batch_scan_image(
    src_path: Path,
    original_filename: str,
    target_storage_dir: Path,
) -> Path:
    """
    Persist an uploaded scan image from staging into persistent application storage.

    - Protects against path traversal.
    - Avoids filename collisions while preserving the original file extension.
    - Ensures destination is strictly inside target_storage_dir.
    """
    clean_name = os.path.basename(original_filename.replace("\\", "/")).strip()
    if (
        not clean_name
        or ".." in original_filename
        or "/" in original_filename
        or "\\" in original_filename
    ):
        if clean_name != original_filename:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid filename '{original_filename}'. Path traversal is not permitted.",
            )

    target_storage_dir = Path(target_storage_dir).resolve()
    target_storage_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(clean_name).stem
    ext = Path(clean_name).suffix.lower()
    if not ext:
        ext = ".jpg"

    dest_path = (target_storage_dir / f"{stem}{ext}").resolve()
    if dest_path.exists():
        unique_suffix = uuid.uuid4().hex[:8]
        dest_path = (target_storage_dir / f"{stem}_{unique_suffix}{ext}").resolve()
        while dest_path.exists():
            unique_suffix = uuid.uuid4().hex[:8]
            dest_path = (target_storage_dir / f"{stem}_{unique_suffix}{ext}").resolve()

    try:
        dest_path.relative_to(target_storage_dir)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="Path traversal is not permitted.",
        )

    shutil.copy2(src_path, dest_path)
    return dest_path


async def process_investigation_batch(
    uploaded_files: "list[UploadFile]",
    db: Session,
    detector: YOLODetector,
    storage_dir: "Path | None" = None,
) -> BatchAnalysisResult:
    """
    Full pipeline to validate and process an investigation batch.

    1. Separates navigation.csv from sonar image files.
    2. Reads and validates the locked navigation.csv format.
    3. Groups multi-row CSV entries by scan_file.
    4. Validates scan-level metadata consistency across repeated rows.
    5. Builds TargetGeometryInput objects from detection rows.
    6. Validates bidirectional 1-to-1 image <-> CSV matching.
    7. Stages uploads in a controlled temporary work directory.
    8. Copies uploaded images to persistent application storage (e.g. data/scans/).
    9. Converts each scan summary into a ScanInput with persistent image paths.
    10. Executes the existing sequential analysis pipeline.
    11. Cleans up the temporary staging directory on both success and failure.
    12. Returns a BatchAnalysisResult response.
    """
    if not uploaded_files:
        raise HTTPException(
            status_code=422,
            detail="Empty batch upload: no files provided.",
        )

    csv_file: "UploadFile | None" = None
    image_files: "dict[str, UploadFile]" = {}

    for f in uploaded_files:
        if not f.filename:
            continue
        clean_name = os.path.basename(f.filename.replace("\\", "/")).strip()
        if not clean_name:
            continue

        if clean_name.lower() == "navigation.csv" or clean_name.lower().endswith(".csv"):
            if csv_file is not None:
                raise HTTPException(
                    status_code=422,
                    detail="Multiple CSV files uploaded. Exactly one navigation.csv is expected.",
                )
            csv_file = f
        else:
            ext = os.path.splitext(clean_name)[1].lower()
            if ext not in SUPPORTED_IMAGE_EXTENSIONS:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Unsupported file format '{ext}' for file '{clean_name}'. "
                        f"Supported formats: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
                    ),
                )
            if clean_name in image_files:
                raise HTTPException(
                    status_code=422,
                    detail=f"Duplicate image file '{clean_name}' uploaded in batch.",
                )
            image_files[clean_name] = f

    if csv_file is None:
        raise HTTPException(
            status_code=422,
            detail="Missing navigation.csv in uploaded batch.",
        )

    if not image_files:
        raise HTTPException(
            status_code=422,
            detail="Batch contains no sonar image files (empty batch).",
        )

    csv_bytes = await csv_file.read()
    scan_summaries = validate_and_parse_navigation_csv(csv_bytes)
    validate_batch_files(scan_summaries, set(image_files.keys()))

    work_dir = Path(tempfile.mkdtemp(prefix="investigation_batch_"))
    scans_dir = work_dir / "scans"
    scans_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Stage uploaded files into temporary staging directory
        staged_paths: dict[str, Path] = {}
        for filename, upload_file in image_files.items():
            dest_file = scans_dir / filename
            content = await upload_file.read()
            dest_file.write_bytes(content)
            staged_paths[filename] = dest_file

        # Copy uploaded images to persistent storage under data/scans
        persistent_storage_dir = Path(storage_dir or settings.scans_dir).resolve()
        persistent_storage_dir.mkdir(parents=True, exist_ok=True)

        persistent_paths: dict[str, str] = {}
        for filename, staged_path in staged_paths.items():
            saved_file = persist_batch_scan_image(
                staged_path, filename, persistent_storage_dir
            )
            persistent_paths[filename] = str(saved_file)

        scan_inputs: "list[ScanInput]" = []
        for summary in scan_summaries:
            filename = summary["scan_file"]
            img_path = persistent_paths[filename]

            scan_in = ScanInput(
                image_path=img_path,
                sonar_latitude=summary["latitude"],
                sonar_longitude=summary["longitude"],
                heading=summary["heading"],
                altitude=summary["altitude"],
                range=None,
                range_type="SLANT",
                relative_bearing=None,
                timestamp=summary["timestamp"],
                data_source=summary["data_source"],
                scan_identity=summary["scan_identity"],
                notes=summary["notes"],
                target_geometries=summary["target_geometries"],
            )
            scan_inputs.append(scan_in)

        result = run_batch_analysis(scan_inputs, db, detector)
        return result

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


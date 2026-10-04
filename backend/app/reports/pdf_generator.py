"""
reports/pdf_generator.py
------------------------
Professional PDF investigation report generation for DRISHTI-SSS sonar scans.

Uses ReportLab Platypus to generate multi-page, deterministic PDF reports
incorporating:
- Report Header & Classification
- Rigorous Provenance & Methodology Statement
- Scan Information
- Sonar Platform Navigation State
- Detection Summary
- Detailed Detection Records (AI-predicted observables vs deterministic geolocation)
- Georeferenced Coordinates Table
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.db.models import InvestigationBatch, Scan as DB_Scan


# ---------------------------------------------------------------------------
# Custom Canvas for Two-Pass Dynamic Page Numbering & Running Decorators
# ---------------------------------------------------------------------------


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas that writes running header and footer with total page count.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._saved_page_states: list[dict[str, Any]] = []

    def showPage(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_decorations(num_pages)
            super().showPage()
        super().save()

    def _draw_decorations(self, page_count: int) -> None:
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))

        # Running top header
        self.drawString(36, 805, "DRISHTI-SSS | Autonomous Sonar Investigation System")
        self.drawRightString(559, 805, "UNCLASSIFIED")
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(36, 800, 559, 800)

        # Running bottom footer
        self.line(36, 45, 559, 45)
        self.drawString(
            36, 33, "CONFIDENTIAL / INTERNAL USE — Geolocation Investigation Report"
        )
        self.drawRightString(559, 33, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


# ---------------------------------------------------------------------------
# PDF Report Generator
# ---------------------------------------------------------------------------


def generate_scan_pdf_report(scan: DB_Scan) -> bytes:
    """
    Build and return a binary PDF investigation report for the given DB Scan.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=50,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#1A365D")   # Deep navy
    secondary_color = colors.HexColor("#2B6CB0") # Slate blue
    dark_neutral = colors.HexColor("#2D3748")    # Charcoal body
    light_bg = colors.HexColor("#F7FAFC")        # Off-white

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=primary_color,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=secondary_color,
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=primary_color,
        spaceBefore=10,
        spaceAfter=4,
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=secondary_color,
        spaceBefore=6,
        spaceAfter=3,
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=dark_neutral,
    )

    body_bold = ParagraphStyle(
        "BodyBold_Custom",
        parent=body_style,
        fontName="Helvetica-Bold",
    )

    meta_label_style = ParagraphStyle(
        "MetaLabel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#4A5568"),
    )

    meta_val_style = ParagraphStyle(
        "MetaVal",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=dark_neutral,
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=dark_neutral,
    )

    provenance_box_style = ParagraphStyle(
        "ProvenanceBox",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11.5,
        textColor=colors.HexColor("#2C5282"),
    )

    story: list[Any] = []

    # ───────────────────────────────────────────────────────────────────────────
    # A. Report Header
    # ───────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("DRISHTI-SSS", title_style))
    story.append(Paragraph("Sonar Investigation Report", subtitle_style))
    story.append(Spacer(1, 6))

    # Header metadata badges
    gen_time_str = (
        scan.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
        if scan.timestamp
        else datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    )
    header_meta_data = [
        [
            Paragraph("<b>Report Type:</b> INVESTIGATION_REPORT", meta_label_style),
            Paragraph(f"<b>Generated At:</b> {gen_time_str}", meta_label_style),
            Paragraph("<b>Classification:</b> UNCLASSIFIED", meta_label_style),
        ]
    ]
    t_header_meta = Table(header_meta_data, colWidths=[170, 200, 150])
    t_header_meta.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EDF2F7")),
                ("PADDING", (0, 0), (-1, -1), 5),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(t_header_meta)
    story.append(Spacer(1, 10))

    # ───────────────────────────────────────────────────────────────────────────
    # B. Provenance / Methodology
    # ───────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("Provenance &amp; Methodology", h1_style))
    story.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=primary_color,
            spaceBefore=1,
            spaceAfter=6,
        )
    )

    provenance_text = (
        "<b>Operational Provenance Distinction:</b><br/>"
        "• <b>Navigation Data:</b> Measured / input platform sensor observables (GNSS, gyrocompass, altimeter).<br/>"
        "• <b>Detection Data:</b> AI-derived object classification and bounding boxes produced by YOLO neural network.<br/>"
        "• <b>Geolocation Data:</b> Deterministically derived via closed-form trigonometry (slant-to-ground range, "
        "relative bearing, platform heading, and destination point projection).<br/>"
        "<b>Notice:</b> Target latitude and longitude coordinates are <u>NOT predicted by AI</u>; they are calculated "
        "deterministically from measured navigation and geometry inputs."
    )
    t_provenance = Table(
        [[Paragraph(provenance_text, provenance_box_style)]],
        colWidths=[520],
    )
    t_provenance.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EBF8FF")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#63B3ED")),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(t_provenance)
    story.append(Spacer(1, 10))

    # ───────────────────────────────────────────────────────────────────────────
    # C. Scan Information & D. Sonar Navigation
    # ───────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("Scan Information &amp; Sonar Navigation", h1_style))
    story.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=primary_color,
            spaceBefore=1,
            spaceAfter=6,
        )
    )

    data_source_str = (
        scan.data_source.value
        if hasattr(scan.data_source, "value")
        else str(scan.data_source)
    ).upper()
    range_type_str = (
        scan.range_type.value
        if hasattr(scan.range_type, "value")
        else str(scan.range_type)
    ).upper()
    scan_ts_str = (
        scan.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
        if scan.timestamp
        else "N/A"
    )

    scan_info_data = [
        [
            Paragraph("Scan ID", meta_label_style),
            Paragraph(str(scan.id), meta_val_style),
            Paragraph("Sonar Latitude", meta_label_style),
            Paragraph(f"{scan.sonar_latitude:.6f}°", meta_val_style),
        ],
        [
            Paragraph("Scan Identity", meta_label_style),
            Paragraph(str(scan.scan_identity), meta_val_style),
            Paragraph("Sonar Longitude", meta_label_style),
            Paragraph(f"{scan.sonar_longitude:.6f}°", meta_val_style),
        ],
        [
            Paragraph("Timestamp", meta_label_style),
            Paragraph(scan_ts_str, meta_val_style),
            Paragraph("Heading", meta_label_style),
            Paragraph(f"{scan.heading:.1f}° True", meta_val_style),
        ],
        [
            Paragraph("Data Source", meta_label_style),
            Paragraph(data_source_str, meta_val_style),
            Paragraph("Altitude", meta_label_style),
            Paragraph(f"{scan.altitude:.2f} m", meta_val_style),
        ],
        [
            Paragraph("Image Path", meta_label_style),
            Paragraph(str(scan.image_path), meta_val_style),
            Paragraph("Range Observable", meta_label_style),
            Paragraph(
                f"{scan.range_m:.2f} m ({range_type_str})"
                if scan.range_m is not None
                else f"N/A ({range_type_str})",
                meta_val_style,
            ),
        ],
        [
            Paragraph("Operator Notes", meta_label_style),
            Paragraph(str(scan.notes) if scan.notes else "None", meta_val_style),
            Paragraph("Relative Bearing", meta_label_style),
            Paragraph(
                f"{scan.relative_bearing:.1f}°"
                if scan.relative_bearing is not None
                else "N/A",
                meta_val_style,
            ),
        ],
    ]

    t_scan_info = Table(scan_info_data, colWidths=[100, 160, 110, 150])
    t_scan_info.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F7FAFC")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#F7FAFC")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t_scan_info)
    story.append(Spacer(1, 10))

    # ───────────────────────────────────────────────────────────────────────────
    # E. Detection Summary
    # ───────────────────────────────────────────────────────────────────────────
    detections_list = sorted(
        scan.detections or [],
        key=lambda d: getattr(d, "id", 0) or 0,
    )
    det_count = len(detections_list)

    story.append(Paragraph("Detection Summary", h1_style))
    story.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=primary_color,
            spaceBefore=1,
            spaceAfter=6,
        )
    )

    if det_count == 0:
        zero_det_table = Table(
            [[Paragraph("No detections found for this scan.", body_bold)]],
            colWidths=[520],
        )
        zero_det_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EDF2F7")),
                    ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
                    ("PADDING", (0, 0), (-1, -1), 10),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ]
            )
        )
        story.append(zero_det_table)
        story.append(Spacer(1, 10))
    else:
        # Summary metrics
        class_counts: dict[str, int] = {}
        conf_values: list[float] = []
        for d in detections_list:
            class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1
            conf_values.append(d.confidence)

        classes_summary_str = ", ".join(
            f"{c} ({cnt})" for c, cnt in sorted(class_counts.items())
        )
        avg_conf = sum(conf_values) / len(conf_values) if conf_values else 0.0
        min_conf = min(conf_values) if conf_values else 0.0
        max_conf = max(conf_values) if conf_values else 0.0

        summary_rows = [
            [
                Paragraph("Total Detections", meta_label_style),
                Paragraph(str(det_count), meta_val_style),
                Paragraph("Class Breakdown", meta_label_style),
                Paragraph(classes_summary_str, meta_val_style),
            ],
            [
                Paragraph("Confidence Range", meta_label_style),
                Paragraph(f"{min_conf:.1%} – {max_conf:.1%}", meta_val_style),
                Paragraph("Average Confidence", meta_label_style),
                Paragraph(f"{avg_conf:.1%}", meta_val_style),
            ],
        ]
        t_det_summary = Table(summary_rows, colWidths=[110, 150, 110, 150])
        t_det_summary.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F7FAFC")),
                    ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#F7FAFC")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("PADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(t_det_summary)
        story.append(Spacer(1, 10))

    # ───────────────────────────────────────────────────────────────────────────
    # F. Detection Details (Per Detection Breakdown)
    # ───────────────────────────────────────────────────────────────────────────
    if det_count > 0:
        story.append(Paragraph("Detection Details", h1_style))
        story.append(
            HRFlowable(
                width="100%",
                thickness=1,
                color=primary_color,
                spaceBefore=1,
                spaceAfter=6,
            )
        )

        for idx, d in enumerate(detections_list):
            det_num = idx + 1
            lat_str = (
                f"{d.target_latitude:.6f}°"
                if d.target_latitude is not None
                else "Not available"
            )
            lon_str = (
                f"{d.target_longitude:.6f}°"
                if d.target_longitude is not None
                else "Not available"
            )
            range_str = (
                f"{d.range_m:.2f} m"
                if d.range_m is not None
                else "Not available"
            )
            ground_range_str = (
                f"{d.ground_range_m:.2f} m"
                if d.ground_range_m is not None
                else "Not available"
            )
            rel_bearing_str = (
                f"{d.relative_bearing:.1f}°"
                if d.relative_bearing is not None
                else "Not available"
            )
            abs_bearing_str = (
                f"{d.absolute_bearing:.1f}°"
                if d.absolute_bearing is not None
                else "Not available"
            )
            bbox_str = f"[{d.bbox_x1:.1f}, {d.bbox_y1:.1f}, {d.bbox_x2:.1f}, {d.bbox_y2:.1f}]"

            det_rows = [
                [
                    Paragraph("Detection #", meta_label_style),
                    Paragraph(f"<b>#{det_num}</b> (ID: {d.id})", meta_val_style),
                    Paragraph("Class Label", meta_label_style),
                    Paragraph(f"<b>{d.class_name}</b>", meta_val_style),
                ],
                [
                    Paragraph("Confidence", meta_label_style),
                    Paragraph(f"{d.confidence:.1%}", meta_val_style),
                    Paragraph("Bounding Box (px)", meta_label_style),
                    Paragraph(bbox_str, meta_val_style),
                ],
                [
                    Paragraph("Range (Slant)", meta_label_style),
                    Paragraph(range_str, meta_val_style),
                    Paragraph("Ground Range", meta_label_style),
                    Paragraph(ground_range_str, meta_val_style),
                ],
                [
                    Paragraph("Range Type", meta_label_style),
                    Paragraph(range_type_str, meta_val_style),
                    Paragraph("Relative Bearing", meta_label_style),
                    Paragraph(rel_bearing_str, meta_val_style),
                ],
                [
                    Paragraph("Absolute Bearing", meta_label_style),
                    Paragraph(abs_bearing_str, meta_val_style),
                    Paragraph("Target Coordinates", meta_label_style),
                    Paragraph(f"Lat: {lat_str}<br/>Lon: {lon_str}", meta_val_style),
                ],
                [
                    Paragraph("Detection Source", meta_label_style),
                    Paragraph("AI_YOLO (Object Detection Model)", meta_val_style),
                    Paragraph("Geolocation Method", meta_label_style),
                    Paragraph("DETERMINISTIC_GEOMETRY (Trigonometric projection)", meta_val_style),
                ],
            ]

            t_single_det = Table(det_rows, colWidths=[110, 150, 110, 150])
            t_single_det.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F7FAFC")),
                        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#F7FAFC")),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("PADDING", (0, 0), (-1, -1), 3.5),
                    ]
                )
            )

            det_block = [
                Paragraph(f"Detection #{det_num}: {d.class_name}", h2_style),
                t_single_det,
                Spacer(1, 6),
            ]
            story.append(KeepTogether(det_block))

    # ───────────────────────────────────────────────────────────────────────────
    # G. Geolocation Results Table
    # ───────────────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 4))
    story.append(Paragraph("Geolocation Results", h1_style))
    story.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=primary_color,
            spaceBefore=1,
            spaceAfter=6,
        )
    )

    if det_count == 0:
        no_geo_table = Table(
            [[Paragraph("No detections found for this scan.", body_style)]],
            colWidths=[520],
        )
        no_geo_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EDF2F7")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                    ("PADDING", (0, 0), (-1, -1), 8),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ]
            )
        )
        story.append(no_geo_table)
    else:
        # Table columns: Detection (40), Class (95), Latitude (100), Longitude (100), Range (90), Ground Range (95)
        # Total = 520
        geo_headers = [
            Paragraph("<b>Det #</b>", table_header_style),
            Paragraph("<b>Class</b>", table_header_style),
            Paragraph("<b>Latitude (WGS-84)</b>", table_header_style),
            Paragraph("<b>Longitude (WGS-84)</b>", table_header_style),
            Paragraph("<b>Range (m)</b>", table_header_style),
            Paragraph("<b>Ground Range (m)</b>", table_header_style),
        ]
        geo_table_data = [geo_headers]

        for idx, d in enumerate(detections_list):
            det_label = f"#{idx + 1}"
            lat_display = (
                f"{d.target_latitude:.6f}°"
                if d.target_latitude is not None
                else "Not available"
            )
            lon_display = (
                f"{d.target_longitude:.6f}°"
                if d.target_longitude is not None
                else "Not available"
            )
            range_display = (
                f"{d.range_m:.2f}"
                if d.range_m is not None
                else "Not available"
            )
            ground_range_display = (
                f"{d.ground_range_m:.2f}"
                if d.ground_range_m is not None
                else "Not available"
            )

            geo_table_data.append(
                [
                    Paragraph(det_label, table_cell_style),
                    Paragraph(d.class_name, table_cell_style),
                    Paragraph(lat_display, table_cell_style),
                    Paragraph(lon_display, table_cell_style),
                    Paragraph(range_display, table_cell_style),
                    Paragraph(ground_range_display, table_cell_style),
                ]
            )

        t_geo_results = Table(
            geo_table_data,
            colWidths=[40, 95, 100, 100, 90, 95],
            repeatRows=1,
        )
        t_geo_results.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), secondary_color),
                    ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                    ("PADDING", (0, 0), (-1, -1), 4),
                    *(
                        [
                            ("BACKGROUND", (0, r), (-1, r), colors.HexColor("#F7FAFC"))
                            for r in range(2, len(geo_table_data), 2)
                        ]
                    ),
                ]
            )
        )
        story.append(t_geo_results)

    # Build PDF with dynamic header/footer decorator
    doc.build(story, canvasmaker=NumberedCanvas)
    return buffer.getvalue()


def generate_batch_pdf_report(batch: InvestigationBatch) -> bytes:
    """
    Generate a deterministic, professional PDF investigation report for an entire batch of sonar scans.

    Includes:
    - Report Header & UNCLASSIFIED status
    - Investigation Batch Information & Provenance
    - Rigorous Provenance & Methodology Statement (Deterministic geometry vs AI detection)
    - Survey Summary & Detection Class Breakdown
    - Scan Summary Table (including zero-detection scans clearly represented)
    - Georeferenced Coordinates and Detection Details Table
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    primary_color = colors.HexColor("#0D3B66")
    secondary_color = colors.HexColor("#006494")
    accent_color = colors.HexColor("#1B998B")
    dark_neutral = colors.HexColor("#1A202C")

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "BatchTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=primary_color,
    )

    subtitle_style = ParagraphStyle(
        "BatchSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        textColor=secondary_color,
    )

    section_heading_style = ParagraphStyle(
        "BatchSectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=primary_color,
        keepWithNext=True,
    )

    meta_label_style = ParagraphStyle(
        "BatchMetaLabel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#4A5568"),
    )

    meta_val_style = ParagraphStyle(
        "BatchMetaVal",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=dark_neutral,
    )

    table_header_style = ParagraphStyle(
        "BatchTableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
    )

    table_cell_style = ParagraphStyle(
        "BatchTableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=dark_neutral,
    )

    provenance_box_style = ParagraphStyle(
        "BatchProvenanceBox",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11.5,
        textColor=colors.HexColor("#2C5282"),
    )

    story: list[Any] = []

    # 1. Report Header
    story.append(Paragraph("DRISHTI-SSS", title_style))
    story.append(Paragraph("Sonar Investigation Report — Multi-Scan Batch", subtitle_style))
    story.append(Spacer(1, 6))

    gen_time_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    header_meta_data = [
        [
            Paragraph("<b>Report Type:</b> BATCH_INVESTIGATION_REPORT", meta_label_style),
            Paragraph(f"<b>Generated At:</b> {gen_time_str}", meta_label_style),
            Paragraph("<b>Classification:</b> UNCLASSIFIED", meta_label_style),
        ]
    ]
    t_header_meta = Table(header_meta_data, colWidths=[180, 190, 150])
    t_header_meta.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EDF2F7")),
                ("PADDING", (0, 0), (-1, -1), 5),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(t_header_meta)
    story.append(Spacer(1, 10))

    # 2. Provenance Disclosure
    prov_text = (
        "<b>OPERATIONAL &amp; SCIENTIFIC PROVENANCE DISCLOSURE:</b><br/>"
        "• <b>Platform Navigation:</b> Sourced from measured navigation records (GNSS, gyrocompass, altimeter).<br/>"
        "• <b>Target Detections:</b> AI-derived object classifications and bounding boxes inferred via neural network.<br/>"
        "• <b>Deterministic Georeferencing:</b> Target geographic coordinates (latitude, longitude) and horizontal ground ranges "
        "are computed using rigorous mathematical trigonometry projecting platform position, heading, and sensor slant/ground range observables. "
        "<b>Target coordinates are strictly deterministic geometry-derived calculations and are NOT predicted or estimated by artificial intelligence.</b>"
    )
    t_prov = Table([[Paragraph(prov_text, provenance_box_style)]], colWidths=[520])
    t_prov.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EBF8FF")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#3182CE")),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(t_prov)
    story.append(Spacer(1, 10))

    # 3. Investigation Batch Overview
    story.append(Paragraph("1. Investigation Batch Overview", section_heading_style))
    story.append(Spacer(1, 4))

    ds_str = (
        batch.data_source.value
        if hasattr(batch.data_source, "value")
        else str(batch.data_source)
    )
    created_str = (
        batch.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        if batch.created_at
        else "N/A"
    )

    scans = sorted(batch.scans or [], key=lambda s: getattr(s, "id", 0) or 0)
    total_scans = batch.total_scans
    successful_scans = len(scans)

    all_detections: list[tuple[Any, Any]] = []
    class_counts: dict[str, int] = {}
    for s in scans:
        for d in (s.detections or []):
            all_detections.append((s, d))
            class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1

    batch_info_data = [
        [
            Paragraph("Batch Identifier:", meta_label_style),
            Paragraph(f"<b>{batch.batch_id}</b>", meta_val_style),
            Paragraph("Created / Ingested:", meta_label_style),
            Paragraph(created_str, meta_val_style),
        ],
        [
            Paragraph("Data Source Provenance:", meta_label_style),
            Paragraph(f"<b>{ds_str}</b>", meta_val_style),
            Paragraph("Total Scans Ingested:", meta_label_style),
            Paragraph(str(total_scans), meta_val_style),
        ],
        [
            Paragraph("Successful Scans:", meta_label_style),
            Paragraph(str(successful_scans), meta_val_style),
            Paragraph("Total Detections:", meta_label_style),
            Paragraph(str(len(all_detections)), meta_val_style),
        ],
    ]
    t_batch_info = Table(batch_info_data, colWidths=[140, 120, 130, 130])
    t_batch_info.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ("PADDING", (0, 0), (-1, -1), 4),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F7FAFC")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#F7FAFC")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(t_batch_info)
    story.append(Spacer(1, 10))

    # 4. Detection Class Breakdown
    story.append(Paragraph("2. Detection Class Summary", section_heading_style))
    story.append(Spacer(1, 4))

    if class_counts:
        class_table_data = [
            [
                Paragraph("Detection Class", table_header_style),
                Paragraph("Target Count", table_header_style),
                Paragraph("Percentage", table_header_style),
            ]
        ]
        tot_d = max(len(all_detections), 1)
        for cname, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            pct = (count / tot_d) * 100
            class_table_data.append(
                [
                    Paragraph(f"<b>{cname}</b>", table_cell_style),
                    Paragraph(str(count), table_cell_style),
                    Paragraph(f"{pct:.1f}%", table_cell_style),
                ]
            )
        t_classes = Table(class_table_data, colWidths=[200, 160, 160])
        t_classes.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), primary_color),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                    ("PADDING", (0, 0), (-1, -1), 4),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    *(
                        [
                            ("BACKGROUND", (0, r), (-1, r), colors.HexColor("#F7FAFC"))
                            for r in range(2, len(class_table_data), 2)
                        ]
                    ),
                ]
            )
        )
        story.append(t_classes)
    else:
        story.append(Paragraph("<i>No targets detected in this batch.</i>", meta_val_style))
    story.append(Spacer(1, 10))

    # 5. Scan Summary Table (including zero-detection scans)
    story.append(Paragraph("3. Survey Scans Breakdown", section_heading_style))
    story.append(Spacer(1, 4))

    scans_table_data = [
        [
            Paragraph("Scan Identity", table_header_style),
            Paragraph("Timestamp", table_header_style),
            Paragraph("Platform Lat / Lon", table_header_style),
            Paragraph("Hdg (°)", table_header_style),
            Paragraph("Alt (m)", table_header_style),
            Paragraph("Detections", table_header_style),
        ]
    ]

    for s in scans:
        s_ts = s.timestamp.strftime("%Y-%m-%d %H:%M") if s.timestamp else "N/A"
        det_cnt = len(s.detections or [])
        pos_str = f"{s.sonar_latitude:.4f}, {s.sonar_longitude:.4f}"
        det_label = f"<b>{det_cnt}</b>" if det_cnt > 0 else "<font color='#718096'>0 (Zero detections)</font>"
        scans_table_data.append(
            [
                Paragraph(f"<b>{s.scan_identity}</b>", table_cell_style),
                Paragraph(s_ts, table_cell_style),
                Paragraph(pos_str, table_cell_style),
                Paragraph(f"{s.heading:.1f}", table_cell_style),
                Paragraph(f"{s.altitude:.1f}", table_cell_style),
                Paragraph(det_label, table_cell_style),
            ]
        )

    t_scans = Table(scans_table_data, colWidths=[120, 95, 130, 55, 55, 65], repeatRows=1)
    t_scans.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), primary_color),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ("PADDING", (0, 0), (-1, -1), 4),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                *(
                    [
                        ("BACKGROUND", (0, r), (-1, r), colors.HexColor("#F7FAFC"))
                        for r in range(2, len(scans_table_data), 2)
                    ]
                ),
            ]
        )
    )
    story.append(t_scans)
    story.append(Spacer(1, 10))

    # 6. Georeferenced Detections Table
    story.append(Paragraph("4. Georeferenced Detections Detail", section_heading_style))
    story.append(Spacer(1, 4))

    if all_detections:
        geo_table_data = [
            [
                Paragraph("Scan", table_header_style),
                Paragraph("Class", table_header_style),
                Paragraph("Conf", table_header_style),
                Paragraph("Target Latitude", table_header_style),
                Paragraph("Target Longitude", table_header_style),
                Paragraph("Range (m)", table_header_style),
                Paragraph("Ground (m)", table_header_style),
            ]
        ]
        for s, d in all_detections:
            lat_str = f"{d.target_latitude:.6f}" if d.target_latitude is not None else "N/A"
            lon_str = f"{d.target_longitude:.6f}" if d.target_longitude is not None else "N/A"
            slant_str = f"{d.range_m:.1f}" if d.range_m is not None else "N/A"
            ground_str = f"{d.ground_range_m:.1f}" if d.ground_range_m is not None else "N/A"
            geo_table_data.append(
                [
                    Paragraph(s.scan_identity, table_cell_style),
                    Paragraph(f"<b>{d.class_name}</b>", table_cell_style),
                    Paragraph(f"{d.confidence:.2f}", table_cell_style),
                    Paragraph(lat_str, table_cell_style),
                    Paragraph(lon_str, table_cell_style),
                    Paragraph(slant_str, table_cell_style),
                    Paragraph(ground_str, table_cell_style),
                ]
            )
        t_geo = Table(geo_table_data, colWidths=[90, 85, 45, 95, 95, 55, 55], repeatRows=1)
        t_geo.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), secondary_color),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                    ("PADDING", (0, 0), (-1, -1), 3.5),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    *(
                        [
                            ("BACKGROUND", (0, r), (-1, r), colors.HexColor("#F7FAFC"))
                            for r in range(2, len(geo_table_data), 2)
                        ]
                    ),
                ]
            )
        )
        story.append(t_geo)
    else:
        story.append(Paragraph("<i>No georeferenced detections recorded in this batch.</i>", meta_val_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    return buffer.getvalue()


"""
reports package
---------------
Investigation report generation utilities (PDF, JSON, GeoJSON).
"""

from app.reports.pdf_generator import (
    generate_batch_pdf_report,
    generate_scan_pdf_report,
)

__all__ = ["generate_scan_pdf_report", "generate_batch_pdf_report"]

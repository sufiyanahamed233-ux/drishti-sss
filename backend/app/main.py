"""
main.py
-------
Drishti SSS FastAPI application entry point.

Phase 3A implements:
- Application factory with metadata
- GET /health      – liveness probe (always 200 if the process is up)
- GET /health/db   – readiness probe (checks PostgreSQL connectivity)

Later phases will register additional routers for scans, detections, and
real-time processing pipelines.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.db.session import check_db_connection


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""

    app = FastAPI(
        title=settings.app_title,
        version=settings.app_version,
        description=(
            "Drishti Side-Scan Sonar — REST API for sonar image ingestion, "
            "YOLO-based object detection, and georeferenced target reporting."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )

    # ── Health endpoints ───────────────────────────────────────────────────────

    @app.get(
        "/health",
        tags=["health"],
        summary="Liveness probe",
        response_description="Service is alive",
    )
    def health_check() -> JSONResponse:
        """
        Return 200 if the FastAPI process is running.

        This endpoint intentionally performs **no** I/O so it can always
        respond, even when the database is unreachable.
        """
        return JSONResponse(
            content={
                "status": "ok",
                "version": settings.app_version,
                "env": settings.app_env,
            }
        )

    @app.get(
        "/health/db",
        tags=["health"],
        summary="Database readiness probe",
        response_description="Database connectivity status",
    )
    def health_db() -> JSONResponse:
        """
        Return 200 if the application can reach PostgreSQL, 503 otherwise.

        Executes a lightweight ``SELECT 1`` against the configured
        ``DATABASE_URL`` and reports the result.
        """
        connected = check_db_connection()
        status_code = 200 if connected else 503
        return JSONResponse(
            status_code=status_code,
            content={
                "status": "ok" if connected else "error",
                "database": "reachable" if connected else "unreachable",
            },
        )

    # ── Database table initialization & schema migration ──────────────────────
    try:
        from app.db import models as _models  # noqa: F401
        from app.db.migration import upgrade_db_schema
        from app.db.session import engine
        upgrade_db_schema(bind=engine)
    except Exception:
        pass

    # ── Mount API routers (Phase 3B) ──────────────────────────────────────────
    from app.api import api_router
    app.include_router(api_router)

    return app


# ---------------------------------------------------------------------------
# Module-level application instance (used by uvicorn and tests)
# ---------------------------------------------------------------------------

app = create_app()

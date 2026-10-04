"""
api package
-----------
Mounts all REST API routers under the /api prefix.
"""

from fastapi import APIRouter

from app.api.analysis import router as analysis_router
from app.api.scans import router as scans_router

api_router = APIRouter(prefix="/api")
api_router.include_router(analysis_router)
api_router.include_router(scans_router)

__all__ = ["api_router"]

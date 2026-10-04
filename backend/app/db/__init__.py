"""
db/__init__.py
--------------
Database sub-package.

Public re-exports
~~~~~~~~~~~~~~~~~
- :data:`Base`    – declarative base for all ORM models
- :func:`get_db`  – FastAPI dependency that yields a scoped session
"""

from .migration import upgrade_db_schema
from .models import Detection, InvestigationBatch, Scan
from .session import Base, get_db

__all__ = ["Base", "get_db", "InvestigationBatch", "Scan", "Detection", "upgrade_db_schema"]

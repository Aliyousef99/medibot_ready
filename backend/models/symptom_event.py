"""SymptomEvent model with dialect-aware types.

Uses generic String/JSON for SQLite and native UUID/JSONB for Postgres.
"""

from sqlalchemy import Column, String, Text, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.types import JSON as SA_JSON
import uuid

from backend.db.session import Base, engine

# --- Dialect-aware column helpers ---
try:
    from sqlalchemy.dialects.postgresql import UUID as PG_UUID
except Exception:
    PG_UUID = None

try:
    from sqlalchemy.dialects.postgresql import JSONB as PG_JSONB
except Exception:
    PG_JSONB = None


def uuid_col_type():
    if engine.dialect.name == "postgresql" and PG_UUID is not None:
        return PG_UUID(as_uuid=True)
    return String(36)


def json_col_type():
    import os
    if os.getenv("FORCE_GENERIC_JSON", "").lower() in ("1", "true", "yes"):
        return SA_JSON
    if engine.dialect.name == "postgresql" and PG_JSONB is not None:
        return PG_JSONB
    return SA_JSON


class SymptomEvent(Base):
    __tablename__ = "symptom_events"

    id = Column(uuid_col_type(), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    raw_text = Column(Text, nullable=False)
    result_json = Column(json_col_type(), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

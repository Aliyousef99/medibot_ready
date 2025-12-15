# backend/models/lab_report.py
import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy import String, DateTime, ForeignKey, text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON as SA_JSON
from backend.db.session import Base, engine
from backend.utils.encryption import EncryptedText, EncryptedJSON

# ---- Dialect-aware column types ----

# UUID: Postgres gets UUID, fallback is String(36)
try:
    from sqlalchemy.dialects.postgresql import UUID as PG_UUID
except Exception:
    PG_UUID = None

def uuid_col_type():
    if engine.dialect.name == "postgresql" and PG_UUID is not None:
        return PG_UUID(as_uuid=True)
    return String(36)

# JSON: Postgres gets JSONB, others get plain JSON
try:
    from sqlalchemy.dialects.postgresql import JSONB as PG_JSONB
except Exception:
    PG_JSONB = None

def json_col_type():
    import os
    if os.getenv("FORCE_GENERIC_JSON", "").lower() in ("1", "true", "yes"):
        return SA_JSON
    if engine.dialect.name == "postgresql" and PG_JSONB is not None:
        return PG_JSONB
    return SA_JSON


class LabReport(Base):
    __tablename__ = "lab_reports"

# lab_report.py

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )

    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    raw_text: Mapped[str] = mapped_column(EncryptedText, nullable=False)
    structured_json: Mapped[Optional[dict]] = mapped_column(EncryptedJSON, nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(EncryptedText, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        server_onupdate=text("CURRENT_TIMESTAMP"),
    )

    user = relationship("User", back_populates="lab_reports")

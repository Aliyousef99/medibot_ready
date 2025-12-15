import os
import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy import (
    String,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    Integer,
    Boolean,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON as SA_JSON
from backend.utils.encryption import EncryptedText, EncryptedJSON

# Import Base/engine from your session module (no circular import)
from backend.db.session import Base, engine

# ---- Dialect-aware column types ----

# UUID column type: normalize to String(36) to match existing schema (avoid UUID/varchar mismatches)
def uuid_col_type():
    return String(36)

# JSON column type: Postgres gets JSONB, others get generic JSON
try:
    from sqlalchemy.dialects.postgresql import JSONB as PG_JSONB
except Exception:
    PG_JSONB = None

def json_col_type():
    # In tests or non-Postgres environments, force generic JSON to avoid JSONB with SQLite
    if os.getenv("FORCE_GENERIC_JSON", "").lower() in ("1", "true", "yes"):
        return SA_JSON
    if engine.dialect.name == "postgresql" and PG_JSONB is not None:
        return PG_JSONB
    return SA_JSON


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("email", name="uq_users_email"),
    )

    id: Mapped[str] = mapped_column(
        uuid_col_type(),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),   # force string not UUID object
    )
    email: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP")
    )

    # relationships
    profile: Mapped["UserProfile"] = relationship(
        "UserProfile",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan"
    )

    lab_reports: Mapped[List["LabReport"]] = relationship(
        "LabReport",
        back_populates="user",
        cascade="all, delete-orphan"
    )

    recommendations: Mapped[List["RecommendationSet"]] = relationship(
        "RecommendationSet",
        back_populates="user",
        cascade="all, delete-orphan"
    )


class UserProfile(Base):
    """
    Optional per-user clinical context we can pass into prompts.
    """
    __tablename__ = "user_profile"

    user_id: Mapped[str] = mapped_column(
        uuid_col_type(),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )

    # Lightweight fields
    age: Mapped[Optional[int]] = mapped_column(EncryptedJSON, nullable=True)
    sex: Mapped[Optional[str]] = mapped_column(EncryptedText, nullable=True)  # "male"|"female"|"other"|None

    # Flexible JSON blobs
    conditions: Mapped[Optional[list]] = mapped_column(EncryptedJSON, nullable=True, default=None)
    medications: Mapped[Optional[list]] = mapped_column(EncryptedJSON, nullable=True, default=None)
    notes: Mapped[Optional[str]] = mapped_column(EncryptedText, nullable=True)
    consent_given: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default=text("0"))
    consent_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        server_onupdate=text("CURRENT_TIMESTAMP")
    )

    user: Mapped["User"] = relationship("User", back_populates="profile")

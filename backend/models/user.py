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
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON as SA_JSON

# Import Base/engine from your session module (no circular import)
from backend.db.session import Base, engine

# ---- Dialect-aware column types ----

# UUID column type: Postgres gets native UUID, others get String(36)
try:
    from sqlalchemy.dialects.postgresql import UUID as PG_UUID
except Exception:
    PG_UUID = None  # not available

def uuid_col_type():
    if engine.dialect.name == "postgresql" and PG_UUID is not None:
        return PG_UUID(as_uuid=True)
    return String(36)

# JSON column type: Postgres gets JSONB, others get generic JSON
try:
    from sqlalchemy.dialects.postgresql import JSONB as PG_JSONB
except Exception:
    PG_JSONB = None

def json_col_type():
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
    age: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    sex: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)  # "male"|"female"|"other"|None

    # Flexible JSON blobs
    conditions: Mapped[Optional[list]] = mapped_column(json_col_type(), nullable=True, default=None)
    medications: Mapped[Optional[list]] = mapped_column(json_col_type(), nullable=True, default=None)
    notes: Mapped[Optional[str]] = mapped_column(String(4000), nullable=True)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        server_onupdate=text("CURRENT_TIMESTAMP")
    )

    user: Mapped["User"] = relationship("User", back_populates="profile")

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, DateTime, text, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.session import Base


class ConversationContext(Base):
    __tablename__ = "conversation_contexts"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    conversation_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)

    # User-editable system prompt additions scoped to the conversation
    system_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # JSON string of extracted lab tests for this conversation
    lab_tests_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

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


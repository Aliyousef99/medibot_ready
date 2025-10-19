import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from ..db.session import Base
import datetime

class RecommendationSet(Base):
    __tablename__ = "recommendation_sets"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True, index=True)
    user_id: Mapped[str] = mapped_column(sa.ForeignKey("users.id"))
    
    risk_tier: Mapped[str] = mapped_column(sa.String, nullable=False)
    actions: Mapped[list[str]] = mapped_column(sa.JSON, nullable=False)
    text: Mapped[str] = mapped_column(sa.Text, nullable=False)
    
    rules_version: Mapped[str] = mapped_column(sa.String, nullable=False, default="1.0")
    llm_used: Mapped[bool] = mapped_column(sa.Boolean, default=False)

    created_at: Mapped[datetime.datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=func.now()
    )
    
    user = relationship("User", back_populates="recommendations")
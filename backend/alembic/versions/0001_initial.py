"""Initial database schema."""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Ensure the enum exists once; drop stray partial attempts and recreate
    op.execute("DROP TYPE IF EXISTS messagerole CASCADE;")
    op.execute("CREATE TYPE messagerole AS ENUM ('user', 'assistant', 'system');")
    from sqlalchemy.dialects import postgresql

    message_role = postgresql.ENUM(
        "user", "assistant", "system", name="messagerole", create_type=False
    )

    op.create_table(
        "users",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("email", sa.String(length=255), nullable=False, index=True),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=120)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )

    op.create_table(
        "user_profile",
        sa.Column("user_id", sa.String(length=36), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("age", sa.Integer),
        sa.Column("sex", sa.String(length=16)),
        sa.Column("conditions", sa.JSON),
        sa.Column("medications", sa.JSON),
        sa.Column("notes", sa.String(length=4000)),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), server_onupdate=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    )

    op.create_table(
        "conversations",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("user_id", sa.String(length=36), nullable=False, index=True),
        sa.Column("title", sa.String(length=255)),
        sa.Column("rolling_summary", sa.String(length=20000)),
        sa.Column("active_lab_id", sa.String(length=36)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), server_onupdate=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    )

    op.create_table(
        "lab_reports",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("user_id", sa.String(length=36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("title", sa.String(length=255)),
        sa.Column("raw_text", sa.String(length=20000), nullable=False),
        sa.Column("structured_json", sa.JSON),
        sa.Column("summary", sa.String(length=20000)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), server_onupdate=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    )

    op.create_table(
        "recommendation_sets",
        sa.Column("id", sa.Integer, primary_key=True, index=True),
        sa.Column("user_id", sa.String(length=36), sa.ForeignKey("users.id")),
        sa.Column("risk_tier", sa.String, nullable=False),
        sa.Column("actions", sa.JSON, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("rules_version", sa.String, nullable=False, server_default="1.0"),
        sa.Column("llm_used", sa.Boolean, server_default=sa.text("false"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "symptom_events",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("user_id", sa.String(length=36), sa.ForeignKey("users.id"), nullable=False, index=True),
        sa.Column("raw_text", sa.Text, nullable=False),
        sa.Column("result_json", sa.JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "messages",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("conversation_id", sa.String(length=36), sa.ForeignKey("conversations.id", ondelete="CASCADE"), index=True),
        sa.Column("user_id", sa.String(length=36), index=True),
        sa.Column("role", message_role, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    )


def downgrade():
    op.drop_table("messages")
    op.drop_table("symptom_events")
    op.drop_table("recommendation_sets")
    op.drop_table("lab_reports")
    op.drop_table("conversations")
    op.drop_table("user_profile")
    op.drop_table("users")
    sa.Enum(name="messagerole").drop(op.get_bind(), checkfirst=True)

"""Add email verification fields to users.

Revision ID: 0004_email_verify
Revises: 0003_user_profile_text
Create Date: 2026-01-12
"""

from alembic import op
import sqlalchemy as sa


revision = "0004_email_verify"
down_revision = "0003_user_profile_text"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("users", sa.Column("email_verified", sa.Boolean(), server_default=sa.text("false"), nullable=False))
    op.add_column("users", sa.Column("email_verification_token_hash", sa.String(length=128), nullable=True))
    op.add_column("users", sa.Column("email_verification_sent_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("users", sa.Column("email_verification_expires_at", sa.DateTime(timezone=True), nullable=True))


def downgrade():
    op.drop_column("users", "email_verification_expires_at")
    op.drop_column("users", "email_verification_sent_at")
    op.drop_column("users", "email_verification_token_hash")
    op.drop_column("users", "email_verified")

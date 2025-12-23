"""Expand user_profile encrypted fields to TEXT.

Revision ID: 0003_user_profile_text
Revises: 0002_encrypt_json_text
Create Date: 2025-12-23
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0003_user_profile_text"
down_revision = "0002_encrypt_json_text"
branch_labels = None
depends_on = None


def upgrade():
    # EncryptedText/EncryptedJSON values exceed prior column sizes; store as TEXT.
    op.alter_column(
        "user_profile",
        "sex",
        existing_type=sa.String(length=16),
        type_=sa.Text(),
        postgresql_using="sex::text",
    )
    op.alter_column(
        "user_profile",
        "age",
        existing_type=sa.Integer(),
        type_=sa.Text(),
        postgresql_using="age::text",
    )
    op.alter_column(
        "user_profile",
        "conditions",
        existing_type=sa.JSON(),
        type_=sa.Text(),
        postgresql_using="conditions::text",
    )
    op.alter_column(
        "user_profile",
        "medications",
        existing_type=sa.JSON(),
        type_=sa.Text(),
        postgresql_using="medications::text",
    )
    op.alter_column(
        "user_profile",
        "notes",
        existing_type=sa.String(length=4000),
        type_=sa.Text(),
        postgresql_using="notes::text",
    )


def downgrade():
    op.alter_column(
        "user_profile",
        "notes",
        existing_type=sa.Text(),
        type_=sa.String(length=4000),
        postgresql_using="NULLIF(notes, '')",
    )
    op.alter_column(
        "user_profile",
        "medications",
        existing_type=sa.Text(),
        type_=sa.JSON(),
        postgresql_using="NULLIF(medications, '')::json",
    )
    op.alter_column(
        "user_profile",
        "conditions",
        existing_type=sa.Text(),
        type_=sa.JSON(),
        postgresql_using="NULLIF(conditions, '')::json",
    )
    op.alter_column(
        "user_profile",
        "age",
        existing_type=sa.Text(),
        type_=sa.Integer(),
        postgresql_using="NULLIF(age, '')::integer",
    )
    op.alter_column(
        "user_profile",
        "sex",
        existing_type=sa.Text(),
        type_=sa.String(length=16),
        postgresql_using="NULLIF(sex, '')",
    )

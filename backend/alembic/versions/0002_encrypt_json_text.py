"""Store encrypted JSON fields as TEXT.

Revision ID: 0002_encrypt_json_text
Revises: 0001_initial
Create Date: 2025-12-23
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0002_encrypt_json_text"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade():
    # EncryptedJSON columns must be TEXT, not JSON/INTEGER.
    op.alter_column(
        "lab_reports",
        "structured_json",
        existing_type=sa.JSON(),
        type_=sa.Text(),
        postgresql_using="structured_json::text",
    )
    op.alter_column(
        "symptom_events",
        "result_json",
        existing_type=sa.JSON(),
        type_=sa.Text(),
        postgresql_using="result_json::text",
    )
    op.alter_column(
        "recommendation_sets",
        "actions",
        existing_type=sa.JSON(),
        type_=sa.Text(),
        postgresql_using="actions::text",
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
        "age",
        existing_type=sa.Integer(),
        type_=sa.Text(),
        postgresql_using="age::text",
    )


def downgrade():
    # Best-effort: cast TEXT back to JSON/INTEGER where possible.
    op.alter_column(
        "user_profile",
        "age",
        existing_type=sa.Text(),
        type_=sa.Integer(),
        postgresql_using="NULLIF(age, '')::integer",
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
        "recommendation_sets",
        "actions",
        existing_type=sa.Text(),
        type_=sa.JSON(),
        postgresql_using="NULLIF(actions, '')::json",
    )
    op.alter_column(
        "symptom_events",
        "result_json",
        existing_type=sa.Text(),
        type_=sa.JSON(),
        postgresql_using="NULLIF(result_json, '')::json",
    )
    op.alter_column(
        "lab_reports",
        "structured_json",
        existing_type=sa.Text(),
        type_=sa.JSON(),
        postgresql_using="NULLIF(structured_json, '')::json",
    )

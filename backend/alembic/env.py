import os
import sys
from logging.config import fileConfig
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool
from alembic import context

# Ensure both repo root and backend package are on sys.path
BACKEND_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = Path(__file__).resolve().parents[2]
for path in (ROOT_DIR, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

ENV_PATH = BACKEND_DIR / ".env"
load_dotenv(ENV_PATH, override=True)

config = context.config
fileConfig(config.config_file_name)

# Grab models metadata
from backend.db.session import DATABASE_URL  # noqa: E402
from backend.models import user, conversation, message, lab_report, symptom_event, recommendation  # noqa: F401, E402
from backend.db.session import Base  # noqa: E402

target_metadata = Base.metadata

def _current_db_url() -> str:
    env_url = os.getenv("DATABASE_URL")
    return env_url if env_url else DATABASE_URL

def run_migrations_offline():
    url = _current_db_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    config_section = config.get_section(config.config_ini_section) or {}
    config_section["sqlalchemy.url"] = _current_db_url()

    connectable = engine_from_config(
        config_section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

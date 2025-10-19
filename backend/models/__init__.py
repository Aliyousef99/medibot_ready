# Ensure SQLAlchemy models are registered and provide a simple DB init helper.

from backend.db.session import Base, engine  # re-exported for convenience

# Import model modules so that their SQLAlchemy models are registered with Base
# before any metadata operations (e.g., create_all) run.
from . import user  # noqa: F401
from . import lab_report  # noqa: F401
from . import symptom_event  # noqa: F401
from . import recommendation  # noqa: F401
from . import message  # noqa: F401
from . import conversation  # noqa: F401


def init_db() -> None:
    """Create all tables for the configured engine.

    This is a minimal helper used by the app startup path. Tests provide
    their own in-memory engine and create schema separately in conftest.
    """
    Base.metadata.create_all(bind=engine)

# backend/models/__init__.py
from backend.db.session import Base, engine, SessionLocal

# Import model modules so SQLAlchemy registers all mappers.
# Important: we import the modules (not the classes) to avoid circular imports.
from . import user  # noqa: F401
from . import lab_report  # noqa: F401
from . import symptom_event  # noqa: F401
from . import recommendation  # noqa: F401


def init_db() -> None:
    """Create tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Yield a DB session and ensure it's closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
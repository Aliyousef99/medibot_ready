from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import create_engine
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://medibot:medibot@localhost:5432/medibot")
SQL_ECHO = (os.getenv("SQL_ECHO", "false") or "false").lower() in {"1", "true", "yes", "on"}

_engine_kwargs = {"echo": SQL_ECHO, "future": True, "pool_pre_ping": True}
if DATABASE_URL.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
elif DATABASE_URL.startswith("postgresql") and "sslmode=" not in DATABASE_URL.lower():
    # Supabase and many hosted Postgres instances require SSL
    _engine_kwargs["connect_args"] = {"sslmode": "require"}

engine = create_engine(DATABASE_URL, **_engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        # Defensive: ensure no aborted transaction is carried over from pool
        try:
            db.rollback()
        except Exception:
            pass
        yield db
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        raise
    finally:
        try:
            db.close()
        except Exception:
            pass

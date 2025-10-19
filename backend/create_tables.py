# backend/create_tables.py
import sys
from pathlib import Path

from dotenv import load_dotenv

# Allow running from the backend/ directory without PYTHONPATH tweaks
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

ENV_PATH = ROOT_DIR / "backend" / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH, override=True)

from backend.models import init_db

if __name__ == "__main__":
    print("Creating tables...")
    init_db()
    print("✅ Tables created.")

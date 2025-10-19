# backend/seed_user.py
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Allow running from backend/ without tweaking PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

ENV_PATH = ROOT_DIR / "backend" / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH, override=True)

from backend.db.session import SessionLocal
from backend.models.user import User
from backend.auth.deps import hash_password

def main():
    email = os.getenv("DEMO_USER_EMAIL", "demo@example.com")
    password = os.getenv("DEMO_USER_PASSWORD", "demo123")
    name = os.getenv("DEMO_USER_NAME", "Demo User")

    if not email or not password:
        raise SystemExit("Set DEMO_USER_EMAIL and DEMO_USER_PASSWORD before seeding")

    with SessionLocal() as db:
        exists = db.query(User).filter(User.email == email).first()
        if exists:
            print(f"User already exists: {email} (id={exists.id})")
            return

        user = User(
            email=email,
            hashed_password=hash_password(password),
            name=name,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"Seeded user: {email} (id={user.id})")

if __name__ == "__main__":
    main()

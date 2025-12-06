"""Authentication dependencies for FastAPI routes."""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session
from sqlalchemy import cast, Text as SAText

from backend.db.session import get_db
from backend.models.user import User
from backend.auth import jwt

_bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Resolve the authenticated user from the Authorization bearer token."""
    # Ensure the session isn't left in aborted state from prior work
    try:
        db.rollback()
    except Exception:
        pass
    print(f"Credentials: {credentials}")
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
        )

    payload = jwt.get_current_user_from_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    subject = payload.get("sub")
    if not subject:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token payload missing subject",
        )

    # Accept either user id or email in sub for flexibility in tests/clients
    qry = db.query(User)
    if isinstance(subject, str) and "@" in subject:
        user = qry.filter(User.email == subject).first()
    else:
        # Cast column to text to avoid UUID vs varchar mismatches on legacy schemas
        user = qry.filter(cast(User.id, SAText) == str(subject)).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    return user


# Re-export helpers for convenience in other modules
hash_password = jwt.hash_password
verify_password = jwt.verify_password
create_access_token = jwt.create_access_token

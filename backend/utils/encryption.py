import base64
import hashlib
import json
import os
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy.types import TypeDecorator, Text


def _build_cipher() -> Fernet:
    """Derive a stable Fernet key from ENCRYPTION_SECRET (or fallback dev secret)."""
    secret = os.getenv("ENCRYPTION_SECRET", "dev-secret-key-change-me").encode("utf-8")
    # Derive a 32-byte key and urlsafe-base64 encode for Fernet
    key = base64.urlsafe_b64encode(hashlib.sha256(secret).digest())
    return Fernet(key)


_CIPHER = _build_cipher()


class EncryptedText(TypeDecorator):
    """Encrypts/decrypts text values transparently using Fernet."""

    impl = Text
    cache_ok = True

    def process_bind_param(self, value: Any, dialect) -> Any:  # type: ignore[override]
        if value is None:
            return None
        if not isinstance(value, str):
            value = str(value)
        token = _CIPHER.encrypt(value.encode("utf-8"))
        return token.decode("utf-8")

    def process_result_value(self, value: Any, dialect) -> Any:  # type: ignore[override]
        if value is None:
            return None
        try:
            raw = _CIPHER.decrypt(value.encode("utf-8"))
            return raw.decode("utf-8")
        except InvalidToken:
            return None


class EncryptedJSON(TypeDecorator):
    """Encrypts/decrypts JSON-serializable values transparently."""

    impl = Text
    cache_ok = True

    def process_bind_param(self, value: Any, dialect) -> Any:  # type: ignore[override]
        if value is None:
            return None
        try:
            payload = json.dumps(value)
        except Exception:
            payload = json.dumps(str(value))
        token = _CIPHER.encrypt(payload.encode("utf-8"))
        return token.decode("utf-8")

    def process_result_value(self, value: Any, dialect) -> Any:  # type: ignore[override]
        if value is None:
            return None
        try:
            raw = _CIPHER.decrypt(value.encode("utf-8")).decode("utf-8")
            return json.loads(raw)
        except InvalidToken:
            return None
        except Exception:
            return None

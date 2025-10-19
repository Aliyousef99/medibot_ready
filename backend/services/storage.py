"""Local storage helpers for uploaded medical documents."""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Tuple

DEFAULT_UPLOAD_DIR = Path(
    os.getenv("UPLOAD_ROOT")
    or (Path(__file__).resolve().parent.parent / "uploads")
)


def _ensure_upload_dir() -> Path:
    DEFAULT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_UPLOAD_DIR


def store_local_upload(data: bytes, original_name: str | None) -> Tuple[str, str]:
    """Persist the raw upload to disk and return (path, filename)."""
    upload_dir = _ensure_upload_dir()
    suffix = Path(original_name or "").suffix.lower()
    safe_suffix = suffix if len(suffix) <= 10 else ""
    filename = f"{uuid.uuid4().hex}{safe_suffix}"
    path = upload_dir / filename
    path.write_bytes(data)
    return str(path), filename


__all__ = ["store_local_upload", "DEFAULT_UPLOAD_DIR"]

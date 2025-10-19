"""File-to-text helpers (PDF/image/text)."""
from __future__ import annotations

import io
from typing import Tuple

import pytesseract
from PIL import Image
from pypdf import PdfReader

SUPPORTED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}


def extract_text_from_bytes(data: bytes, filename: str, content_type: str) -> Tuple[str, str]:
    lowered = (filename or "").lower()
    mt = content_type or ""

    if mt == "application/pdf" or lowered.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages).strip()
        if not text:
            raise ValueError("No text extracted from PDF; consider adding OCR fallback")
        return text, "pdf"

    if mt.startswith("image/") or any(lowered.endswith(ext) for ext in SUPPORTED_IMAGE_EXT):
        img = Image.open(io.BytesIO(data))
        text = pytesseract.image_to_string(img, lang="eng")
        if not text.strip():
            raise ValueError("OCR produced empty output")
        return text, "ocr"

    try:
        text = data.decode("utf-8")
        return text, "text"
    except UnicodeDecodeError as exc:
        raise ValueError("Unable to decode file as UTF-8 text") from exc


__all__ = ["extract_text_from_bytes"]

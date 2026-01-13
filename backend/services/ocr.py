"""File-to-text helpers (PDF/image/text)."""
from __future__ import annotations

import io
from typing import Tuple

import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
from pypdf import PdfReader

SUPPORTED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}


class PdfTextTooShortError(RuntimeError):
    """Raised when primary PDF text extraction is empty or too short."""


def extract_text_from_pdf(data: bytes) -> Tuple[str, str]:
    """Extract text from PDF with OCR fallback."""
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages).strip()
        if len(text) < 50:
            raise PdfTextTooShortError("PDF text extraction too short; fallback to OCR")
        return text, "pdf"
    except PdfTextTooShortError:
        images = convert_from_bytes(data)
        ocr_pages = [pytesseract.image_to_string(img, lang="eng") for img in images]
        ocr_text = "\n".join(ocr_pages).strip()
        if not ocr_text:
            raise ValueError("OCR produced empty output for PDF")
        return ocr_text, "pdf_ocr_fallback"


def extract_text_from_bytes(data: bytes, filename: str, content_type: str) -> Tuple[str, str]:
    lowered = (filename or "").lower()
    mt = content_type or ""

    if mt == "application/pdf" or lowered.endswith(".pdf"):
        return extract_text_from_pdf(data)

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


__all__ = ["extract_text_from_bytes", "extract_text_from_pdf", "PdfTextTooShortError"]

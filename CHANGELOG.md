# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by Keep a Changelog and adheres to semantic, human‑readable entries.

## [v1.0.0-beta] - 2025-10-16

### Added
- Initial V1 Beta release of MediBuddy: React (Vite) frontend, FastAPI backend, OCR (Tesseract), PDF text extraction, and Gemini‑powered explanations.
- Authentication with JWT and profile management (age, sex, conditions, medications).
- Lab text parsing with HuggingFace NER (BioBERT by default, fallback to BERT NER) plus heuristic supplementation.

### Security / Hardening
- Frontend: Sanitized all AI responses with DOMPurify to mitigate XSS when rendering untrusted model output.
- Backend: Replaced prints with structured JSON logging (timestamp, level, function, message).
- Backend: Input validation via Pydantic `max_length=5000` on user text fields (`ParseRequest.text`, `ChatRequest.message/question`).

### Known Limitations (V1 Beta)
- English‑only support; multilingual support is planned for V2.
- No voice commands; V1 supports text input only.

[v1.0.0-beta]: https://example.com/releases/v1.0.0-beta


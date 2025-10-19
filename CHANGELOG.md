# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by Keep a Changelog and adheres to semantic, human-readable entries.

## [v1.1.0-beta] - 2025-10-19

### Added
- Backend: Inline lab extractor now runs early in `/api/chat` when messages look like lab reports (≥200 chars and includes unit token like `ng/mL|mmol/L|mg/dL` or bracketed range `[...]`).
- Backend: `ChatOut` now includes `user_view` with `{ summary, abnormal[], normal[], recommendation, confidence }` for UI consumption.
- Backend: Single structured log for inline extraction: `INFO {"function":"inline_lab_extract","parsed":<int>,"abnormal":<int>,"confidence":<float>}`.
- Backend: Date hint appended to summary when `ReceivedDate/Time dd/mm/yyyy` is detected.
- Frontend: Chat renders `response.user_view` by default (Title, Abnormal, Normal, one-line Recommendation, Confidence). Falls back to prior behavior if absent.
- Frontend: Developer Mode toggle (Ctrl+D) persists via `localStorage` and reveals a collapsible JSON viewer of the raw response and analysis panels.

### Changed
- Backend: Inline extractor in `/api/chat` no longer writes to the database; it is strictly for chat UX. Any structured object is attached to the request context only.
- Backend: Tightened extractor regex to accept trailing dots (e.g., `4.7.`), split glued flags (e.g., `1270H.1` → `1270.1` with `H` flag), recognize optional `(unit)` after name, and require a recognizable unit to reduce false positives.
- Frontend: Removed placeholder demo assistant message from initial seed.

### Acceptance Verification
- For ferritin/lipid samples: parsed ≥ 3, abnormal ≥ 1 (Ferritin), total cholesterol and triglycerides normal/desirable, confidence ≥ 0.6; UI shows only structured lists and confidence unless Dev Mode is enabled.

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
[v1.1.0-beta]: https://example.com/releases/v1.1.0-beta

## [v1.1.1-beta] - 2025-10-19

### Changed
- Backend: Assistant prompt relaxed to allow educational follow-up questions about abnormal results (e.g., causes of high ferritin) while still avoiding diagnosis/prescriptions and including care-seeking caveats.
- Frontend: Only render the structured Lab Summary card when `response.user_view` exists; otherwise show a normal assistant chat bubble. Prevents follow-up Q&A from being formatted as lab summaries.
- Frontend: Added a visible divider line between Abnormal and Normal lists in the Lab Summary card for easier scanning.

### Fixed/Filtered
- Backend/Extractor: Filtered out legend/threshold headings (e.g., "Desirable", "Normal", "Borderline High", "High:>") so they are not parsed as analytes. Tightened acceptance to require a recognizable unit on parsed lines.
- Frontend: Added defensive filtering in the Lab Summary renderer so only medical analytes/tests are listed.

[v1.1.1-beta]: https://example.com/releases/v1.1.1-beta

## [v1.1.2-beta] - 2025-10-19

### Changed
- Frontend: Abnormal results now have clearer visual emphasis (red section heading, count badge, and dot indicator) to improve scanability.
- Frontend: Inserted a divider between Abnormal and Normal lists when both exist for clearer separation.
- Frontend: Removed the separate "Analyze Symptoms" button; analysis is now fused into the Send flow using `symptom_analysis` from `/api/chat`. Cleaned unused API export.

[v1.1.2-beta]: https://example.com/releases/v1.1.2-beta

## [v1.1.3-beta] - 2025-10-19

### Added
- Backend: `user_view.explanation` included in `/api/chat` output for lab-like messages. Produces concise but specific explanations and escalates clearly when values are critically outside range.

### Changed
- Backend (inline extractor):
  - Preserve decimals when cleaning numbers; no more `5.2 -> 52` issues.
  - Normalize en/em dashes inside ranges (e.g., `[12–17.5]` -> `[12-17.5]`).
  - Recognize additional units: `%`, `fL`, and scientific counts `x10^N/L` (including `×10⁹/L` via Unicode normalization and superscripts -> `10^9`).
  - Filter legend/threshold headings (`Desirable`, `Normal`, `Borderline High`, `High:>`, etc.) from being parsed as analytes.
  - Critical detection widened: flags very low (≤ 0.5× lower ref) in addition to very high (≥ 2× upper ref) for escalation copy.
  - More robust loader: fall back to `backend.services.lab_parser` if top-level import fails.
- Frontend (LabSummaryCard):
  - Shows the full Explanation text inline with the lab summary (what Dev Mode showed), not only a brief snippet.
  - Keeps structured card rendering only when `response.user_view` exists; follow-up Q&A shows as normal chat messages.
  - Abnormal section styling and divider (from v1.1.2-beta) retained.

### Fixed
- CBC parsing correctness: Hematocrit `%`, MCV `fL`, Platelets/WBC `×10⁹/L` and `x10^9/L` now parse; reference ranges with typographic dashes are handled; decimals preserved.

[v1.1.3-beta]: https://example.com/releases/v1.1.3-beta

## 1. Introduction

- Purpose: AI‑assisted medical companion that parses lab reports, extracts entities, explains results, and supports chat/Q&A with profile context.
- Primary stacks:
  - Backend: FastAPI, SQLAlchemy ORM (SQLite by default), Pydantic v2, httpx, slowapi rate limiting.
  - Frontend: React 18 + Vite + Tailwind; Axios with JWT interceptor.
  - AI/OCR: Optional Transformers (BioBERT/NER) with heuristic fallbacks, Google Gemini via REST, PyPDF and Tesseract OCR.
- Architecture: SPA → REST API → services (OCR, parsing, rules, optional NER/LLM) → DB. Modular services exist under `backend/services/**`, with some legacy/monolithic logic in `backend/app.py`.

## 2. Object Design Trade-offs

- FastAPI + SQLAlchemy selected for lightweight, async‑friendly API development and simple persistence; defaults to SQLite with easy swap to Postgres via `DATABASE_URL` (backend/db/session.py).
- Parsing strategy prefers deterministic heuristics with optional Transformer NER for resilience and light runtime deps. Two parsers coexist:
  - Heuristic table/keyword parser in `backend/services/lab_parser.py` (modular, used by pipelines).
  - A newer, expanded parser inside `backend/app.py` with placeholder NER hooks. Duplication increases maintenance cost.
- LLM access via direct REST to Gemini instead of heavyweight SDK, reducing dependencies but requiring manual payload shaping and error handling.
- Mixed design: feature‑routers in `backend/routes/**` (auth, history, symptoms, recs, chat) alongside a combined orchestration endpoint in `backend/app.py` (`/api/chat`). Coexistence can confuse clients if not documented.
- Rate limiting with slowapi plus graceful no‑op fallback keeps dev ergonomics while enabling prod hardening.

## 3. Interface Documentation Guidelines

- Protocols: REST, JSON. Auth via Bearer JWT in `Authorization` header. Frontend stores auth in `localStorage` and injects on requests (`frontend/src/services/api.ts`).
- Key endpoints (paths → module):
  - Auth: `POST /api/auth/register`, `POST /api/auth/login` → `backend/routes/auth_routes.py`.
  - Profile: `GET/PUT /api/profile/` → `backend/routers/profile.py`.
  - Lab parsing: `POST /api/parse_lab` → `backend/app.py` (heuristic + placeholder NER inside app file).
  - Explain labs: `POST /api/explain` → `backend/app.py` (Gemini REST or offline fallback); sets `X-Lab-Id` header on success.
  - OCR: `POST /api/extract_text` → `backend/app.py` (PDF via PyPDF; images via Tesseract; strict MIME/size checks).
  - Symptoms (modular): `POST /api/symptoms/analyze`, `POST /api/symptoms/parse` → `backend/routes/symptoms_routes.py` (lexicon‑based pipeline in `backend/services/symptoms.py`).
  - Chat (combined): `POST /api/chat` → `backend/app.py` (builds profile+lab context, inline lab extraction, triage+recommendations, Gemini explanation).
  - Chat (conversations): `POST /api/chat/start`, `POST /api/chat/send`, `GET /api/chat/{id}/history` → `backend/routes/chat_routes.py` (persists messages; LLM call stubbed).
  - History (labs): `GET/POST/GET/DELETE /api/history/labs[/…]` → `backend/routes/history_routes.py` (persists `LabReport`).
  - Recommendations: `POST /api/recommendations/generate` → `backend/routes/recs_routes.py` (rules in `backend/services/recs.py`).
  - Models list: `GET /api/list_models` → `backend/app.py` (Gemini discovery; requires `GEMINI_API_KEY`).
- Frontend usage:
  - `postChatMessage('/api/chat')` returns a composite payload with `symptom_analysis`, `local_recommendations`, `ai_explanation`, and optional `user_view` (`frontend/src/services/api.ts`).
  - History and recs flows call `/api/history/*` and `/api/recommendations/generate` (`frontend/src/pages/History.tsx`).

## 4. Engineering Standards

- Structure: `backend/{app.py,routes,routers,services,auth,db,models,schemas,tests}` and `frontend/src/{components,pages,services}` are consistent and clear.
- Logging: JSON formatter with `trace_id` middleware (`backend/middleware/tracing.py`, `backend/app.py`); rate limit handler returns 429 with `Retry-After`.
- Error envelope: Standardized handlers exist in `backend/utils/exceptions.py` but are not registered on the FastAPI app; tests expect envelope fields (`code/message/trace_id`). This is a gap.
- Validation: Pydantic v2 models across request/response schemas; file uploads size/type validated.
- Tests: Pytest suite under `backend/tests` covers rate limits, security envelope, chat, symptoms, and recs. CI runs on GitHub Actions (`.github/workflows/CI.yml`).
- Frontend: TypeScript + React; no ESLint/Prettier config checked in; DOMPurify used to sanitize rendered HTML in chat UI.

## 5. Detailed Component Design

- 5.1 User Authentication Module
  - Files: `backend/routes/auth_routes.py`, `backend/auth/{deps.py,jwt.py}`, models in `backend/models/user.py`.
  - JWT: Created via `create_access_token` (python‑jose HS256); validated by `get_current_user` (HTTPBearer). Passwords hashed with `passlib` PBKDF2, bcrypt fallback; legacy plaintext tolerated as final fallback.
  - Demo seed: `_maybe_seed_demo_user()` on startup (`backend/app.py`) creates `demo@example.com` if missing.
  - Missing: Refresh tokens/rotation, lockout/2FA, and CSRF mitigations for non‑JWT flows. `deps.py` prints credentials to stdout (debug leak risk).

- 5.2 Symptom Analysis Module
  - Two paths exist:
    - Heuristic mapper `backend/services/symptom_analysis.py:analyze_text` used by `/api/analyze_symptoms` and the combined chat pipeline. Extracts canonical symptoms and suggests likely tests; returns confidence.
    - Lexicon+NER pipeline `backend/services/symptoms.py` with resources lexicon, negation, urgency classifier; routed by `/api/symptoms/*` and persists `SymptomEvent`. Optional NER fallback via `backend/services/ner.py`.
  - Persistence: `SymptomEvent` model (`backend/models/symptom_event.py`) recorded by both paths in different places.
  - Status: Functional heuristics; duplication between modules; NER optional.

- 5.3 Medical Report Interpretation Module
  - OCR: `POST /api/extract_text` in `backend/app.py` accepts PDF/images, enforces size/MIME, uses PyPDF or Tesseract; returns `{text}`.
  - Parsing:
    - Modular parser `backend/services/lab_parser.py` (row patterns, bracket ranges, status, conditions) used by `backend/services/report_pipeline.py`.
    - In‑file parser `backend/app.py:parse_lab_text` enhances heuristics and provides placeholders for NER integration; `/api/parse_lab` persists `LabReport` and caches structured JSON.
  - NER/Glossary enrich: `backend/services/ner.py` (optional Transformers pipeline; heuristic fallback) and `backend/services/glossary.py` (local definitions + Gemini fetch).
  - Explain: `POST /api/explain` calls Gemini 2.5‑flash via REST (httpx) or returns an offline summary fallback; persists `LabReport` and exposes `X-Lab-Id` response header.
  - Status: Heuristic extraction is robust; NER is optional and partially stubbed in `backend/app.py` path; the modular pipeline is more cohesive.

- 5.4 Personalized Recommendation Engine
  - Rules engine: `backend/services/recs.py` loads YAML rules (`backend/config/clinical_rules.yaml`) to compute risk tier and build action lists.
  - API: `POST /api/recommendations/generate` creates a `RecommendationSet` row and returns it (`backend/routes/recs_routes.py`).
  - LLM copy: `render_patient_copy` attempts Gemini‑based rewrite when `GEMINI_API_KEY` is set; otherwise uses a templated fallback. The referenced `gemini.rewrite_patient_copy` is not implemented in `backend/services/gemini.py` (only `generate_chat` exists) → incomplete integration.

- 5.5 Database and Models
  - SQLAlchemy models: `User`, `UserProfile`, `LabReport`, `SymptomEvent`, `RecommendationSet`, `Conversation`, `Message` (`backend/models/**`). Dialect‑aware UUID/JSON types; timestamps defaulted server‑side.
  - Engine/session: `backend/db/session.py` (SQLite default, `check_same_thread=False`), `init_db()` in `backend/models/__init__.py` creates schema; `backend/create_tables.py` helper.
  - Relationships: Users ↔ Profile (1–1), Users ↔ LabReports/Recommendations (1–N), Conversations ↔ Messages (1–N).

- 5.6 Third‑Party Integrations
  - Gemini: direct REST calls for chat explanation and labs explanation (`backend/app.py`); list models endpoint; API key via env. No official SDK dependency.
  - Transformers: optional NER (BioBERT or fallback) with graceful degradation to heuristics (`backend/services/ner.py`).
  - OCR/PDF: Tesseract via `pytesseract`, PDF via `pypdf`.
  - Security libs: `python-jose` for JWT, `passlib` for hashing.

## 6. UML-Level Notes (Informal)

- Frontend (React) → Axios → Backend (FastAPI) → Services (OCR, lab parser, symptoms, recs, glossary, NER) → DB (SQLAlchemy).
- Combined chat flow (`/api/chat`):
  - Build user context (profile, latest lab, recent symptoms/history) → inline lab extraction for pasted text → rule triage (`red_flag_triage`, `lab_triage`) + recommendations → optionally call Gemini for explanation → return composite payload.
- Async usage: httpx AsyncClient for Gemini; slowapi middlewares; pure functions in services enable testing.

## 7. Current State of Development

- ✅ Fully functional / tested
  - Auth (JWT, hashing, register/login): `backend/routes/auth_routes.py`, tests in `backend/tests` (auth used via overrides).
  - Profiles and persistence: `backend/routers/profile.py`, `backend/models/**`.
  - OCR and heuristic lab parsing: `/api/extract_text`, `/api/parse_lab`, `backend/services/lab_parser.py` and caching/persistence into `LabReport`.
  - Recommendations (rules engine) and route: `backend/services/recs.py`, `backend/routes/recs_routes.py`.
  - History endpoints and UI consumption: `backend/routes/history_routes.py`, `frontend/src/pages/History.tsx`.
  - Rate limiting + tracing middleware.

- ⚠️ Partially implemented
  - Chat: `/api/chat` integrates many parts and calls Gemini when configured; works end‑to‑end but depends on API key and returns fallback explanation otherwise.
  - Symptom analysis: two distinct implementations (`services/symptom_analysis.py` and `services/symptoms.py`) and route duplication (`/api/analyze_symptoms` in app vs `/api/symptoms/*`). Needs consolidation.
  - NER: Optional HF pipeline available in `backend/services/ner.py`; the app‑inline NER hooks are placeholders.
  - Error envelope: Handlers exist but not registered on `app`; tests reference envelope behavior—wiring is missing in `backend/app.py`.
  - LLM copywriter for recs: function referenced but not implemented (`gemini.rewrite_patient_copy`).

- 🚧 Placeholder or stubbed
  - `backend/services/gemini.py:generate_chat` returns empty string; chat_routes relies on it for replies.
  - `backend/services/summarizer.py` returns trivial summary; used for conversation trimming.
  - Some legacy top‑level scripts/files (e.g., `symptoms.py`, `ner.py` at repo root) appear experimental and unused by the main app.

## 8. Recommendations for Next Steps

- Consolidate lab parsing: Remove duplication by routing all parsing through `backend/services/lab_parser.py` and deleting/isolating the app‑inline parser. Keep NER integration behind a single service API.
- Wire error envelope: Register handlers from `backend/utils/exceptions.py` in `backend/app.py` for `HTTPException` and generic exceptions to meet test/docs expectations and consistently return `{code,message,trace_id}`.
- Unify symptom modules: Choose one pipeline (`services/symptoms.py` preferred for richer structure) and make both routes consume the same service; remove `/api/analyze_symptoms` in app or alias it.
- Complete Gemini integration: Implement `rewrite_patient_copy(actions)` in `backend/services/gemini.py` and consider moving shared Gemini REST logic there (used by chat and explain) with timeouts and retries.
- Security hardening: Remove debug prints (credentials in `backend/auth/deps.py`), add token refresh/rotation, consider per‑user rate buckets consistently, and sanitize all LLM‑rendered content in UI (already uses DOMPurify).
- Tooling and quality: Add ESLint/Prettier config for frontend; add ruff/black/mypy for backend; expand pytest coverage around error envelopes, `/api/explain` headers, and duplicate routes; include minimal vitest for UI API adapters.
- Config/docs: Provide `backend/.env.example`, document required OCR dependencies per OS, and clarify that only English is supported (already in README).


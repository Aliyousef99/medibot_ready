
# MediBot: Your AI-Powered Health Assistant 🩺

An AI-powered chatbot that helps you understand your medical lab reports. It uses a combination of local NER and the Gemini API to provide explanations and answer your questions.

- **Frontend**: React (Vite) + Tailwind. Classic 3‑pane chat UI.
- **Backend**: FastAPI.
  - `POST /api/parse_lab` — HuggingFace NER (BioBERT by default, fallback to BERT NER) with heuristic supplementation to extract tests/values/units/status/conditions.
  - `POST /api/explain` — calls **Gemini** with your `GEMINI_API_KEY` to produce a simplified explanation (with disclaimer).
  - `POST /api/extract_text` — extracts text from **PDF** (PyPDF) or **images** (Tesseract OCR).

> Note: To truly use **BioBERT**, swap the heuristic parser with a HuggingFace pipeline on your infra. This template keeps things light and dependency‑free of huge models by default.

---

## Features ✨

- Medical lab report upload via PDF or image with OCR fallback handled by the backend.
- Structured lab parsing that extracts key fields (test, value, unit, status, condition) from raw text.
- AI explanation flow that calls Gemini and returns a patient-friendly summary plus a safety disclaimer.
- Classic three-pane chat UI with conversation history, chat input, and info sidebar, wired to backend APIs.
- Environment-driven configuration for API keys and ports, including runtime overrides for frontend API base URL.
- HuggingFace-powered BioBERT pipeline for lab entities with automatic fallback to a smaller model and heuristic supplementation.

---

## Quick Start 🚀

### 1) Backend

```bash
cd backend
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt

# Copy env template and set your key
cp .env.example .env
# then tweak `.env` as needed (API keys, demo user overrides, optional Postgres URL, etc.)

# Run
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 2) Frontend

```bash
cd ../frontend
# pnpm preferred, but npm works too
npm i
npm run dev
```

Open the shown local URL (usually http://localhost:5173). The UI expects the backend at `http://localhost:8000`. You can change this by setting `window.__CHAT_API_BASE__` in the dev console or editing `src/components/ClassicChatbotUI.tsx` (constant `API_BASE`).

---

## Backend API 🤖

- `POST /api/parse_lab` — Runs HuggingFace BioBERT NER (with fallback model) to detect labs, aligns values/units via table heuristics, and enriches with conditions metadata.
- `POST /api/explain` — Sends structured labs to Gemini 2.5 Flash for human-friendly output. Optional offline fallback summarises results if no API key is present.
- `POST /api/extract_text` — Accepts PDF or image uploads and returns extracted text (PyPDF or Tesseract OCR).
- `POST /api/auth/login` — Username/password login returning a JWT. Demo user (`demo@example.com` / `demo123`) is auto-seeded on startup when missing.
- `GET /api/list_models` — Lists Gemini models (requires valid `GEMINI_API_KEY`).

---

## Environment 🛠️

Create `backend/.env` (see `.env.example`). Key fields:

```
GEMINI_API_KEY=YOUR_KEY            # required for Gemini calls
HUGGINGFACE_TOKEN=hf_xxx           # optional, set if the models need auth
BIOBERT_MODEL_NAME=d4data/biobert_ner
NER_FALLBACK_MODEL=dslim/bert-base-NER
DEMO_USER_EMAIL=demo@example.com   # auto-seeded on startup
DEMO_USER_PASSWORD=demo123
DATABASE_URL=sqlite:///./medibot.db # defaults to SQLite; point to Postgres via URL
```

- **Gemini** used: REST call to `generateContent` (2.5‑flash by default; adjust as you wish).
- **Tesseract OCR**: Make sure **Tesseract** is installed on your machine and available on PATH for OCR to work.
  - Windows: https://github.com/UB-Mannheim/tesseract/wiki
  - macOS: `brew install tesseract`
  - Linux: `sudo apt-get install tesseract-ocr`

If Tesseract or PDF text extraction isn't available, the endpoint will return an error message.

---

## ## Local Database & Demo User

- The backend defaults to a local SQLite file (`medibot.db`) with `check_same_thread=False` for multi-threaded use. Override via `DATABASE_URL` for Postgres or another RDBMS.
- On startup, the backend seeds a demo user using the env vars above. Adjust or disable by clearing `DEMO_USER_EMAIL`/`DEMO_USER_PASSWORD`.
- Password hashing upgraded to PBKDF2 (with fallback to bcrypt/plaintext for legacy records). Existing user rows continue to work; new ones get PBKDF2 hashes automatically.

---

## Swapping Heuristic NER with BioBERT 🔄

Inside `backend/app.py` see `parse_lab_text()`:
- Replace the heuristic with a HF pipeline:
  ```python
  from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
  model_name = "dslim/bert-base-NER"  # or your fine-tuned BioBERT
  nlp = pipeline("ner", model=model_name, aggregation_strategy="simple")
  ents = nlp(text)
  ```
Map entities to your schema (Lab Test, Value, Unit, Status, Condition). This repo now loads BioBERT automatically (with a fallback model) and merges with heuristics for robust structured output.

---

## Security Notes 🔒

- Do not expose your Gemini API key client-side.
- Add auth/rate-limits before deploying publicly.
- This project is for **educational** purposes; see the disclaimer in the explanation output.

## V1 Beta: Scope and Known Limitations 📋

- English-Only Support: The V1 beta is designed for and officially supports English language input and output only. Multilingual support is planned for a future release (V2).
- No Voice Commands: Voice command functionality is not included in the V1 release. The application currently supports text-based input only.

---

## Changelog 📜

See the full release history in [CHANGELOG.md](CHANGELOG.md).

Latest: v1.0.0-beta — initial V1 beta with structured logging, input validation, and frontend sanitization.

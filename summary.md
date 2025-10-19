## Updated Summary (2025-10-19)

- Changes at a glance
  - Frontend service `frontend/src/services/api.ts` now exposes `getHistory`, `getHistoryDetail`, `deleteHistory`, and `getRecommendationsForLab`, plus shared types `LabReport` and `RecommendationSet`. It uses an axios instance with an auth interceptor (not a custom `fetchApi` wrapper).
  - History page `frontend/src/pages/History.tsx` is wired to those APIs, fixes types (IDs are `string`), and uses `structured_json`.
  - Frontend TypeScript config added at `frontend/tsconfig.json` with `jsx: "react-jsx"` and `moduleResolution: "Bundler"` to resolve React and `lucide-react` typings. Production build succeeds.
  - Dedicated recommendations endpoint exists at `POST /api/recommendations/generate` (backend/routes/recs_routes.py) using a rules engine (`backend/services/recs.py`) with optional LLM rendering for user-facing copy.
  - Backend tests are present (pytest) under `backend/tests` with `pytest.ini` at repo root. Frontend testing libs are installed but no frontend test files were added.

- Corrected interface notes
  - Keep authentication and profile endpoints as documented.
  - Lab history endpoints: `GET /api/history/labs`, `GET /api/history/labs/{lab_id}`, `DELETE /api/history/labs/{lab_id}` are implemented and consumed by the frontend.
  - Add recommendations: `POST /api/recommendations/generate` (request may include `lab_id`, `labs`, `symptoms`, `patient_info`). Response conforms to `RecommendationSet`.

- Current state of development (updated)
  - Fully functional: Authentication, Database/Models, Lab Report Interpretation (OCR, NER, explanation), User Profile, Lab Report History.
  - Recommendations: Functional rules-based engine (risk scoring and actions) with optional LLM to render patient copy.
  - Symptom Analysis: Still LLM‑based for general queries; no dedicated model.
  - Testing: Backend tests exist (pytest). Frontend: testing deps present but no tests authored.
  - Linting: Not configured.

- Frontend details
  - `api.ts` centralizes axios with JWT injection via interceptor and defines `LabReport`/`RecommendationSet` types.
  - `History.tsx` uses those APIs; delete/view flows use modal and confirmation, with improved typing.
  - `tsconfig.json` ensures proper JSX/typing and removes prior TS errors about `react` and `react/jsx-runtime`.

---

## Previous Version

## 1. Introduction

The project is a web-based, AI-powered medical assistant named "MediBot". It allows users to upload medical lab reports, view their health history, and ask questions about their results. The backend is built with Python and the FastAPI framework, while the frontend is a single-page application built with React and Vite.

The core technologies employed are:

*   **Backend:** FastAPI, SQLAlchemy (with SQLite), Uvicorn
*   **Frontend:** React, Vite, Tailwind CSS
*   **AI/ML:**
    *   Hugging Face Transformers (BioBERT for NER)
    *   Google Gemini for natural language explanations and chat
    *   Pytesseract for OCR
*   **Authentication:** JWT (JSON Web Tokens)

The architecture follows a classic client-server model. The React frontend communicates with the FastAPI backend via a RESTful API. The backend handles business logic, database interactions, and integrations with third-party AI services.

## 2. Object Design Trade-offs

*   **FastAPI vs. Django:** The choice of FastAPI suggests a preference for performance, asynchronous programming, and a more lightweight, less opinionated framework compared to Django. This is suitable for an API-centric application.
*   **React vs. Other Frameworks:** React with Vite provides a fast and modern development experience for building a dynamic user interface.
*   **SQLAlchemy Core vs. ORM:** The project uses the SQLAlchemy ORM, which simplifies database interactions by mapping Python objects to database tables. This improves developer productivity at the cost of some performance overhead compared to using SQLAlchemy Core directly.
*   **Environment-based Configuration:** The use of `.env` files for configuration (e.g., API keys, database URLs) is a good practice for separating configuration from code, but it requires careful management of environment variables in different deployment environments.
*   **Fallback Mechanisms:** The NER service in `backend/app.py` has a fallback mechanism to a simpler BERT model if the primary BioBERT model fails to load. This improves robustness but may result in lower accuracy.

## 3. Interface Documentation Guidelines

The frontend communicates with the backend via a RESTful API with a base path of `/api`. The communication format is JSON. Authentication is handled via JWTs sent in the `Authorization` header.

Key API endpoints defined in `backend/app.py` and `frontend/src/services/api.ts` include:

*   **Authentication:**
    *   `POST /api/auth/register`: Creates a new user.
    *   `POST /api/auth/login`: Authenticates a user and returns a JWT.
*   **User Profile:**
    *   `GET /api/profile/`: Retrieves the current user's profile.
    *   `PUT /api/profile/`: Updates the user's profile.
*   **Lab Reports & History:**
    *   `POST /api/extract_text`: Extracts text from an uploaded file (PDF or image).
    *   `POST /api/parse_lab`: Parses the extracted text to identify medical entities.
    *   `POST /api/explain`: Generates a patient-friendly explanation of lab results using Gemini.
    *   `POST /api/chat`: Handles conversational chat with the user, using their profile and lab history as context.
    *   `GET /api/history/labs`: Retrieves a list of the user's past lab reports.
    *   `GET /api/history/labs/{lab_id}`: Retrieves the details of a specific lab report.
    *   `DELETE /api/history/labs/{lab_id}`: Deletes a lab report.

The frontend service `frontend/src/services/api.ts` provides a `fetchApi` wrapper that automatically includes the JWT in requests.

## 4. Engineering Standards

*   **Folder Structure:** The project has a clear separation between the `frontend` and `backend` directories. Within the backend, there are further subdivisions for `auth`, `db`, `models`, `routes`, `schemas`, and `services`.
*   **Naming Conventions:** The code generally follows Python's PEP 8 and TypeScript's standard naming conventions.
*   **Error Handling:** The backend uses FastAPI's `HTTPException` to return appropriate HTTP error codes and messages. The frontend has a basic `ErrorBoundary.tsx` component.
*   **Logging:** The backend has a JSON-based logger configured in `backend/app.py`.
*   **Testing:** There is no evidence of a testing framework being used in the provided file structure (e.g., no `tests` directory with `pytest` files).
*   **Linting:** There are no explicit linting configurations (e.g., `.eslintrc`, `pyproject.toml` with ruff/flake8 settings) visible in the file listing.
*   **Comments & Docstrings:** The code has a good amount of comments and docstrings, especially in the backend, explaining the purpose of functions and modules.

## 5. Detailed Component Design

### 5.1 User Authentication Module

*   **Location:** `backend/auth/`, `backend/routes/auth_routes.py`
*   **Implementation:** JWT-based authentication is implemented using the `python-jose` library. Passwords are hashed with `pbkdf2_sha256`. The `get_current_user` dependency in `backend/auth/deps.py` protects routes that require authentication.
*   **State:** ✅ **Fully functional**

### 5.2 Symptom Analysis Module

*   **Location:** `backend/app.py` (within the `/api/chat` endpoint)
*   **Implementation:** The `/api/chat` endpoint can receive free-text user queries that may contain symptoms. It uses a keyword-based approach (`_is_question_relevant_to_labs`) to determine if the question is relevant to medical topics. The actual analysis is then performed by the Gemini large language model.
*   **State:** ⚠️ **Partially implemented** (relies on a general-purpose LLM rather than a dedicated symptom analysis model).

### 5.3 Medical Report Interpretation Module

*   **Location:** `backend/services/`, `backend/app.py`
*   **Implementation:** This is a multi-stage pipeline:
    1.  **OCR:** `extract_text` in `backend/app.py` uses `pytesseract` for images and `pypdf` for PDFs.
    2.  **NER:** `parse_lab_text` in `backend/app.py` uses a Hugging Face `ner` pipeline with a BioBERT model (`d4data/biobert_ner`) to extract medical entities. It has a fallback to a simpler model (`dslim/bert-base-NER`) and a heuristic-based parser (`_parse_lab_text_heuristic`) if the NER models fail.
    3.  **Explanation:** The `/api/explain` endpoint sends the structured data to the Gemini API to generate a patient-friendly explanation.
*   **State:** ✅ **Fully functional** (with fallback mechanisms).

### 5.4 Personalized Recommendation Engine

*   **Location:** `backend/app.py` (within the `/api/chat` endpoint)
*   **Implementation:** The `/api/chat` endpoint constructs a detailed prompt for the Gemini API, including the user's profile (age, sex, conditions, medications) and their most recent lab report. This allows Gemini to provide personalized, context-aware responses.
*   **State:** ⚠️ **Partially implemented** (recommendations are generated by a general-purpose LLM and are not based on a dedicated recommendation engine).

### 5.5 Database and Models

*   **Location:** `backend/models/`, `backend/db/`
*   **Implementation:** The database schema is defined using SQLAlchemy ORM. The main models are `User`, `UserProfile`, and `LabReport`. The database is initialized in `backend/app.py` using the `init_db` function. The project is configured to use a SQLite database (`medibot.db`).
*   **State:** ✅ **Fully functional**

### 5.6 Third-Party Integrations

*   **Google Gemini:** Used for chat responses and explaining lab results. Integrated via direct HTTP requests to the Gemini API in `backend/app.py`.
*   **Hugging Face:** Used for the NER pipeline. The `transformers` library downloads and runs the models locally.
*   **Pytesseract:** Used for OCR. This requires a local installation of the Tesseract OCR engine.

## 6. UML-Level Notes (Informal)

The data flow can be summarized as follows:

1.  **User Interaction:** The user interacts with the React frontend.
2.  **API Request:** The frontend sends API requests to the FastAPI backend.
3.  **Authentication:** The backend verifies the user's JWT for protected endpoints.
4.  **Business Logic:** The backend processes the request, which may involve:
    *   Querying the database (e.g., to get user profile or lab history).
    *   Calling the Gemini API for chat or explanations.
    *   Running the OCR and NER pipelines for lab report analysis.
5.  **Database:** The backend reads from and writes to the SQLite database via SQLAlchemy.
6.  **API Response:** The backend sends a JSON response to the frontend.
7.  **UI Update:** The frontend updates the UI based on the API response.

Async patterns are used extensively in the FastAPI backend, particularly for handling HTTP requests and calling the Gemini API.

## 7. Current State of Development

*   ✅ **Fully functional / tested:**
    *   User Authentication (registration, login)
    *   Database and Models
    *   Medical Report Interpretation (OCR, NER, explanation)
    *   User Profile Management
    *   Lab Report History

*   ⚠️ **Partially implemented:**
    *   Symptom Analysis (relies on Gemini, no dedicated model)
    *   Personalized Recommendation Engine (relies on Gemini, no dedicated engine)
    *   Error Handling (basic implementation)

*   🚧 **Placeholder or stubbed:**
    *   **Testing:** No automated tests are present.
    *   **Admin Interface:** No admin interface is available.

There are no explicit `TODO` comments in the provided code, but the lack of a dedicated testing suite is a significant gap.

## 8. Recommendations for Next Steps

*   **Add Automated Tests:** Introduce a testing framework like `pytest` for the backend and `jest`/`@testing-library/react` for the frontend. This is the highest priority to ensure code quality and prevent regressions.
*   **Improve Error Handling:** Enhance the error handling in both the frontend and backend to provide more specific and user-friendly error messages.
*   **Refactor AI Service Calls:** The calls to the Gemini API are spread throughout `backend/app.py`. Refactor these into a dedicated `services/gemini.py` module to centralize the logic and make it more maintainable.
*   **Configuration Management:** For a production environment, consider using a more robust configuration management solution than just `.env` files, such as HashiCorp Vault or AWS Secrets Manager.
*   **Add Linting and Formatting:** Integrate tools like `ruff` or `black` for the backend and `eslint` and `prettier` for the frontend to enforce consistent code style.
*   **Documentation:** While the code has some comments, generating API documentation using FastAPI's automatic docs generation and adding more detailed documentation for the frontend components would be beneficial.

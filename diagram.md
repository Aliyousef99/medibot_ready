At Level 0, the system is represented as a single process interacting with two primary external entities — Users and AI Services (Gemini, Hugging Face). Users upload medical reports or submit chat queries, which are processed by the MediBot system and returned as explanations or structured data.

At Level 1, the data movement expands into core subsystems:

User Upload Flow: A user uploads a file through the frontend. The file is sent to the /api/extract_text endpoint. The backend OCR service extracts raw text and stores it temporarily in the database.

Lab Parsing Flow: The extracted text is passed to /api/parse_lab, where the parsing service structures the data and persists it into the LabReport table.

Explanation Flow: The structured JSON output is forwarded to the /api/explain endpoint, which calls the Gemini API. The generated explanation is then stored as part of the same lab report record.

Recommendation Flow: Parsed values are analyzed by the recs.py service using rule-based YAML logic. The generated recommendations are stored in the RecommendationSet table and optionally rephrased through Gemini.

Chat Flow: When users send a message through /api/chat, the request is validated, linked to an existing conversation, and a response is generated via Gemini. The conversation and all messages are logged in the database for history retrieval.

The flow can be summarized as follows:

Frontend (React) → FastAPI Routes (Auth, OCR, Parse, Explain, Chat) → Services (OCR, Parser, Gemini, Recommendations) → Database (SQLAlchemy).
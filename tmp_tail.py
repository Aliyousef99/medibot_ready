                "engine": "heuristic-ner-placeholder",
                "error": f"parse_lab failed: {e.__class__.__name__}: {e}",
            },
        }

@app.get("/api/list_models")
@limiter.limit("5/minute")
async def list_models(request: Request):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=400, detail="No GEMINI_API_KEY set")
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            logger.error("list_models failed")
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text) from e

@app.post("/api/chat", response_model=ChatOut)
@limiter.limit("15/minute")
async def chat(
    request: Request,
    payload: ChatIn,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ChatOut:
    from backend.utils.app import build_chat_context
    from backend.services import symptoms as symp

    # Run deterministic symptom parsing first
    try:
        pipeline_json = symp.summarize_to_json(payload.message).model_dump()
    except Exception as _e:
        pipeline_json = {"symptoms": [], "overall_confidence": 0.0}

    context, notice = build_chat_context(db, user, payload.message)
    prompt = (
        "You are a helpful medical assistant. Use the provided context to answer the user's question. "
        "The context includes the user's profile, their latest lab report, and their latest symptom summary. "
        "If context is missing, inform the user. Do not provide medical advice. "
        "Keep responses concise and easy to understand.\n\n"
        f"{context}\n\n--- Symptom Parser JSON (low confidence may be noisy) ---\n"
        f"{json.dumps(pipeline_json, ensure_ascii=False)}\n"
    )

    if pipeline_json.get("overall_confidence", 0.0) >= 0.5:
        # Avoid Gemini; return structured parse
        return ChatOut(response="Structured symptom analysis detected.", pipeline=pipeline_json)

    if not GEMINI_API_KEY:
        return ChatOut(response="Chat is not configured. No API key found.", pipeline=pipeline_json)

    try:
        async with httpx.AsyncClient(timeout=25) as client:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
            r = await client.post(
                url,
                params={"key": GEMINI_API_KEY},
                headers={"Content-Type": "application/json"},
                json={"contents": [{"role": "user", "parts": [{"text": prompt.strip()}]}]},
            )
            r.raise_for_status()
            data = r.json()
            text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "") or "Could not generate a response."
            return ChatOut(response=text.strip())
    except Exception as e:
        logger.exception("Chat endpoint failed")
        raise HTTPException(status_code=500, detail=f"Could not get a response from the chat model: {e}")



@router.post("/explain", response_model=ExplainOut)
@limiter.limit("2/minute")
async def explain(request: Request, payload: ExplainIn) -> ExplainOut:
    structured = payload.structured

    # fallback if no key
    if not GEMINI_API_KEY:
        lines = []
        for t in (structured.get("tests") or [])[:10]:
            name = t.get("name", "Test")
            val  = t.get("value", "")
            unit = t.get("unit", "")
            status = t.get("status", "unspecified")
            lines.append(f"- {name}: {val} {unit} — {status}")
        txt = ("Here's a brief summary of your labs.\n\n" +
               "\n".join(lines) +
               "\n\nThis is educational only; please consult a clinician.")
        return ExplainOut(explanation=txt)

    # call Gemini
    prompt = (
        "You are a clinician assistant. Given structured lab data, produce a concise, "
        "patient-friendly explanation (<=200 words). Not medical advice.\n\n" +
        f"{json.dumps(structured, ensure_ascii=False)}"
    )
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
            )
            r = await client.post(
                url,
                params={"key": GEMINI_API_KEY},
                headers={"Content-Type": "application/json"},
                json={"contents": [{"role": "user", "parts": [{"text": prompt.strip()}]}]},
            )
            r.raise_for_status()
            data = r.json()
            text = (
                data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
            ) or "I couldn’t generate a response."
            return ExplainOut(explanation=text.strip())
    except Exception as e:
        logger.exception("explain failed")
        return ExplainOut(explanation=f"⚠️ Gemini error: {e}")

# Register the local /api/explain route after it has been attached to the router
app.include_router(router)

def _sanitize_filename(name: str) -> str:
    name = os.path.basename(name or "file")
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return name or "file"


MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "10"))


@app.post("/api/extract_text")
@limiter.limit("2/minute")
async def extract_text(request: Request, file: UploadFile = File(...)):
    # Read into memory; never write to disk
    data = await file.read()
    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds the {MAX_FILE_MB}MB limit",
        )

    mt = (file.content_type or "").lower()
    name = _sanitize_filename(file.filename or "file")

    # Enforce allowed MIME types strictly
    if not (mt == "application/pdf" or mt.startswith("image/")):
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {mt or 'unknown'}")

    # PDF path
    if mt == "application/pdf" or name.lower().endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(data))
            text = [page.extract_text() or "" for page in reader.pages]
            joined = "\n".join(text).strip()
            if not joined:
                raise ValueError("No text found in PDF")
            return {"text": joined}
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"PDF extraction failed: {e}")

    # Image OCR path
    try:
        img = Image.open(io.BytesIO(data))
        txt = pytesseract.image_to_string(img, lang="eng")
        if not txt.strip():
            raise ValueError("OCR found no text")
        return {"text": txt}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OCR failed: {e}")

# -------------------- Table/line parser helpers (new) --------------------

RANGE_BRACKET = re.compile(r"[\[](?P<body>[^\\\]]+)[\]]")  # [<=5.2], [10-120], [200-239mg/dL]
NUM = r"\\d+(?:\\.\\d+)?"

def clean_value_token(tok: str) -> str:
    """Strip trailing punctuation and glued flags like '1270H.1' -> '1270' and return numeric string."""
    t = tok.strip()
    # split glued H/L flags next to number (e.g., 1270H.1 or 1270H)
    t = re.sub(r"(?i)\b(" + NUM + r")[HhLl](?:\\."+")?", r"\1", t)  # tolerate an extra dot after H/L
    # remove trailing dots and commas
    t = re.sub(r"[.,]+", "", t)
    # keep only leading numeric
    m = re.match(r"^\\s*(" + NUM + ")", t)
    return m.group(1) if m else ""

def detect_flag(token_or_line: str) -> Optional[str]:
    s = token_or_line.lower()
    if re.search(r"\bhigh\b|\(high\)|\bH\b", s): return "high"
    if re.search(r"\blow\b|\(low\)|\bL\b", s): return "low"
    return None


def parse_bracket_range(line: str) -> Optional[Dict[str, Any]]:
    """
    Canonicalize bracket ranges to one of:
      {"kind":"lte","v":N,"unit":"..."}
      {"kind":"lt","v":N,"unit":"..."}
      {"kind":"gte","v":N,"unit":"..."}
      {"kind":"gt","v":N,"unit":"..."}
      {"kind":"between","lo":A,"hi":B,"unit":"..."}
    """
    m = RANGE_BRACKET.search(line)
    if not m:
        return None
    body = m.group("body").replace(" ", "")
    # <=, <, >=, >
    m2 = re.match(r"^(<=|<|>=|>)(" + NUM + r")([a-zA-Zµμ%/\\^-]+)?$", body)
    if m2:
        op, val, unit_tail = m2.groups()
        unit = normalize_unit_text(unit_tail or "")
        v = float(val)
        op_map = {"<=": "lte", "<": "lt", ">=": "gte", ">": "gt"}
        return {"kind": op_map[op], "v": v, "unit": unit}
    # a-b (optional unit)
    m3 = re.match(r"^(" + NUM + r")-(" + NUM + r")([a-zA-Zµμ%/\\^-]+)?$", body)
    if m3:
        lo, hi, unit_tail = m3.groups()
        unit = normalize_unit_text(unit_tail or "")
        return {"kind": "between", "lo": float(lo), "hi": float(hi), "unit": unit}
    return None

def compare_to_range(value: float, rng: Dict[str, Any]) -> Optional[str]:
    """Return 'high'/'low'/'normal' from value vs canonical range."""
    if not rng:
        return None
    k = rng.get("kind")
    if k == "lte":
        return "high" if value > rng["v"] else "normal"
    if k == "lt":
        return "high" if value >= rng["v"] else "normal"
    if k == "gte":
        return "low"  if value < rng["v"] else "normal"
    if k == "gt":
        return "low"  if value <= rng["v"] else "normal"
    if k == "between":
        lo, hi = rng["lo"], rng["hi"]
        if value < lo: return "low"
        if value > hi: return "high"
        return "normal"
    return None

NAME_UNIT_IN_PARENS = re.compile(r"^(?P<name>.+?)\((?P<unit>[^)]+)\)\s*$", re.IGNORECASE)

ROW_PATTERN = re.compile(
    r"""^
    (?P<proc>[A-Za-z][A-Za-z0-9\s/_\-]+\((?:[^)]+\))?)"""   # Procedure name (might include (unit))
    """\s+"""
    """(?P<valtok>[-+]?""" + NUM + r"(?:[HhLl](?:\.)?)?|\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:[HhLl](?:\.)?)?)"""  # value token with optional H/L
    """\s*\.?"""                                               # optional trailing dot after value
    """\s+"""
    """(?P<unit>[^\[\]\s]+(?:/[^\[\]\s]+)?)?"""                # unit token (e.g., mg/dL)
    """(?P<rest>.*)$"""
    , re.VERBOSE
)

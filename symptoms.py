# backend/schemas/symptoms.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class SymptomSpan(BaseModel):
    text: str
    start: int
    end: int
    label: str
    score: float
    negated: bool = False

class SymptomAnalysisResult(BaseModel):
    symptoms: List[SymptomSpan]
    urgency: Literal["urgent", "moderate", "low", "unknown"] = "unknown"
    summary: str

class SymptomAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

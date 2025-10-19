# backend/schemas/symptoms.py
from pydantic import BaseModel, Field
from typing import List, Optional


class SymptomSpan(BaseModel):
    """Represents a single detected symptom entity."""

    text: str = Field(..., description="The text of the symptom.")
    start: int = Field(..., description="Start character index of the symptom.")
    end: int = Field(..., description="End character index of the symptom.")
    score: float = Field(..., description="Confidence score from the NER model.")
    label: str = Field(..., description="Entity label (e.g., 'SYMPTOM').")
    negated: bool = Field(False, description="Whether the symptom is negated.")


class SymptomAnalysisRequest(BaseModel):
    """Request model for the symptom analysis endpoint."""

    text: str = Field(..., max_length=5000, description="User-provided text describing symptoms.")


class SymptomAnalysisResult(BaseModel):
    """Response model for the symptom analysis endpoint."""

    symptoms: List[SymptomSpan] = Field(..., description="List of detected symptoms.")
    urgency: str = Field(..., description="Urgency classification (e.g., 'normal', 'urgent').")
    summary: str = Field(..., description="A brief summary of the analysis.")
    engine: str = Field(..., description="The NER engine used for analysis.")


class SymptomParsedItem(BaseModel):
    name: str
    canonical: str
    negated: bool = False
    onset: Optional[str] = None
    duration: Optional[str] = None
    severity: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class SymptomParseResult(BaseModel):
    symptoms: List[SymptomParsedItem]
    overall_confidence: float = Field(0.0, ge=0.0, le=1.0)

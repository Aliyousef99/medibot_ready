# backend/schemas.py
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Any
from datetime import datetime

# ---------- User ----------
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserOut(BaseModel):
    id: str
    email: EmailStr
    name: Optional[str]

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

# ---------- User Profile ----------
class UserProfileIn(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    sex: Optional[str] = Field(default=None, description='male|female|other')
    conditions: Optional[list[str]] = None
    medications: Optional[list[str]] = None
    notes: Optional[str] = None
    consent_given: Optional[bool] = Field(default=None, description="Explicit consent to process/store data")

class UserProfileOut(UserProfileIn):
    user_id: str
    consent_at: Optional[datetime] = None

    class Config:
        from_attributes = True

# ---------- Lab Reports ----------
class LabReportCreate(BaseModel):
    title: Optional[str] = None
    raw_text: str
    structured_json: Optional[dict] = None
    summary: Optional[str] = None

class LabReportOut(BaseModel):
    id: str
    user_id: str
    title: Optional[str] = None
    raw_text: str
    structured_json: Optional[dict] = None
    summary: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

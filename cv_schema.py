"""
cv_schema.py
Pydantic schemas for CV upload and analysis response.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class Education(BaseModel):
    degree: str
    field: Optional[str] = None
    institution: str
    graduation_year: Optional[int] = None


class Experience(BaseModel):
    job_title: str
    company: Optional[str] = "N/A"
    duration: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    responsibilities: Optional[str] = None

    @field_validator("company", mode="before")
    @classmethod
    def validate_company(cls, v):
        """Ensure company is never None."""
        return v if v else "N/A"

    @field_validator("duration", mode="before")
    @classmethod
    def calculate_duration(cls, v, info):
        """Calculate duration from start_date and end_date if not provided."""
        if v:
            return v
        data = info.data
        start_date = data.get("start_date")
        end_date = data.get("end_date")
        if start_date and end_date:
            return f"{start_date} - {end_date}"
        elif start_date:
            return f"{start_date} - Present"
        return "N/A"


class CVAnalysisResponse(BaseModel):
    full_name: str = ""
    email: str = ""
    phone: Optional[str] = None
    location: Optional[str] = None
    languages: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)
    summary: Optional[str] = None
    database_saved: bool = False
    message: str = "CV analyzed successfully"


class CVReviewResponse(BaseModel):
    """
    Response returned by POST /cv-box
    Contains both the structured parsed data AND the GROQ-generated human review.
    """
    # ── Parsed CV fields ──────────────────────────────────────────────────────
    full_name:      str            = ""
    email:          str            = ""
    phone:          Optional[str]  = None
    location:       Optional[str]  = None
    languages:      List[str]      = Field(default_factory=list)
    skills:         List[str]      = Field(default_factory=list)
    certifications: List[str]      = Field(default_factory=list)
    education:      List[Education]  = Field(default_factory=list)
    experience:     List[Experience] = Field(default_factory=list)
    summary:        Optional[str]  = None
    database_saved: bool           = False
    message:        str            = "CV analyzed successfully"

    # ── GROQ review ───────────────────────────────────────────────────────────
    cv_review:      str            = ""

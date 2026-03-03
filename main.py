"""
main.py
Job Matching AI Service — all three features implemented:

  POST /cv-box              Feature 1: CV parsing + GROQ review
  POST /match               Feature 2: Job matching + personalized roadmap (max 10 jobs)
  POST /interview-questions Feature 3: Generate 10 MCQ questions + save to DB
"""

from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from pydantic import BaseModel, validator, Field
from typing import List, Optional
from contextlib import asynccontextmanager
import json
import logging

from embedding_service import EmbeddingService
from scoring import fuzzy_score, hybrid_score, get_matched_missing
from groq_service import generate_general_prompt, generate_cv_review, generate_interview_questions
from database import Database
from cv_analyzer import CVAnalyzer
from cv_schema import Education, Experience, CVReviewResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Rate Limiter ─────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])

# ─── Service instances ────────────────────────────────────────────────────────
_embedder: Optional[EmbeddingService] = None
_db:       Optional[Database]         = None

MAX_ROADMAP_JSON_BYTES = 1_000_000   # 1 MB cap


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embedder, _db
    logger.info("🚀 Starting Job Matching AI Service…")
    _embedder = EmbeddingService()
    _db       = Database()
    logger.info("✅ Service ready!")
    yield
    logger.info("🛑 Shutting down…")


app = FastAPI(
    title="Job Matching AI Service",
    description="CV parsing, job matching with roadmap, and interview question generation.",
    version="4.0.0",
    lifespan=lifespan,
)

# ─── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Max 10 requests/minute."},
    )


# ─── Dependency injection ─────────────────────────────────────────────────────
def get_embedder() -> EmbeddingService:
    if _embedder is None:
        raise HTTPException(503, "Embedding service not initialized yet")
    return _embedder

def get_db() -> Database:
    if _db is None:
        raise HTTPException(503, "Database service not initialized yet")
    return _db


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 1 — CV Parsing + GROQ Review
# POST /cv-box
# ═══════════════════════════════════════════════════════════════════════════════

ALLOWED_CV_EXTENSIONS = {"pdf", "docx", "doc", "txt"}
MAX_CV_FILE_SIZE      = 10 * 1024 * 1024   # 10 MB


@app.post("/cv-box", response_model=CVReviewResponse, tags=["CV Analysis"])
@limiter.limit("10/minute")
async def upload_cv(
    request: Request,
    file: UploadFile = File(...),
    application_user_id: str = Form(...),      # required per spec
    db: Database = Depends(get_db),
):
    """
    Upload a CV (PDF / DOCX / TXT).

    Processing:
      1. Parse CV → structured JSON
      2. Delete old CV data for this user from DB
      3. Save new parsed data to DB
      4. Use GROQ to generate a human-readable review

    Returns structured CV fields + cv_review text.
    """
    # ── Validate file ─────────────────────────────────────────────────────────
    file_extension = file.filename.split(".")[-1].lower() if file.filename else ""
    if file_extension not in ALLOWED_CV_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file_extension}'. Allowed: {', '.join(ALLOWED_CV_EXTENSIONS)}",
        )

    file_content = await file.read()
    if len(file_content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(file_content) > MAX_CV_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds the 10 MB limit ({len(file_content):,} bytes).",
        )

    # ── Step 1: Parse CV ──────────────────────────────────────────────────────
    try:
        analyzer = CVAnalyzer()
        analysis = analyzer.analyze(file_content, file_extension)
    except ValueError as ve:
        logger.error(f"CV analysis validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"CV analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"CV analysis failed: {e}")

    # Normalize experience fields
    for exp in analysis.get("experience", []):
        if not exp.get("duration"):
            start = exp.get("start_date", "")
            end   = exp.get("end_date", "")
            if start and end:
                exp["duration"] = f"{start} - {end}"
            elif start:
                exp["duration"] = f"{start} - Present"
            else:
                exp["duration"] = "N/A"
        if not exp.get("company"):
            exp["company"] = "N/A"

    # ── Step 2 & 3: Delete old + save new to DB ───────────────────────────────
    database_saved = False
    message        = "CV analyzed successfully"
    try:
        db_result      = db.upsert_cv_data(analysis, application_user_id)
        database_saved = db_result.get("saved", False)
        message        = db_result.get("message", message)
    except Exception as db_err:
        logger.error(f"CV DB persist error: {db_err}")
        raise HTTPException(status_code=500, detail=f"Database operation failed: {db_err}")

    # ── Step 4: Generate GROQ review ──────────────────────────────────────────
    try:
        cv_review = generate_cv_review(analysis)
    except Exception as review_err:
        logger.error(f"CV review generation error: {review_err}")
        cv_review = "Review generation failed. Please retry."

    return CVReviewResponse(
        full_name      = analysis.get("full_name", ""),
        email          = analysis.get("email", ""),
        phone          = analysis.get("phone"),
        location       = analysis.get("location"),
        languages      = analysis.get("languages", []),
        skills         = analysis.get("skills", []),
        certifications = analysis.get("certifications", []),
        education      = analysis.get("education", []),
        experience     = analysis.get("experience", []),
        summary        = analysis.get("summary"),
        database_saved = database_saved,
        message        = message,
        cv_review      = cv_review,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 2 — Job Matching + Personalized Roadmap
# POST /match
# ═══════════════════════════════════════════════════════════════════════════════

class Job(BaseModel):
    job_id:          str
    job_title:       str
    job_description: str
    job_skills:      List[str]

    @validator("job_skills")
    def skills_not_empty(cls, v):
        if not v:
            raise ValueError("job_skills cannot be empty")
        return v


class MatchRequest(BaseModel):
    user_id:     str
    user_skills: List[str] = Field(default_factory=list)  # optional — fetched from DB if empty
    jobs:        List[Job]

    @validator("user_id")
    def validate_user_id(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("user_id cannot be empty")
        return v

    @validator("jobs")
    def jobs_not_empty(cls, v):
        if not v:
            raise ValueError("jobs list cannot be empty")
        if len(v) > 10:
            raise ValueError("Maximum 10 jobs per request")
        return v


class JobResult(BaseModel):
    job_id:            str
    job_title:         str
    match_percentage:  float
    matched_skills:    List[str]
    missing_skills:    List[str]


class MatchResponse(BaseModel):
    user_id:           str
    total_jobs:        int
    prompt_db_id:      int
    generation_failed: bool
    results:           List[JobResult]


@app.post("/match", response_model=MatchResponse, tags=["Job Matching"])
@limiter.limit("10/minute")
async def match_jobs(
    request:  Request,
    body:     MatchRequest,
    embedder: EmbeddingService = Depends(get_embedder),
    db:       Database         = Depends(get_db),
):
    """
    Match up to 10 jobs against user skills (fetched from DB by user_id).
    Generates a personalized learning roadmap and saves it to DB.
    """
    # ── Resolve user skills (from DB, fall back to request body) ─────────────
    user_skills = db.get_user_skills(body.user_id)
    if not user_skills:
        # fall back to skills provided in request (if any)
        user_skills = [s.strip() for s in body.user_skills if s.strip()]
    if not user_skills:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No skills found for user '{body.user_id}' in the database. "
                "Please upload a CV first via POST /cv-box."
            ),
        )

    logger.info(f"[{body.user_id}] Processing {len(body.jobs)} jobs with {len(user_skills)} skills")

    # ── Step 1: Score all jobs ────────────────────────────────────────────────
    scored_jobs = []
    for job in body.jobs:
        semantic         = embedder.semantic_score(user_skills, job.job_skills)
        fuzz             = fuzzy_score(user_skills, job.job_skills)
        matched, missing = get_matched_missing(user_skills, job.job_skills)
        score            = hybrid_score(semantic, fuzz, len(matched), len(job.job_skills))

        scored_jobs.append({
            "job_id":           job.job_id,
            "title":            job.job_title,
            "description":      job.job_description,
            "match_percentage": score,
            "matched":          matched,
            "missing":          missing,
        })

    scored_jobs.sort(key=lambda x: x["match_percentage"], reverse=True)
    logger.info(f"[{body.user_id}] Scoring done.")

    # ── Step 2: Generate personalized roadmap ────────────────────────────────
    try:
        roadmap = generate_general_prompt(user_skills, scored_jobs)
    except Exception as e:
        logger.error(f"[{body.user_id}] Groq roadmap generation failed: {e}")
        raise HTTPException(502, f"LLM service error: {e}")

    generation_failed = roadmap.get("generation_failed", False)

    # ── Step 3: Validate size + save roadmap to DB ────────────────────────────
    roadmap_json = json.dumps(roadmap, ensure_ascii=False)
    if len(roadmap_json.encode("utf-8")) > MAX_ROADMAP_JSON_BYTES:
        logger.warning(f"[{body.user_id}] Roadmap JSON too large, truncating modules")
        roadmap["modules"] = roadmap["modules"][:6]

    try:
        prompt_db_id = db.save_prompt_roadmap(
            title=roadmap.get("roadmap_title", "Roadmap"),
            description=roadmap,
            user_id=body.user_id,
        )
    except Exception as e:
        logger.error(f"[{body.user_id}] DB save roadmap error: {e}")
        raise HTTPException(500, f"Database error: {e}")

    # ── Step 4: Build response ────────────────────────────────────────────────
    results = [
        JobResult(
            job_id           = j["job_id"],
            job_title        = j["title"],
            match_percentage = j["match_percentage"],
            matched_skills   = j["matched"],
            missing_skills   = j["missing"],
        )
        for j in scored_jobs
    ]

    logger.info(f"[{body.user_id}] ✅ Done. {len(results)} results. generation_failed={generation_failed}")
    return MatchResponse(
        user_id           = body.user_id,
        total_jobs        = len(results),
        prompt_db_id      = prompt_db_id,
        generation_failed = generation_failed,
        results           = results,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 3 — Interview Questions (10 MCQ)
# POST /interview-questions
# ═══════════════════════════════════════════════════════════════════════════════

class InterviewJobInput(BaseModel):
    job_id:          str
    job_title:       str
    job_description: str
    job_skills:      List[str]

    @validator("job_skills")
    def skills_not_empty(cls, v):
        if not v:
            raise ValueError("job_skills cannot be empty")
        return v


class InterviewRequest(BaseModel):
    job: InterviewJobInput


@app.post("/interview-questions", status_code=200, tags=["Interview Questions"])
@limiter.limit("10/minute")
async def generate_interview_questions_endpoint(
    request: Request,
    body:    InterviewRequest,
    db:      Database = Depends(get_db),
):
    """
    Generate 10 MCQ interview questions for a given job using GROQ,
    save them to the database, and return HTTP 200 OK.
    """
    job = body.job
    logger.info(f"Generating interview questions for job_id={job.job_id}, title='{job.job_title}'")

    # ── Generate questions via GROQ ───────────────────────────────────────────
    try:
        questions = generate_interview_questions(job.dict())
    except Exception as e:
        logger.error(f"Interview question generation error: {e}")
        raise HTTPException(502, f"LLM service error: {e}")

    # ── Save to DB ────────────────────────────────────────────────────────────
    try:
        saved_id = db.save_interview_questions(
            job_id    = job.job_id,
            job_title = job.job_title,
            questions = questions,
        )
    except Exception as e:
        logger.error(f"Interview questions DB save error: {e}")
        raise HTTPException(500, f"Database error: {e}")

    logger.info(f"✅ Interview questions saved: db_id={saved_id}, job_id={job.job_id}")
    return {"message": "Interview questions generated and saved successfully", "db_id": saved_id, "total_questions": len(questions)}


# ═══════════════════════════════════════════════════════════════════════════════
# Health Check
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["Health"])
async def health(
    embedder: EmbeddingService = Depends(get_embedder),
    db:       Database         = Depends(get_db),
):
    return {
        "status":               "ok",
        "model":                "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_cache_size": embedder.cache_size,
    }

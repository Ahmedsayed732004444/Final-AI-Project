"""
database.py
SQL Server database layer — aligned with the C# / EF Core schema.

Real tables used (already managed by EF Core migrations in the .NET project):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  C# Entity          │  Table Name          │  Purpose               │
  ├─────────────────────┼──────────────────────┼────────────────────────┤
  │  ModelExtration     │  ModelExtrations    │  Parsed CV per user    │
  │    └─ Education     │  Education           │  Owned (separate rows) │
  │    └─ Experience    │  Experience          │  Owned (separate rows) │
  │  RoadmapJson        │  RoadmapJsons        │  LLM roadmap per user  │
  │  JobInterview       │  JobInterviews       │  MCQ question per job  │
  │    └─ JobInterviewOption │ JobInterviewOption │ 4 options per Q   │
  └─────────────────────────────────────────────────────────────────────┘

NOTE: This service never creates or migrates tables — that is EF Core's job.
      All table/column names here mirror the C# entity conventions exactly.
"""

import json
import logging
import os
import pyodbc
from typing import Any, Dict, List
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def _get_connection() -> pyodbc.Connection:
    server   = os.getenv("DB_SERVER")
    database = os.getenv("DB_NAME")
    driver   = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
    trusted  = os.getenv("DB_TRUSTED", "no").lower() in ("yes", "true", "1")
    username = os.getenv("DB_USER", "")
    password = os.getenv("DB_PASSWORD", "")

    # Base connection string — works for both local and remote
    conn_str = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        "Encrypt=yes;"
        "TrustServerCertificate=yes;"
        "Connection Timeout=30;"
    )

    if trusted:
        # Windows Authentication (Integrated Security=True)
        # Used for local SQL Server instances like .\Ahmed1
        conn_str += "Trusted_Connection=yes;"
    else:
        # SQL Server Authentication (username + password)
        # Used for remote/cloud servers
        conn_str += f"UID={username};PWD={password};"

    logger.debug(f"Connecting → SERVER={server}, DB={database}, WindowsAuth={trusted}")
    return pyodbc.connect(conn_str)


class Database:
    """
    Thin SQL Server wrapper.
    Every public method opens its own connection and closes it on exit.
    No table creation here — EF Core owns all schema migrations.
    """

    def __init__(self) -> None:
        # In-memory cache for frequently read data (e.g., user skills).
        # Safe across requests within the same process and provides a
        # lightweight fallback if the database becomes temporarily unavailable.
        self._skills_cache: dict[str, list] = {}

    # ═══════════════════════════════════════════════════════════════════════════
    # FEATURE 1 — CV Parsing → ModelExtrations + Education + Experience
    # ═══════════════════════════════════════════════════════════════════════════

    def upsert_cv_data(self, analysis: Dict[str, Any], application_user_id: str) -> Dict[str, Any]:
        """
        Delete the existing ModelExtration row for this user (cascades to
        Education and Experience via EF Core owned-entity convention),
        then insert a fresh set of rows.

        C# entities involved:
          ModelExtration      → ModelExtrations table
          Education (Owned)   → Education table   (ModelExtrationId FK)
          Experience (Owned)  → Experience table  (ModelExtrationId FK)

        Skills / Certifications / Languages are stored as JSON strings
        because EF Core maps List<string> to a JSON column in newer EF or
        via value converters. We store them the same way.
        """
        try:
            with _get_connection() as conn:
                cur = conn.cursor()

                # ── 1. Delete old ModelExtration (owned entities cascade) ─────
                # Education and Experience rows have a FK to ModelExtrations.Id
                # so we delete them first to avoid FK violations.
                cur.execute("""
                    DELETE e FROM Education e
                    INNER JOIN ModelExtrations me ON e.ModelExtrationId = me.Id
                    WHERE me.ApplicationUserId = ?
                """, application_user_id)

                cur.execute("""
                    DELETE ex FROM Experience ex
                    INNER JOIN ModelExtrations me ON ex.ModelExtrationId = me.Id
                    WHERE me.ApplicationUserId = ?
                """, application_user_id)

                cur.execute(
                    "DELETE FROM ModelExtrations WHERE ApplicationUserId = ?",
                    application_user_id,
                )

                # ── 2. Insert fresh ModelExtration row ────────────────────────
                # Skills, Certifications, Languages stored as JSON strings
                # matching EF Core's value-converter pattern for List<string>.
                cur.execute("""
                    INSERT INTO ModelExtrations
                        (ApplicationUserId, FullName, Email, Phone, Location,
                         Summary, Skills, Certifications, Languages)
                    OUTPUT INSERTED.Id
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    application_user_id,
                    analysis.get("full_name", ""),
                    analysis.get("email", ""),
                    analysis.get("phone") or "",
                    analysis.get("location") or "",
                    analysis.get("summary") or "",
                    json.dumps(analysis.get("skills", []),         ensure_ascii=False),
                    json.dumps(analysis.get("certifications", []), ensure_ascii=False),
                    json.dumps(analysis.get("languages", []),      ensure_ascii=False),
                )
                row = cur.fetchone()
                model_extration_id = row[0] if row else None

                if model_extration_id is None:
                    raise RuntimeError("Failed to retrieve inserted ModelExtration Id")

                # ── 3. Insert Education rows (bulk) ───────────────────────────
                # C# Education: Degree, Field, Institution, Year (all strings)
                education_rows = [
                    (
                        model_extration_id,
                        edu.get("degree", ""),
                        edu.get("field") or "",
                        edu.get("institution", ""),
                        str(edu.get("graduation_year") or ""),
                    )
                    for edu in analysis.get("education", [])
                ]
                if education_rows:
                    cur.fast_executemany = True
                    cur.executemany(
                        """
                        INSERT INTO Education
                            (ModelExtrationId, Degree, Field, Institution, Year)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        education_rows,
                    )

                # ── 4. Insert Experience rows (bulk) ──────────────────────────
                # C# Experience: JobTitle, Company, StartDate, EndDate, Description
                experience_rows = [
                    (
                        model_extration_id,
                        exp.get("job_title", ""),
                        exp.get("company") or "N/A",
                        exp.get("start_date") or "",
                        exp.get("end_date") or "",
                        exp.get("responsibilities") or "",
                    )
                    for exp in analysis.get("experience", [])
                ]
                if experience_rows:
                    cur.fast_executemany = True
                    cur.executemany(
                        """
                        INSERT INTO Experience
                            (ModelExtrationId, JobTitle, Company, StartDate, EndDate, Description)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        experience_rows,
                    )

                conn.commit()
                # Warm the in-memory skills cache with latest skills
                self._skills_cache[application_user_id] = analysis.get("skills", []) or []
                logger.info(
                    f"CV saved → ModelExtrations.Id={model_extration_id}, "
                    f"user={application_user_id}"
                )
                return {"saved": True, "message": "CV data saved successfully"}

        except Exception as e:
            logger.error(f"upsert_cv_data error: {e}")
            # Fallback: leave cache untouched and signal failure to caller
            return {"saved": False, "message": f"DB error: {e}"}

    def get_user_skills(self, user_id: str) -> List[str]:
        """
        Read Skills JSON string from ModelExtrations and return as a Python list.
        Returns [] if the user has no CV record yet.
        """
        # Fast path: serve from in-memory cache when available
        cached = self._skills_cache.get(user_id)
        if cached is not None:
            return list(cached)

        try:
            with _get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT Skills FROM ModelExtrations WHERE ApplicationUserId = ?",
                    user_id,
                )
                row = cur.fetchone()
                if row and row[0]:
                    skills = json.loads(row[0])
                    # Cache result for subsequent requests
                    self._skills_cache[user_id] = skills or []
                    return skills
                # Cache negative lookups to avoid repeated DB hits
                self._skills_cache[user_id] = []
                return []
        except Exception as e:
            logger.error(f"get_user_skills error: {e}")
            # Fallback to last known cached value (if any)
            cached = self._skills_cache.get(user_id)
            return list(cached) if cached is not None else []

    # ═══════════════════════════════════════════════════════════════════════════
    # FEATURE 2 — Job Matching Roadmap → RoadmapJsons
    # ═══════════════════════════════════════════════════════════════════════════

    def save_prompt_roadmap(self, title: str, description: Any, user_id: str) -> int:
        """
        Persist the LLM-generated roadmap JSON into RoadmapJsons.
        RoadmapData column stores the full roadmap JSON string.

        C# entity: RoadmapJson { Id, ApplicationUserId, RoadmapData }

        Returns the auto-generated Id.
        """
        try:
            roadmap_data = (
                json.dumps(description, ensure_ascii=False)
                if not isinstance(description, str)
                else description
            )

            with _get_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO RoadmapJsons (ApplicationUserId, RoadmapData, CreatedAt, IsSaved)
                    OUTPUT INSERTED.Id
                    VALUES (?, ?, GETDATE(), 0)
                """, user_id, roadmap_data)
                row = cur.fetchone()
                conn.commit()
                db_id = row[0] if row else -1
                logger.info(f"Roadmap saved → RoadmapJsons.Id={db_id}, user={user_id}")
                return db_id
        except Exception as e:
            logger.error(f"save_prompt_roadmap error: {e}")
            raise

    # ═══════════════════════════════════════════════════════════════════════════
    # FEATURE 3 — Interview Questions → JobInterviews + JobInterviewOption
    # ═══════════════════════════════════════════════════════════════════════════

    def save_interview_questions(self, job_id: str, job_title: str, questions: List[Dict]) -> int:
        """
        Persist 10 MCQ questions into JobInterviews + JobInterviewOption.

        C# entities:
          JobInterview       { Id, Question, JobId }
          JobInterviewOption { Id, JobInterviewId, OptionText, IsCorrect }

        Each question → 1 JobInterviews row + 4 JobInterviewOption rows.

        Returns the Id of the LAST inserted JobInterview row
        (kept consistent with the original int return contract).
        """
        if not questions:
            raise ValueError("questions list is empty")

        last_id = -1
        try:
            with _get_connection() as conn:
                cur = conn.cursor()

                # ── Delete old questions + options for this job ───────────
                cur.execute("""
                    DELETE FROM JobInterviewOption
                    WHERE JobInterviewId IN (
                        SELECT Id FROM JobInterviews WHERE JobId = ?
                    )
""", job_id)
                cur.execute(
                    "DELETE FROM JobInterviews WHERE JobId = ?",
                    job_id,
                )
                logger.info(f"Deleted old interview questions for job_id={job_id}")

                for q in questions:
                    question_text = q.get("question", "")
                    correct_key   = q.get("correct_answer", "A").upper()
                    options_dict  = q.get("options", {})

                    # ── Insert JobInterview row ───────────────────────────────
                    cur.execute(
                        """
                        INSERT INTO JobInterviews (Question, JobId)
                        OUTPUT INSERTED.Id
                        VALUES (?, ?)
                        """,
                        question_text,
                        job_id,
                    )
                    ji_row = cur.fetchone()
                    ji_id  = ji_row[0] if ji_row else None

                    if ji_id is None:
                        raise RuntimeError("Failed to retrieve inserted JobInterview Id")

                    last_id = ji_id

                    # ── Insert 4 JobInterviewOption rows (bulk per question) ──
                    option_rows = [
                        (ji_id, options_dict.get(option_key, ""), 1 if option_key == correct_key else 0)
                        for option_key in ["A", "B", "C", "D"]
                    ]
                    cur.fast_executemany = True
                    cur.executemany(
                        """
                        INSERT INTO JobInterviewOption
                            (JobInterviewId, OptionText, IsCorrect)
                        VALUES (?, ?, ?)
                        """,
                        option_rows,
                    )

                conn.commit()
                logger.info(
                    f"Interview questions saved → {len(questions)} questions, "
                    f"job_id={job_id}, last JobInterviews.Id={last_id}"
                )
                return last_id

        except Exception as e:
            logger.error(f"save_interview_questions error: {e}")
            raise
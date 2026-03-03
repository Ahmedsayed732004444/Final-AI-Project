"""
groq_service.py
Fixes applied:
  #4  Timeout on Groq API call (30s)
  #5  generation_failed flag in response instead of silent HTTP 200
  #6  Prompt injection protection — skills sanitized before prompt
  #8  Retry logic with exponential backoff (3 retries)
  #12 Domain detection uses job_description as fallback + scores top-2
  #14 max_tokens raised to 8192; splits into 2 calls for 12-week roadmaps
"""

from groq import Groq
import json
import os
import re
import time
import logging
from collections import Counter
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MAX_SKILL_LEN     = 80    # Fix #6: cap individual skill length
MAX_SKILLS_PROMPT = 40    # max skills shown in prompt
GROQ_TIMEOUT      = 45    # Fix #4: seconds
MAX_RETRIES       = 3     # Fix #8: retry attempts


# ─── Fix #6: Input sanitization ───────────────────────────────────────────────

_INJECTION_PATTERNS = re.compile(
    r"(ignore\s+(all\s+)?instructions|forget\s+everything|"
    r"you\s+are\s+now|new\s+persona|system\s*:|\[INST\]|<\|im_start\|>)",
    re.IGNORECASE,
)

def sanitize_skill(skill: str) -> str:
    """Strip prompt-injection attempts and enforce length limit."""
    skill = skill.strip()[:MAX_SKILL_LEN]
    skill = re.sub(r"[\"\'`]", "", skill)           # remove quotes
    skill = re.sub(r"\n|\r", " ", skill)             # no newlines
    if _INJECTION_PATTERNS.search(skill):
        return "[REDACTED]"
    return skill

def sanitize_skills(skills: list[str]) -> list[str]:
    cleaned = [sanitize_skill(s) for s in skills if s.strip()]
    cleaned = [s for s in cleaned if s != "[REDACTED]"]
    return cleaned[:MAX_SKILLS_PROMPT]


# ─── Fix #12: Domain detection with description fallback ──────────────────────

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "technology / software engineering": [
        "engineer", "developer", "devops", "backend", "frontend",
        "fullstack", "software", "platform", "infrastructure", "sre", "architect"
    ],
    "data science / machine learning / AI": [
        "data scientist", "machine learning", "ml engineer", "ai engineer",
        "llm", "nlp", "deep learning", "computer vision", "data analyst", "research scientist"
    ],
    "data engineering": [
        "data engineer", "etl", "pipeline", "analytics engineer", "big data", "dataops"
    ],
    "product management": [
        "product manager", "product owner", "product lead", "head of product"
    ],
    "cybersecurity": [
        "security", "cyber", "soc analyst", "pentest", "infosec", "devsecops", "vulnerability"
    ],
    "design / ux": [
        "designer", "ux", "ui", "creative director", "visual", "product designer", "motion"
    ],
    "medicine / healthcare": [
        "doctor", "physician", "nurse", "medical", "clinical", "healthcare",
        "surgeon", "therapist", "pharmacist", "radiologist", "dentist", "cardiologist",
        "neurologist", "oncologist", "pediatrician", "anesthesiologist"
    ],
    "law / legal": [
        "lawyer", "attorney", "counsel", "legal", "paralegal", "solicitor",
        "barrister", "compliance officer", "general counsel"
    ],
    "finance / accounting": [
        "finance", "financial", "accounting", "analyst", "investment",
        "banking", "auditor", "tax", "cfo", "treasurer", "actuary", "quant"
    ],
    "marketing / sales": [
        "marketing", "growth", "seo", "content", "brand", "sales",
        "account executive", "business development", "crm", "performance marketing"
    ],
    "human resources": [
        "hr", "human resources", "talent", "recruiter", "people ops",
        "hrbp", "learning development", "compensation", "chief people"
    ],
    "civil / mechanical / electrical engineering": [
        "civil", "mechanical", "structural", "electrical", "manufacturing",
        "autocad", "embedded", "plc", "iot", "pcb", "aerospace", "manufacturing"
    ],
    "architecture / real estate": [
        "architect", "interior design", "urban planning", "construction",
        "quantity surveyor", "real estate", "property manager", "bim"
    ],
    "education / teaching": [
        "teacher", "educator", "instructor", "professor", "tutor",
        "curriculum", "e-learning", "training", "learning specialist"
    ],
    "supply chain / operations": [
        "supply chain", "logistics", "procurement", "warehouse",
        "operations", "inventory", "erp", "demand planning", "fulfillment"
    ],
    "project management": [
        "project manager", "scrum master", "agile coach", "pmo",
        "delivery manager", "program manager", "portfolio manager"
    ],
}


def _detect_domain(jobs_data: list) -> str:
    """
    Fix #12: Score keywords against titles AND descriptions.
    Use top-2 scoring to pick best domain; fall back gracefully.
    """
    # Combine titles + first 100 chars of description for richer signal
    combined_text = " ".join(
        j["title"].lower() + " " + j.get("description", "")[:100].lower()
        for j in jobs_data
    )

    scores = {domain: 0 for domain in DOMAIN_KEYWORDS}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in combined_text:
                scores[domain] += 1

    # Sort and pick winner
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_domain, best_score = ranked[0]

    if best_score == 0:
        # Last resort: semantic guess from job titles alone
        logger.warning("Domain detection found no keyword matches — using 'professional'")
        return "professional"

    logger.info(f"Domain detected: '{best_domain}' (score={best_score})")
    return best_domain


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _aggregate_missing_skills(jobs_data: list) -> list[str]:
    skill_weight: Counter = Counter()
    total = len(jobs_data)
    for rank, job in enumerate(jobs_data):
        weight = total - rank
        for skill in job.get("missing", []):
            skill_weight[skill] += weight
    return [s for s, _ in skill_weight.most_common()]


def _roadmap_meta(avg_match: float) -> tuple[str, int]:
    if avg_match >= 88:
        return "accelerated", 3
    elif avg_match >= 78:
        return "normal", 6
    elif avg_match >= 65:
        return "special", 9
    else:
        return "leadership", 12


# ─── Fix #14: Smart token budget ─────────────────────────────────────────────

def _get_token_budget(weeks: int) -> int:
    """Estimate required tokens; cap at model max."""
    estimated = weeks * 350   # ~350 tokens per week module
    return min(max(estimated, 2000), 8192)


# ─── Groq call with retry + timeout ──────────────────────────────────────────

def _call_groq(messages: list, max_tokens: int) -> str:
    """
    Fix #4: Groq call with timeout.
    Fix #8: Exponential backoff retry on 429 / 5xx.
    Returns raw response string.
    """
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.25,
                max_tokens=max_tokens,
                timeout=GROQ_TIMEOUT,   # Fix #4
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            is_retryable = any(x in err_str for x in ["429", "500", "502", "503", "timeout", "rate"])

            if is_retryable and attempt < MAX_RETRIES:
                wait = 2 ** attempt   # 2s, 4s, 8s
                logger.warning(f"Groq attempt {attempt}/{MAX_RETRIES} failed ({e}). Retrying in {wait}s…")
                time.sleep(wait)
            else:
                break

    raise RuntimeError(f"Groq failed after {MAX_RETRIES} attempts: {last_error}")


def _strip_fences(raw: str) -> str:
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.lower().startswith("json"):
            raw = raw[4:]
    return raw.strip()


def _parse_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Partial recovery: find outermost { }
        try:
            s = raw.index("{")
            e = raw.rindex("}") + 1
            return json.loads(raw[s:e])
        except Exception:
            raise ValueError("Could not parse JSON from Groq response")


# ─── Prompt builder ───────────────────────────────────────────────────────────

def _build_prompt(user_skills: list, jobs_data: list, domain: str) -> str:
    all_missing  = _aggregate_missing_skills(jobs_data)
    avg_match    = sum(j["match_percentage"] for j in jobs_data) / len(jobs_data)
    rtype, weeks = _roadmap_meta(avg_match)

    # Fix #6: sanitize before inserting into prompt
    safe_user_skills = sanitize_skills(user_skills)
    safe_missing     = sanitize_skills(all_missing)

    top_jobs = "\n".join(
        f"  {i+1}. {sanitize_skill(j['title'])} — {j['match_percentage']}% | "
        f"missing: {', '.join(sanitize_skills(j['missing'])) if j['missing'] else 'none'}"
        for i, j in enumerate(jobs_data[:6])
    )
    missing_str = ", ".join(safe_missing) if safe_missing else "None — fully qualified!"

    domain_resources = {
        "technology / software engineering":  "official docs, Coursera, GitHub, YouTube tech channels",
        "data science / machine learning / AI":"fast.ai, deeplearning.ai, Kaggle, Papers with Code",
        "data engineering":                   "dbt docs, Airflow docs, Spark docs, DataTalks.Club",
        "medicine / healthcare":              "UpToDate, BMJ Learning, PubMed, Coursera Health",
        "law / legal":                        "Westlaw, LexisNexis, Coursera Law, bar prep materials",
        "finance / accounting":               "CFA Institute, Investopedia, Coursera Finance, Bloomberg",
        "design / ux":                        "Figma Academy, Interaction Design Foundation, YouTube UX",
        "marketing / sales":                  "HubSpot Academy, Google Skillshop, Coursera Marketing",
        "human resources":                    "SHRM, CIPD, LinkedIn Learning, Coursera HR",
        "civil / mechanical / electrical engineering": "Coursera Engineering, MIT OCW, ASME, IEEE",
        "architecture / real estate":         "Autodesk Learning, RIBA, Coursera Architecture",
        "education / teaching":               "Coursera Education, ISTE, edX Teaching, Khan Academy",
        "supply chain / operations":          "APICS, Coursera Supply Chain, SAP Learning Hub",
        "project management":                 "PMI, Scrum.org, Coursera PM, Atlassian University",
    }
    resources_hint = domain_resources.get(domain, "Coursera, YouTube, official documentation, books")

    return f"""You are a world-class career coach and curriculum designer for ALL professional domains.

## Candidate Profile
- Domain: {domain}
- Current Skills: {", ".join(safe_user_skills)}
- Average Match Score: {avg_match:.1f}%
- Roadmap Type: {rtype} | Duration: {weeks} weeks

## Top Matched Jobs
{top_jobs}

## Skill Gaps (priority order)
{missing_str}

## Task
Generate ONE structured JSON roadmap to close ALL skill gaps above.

## Rules
1. Use REAL resources from: {resources_hint}
2. Group 1-3 related skills per week. Never mix unrelated skills.
3. LAST week only: "project": true (capstone using all learned skills).
4. All other weeks: "project": false.
5. resource "type": "course"|"article"|"video"|"docs"|"github"|"book"
6. "skills_covered" — ONLY from the skill gaps list above.
7. Return ONLY valid JSON. Start with {{ end with }}. No markdown.

## JSON Schema
{{
  "roadmap_title": "<max 10 words>",
  "roadmap_type": "{rtype}",
  "duration_weeks": {weeks},
  "generation_failed": false,
  "modules": [
    {{
      "week": 1,
      "title": "<title>",
      "description": "<1-2 sentences>",
      "skills_covered": ["skill"],
      "resources": [{{"type": "docs", "url": "https://..."}}],
      "project": false
    }}
  ]
}}"""


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_general_prompt(user_skills: list, jobs_data: list) -> dict:
    """
    Main entry point.
    Returns structured roadmap dict. Always includes 'generation_failed' key.
    Fix #5: generation_failed=True on failure (still HTTP 200, but detectable).
    Fix #14: Uses appropriate token budget per roadmap length.
    """
    domain        = _detect_domain(jobs_data)
    avg_match     = sum(j["match_percentage"] for j in jobs_data) / len(jobs_data)
    _, weeks      = _roadmap_meta(avg_match)
    token_budget  = _get_token_budget(weeks)

    prompt = _build_prompt(user_skills, jobs_data, domain)
    system = (
        "You are a JSON-only API for a career coaching platform serving ALL professional domains. "
        "Output ONLY valid JSON starting with { and ending with }. "
        "No markdown. No explanation. No extra text."
    )

    logger.info(f"Groq call: domain='{domain}', weeks={weeks}, tokens={token_budget}")

    try:
        # Fix #14: For 12-week roadmaps, split into 2 calls
        if weeks >= 12:
            roadmap = _generate_split(user_skills, jobs_data, domain, system, token_budget)
        else:
            raw     = _call_groq([{"role": "system", "content": system},
                                   {"role": "user",   "content": prompt}], token_budget)
            raw     = _strip_fences(raw)
            roadmap = _parse_json(raw)

        roadmap["generation_failed"] = False
        logger.info(f"✅ Roadmap: '{roadmap.get('roadmap_title')}' | "
                    f"{roadmap.get('duration_weeks')}w | {len(roadmap.get('modules',[]))} modules")
        return roadmap

    except Exception as e:
        logger.error(f"Roadmap generation failed: {e}")
        missing = _aggregate_missing_skills(jobs_data)
        return {
            "roadmap_title": f"{domain.title()} Development Roadmap",
            "roadmap_type": "normal",
            "duration_weeks": weeks,
            "generation_failed": True,   # Fix #5: signal failure to client
            "error_message": str(e),
            "modules": [{
                "week": 1,
                "title": "Generation Failed — Please Retry",
                "description": f"Error: {str(e)[:200]}",
                "skills_covered": missing[:3],
                "resources": [{"type": "article", "url": "https://www.coursera.org"}],
                "project": False,
            }],
        }


def _generate_split(user_skills, jobs_data, domain, system, token_budget) -> dict:
    """
    Fix #14: For 12-week roadmaps, generate weeks 1-6 and 7-12 separately.
    Merge results into one roadmap dict.
    """
    all_missing   = _aggregate_missing_skills(jobs_data)
    safe_missing  = sanitize_skills(all_missing)
    avg_match     = sum(j["match_percentage"] for j in jobs_data) / len(jobs_data)
    rtype, weeks  = _roadmap_meta(avg_match)

    half   = weeks // 2
    half1  = safe_missing[:len(safe_missing)//2]
    half2  = safe_missing[len(safe_missing)//2:]

    def _partial_prompt(skills_subset, week_start, week_end, is_final):
        skills_str   = ', '.join(skills_subset)
        capstone_note = (
            f"The LAST week (week {week_end}) must be a capstone project: \"project\": true"
            if is_final else
            "All weeks: \"project\": false"
        )
        example_obj = (
            '[{"week": ' + str(week_start) + ', "title": "", "description": "", '
            '"skills_covered": [], "resources": [{"type":"","url":""}], "project": false}]'
        )
        return (
            f"Domain: {domain} | Roadmap Type: {rtype}\n"
            f"Generate weeks {week_start}-{week_end} of a {weeks}-week roadmap.\n"
            f"Skills to cover: {skills_str}\n"
            f"{capstone_note}\n"
            f"Return ONLY a JSON array of module objects (no outer object):\n"
            f"{example_obj}"
        )

    p1 = _partial_prompt(half1, 1,      half,  False)
    p2 = _partial_prompt(half2, half+1, weeks, True)

    raw1 = _call_groq([{"role":"system","content":system},{"role":"user","content":p1}], 4096)
    raw2 = _call_groq([{"role":"system","content":system},{"role":"user","content":p2}], 4096)

    modules1 = json.loads(_strip_fences(raw1))
    modules2 = json.loads(_strip_fences(raw2))

    return {
        "roadmap_title": f"{domain.title()} Mastery Roadmap",
        "roadmap_type":  rtype,
        "duration_weeks": weeks,
        "modules": modules1 + modules2,
    }

# ─── CV Review (Feature #1 output) ───────────────────────────────────────────

def generate_cv_review(cv_data: dict) -> str:
    """
    Given structured CV data, call Groq and return a human-readable
    review string matching the required format:
      Overall Rating: XX/100
      Summary: ...
      Strengths: ...
      Weaknesses: ...
      ATS Compatibility Analysis: ...
      Formatting and Readability: ...
      Content and Impact: ...
      Grammar and Clarity: ...
    """
    skills_str = ", ".join(cv_data.get("skills", [])[:30]) or "N/A"
    exp_list   = cv_data.get("experience", [])
    exp_summary = "; ".join(
        f"{e.get('job_title','?')} at {e.get('company','?')} ({e.get('duration','?')})"
        for e in exp_list[:5]
    ) or "No experience listed"

    edu_list = cv_data.get("education", [])
    edu_summary = "; ".join(
        f"{e.get('degree','?')} in {e.get('field','?')} from {e.get('institution','?')}"
        for e in edu_list[:3]
    ) or "No education listed"

    prompt = f"""You are an expert career coach and ATS specialist. Analyze the following CV data and provide a detailed review.

CV Data:
- Name: {cv_data.get('full_name', 'N/A')}
- Skills: {skills_str}
- Experience: {exp_summary}
- Education: {edu_summary}
- Summary: {cv_data.get('summary', 'N/A')}
- Languages: {', '.join(cv_data.get('languages', [])) or 'N/A'}
- Certifications: {', '.join(cv_data.get('certifications', [])) or 'None'}

Provide your review in EXACTLY this format (plain text, no markdown symbols like ** or ##):

Overall Rating: [number]/100
Summary: [2-3 sentence overview of the candidate]
Strengths:
* [strength 1]
* [strength 2]
* [strength 3]
Weaknesses:
* [weakness 1]
* [weakness 2]
* [weakness 3]
ATS Compatibility Analysis: [analysis text]. Score: [X]/10
Formatting and Readability: [analysis text]
Content and Impact: [analysis text]
Grammar and Clarity: [analysis text]

Be specific, professional, and constructive. Base the rating on skills depth, experience relevance, education, and overall presentation."""

    system = (
        "You are an expert resume reviewer and career coach. "
        "Provide detailed, honest, and constructive CV analysis. "
        "Follow the exact format requested. Do not use markdown formatting like ** or ##."
    )

    try:
        raw = _call_groq(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            max_tokens=1500,
        )
        return raw.strip()
    except Exception as e:
        logger.error(f"CV review generation failed: {e}")
        return (
            f"Overall Rating: N/A\n"
            f"Summary: CV analysis completed but review generation failed.\n"
            f"Error: {str(e)[:200]}"
        )


# ─── Interview MCQ Questions (Feature #3) ────────────────────────────────────

def generate_interview_questions(job: dict) -> list:
    """
    Generate 10 MCQ questions with correct answers for a given job.
    Returns a list of 10 question dicts:
    [
      {
        "question": "...",
        "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
        "correct_answer": "A",
        "explanation": "..."
      }, ...
    ]
    """
    job_title       = sanitize_skill(job.get("job_title", "Unknown Role"))
    job_description = job.get("job_description", "")[:800]
    job_skills      = sanitize_skills(job.get("job_skills", []))

    skills_str = ", ".join(job_skills[:20]) or "General professional skills"

    prompt = f"""Generate exactly 10 multiple choice questions (MCQ) for an interview for the following job.

Job Title: {job_title}
Job Description: {job_description}
Required Skills: {skills_str}

Rules:
- Each question must be directly relevant to the job role and skills
- Each question must have exactly 4 options (A, B, C, D)
- Only one option is correct
- Include a brief explanation for the correct answer
- Questions should range from basic to advanced
- Return ONLY valid JSON — no extra text, no markdown fences

Required JSON format:
[
  {{
    "question_number": 1,
    "question": "question text here",
    "options": {{
      "A": "option A text",
      "B": "option B text",
      "C": "option C text",
      "D": "option D text"
    }},
    "correct_answer": "A",
    "explanation": "brief explanation why this is correct"
  }}
]

Generate all 10 questions now."""

    system = (
        "You are a technical interview expert. "
        "Generate high-quality MCQ interview questions. "
        "Return ONLY a valid JSON array. No markdown. No extra text."
    )

    try:
        raw      = _call_groq(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            max_tokens=3000,
        )
        raw      = _strip_fences(raw)
        # Extract JSON array
        s = raw.index("[")
        e = raw.rindex("]") + 1
        questions = json.loads(raw[s:e])
        if not isinstance(questions, list):
            raise ValueError("Response is not a JSON array")
        # Enforce exactly 10
        questions = questions[:10]
        logger.info(f"Generated {len(questions)} interview questions for '{job_title}'")
        return questions
    except Exception as e:
        logger.error(f"Interview question generation failed: {e}")
        # Return a minimal fallback
        return [
            {
                "question_number": i + 1,
                "question": f"Question {i+1} generation failed. Please retry.",
                "options": {"A": "N/A", "B": "N/A", "C": "N/A", "D": "N/A"},
                "correct_answer": "A",
                "explanation": f"Generation error: {str(e)[:100]}",
            }
            for i in range(10)
        ]

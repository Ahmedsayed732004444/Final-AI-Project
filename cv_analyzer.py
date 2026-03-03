"""
cv_analyzer.py
Orchestrates file parsing and LLM-based CV analysis.
"""

import logging
import json
import re
from typing import Dict, Any

from file_parser import FileParser

logger = logging.getLogger(__name__)


def safe_json_parse(content: str) -> Dict[str, Any]:
    """Safely parse JSON from LLM response, stripping markdown fences."""
    content = content.strip()
    # Strip ```json ... ``` fences
    if content.startswith("```"):
        parts = content.split("```")
        content = parts[1] if len(parts) > 1 else content
        if content.lower().startswith("json"):
            content = content[4:]
    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to extract outermost { }
        try:
            s = content.index("{")
            e = content.rindex("}") + 1
            return json.loads(content[s:e])
        except Exception:
            raise ValueError("Could not parse JSON from LLM response")


class CVAnalyzer:
    def __init__(self):
        self.file_parser = FileParser()

    def analyze(self, file_content: bytes, file_extension: str) -> Dict[str, Any]:
        """Parse the file and run LLM analysis. Returns structured CV dict."""
        try:
            cv_text = self.file_parser.parse_file(file_content, file_extension)

            if not cv_text or len(cv_text.strip()) == 0:
                raise ValueError("Empty CV content after parsing")

            result = self._analyze_with_groq(cv_text)
            return result

        except Exception as e:
            logger.error(f"CV analysis error: {e}")
            raise

    def _analyze_with_groq(self, cv_text: str) -> Dict[str, Any]:
        """Send CV text to Groq and return structured data."""
        from groq import Groq
        import os
        from dotenv import load_dotenv

        load_dotenv()
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

        prompt = f"""Extract structured information from the following CV text. Return ONLY valid JSON with this exact structure:

{{
  "full_name": "string",
  "email": "string",
  "phone": "string or null",
  "location": "string (city, country) or null",
  "languages": ["English", "Arabic", "French"],
  "skills": ["skill1", "skill2"],
  "certifications": ["cert1", "cert2"] or [],
  "education": [
    {{
      "degree": "string",
      "field": "string or null",
      "institution": "string",
      "graduation_year": number or null
    }}
  ],
  "experience": [
    {{
      "job_title": "string",
      "company": "string",
      "start_date": "string (e.g., Jan 2020) or null",
      "end_date": "string (e.g., Dec 2022) or Present or null",
      "responsibilities": "string or null"
    }}
  ],
  "summary": "string or null"
}}

IMPORTANT EXTRACTION RULES:
- Extract ALL languages mentioned (English, Arabic, French, German, etc.)
- Extract location/address if mentioned (city, country)
- Extract ALL certifications, licenses, or credentials
- For experience: extract start_date and end_date separately
- For education: extract field of study if mentioned
- If information is not found, use null or empty array []

CV Text:
{cv_text}

Return ONLY the JSON object, no additional text."""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4096,
                timeout=45,
            )
            content = response.choices[0].message.content
            parsed = safe_json_parse(content)

        except Exception as e:
            logger.warning(f"First Groq attempt failed: {e}. Retrying…")
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=4096,
                    timeout=45,
                )
                content = response.choices[0].message.content
                parsed = safe_json_parse(content)
            except Exception as retry_err:
                logger.error(f"Groq retry failed: {retry_err}")
                raise ValueError(f"LLM service failed: {retry_err}")

        # Ensure defaults
        parsed.setdefault("languages", [])
        parsed.setdefault("location", None)
        parsed.setdefault("certifications", [])
        parsed.setdefault("skills", [])
        parsed.setdefault("education", [])
        parsed.setdefault("experience", [])
        parsed.setdefault("summary", None)

        return parsed

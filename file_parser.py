"""
file_parser.py
Parses uploaded CV files (PDF, DOCX, TXT) and returns plain text.
"""

import io
import logging

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"pdf", "docx", "doc", "txt"}


class FileParser:
    def parse_file(self, file_content: bytes, file_extension: str) -> str:
        """Parse file content and return extracted text."""
        ext = file_extension.lower().strip(".")

        if ext == "pdf":
            return self._parse_pdf(file_content)
        elif ext in ("docx", "doc"):
            return self._parse_docx(file_content)
        elif ext == "txt":
            return self._parse_txt(file_content)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

    def _parse_pdf(self, content: bytes) -> str:
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content))
            text = "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
            return text.strip()
        except ImportError:
            raise RuntimeError("pypdf is not installed. Run: pip install pypdf")
        except Exception as e:
            logger.error(f"PDF parsing error: {e}")
            raise ValueError(f"Failed to parse PDF: {e}")

    def _parse_docx(self, content: bytes) -> str:
        try:
            import docx
            doc = docx.Document(io.BytesIO(content))
            text = "\n".join(para.text for para in doc.paragraphs)
            return text.strip()
        except ImportError:
            raise RuntimeError("python-docx is not installed. Run: pip install python-docx")
        except Exception as e:
            logger.error(f"DOCX parsing error: {e}")
            raise ValueError(f"Failed to parse DOCX: {e}")

    def _parse_txt(self, content: bytes) -> str:
        try:
            return content.decode("utf-8", errors="ignore").strip()
        except Exception as e:
            logger.error(f"TXT parsing error: {e}")
            raise ValueError(f"Failed to parse TXT: {e}")

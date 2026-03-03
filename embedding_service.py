"""
embedding_service.py
Fix #3: Persistent Redis cache instead of in-memory dict.
         Falls back to in-memory if Redis unavailable.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Optional
import hashlib
import pickle
import re
import os
import logging

logger = logging.getLogger(__name__)

SKILL_CONTEXT_MAP: dict[str, str] = {
    # Tech / AI
    "kubernetes":           "Kubernetes container orchestration DevOps",
    "machine learning":     "machine learning predictive models algorithms",
    "deep learning":        "deep learning neural networks AI",
    "artificial intelligence": "artificial intelligence automation AI",
    "natural language processing": "NLP text processing language AI",
    "computer vision":      "computer vision image recognition CNN",
    "llms":                 "large language models GPT AI generation",
    "amazon web services":  "AWS Amazon cloud infrastructure computing",
    "google cloud platform":"GCP Google cloud services computing",
    "microsoft azure":      "Azure Microsoft cloud services",
    "ci/cd":                "continuous integration deployment DevOps pipeline",
    "pytorch":              "PyTorch deep learning neural network framework",
    "tensorflow":           "TensorFlow machine learning Google framework",
    "retrieval augmented generation": "RAG vector search LLM knowledge",
    "vector databases":     "vector database embeddings similarity search AI",
    "postgresql":           "PostgreSQL relational database SQL RDBMS",
    "mongodb":              "MongoDB NoSQL document database",
    "mlops":                "MLOps ML operations model deployment monitoring",
    "cuda":                 "CUDA GPU parallel computing NVIDIA deep learning",
    "dbt":                  "dbt data build tool SQL analytics transformation",
    "airflow":              "Apache Airflow workflow orchestration data pipeline",
    "spark":                "Apache Spark distributed data processing big data",
    "kafka":                "Apache Kafka event streaming real-time messaging",
    "terraform":            "Terraform infrastructure as code IaC provisioning",
    "rest apis":            "REST API HTTP web services backend integration",
    "system design":        "distributed systems architecture scalability",
    "cloud architecture":   "cloud infrastructure multi-cloud design",
    "microservices":        "microservices distributed architecture APIs",
    "huggingface":          "HuggingFace transformers NLP models",
    "langchain":            "LangChain LLM AI framework orchestration",
    "redis":                "Redis cache in-memory key-value store",
    "elasticsearch":        "Elasticsearch full-text search indexing",
    "docker":               "Docker containers containerization",
    "react":                "React JavaScript frontend UI library",
    "react native":         "React Native mobile app iOS Android",
    "next.js":              "Next.js React server-side rendering SSR",
    "vue.js":               "Vue.js JavaScript frontend framework",
    "angular":              "Angular TypeScript frontend framework",
    "django":               "Django Python web backend framework",
    "fastapi":              "FastAPI Python REST API async",
    "spring boot":          "Spring Boot Java enterprise backend",
    "html/css":             "HTML CSS frontend web design",
    "tailwind css":         "Tailwind CSS utility-first styling",
    "figma":                "Figma UI/UX design prototyping tool",
    # Medicine
    "intensive care":           "ICU critical care medicine ventilator",
    "emergency medicine":       "emergency room ER acute trauma care",
    "general practice":         "primary care family medicine GP",
    "obstetrics and gynecology":"OB/GYN women health pregnancy",
    "cardiology":               "cardiology heart cardiovascular disease",
    "neurology":                "neurology brain nervous system disorders",
    "oncology":                 "oncology cancer chemotherapy treatment",
    "pediatrics":               "pediatrics children medicine child health",
    "anesthesiology":           "anesthesia sedation surgery pain management",
    "physical therapy":         "physical therapy physiotherapy rehabilitation",
    "electronic health records":"EHR EMR digital patient medical records",
    "clinical research":        "clinical trials research protocol study",
    "pharmacology":             "pharmacology drugs medications treatment",
    "radiology":                "radiology imaging X-ray MRI CT diagnosis",
    # Law
    "intellectual property law":    "IP patents trademarks copyright law",
    "mergers and acquisitions":     "M&A corporate deals transactions",
    "corporate law":                "corporate governance business entity law",
    "contract law":                 "contracts legal agreements obligations",
    "litigation":                   "court trial dispute legal proceedings",
    "compliance":                   "regulatory compliance legal requirements",
    "anti-money laundering":        "AML financial crime prevention",
    "know your customer":           "KYC identity verification compliance",
    "due diligence":                "legal financial investigation review",
    "legal research":               "legal research case law statutes",
    # Finance
    "financial analysis":       "financial analysis ratios valuation",
    "financial modeling":       "financial modeling Excel DCF forecast",
    "investment banking":       "investment banking deals capital markets",
    "private equity":           "private equity PE fund portfolio",
    "venture capital":          "VC startup investment funding",
    "quantitative finance":     "quant trading mathematical models finance",
    "bloomberg terminal":       "Bloomberg financial data markets",
    "cryptocurrency":           "crypto Bitcoin blockchain digital assets",
    "risk management":          "risk assessment financial operational",
    "portfolio management":     "investment portfolio asset allocation",
    "taxation":                 "tax accounting fiscal planning compliance",
    "auditing":                 "audit financial statements control",
    "ifrs":                     "IFRS international financial standards",
    "gaap":                     "GAAP US accounting principles",
    # Marketing
    "seo":                      "SEO search engine optimization ranking",
    "pay-per-click advertising":"PPC Google Ads paid digital advertising",
    "social media advertising": "social media ads Facebook Instagram paid",
    "social media marketing":   "social media marketing strategy",
    "content marketing":        "content blog articles strategy",
    "email marketing":          "email campaigns newsletters automation",
    "crm":                      "CRM customer relationship Salesforce",
    "copywriting":              "copywriting persuasive writing sales",
    "branding":                 "brand identity strategy design",
    "market research":          "market research consumer insights analysis",
    "cro":                      "conversion rate optimization A/B testing",
    "business development":     "biz dev partnerships growth revenue",
    # HR
    "recruitment":              "recruitment talent hiring interviews",
    "learning and development": "L&D training employee development",
    "performance management":   "performance reviews appraisals KPIs",
    "compensation and benefits":"salary benefits rewards HR",
    "organizational development":"OD change management culture",
    "human resources":          "HR people management policies",
    "diversity equity inclusion":"DEI diversity inclusion culture",
    # Engineering
    "autocad":                  "AutoCAD 2D 3D mechanical drawing",
    "solidworks":               "SolidWorks 3D CAD mechanical design",
    "revit":                    "Revit BIM architectural building",
    "building information modeling":"BIM construction architecture digital",
    "finite element analysis":  "FEA structural analysis simulation",
    "computational fluid dynamics":"CFD fluid simulation engineering",
    "plc programming":          "PLC industrial automation control",
    "embedded systems":         "embedded firmware microcontroller hardware",
    "internet of things":       "IoT sensors connected devices smart",
    "pcb design":               "PCB circuit board electronics hardware",
    "quality control":          "QC quality inspection manufacturing",
    "lean manufacturing":       "lean waste reduction efficiency manufacturing",
    # Soft skills
    "leadership":               "leadership team management direction strategy",
    "communication":            "communication presentation writing interpersonal",
    "project management":       "project planning execution delivery",
    "product management":       "product roadmap strategy user needs",
    "negotiation":              "negotiation deal persuasion conflict",
    "strategic thinking":       "strategic planning vision long-term",
    "stakeholder management":   "stakeholder engagement communication",
}


def expand_skill(skill: str) -> str:
    key = skill.strip().lower()
    key = re.sub(r"[_\-]+", " ", key)
    key = re.sub(r"\s+", " ", key)
    return SKILL_CONTEXT_MAP.get(key, skill)


class _RedisCache:
    """Thin Redis wrapper for embedding cache. Auto-serializes numpy arrays."""

    def __init__(self, host: str, port: int, db: int, ttl: int = 86400 * 7):
        import redis
        self._r   = redis.Redis(host=host, port=port, db=db,
                                socket_connect_timeout=2, socket_timeout=2)
        self._ttl = ttl
        self._r.ping()   # raises if unavailable
        logger.info(f"Redis embedding cache connected at {host}:{port}")

    def get(self, key: str) -> Optional[np.ndarray]:
        raw = self._r.get(key)
        return pickle.loads(raw) if raw else None

    def set(self, key: str, value: np.ndarray) -> None:
        self._r.setex(key, self._ttl, pickle.dumps(value))

    def __contains__(self, key: str) -> bool:
        return bool(self._r.exists(key))

    @property
    def size(self) -> int:
        try:
            return self._r.dbsize()
        except Exception:
            return -1


class _MemoryCache:
    """Fallback in-memory cache (single process, lost on restart)."""

    def __init__(self):
        self._d: Dict[str, np.ndarray] = {}
        logger.warning("Using in-memory embedding cache (no persistence across restarts). "
                       "Set REDIS_HOST env var to enable Redis cache.")

    def get(self, key: str) -> Optional[np.ndarray]:
        return self._d.get(key)

    def set(self, key: str, value: np.ndarray) -> None:
        self._d[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._d

    @property
    def size(self) -> int:
        return len(self._d)


class EmbeddingService:
    def __init__(self):
        logger.info("Loading MiniLM model…")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Fix #3: Try Redis first, fall back to memory
        redis_host = os.getenv("REDIS_HOST", "")
        if redis_host:
            try:
                self._cache = _RedisCache(
                    host=redis_host,
                    port=int(os.getenv("REDIS_PORT", "6379")),
                    db=int(os.getenv("REDIS_EMBED_DB", "1")),
                )
            except Exception as e:
                logger.warning(f"Redis unavailable ({e}), falling back to in-memory cache.")
                self._cache = _MemoryCache()
        else:
            self._cache = _MemoryCache()

        logger.info("MiniLM model loaded ✅")

    def _cache_key(self, text: str) -> str:
        return "emb:" + hashlib.md5(text.lower().strip().encode()).hexdigest()

    def embed(self, text: str) -> np.ndarray:
        expanded = expand_skill(text)
        key = self._cache_key(expanded)
        v = self._cache.get(key)
        if v is None:
            v = self.model.encode(expanded, convert_to_numpy=True)
            self._cache.set(key, v)
        return v

    def embed_list(self, items: List[str]) -> np.ndarray:
        if not items:
            return np.zeros(384)
        seen, unique = set(), []
        for s in items:
            k = s.strip().lower()
            if k not in seen:
                seen.add(k); unique.append(s)

        expanded      = [expand_skill(s) for s in unique]
        uncached_texts, uncached_keys = [], []
        for text in expanded:
            k = self._cache_key(text)
            if k not in self._cache:
                uncached_texts.append(text); uncached_keys.append(k)

        if uncached_texts:
            vecs = self.model.encode(uncached_texts, convert_to_numpy=True, batch_size=64)
            for k, v in zip(uncached_keys, vecs):
                self._cache.set(k, v)

        vecs = np.array([self._cache.get(self._cache_key(t)) for t in expanded])
        return np.mean(vecs, axis=0)

    def semantic_score(self, user_skills: List[str], job_skills: List[str]) -> float:
        u = self.embed_list(user_skills).reshape(1, -1)
        j = self.embed_list(job_skills).reshape(1, -1)
        return float(np.clip(cosine_similarity(u, j)[0][0], 0.0, 1.0))

    @property
    def cache_size(self) -> int:
        return self._cache.size
# 🧠 Job Matching AI Service

A FastAPI service that matches user skills to jobs using **MiniLM semantic embeddings** and generates personalized learning roadmaps via **Groq LLM**.

---

## 🏗️ Architecture

```
POST /match
     │
     ▼
1. Normalize Skills  →  "ML" → "machine learning", "JS" → "javascript"
     │
     ▼
2. MiniLM Embeddings + In-Memory Cache
   └── model: all-MiniLM-L6-v2 (90MB, runs locally, free)
   └── repeated skills served from cache (~5ms vs ~10ms)
     │
     ▼
3. Hybrid Score per Job
   └── 60% Semantic Similarity  (cosine via MiniLM)
   └── 40% Fuzzy Match          (rapidfuzz partial_ratio)
     │
     ▼
4. Groq LLM — Single Batch Request
   └── ALL jobs in ONE prompt → saves quota & latency
   └── model: llama-3.3-70b-versatile
     │
     ▼
5. Save to SQL Server  →  PrompetRoadMaps (Title, Description)
     │
     ▼
6. Return sorted results (highest match first)
```

---

## ⚡ Performance

| Step | Time |
|---|---|
| Normalize | ~5ms |
| Embeddings (first request) | ~200ms |
| Embeddings (cached) | ~5ms |
| Hybrid Scoring (10 jobs) | ~10ms |
| Groq Batch (10 roadmaps) | ~3000ms |
| DB Save (10 inserts) | ~200ms |
| **Total (cold)** | **~3.5s** |
| **Total (warm cache)** | **~3.2s** |

---

## 🚀 Setup & Run

### 1. Install system dependency (ODBC Driver for SQL Server)

**Ubuntu/Debian:**
```bash
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
curl https://packages.microsoft.com/config/ubuntu/22.04/prod.list > /etc/apt/sources.list.d/mssql-release.list
apt-get update
ACCEPT_EULA=Y apt-get install -y msodbcsql17
```

**macOS:**
```bash
brew install unixodbc
brew tap microsoft/mssql-release https://github.com/Microsoft/homebrew-mssql-release
brew install msodbcsql17
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
Edit `.env` and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here   # Get free at console.groq.com
```

### 4. Run the service
```bash
uvicorn main:app --reload --port 8000
```

---

## 📬 API Usage

### POST /match

**Request:**
```json
{
  "user_skills": ["Python", "ML", "REST APIs", "SQL"],
  "jobs": [
    {
      "job_title": "ML Engineer",
      "job_description": "Build and deploy machine learning models at scale.",
      "job_skills": ["Python", "Machine Learning", "Docker", "Kubernetes", "FastAPI"]
    },
    {
      "job_title": "Backend Developer",
      "job_description": "Design REST APIs and manage databases.",
      "job_skills": ["Python", "FastAPI", "PostgreSQL", "Redis", "Docker"]
    }
  ]
}
```

**Response:**
```json
{
  "total_jobs": 2,
  "results": [
    {
      "job_title": "ML Engineer",
      "match_percentage": 74.5,
      "matched_skills": ["Python", "Machine Learning"],
      "missing_skills": ["Docker", "Kubernetes", "FastAPI"],
      "roadmap_title": "ML Engineer Skill Gap Roadmap",
      "roadmap": "Week 1-2: Learn Docker...\nWeek 3-4: Kubernetes...\nFinal Project: ...",
      "roadmap_db_id": 42
    }
  ]
}
```

### GET /health
```json
{
  "status": "ok",
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "embedding_cache_size": 128
}
```

---

## 📁 Project Structure

```
job_matching_service/
├── main.py                # FastAPI app + /match endpoint
├── embedding_service.py   # MiniLM model + in-memory cache
├── scoring.py             # hybrid score + fuzzy + normalization
├── groq_service.py        # batch prompt builder + Groq API call
├── database.py            # SQL Server connection + save roadmap
├── .env                   # credentials (do not commit to git!)
├── requirements.txt
└── README.md
```

---

## 🔑 Key Design Decisions

| Decision | Reason |
|---|---|
| MiniLM not BERT Large | 14x lighter, 10x faster, 95% same accuracy |
| In-Memory Cache | Skills repeat across requests → near-zero re-embedding cost |
| Single Groq Batch | 1 API call instead of 10 → saves quota (6000 req/day free) |
| Hybrid Score 60/40 | Semantic understands meaning + Fuzzy catches typos |
| Sort DESC | Best-matching job appears first in response |

---

## 🆓 Free Tier Limits

- **Groq:** 6,000 requests/day, 6,000 tokens/minute — [console.groq.com](https://console.groq.com)
- **MiniLM:** Completely free, runs locally, no API calls

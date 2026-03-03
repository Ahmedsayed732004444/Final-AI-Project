"""
scoring.py — Universal Job Skill Matching
Fixes applied:
  #1  Ambiguous abbreviations → resolved via domain-aware context or removed
  #2  False positives (React→React Native, Java→JS, SQL→NoSQL) → fixed via
      length-ratio penalty + exact-normalized match requirement
  #13 match_percentage inflated → penalized by coverage ratio
"""

from rapidfuzz import fuzz
from typing import List, Tuple
import re

# ─── Synonym Map ──────────────────────────────────────────────────────────────
# Rules:
#   - Short ambiguous abbreviations (tf, cv, pt, rn, pm) are REMOVED.
#     Users should write full skill names; fuzzy will handle minor typos.
#   - Multi-word aliases and clear unambiguous abbreviations are kept.
#   - "Skill grouping" maps semantically equivalent skills to one canonical form.

SKILL_SYNONYMS: dict[str, str] = {
    # POLICY: Short ambiguous abbreviations (tf, cv, pt, rn, np, ps, ae, od, bd, ci, cd, es)
    # are intentionally EXCLUDED. They are too context-dependent across domains.
    # Users should write full skill names. Fuzzy matching handles minor typos.
    # Unambiguous multi-char abbreviations (k8s, gcp, aws, mlops, etc.) are kept.


    # ── Programming Languages ──────────────────────────────────────────────────
    "pythn":                     "python",
    "pyton":                     "python",
    "pyhton":                    "python",
    "python3":                   "python",
    "py":                        "python",        # unambiguous
    "golang":                    "go",
    "go lang":                   "go",
    "cpp":                       "c++",
    "c plus plus":               "c++",
    "c sharp":                   "c#",
    "csharp":                    "c#",
    "dotnet":                    ".net",
    "dot net":                   ".net",
    "ruby on rails":             "ruby on rails",
    "ror":                       "ruby on rails",
    "rails":                     "ruby on rails",
    "r language":                "r",
    "r lang":                    "r",
    "bash scripting":            "bash",
    "shell scripting":           "bash",
    "powershell scripting":      "powershell",
    "dart flutter":              "flutter",

    # ── AI / ML ────────────────────────────────────────────────────────────────
    "machine learning":          "machine learning",
    "ml engineering":            "machine learning",
    "artificial intelligence":   "artificial intelligence",
    "deep learning":             "deep learning",
    "deep learing":              "deep learning",   # typo
    "deep lerning":              "deep learning",   # typo
    "depp learning":             "deep learning",   # typo
    "natural language processing": "natural language processing",
    "large language model":      "llms",
    "large language models":     "llms",
    "generative ai":             "generative ai",
    "gen ai":                    "generative ai",
    "genai":                     "generative ai",
    "retrieval augmented generation": "retrieval augmented generation",
    "tensorflow":                "tensorflow",
    "pytorch":                   "pytorch",
    "xgboost":                   "xgboost",
    "xgb":                       "xgboost",
    "scikit-learn":              "scikit-learn",
    "scikit learn":              "scikit-learn",
    "sklearn":                   "scikit-learn",
    "huggingface":               "huggingface",
    "hugging face":              "huggingface",
    "langchain":                 "langchain",
    "lang chain":                "langchain",
    "vector database":           "vector databases",
    "vector db":                 "vector databases",
    "vectordb":                  "vector databases",
    "reinforcement learning":    "reinforcement learning",
    "transfer learning":         "transfer learning",
    "fine-tuning":               "fine-tuning",
    "fine tuning":               "fine-tuning",
    "computer vision":           "computer vision",  # explicit — not "cv"

    # ── Databases ──────────────────────────────────────────────────────────────
    "postgres":                  "postgresql",
    "postgre":                   "postgresql",
    "postgresql":                "postgresql",
    "mongodb":                   "mongodb",
    "mongo":                     "mongodb",
    "elasticsearch":             "elasticsearch",
    "elastic search":            "elasticsearch",
    "elastic":                   "elasticsearch",
    "nosql":                     "nosql",
    "sql server":                "microsoft sql server",
    "mssql":                     "microsoft sql server",
    "ms sql":                    "microsoft sql server",
    "bigquery":                  "bigquery",
    "big query":                 "bigquery",
    "dynamodb":                  "dynamodb",
    "dynamo db":                 "dynamodb",
    "dynamo":                    "dynamodb",
    "oracle database":           "oracle",
    "oracle db":                 "oracle",
    "cassandra":                 "apache cassandra",
    "apache cassandra":          "apache cassandra",
    "neo4j":                     "neo4j",
    "graph database":            "graph databases",
    "graph databases":           "graph databases",
    "influxdb":                  "influxdb",
    "amazon redshift":           "amazon redshift",
    "redshift":                  "amazon redshift",
    "snowflake":                 "snowflake",

    # ── Cloud ──────────────────────────────────────────────────────────────────
    "aws":                       "amazon web services",
    "amazon aws":                "amazon web services",
    "amazon web services":       "amazon web services",
    "gcp":                       "google cloud platform",
    "google cloud":              "google cloud platform",
    "google cloud platform":     "google cloud platform",
    "microsoft azure":           "microsoft azure",
    "azure":                     "microsoft azure",
    "ms azure":                  "microsoft azure",
    "ibm cloud":                 "ibm cloud",
    "aws lambda":                "aws lambda",
    "lambda":                    "aws lambda",
    "aws s3":                    "aws s3",
    "aws ec2":                   "aws ec2",
    "serverless computing":      "serverless computing",
    "serverless":                "serverless computing",
    "faas":                      "serverless computing",

    # ── DevOps / Infra ─────────────────────────────────────────────────────────
    "kubernetes":                "kubernetes",
    "k8s":                       "kubernetes",
    "kube":                      "kubernetes",
    "ci/cd":                     "ci/cd",
    "cicd":                      "ci/cd",
    "ci cd":                     "ci/cd",
    "continuous integration":    "ci/cd",
    "continuous deployment":     "ci/cd",
    "continuous delivery":       "ci/cd",
    "terraform":                 "terraform",
    "infrastructure as code":    "infrastructure as code",
    "infra as code":             "infrastructure as code",
    "iac":                       "infrastructure as code",
    "ansible":                   "ansible",
    "jenkins":                   "jenkins",
    "github actions":            "github actions",
    "gh actions":                "github actions",
    "gitlab ci":                 "gitlab ci/cd",
    "gitlab cicd":               "gitlab ci/cd",
    "gitlab ci/cd":              "gitlab ci/cd",
    "helm":                      "helm",
    "istio":                     "istio",
    "service mesh":              "service mesh",
    "prometheus":                "prometheus",
    "grafana":                   "grafana",
    "datadog":                   "datadog",
    "elk stack":                 "elk stack",
    "elk":                       "elk stack",

    # ── Data Engineering ───────────────────────────────────────────────────────
    "airflow":                   "airflow",
    "apache airflow":            "airflow",
    "apache spark":              "spark",
    "apache kafka":              "kafka",
    "dbt":                       "dbt",
    "data build tool":           "dbt",
    "etl":                       "etl",
    "elt":                       "elt",
    "data pipelines":            "data pipelines",
    "data pipeline":             "data pipelines",
    "apache flink":              "apache flink",
    "flink":                     "apache flink",
    "apache hive":               "apache hive",
    "hive":                      "apache hive",
    "hadoop":                    "hadoop",
    "hdfs":                      "hadoop",
    "data warehousing":          "data warehousing",
    "data warehouse":            "data warehousing",
    "data lakehouse":            "data lakehouse",
    "delta lake":                "delta lake",

    # ── Web & Frontend ─────────────────────────────────────────────────────────
    "react":                     "react",
    "reactjs":                   "react",
    "react js":                  "react",
    "react native":              "react native",   # distinct from react
    "vue.js":                    "vue.js",
    "vuejs":                     "vue.js",
    "vue js":                    "vue.js",
    "angular":                   "angular",
    "angularjs":                 "angular",
    "next.js":                   "next.js",
    "nextjs":                    "next.js",
    "next js":                   "next.js",
    "nuxt.js":                   "nuxt.js",
    "nuxtjs":                    "nuxt.js",
    "svelte":                    "svelte",
    "html/css":                  "html/css",
    "html css":                  "html/css",
    "tailwind css":              "tailwind css",
    "tailwindcss":               "tailwind css",
    "tailwind":                  "tailwind css",
    "sass/scss":                 "sass/scss",
    "sass":                      "sass/scss",
    "scss":                      "sass/scss",

    # ── Backend / APIs ─────────────────────────────────────────────────────────
    "rest apis":                 "rest apis",
    "rest api":                  "rest apis",
    "restful":                   "rest apis",
    "restful api":               "rest apis",
    "fastapi":                   "fastapi",
    "spring boot":               "spring boot",
    "spring":                    "spring boot",
    "express.js":                "express.js",
    "expressjs":                 "express.js",
    "express js":                "express.js",
    "asp.net":                   "asp.net",
    "asp net":                   "asp.net",
    "aspnet":                    "asp.net",
    "nest.js":                   "nest.js",
    "nestjs":                    "nest.js",

    # ── Mobile ─────────────────────────────────────────────────────────────────
    "android development":       "android development",
    "ios development":           "ios development",
    "progressive web apps":      "progressive web apps",
    "pwa":                       "progressive web apps",
    "xamarin":                   "xamarin",
    "ionic":                     "ionic",

    # ── Security ───────────────────────────────────────────────────────────────
    "cybersecurity":             "security",
    "cyber security":            "security",
    "infosec":                   "security",
    "information security":      "security",
    "application security":      "application security",
    "appsec":                    "application security",
    "penetration testing":       "penetration testing",
    "pen testing":               "penetration testing",
    "pentesting":                "penetration testing",
    "ethical hacking":           "penetration testing",
    "devsecops":                 "devsecops",
    "zero trust security":       "zero trust security",
    "zero trust":                "zero trust security",
    "ssl/tls":                   "ssl/tls",
    "ssl tls":                   "ssl/tls",
    "identity and access management": "identity and access management",
    "iam":                       "identity and access management",

    # ── Compliance Grouping ────────────────────────────────────────────────────
    "regulatory compliance":     "compliance",
    "gdpr compliance":           "compliance",
    "gdpr":                      "compliance",
    "ccpa":                      "compliance",
    "ccpa compliance":           "compliance",
    "hipaa":                     "compliance",
    "hipaa compliance":          "compliance",
    "sox":                       "compliance",
    "pci dss":                   "compliance",
    "aml compliance":            "anti-money laundering",
    "anti money laundering":     "anti-money laundering",
    "anti-money laundering":     "anti-money laundering",

    # ── Design / UX ────────────────────────────────────────────────────────────
    "ux design":                 "ux design",
    "user experience":           "ux design",
    "ui design":                 "ui design",
    "user interface design":     "ui design",
    "ui/ux design":              "ui/ux design",
    "ui ux":                     "ui/ux design",
    "ui ux design":              "ui/ux design",
    "figma":                     "figma",
    "adobe xd":                  "adobe xd",
    "prototyping":               "prototyping",
    "wireframing":               "wireframing",
    "user research":             "user research",
    "design systems":            "design systems",
    "design system":             "design systems",
    "motion design":             "motion design",
    "3d modeling":               "3d modeling",
    "adobe illustrator":         "adobe illustrator",
    "illustrator":               "adobe illustrator",
    "photoshop":                 "photoshop",
    "adobe photoshop":           "photoshop",
    "adobe premiere":            "adobe premiere",
    "premiere pro":              "adobe premiere",
    "after effects":             "after effects",
    "adobe after effects":       "after effects",
    "canva":                     "canva",
    "indesign":                  "indesign",

    # ── Data / Analytics / BI ──────────────────────────────────────────────────
    "power bi":                  "power bi",
    "powerbi":                   "power bi",
    "tableau":                   "tableau",
    "looker":                    "looker",
    "google analytics":          "google analytics",
    "excel":                     "excel",
    "microsoft excel":           "excel",
    "ms excel":                  "excel",
    "google sheets":             "google sheets",
    "statistics":                "statistics",
    "statistical analysis":      "statistics",
    "stats":                     "statistics",
    "data analysis":             "data analysis",
    "data analytics":            "data analysis",
    "a/b testing":               "a/b testing",
    "ab testing":                "a/b testing",
    "experimentation":           "a/b testing",
    "rstudio":                   "rstudio",
    "r studio":                  "rstudio",

    # ── Advertising / Marketing Grouping ──────────────────────────────────────
    "pay-per-click advertising": "pay-per-click advertising",
    "ppc":                       "pay-per-click advertising",
    "pay per click":             "pay-per-click advertising",
    "google ads":                "pay-per-click advertising",
    "adwords":                   "pay-per-click advertising",
    "social media advertising":  "social media advertising",
    "facebook ads":              "social media advertising",
    "meta ads":                  "social media advertising",
    "instagram ads":             "social media advertising",
    "linkedin ads":              "social media advertising",
    "tiktok ads":                "social media advertising",
    "seo":                       "seo",
    "search engine optimization":"seo",
    "sem":                       "sem",
    "search engine marketing":   "sem",
    "social media marketing":    "social media marketing",
    "smm":                       "social media marketing",
    "content marketing":         "content marketing",
    "email marketing":           "email marketing",
    "crm":                       "crm",
    "hubspot":                   "hubspot",
    "salesforce":                "salesforce",
    "marketing automation":      "marketing automation",
    "copywriting":               "copywriting",
    "branding":                  "branding",
    "brand strategy":            "branding",
    "public relations":          "public relations",
    "market research":           "market research",
    "consumer insights":         "market research",
    "consumer research":         "market research",
    "competitive analysis":      "market research",
    "cro":                       "cro",
    "conversion rate optimization": "cro",
    "funnel optimization":       "cro",
    "business development":      "business development",
    "b2b sales":                 "b2b sales",
    "b2c sales":                 "b2c sales",

    # ── Finance / Accounting ───────────────────────────────────────────────────
    "financial analysis":        "financial analysis",
    "fin analysis":              "financial analysis",
    "financial modeling":        "financial modeling",
    "fin modeling":              "financial modeling",
    "discounted cash flow":      "discounted cash flow",
    "dcf":                       "discounted cash flow",
    "mergers and acquisitions":  "mergers and acquisitions",
    "m&a":                       "mergers and acquisitions",
    "investment banking":        "investment banking",
    "private equity":            "private equity",
    "venture capital":           "venture capital",
    "quantitative finance":      "quantitative finance",
    "bloomberg terminal":        "bloomberg terminal",
    "cryptocurrency":            "cryptocurrency",
    "crypto":                    "cryptocurrency",
    "blockchain":                "blockchain",
    "portfolio management":      "portfolio management",
    "risk management":           "risk management",
    "taxation":                  "taxation",
    "tax":                       "taxation",
    "auditing":                  "auditing",
    "internal auditing":         "auditing",
    "audit":                     "auditing",
    "bookkeeping":               "bookkeeping",
    "ifrs":                      "ifrs",
    "gaap":                      "gaap",
    "profit and loss":           "profit and loss",
    "p&l":                       "profit and loss",
    "foreign exchange":          "foreign exchange",
    "forex":                     "foreign exchange",
    "defi":                      "decentralized finance",
    "decentralized finance":     "decentralized finance",

    # ── Medicine / Healthcare ──────────────────────────────────────────────────
    "intensive care":            "intensive care",
    "icu":                       "intensive care",
    "emergency medicine":        "emergency medicine",
    "emergency room":            "emergency medicine",
    "general practice":          "general practice",
    "obstetrics and gynecology": "obstetrics and gynecology",
    "ob/gyn":                    "obstetrics and gynecology",
    "ob gyn":                    "obstetrics and gynecology",
    "orthopedics":               "orthopedics",
    "orthopaedics":              "orthopedics",
    "cardiology":                "cardiology",
    "neurology":                 "neurology",
    "oncology":                  "oncology",
    "pediatrics":                "pediatrics",
    "paediatrics":               "pediatrics",
    "peds":                      "pediatrics",
    "paeds":                     "pediatrics",
    "anesthesiology":            "anesthesiology",
    "anaesthesiology":           "anesthesiology",
    "anaesthesia":               "anesthesiology",
    "anesthesia":                "anesthesiology",
    "physical therapy":          "physical therapy",
    "physiotherapy":             "physical therapy",
    "occupational therapy":      "occupational therapy",
    "cpr/first aid":             "cpr/first aid",
    "cpr":                       "cpr/first aid",
    "first aid":                 "cpr/first aid",
    "basic life support":        "basic life support",
    "bls":                       "basic life support",
    "advanced cardiac life support": "advanced cardiac life support",
    "acls":                      "advanced cardiac life support",
    "electronic medical records":"electronic health records",
    "emr":                       "electronic health records",
    "ehr":                       "electronic health records",
    "icd coding":                "medical coding",
    "medical coding":            "medical coding",
    "clinical research":         "clinical research",
    "clinical trials":           "clinical research",
    "pharmacology":              "pharmacology",
    "radiology":                 "radiology",
    "telemedicine":              "telemedicine",
    "telehealth":                "telemedicine",
    "registered nurse":          "registered nurse",
    "nurse practitioner":        "nurse practitioner",
    "physician assistant":       "physician assistant",
    "cognitive behavioral therapy": "cognitive behavioral therapy",

    # ── Law / Legal ────────────────────────────────────────────────────────────
    "intellectual property law": "intellectual property law",
    "ip law":                    "intellectual property law",
    "corporate law":             "corporate law",
    "contract law":              "contract law",
    "contract drafting":         "contract drafting",
    "due diligence":             "due diligence",
    "litigation":                "litigation",
    "dispute resolution":        "dispute resolution",
    "arbitration":               "arbitration",
    "legal research":            "legal research",
    "westlaw":                   "westlaw",
    "lexisnexis":                "lexisnexis",
    "know your customer":        "know your customer",
    "kyc":                       "know your customer",
    "tax law":                   "tax law",
    "employment law":            "employment law",
    "labor law":                 "employment law",
    "immigration law":           "immigration law",
    "paralegal":                 "paralegal",
    "securities law":            "securities law",
    "patent law":                "patent law",

    # ── HR ─────────────────────────────────────────────────────────────────────
    "human resources":           "human resources",
    "recruitment":               "recruitment",
    "recruiting":                "recruitment",
    "talent acquisition":        "recruitment",
    "headhunting":               "recruitment",
    "learning and development":  "learning and development",
    "l&d":                       "learning and development",
    "training":                  "learning and development",
    "employee training":         "learning and development",
    "corporate training":        "learning and development",
    "training development":      "learning and development",
    "performance management":    "performance management",
    "compensation and benefits": "compensation and benefits",
    "c&b":                       "compensation and benefits",
    "employee relations":        "employee relations",
    "organizational development":"organizational development",
    "workforce planning":        "workforce planning",
    "succession planning":       "succession planning",
    "diversity equity inclusion":"diversity equity inclusion",
    "dei":                       "diversity equity inclusion",
    "employer branding":         "employer branding",
    "onboarding":                "onboarding",
    "payroll management":        "payroll management",

    # ── Engineering (Civil / Mechanical / Electrical) ──────────────────────────
    "autocad":                   "autocad",
    "auto cad":                  "autocad",
    "cad design":                "cad design",
    "solidworks":                "solidworks",
    "catia":                     "catia",
    "revit":                     "revit",
    "building information modeling": "building information modeling",
    "bim":                       "building information modeling",
    "ansys":                     "ansys",
    "finite element analysis":   "finite element analysis",
    "fea":                       "finite element analysis",
    "fem":                       "finite element analysis",
    "computational fluid dynamics": "computational fluid dynamics",
    "cfd":                       "computational fluid dynamics",
    "simulink":                  "simulink",
    "plc programming":           "plc programming",
    "plc":                       "plc programming",
    "scada":                     "scada",
    "embedded systems":          "embedded systems",
    "internet of things":        "internet of things",
    "iot":                       "internet of things",
    "pcb design":                "pcb design",
    "pcb":                       "pcb design",
    "fpga":                      "fpga",
    "vhdl":                      "vhdl",
    "verilog":                   "verilog",
    "3d printing":               "3d printing",
    "cnc machining":             "cnc machining",
    "cnc":                       "cnc machining",
    "lean manufacturing":        "lean manufacturing",
    "quality control":           "quality control",
    "iso standards":             "iso standards",

    # ── Architecture / Real Estate ─────────────────────────────────────────────
    "architecture":              "architecture",
    "architectural design":      "architecture",
    "interior design":           "interior design",
    "urban planning":            "urban planning",
    "construction management":   "construction management",
    "quantity surveying":        "quantity surveying",
    "structural engineering":    "structural engineering",
    "mep engineering":           "mep engineering",
    "sustainable design":        "sustainable design",
    "leed certification":        "leed certification",
    "leed":                      "leed certification",

    # ── Education ─────────────────────────────────────────────────────────────
    "curriculum development":    "curriculum development",
    "curriculum design":         "curriculum development",
    "lesson planning":           "lesson planning",
    "classroom management":      "classroom management",
    "e-learning":                "e-learning",
    "elearning":                 "e-learning",
    "learning management systems": "learning management systems",
    "lms":                       "learning management systems",
    "instructional design":      "instructional design",

    # ── Supply Chain / Operations ──────────────────────────────────────────────
    "supply chain management":   "supply chain management",
    "supply chain":              "supply chain management",
    "scm":                       "supply chain management",
    "logistics":                 "logistics",
    "procurement":               "procurement",
    "inventory management":      "inventory management",
    "warehouse management":      "warehouse management",
    "warehouse management systems": "warehouse management systems",
    "wms":                       "warehouse management systems",
    "enterprise resource planning": "enterprise resource planning",
    "erp":                       "enterprise resource planning",
    "sap":                       "sap",
    "demand planning":           "demand planning",
    "vendor management":         "vendor management",

    # ── Project Management ─────────────────────────────────────────────────────
    "project management":        "project management",
    "agile":                     "agile",
    "scrum":                     "scrum",
    "kanban":                    "kanban",
    "lean methodology":          "lean methodology",
    "six sigma":                 "six sigma",
    "prince2":                   "prince2",
    "jira":                      "jira",
    "confluence":                "confluence",
    "stakeholder management":    "stakeholder management",

    # ── Product Management ─────────────────────────────────────────────────────
    "product management":        "product management",
    "product roadmap":           "product management",
    "product strategy":          "product management",
    "okrs":                      "okrs",
    "okr":                       "okrs",
    "kpis":                      "kpis",
    "kpi":                       "kpis",

    # ── Soft Skills ────────────────────────────────────────────────────────────
    "leadership":                "leadership",
    "communication":             "communication",
    "comms":                     "communication",
    "presentation skills":       "presentation skills",
    "public speaking":           "public speaking",
    "negotiation":               "negotiation",
    "problem solving":           "problem solving",
    "critical thinking":         "critical thinking",
    "teamwork":                  "teamwork",
    "collaboration":             "teamwork",
    "time management":           "time management",
    "emotional intelligence":    "emotional intelligence",
    "mentoring":                 "mentoring",
    "coaching":                  "coaching",
    "conflict resolution":       "conflict resolution",
    "strategic thinking":        "strategic thinking",
    "analytical thinking":       "analytical thinking",

    # ── General Tech ──────────────────────────────────────────────────────────
    "system design":             "system design",
    "distributed systems":       "distributed systems",
    "cloud architecture":        "cloud architecture",
    "cloud arch":                "cloud architecture",
    "microservices":             "microservices",
    "high availability":         "high availability",
    "load balancing":            "load balancing",
    "networking":                "networking",
    "tcp/ip":                    "tcp/ip",
    "tcp ip":                    "tcp/ip",
    "git":                       "git",
    "version control":           "git",
    "github":                    "github",
    "gitlab":                    "gitlab",
    "linux":                     "linux",
    "unix":                      "linux",
    "windows server":            "windows server",
    "virtualization":            "virtualization",
    "vmware":                    "vmware",
    "redis":                     "redis",
    "rabbitmq":                  "rabbitmq",
    "celery":                    "celery",
    "unit testing":              "unit testing",
    "test driven development":   "test driven development",
    "tdd":                       "test driven development",
    "quality assurance":         "quality assurance",
    "qa":                        "quality assurance",
    "technical writing":         "technical writing",
    "swagger/openapi":           "swagger/openapi",
    "swagger":                   "swagger/openapi",
    "openapi":                   "swagger/openapi",
    "mlops":                     "mlops",
    "mlflow":                    "mlflow",
    "model monitoring":          "model monitoring",
    "feature store":             "feature store",
    "cuda":                      "cuda",
    "tensorrt":                  "tensorrt",
    "opencv":                    "opencv",
    "open cv":                   "opencv",
}

# ── False-positive protection: pairs that must NOT match despite string overlap ─
# Format: frozenset({normalized_a, normalized_b})
MUST_NOT_MATCH: set[frozenset] = {
    frozenset({"react",            "react native"}),
    frozenset({"java",             "javascript"}),
    frozenset({"sql",              "nosql"}),
    frozenset({"machine learning", "mlops"}),
    frozenset({"docker",           "docker swarm"}),
    frozenset({"spark",            "spark ar"}),
    frozenset({"python",           "python 2"}),   # if someone explicitly writes python 2
    frozenset({"nursing",          "nurse practitioner"}),
    frozenset({"nursing",          "registered nurse"}),
    frozenset({"security",         "security operations"}),
    frozenset({"marketing",        "email marketing"}),
    frozenset({"marketing",        "content marketing"}),
    frozenset({"marketing",        "social media marketing"}),
    frozenset({"design",           "system design"}),
    frozenset({"testing",          "a/b testing"}),
    frozenset({"r",                "ruby"}),
    frozenset({"r",                "rust"}),
    frozenset({"go",               "google cloud platform"}),
    frozenset({"c",                "c++"}),
    frozenset({"c",                "c#"}),
    frozenset({"c++",              "c#"}),
    frozenset({"kubernetes",       "docker"}),
    frozenset({"react",            "react.js"}),   # react.js is the same as react → allow
}

# Correct the above — react and react.js ARE the same
MUST_NOT_MATCH.discard(frozenset({"react", "react.js"}))

FUZZY_MATCH_THRESHOLD = 82
SEMANTIC_WEIGHT       = 0.60
FUZZY_WEIGHT          = 0.40


def normalize(skill: str) -> str:
    """Lowercase, remove noise, expand synonyms, handle plurals."""
    s = skill.strip().lower()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    if s in SKILL_SYNONYMS:
        return SKILL_SYNONYMS[s]
    if s.endswith("s") and s[:-1] in SKILL_SYNONYMS:
        return SKILL_SYNONYMS[s[:-1]]
    return s


def _length_ratio(a: str, b: str) -> float:
    """
    Penalty for large length differences.
    'React' (5 chars) vs 'React Native' (12 chars) → ratio = 5/12 = 0.42
    Used to downweight partial matches where one string is a strict prefix of the other.
    """
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def best_fuzzy_match(skill: str, candidates: List[str]) -> float:
    """
    Safe fuzzy scoring:
    1. Exact normalized match → 100 immediately
    2. MUST_NOT_MATCH block → 0
    3. Length-ratio penalty on partial_ratio to prevent prefix false-positives
    4. token_set_ratio as secondary scorer
    Returns 0–100.
    """
    ns = normalize(skill)
    best = 0.0

    for c in candidates:
        nc = normalize(c)

        # ── Exact normalized match ─────────────────────────────────────────────
        if ns == nc:
            return 100.0

        # ── Hard block ────────────────────────────────────────────────────────
        if frozenset({ns, nc}) in MUST_NOT_MATCH:
            continue

        # ── Fuzzy scoring with length-ratio penalty ───────────────────────────
        pr   = fuzz.partial_ratio(ns, nc)
        tsr  = fuzz.token_set_ratio(ns, nc)
        lr   = _length_ratio(ns, nc)

        # partial_ratio is penalized when strings differ heavily in length
        # (prevents "React" from fully matching "React Native")
        pr_adjusted = pr * (0.5 + 0.5 * lr)   # at lr=1 → no penalty; at lr=0.3 → 65% of pr

        score = max(pr_adjusted, tsr * lr)      # tsr also penalized by length ratio

        if score > best:
            best = score

    return min(best, 100.0)


def fuzzy_score(user_skills: List[str], job_skills: List[str]) -> float:
    """Average best fuzzy score for all job skills. Returns 0.0–1.0."""
    if not user_skills or not job_skills:
        return 0.0
    return sum(best_fuzzy_match(js, user_skills) for js in job_skills) / (len(job_skills) * 100.0)


def hybrid_score(
    semantic: float,
    fuzzy: float,
    matched_count: int,
    total_job_skills: int,
) -> float:
    """
    Final match percentage.
    Fix #13: Multiplied by coverage ratio to prevent inflated scores
    when user only shares 1 skill with a 10-skill job.

    Formula:
        raw      = 0.6*semantic + 0.4*fuzzy
        coverage = matched_count / total_job_skills
        final    = raw * (0.5 + 0.5*coverage)   ← coverage dampens but doesn't zero
    """
    if total_job_skills == 0:
        return 0.0
    raw      = SEMANTIC_WEIGHT * semantic + FUZZY_WEIGHT * fuzzy
    coverage = matched_count / total_job_skills
    adjusted = raw * (0.5 + 0.5 * coverage)
    return round(adjusted * 100, 2)


def get_matched_missing(
    user_skills: List[str],
    job_skills: List[str],
    threshold: int = FUZZY_MATCH_THRESHOLD,
) -> Tuple[List[str], List[str]]:
    """
    Fuzzy-based classification with false-positive protection.
    Works for any professional domain.
    """
    matched, missing = [], []
    for job_skill in job_skills:
        score = best_fuzzy_match(job_skill, user_skills)
        (matched if score >= threshold else missing).append(job_skill)
    return matched, missing
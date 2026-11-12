"""
SkillSync Taxonomy & Alias Configuration

Contains skill taxonomies, canonical alias mappings, and generic term filters
decoupled from the core NLP matching engine.
"""

# ============================================================
# CANONICAL SKILL ALIASES
# ============================================================

SKILL_ALIASES = {
    # Programming Languages
    "js": "javascript",
    "jscript": "javascript",
    "ts": "typescript",
    "py": "python",
    "golang": "go",
    "cpp": "c++",
    "cplusplus": "c++",
    "c_plus_plus": "c++",
    "csharp": "c#",
    "c_sharp": "c#",
    "c #": "c#",
    "c ++": "c++",

    # Machine Learning & AI
    "ml": "machine learning",
    "dl": "deep learning",
    "ai": "artificial intelligence",
    "genai": "generative ai",
    "gen ai": "generative ai",
    "llm": "large language models",
    "llms": "large language models",
    "large language model": "large language models",
    "nlp": "natural language processing",
    "sklearn": "scikit-learn",
    "scikitlearn": "scikit-learn",
    "tf": "tensorflow",
    "xg boost": "xgboost",
    "ml flow": "mlflow",

    # Frameworks & Libraries
    "react.js": "react",
    "reactjs": "react",
    "next.js": "next.js",
    "nextjs": "next.js",
    "vue.js": "vue.js",
    "vuejs": "vue.js",
    "node": "node.js",
    "nodejs": "node.js",
    "express.js": "express",
    "expressjs": "express",
    "spring-boot": "spring boot",
    "springboot": "spring boot",
    "fast-api": "fastapi",
    "tailwind": "tailwind css",

    # API & Architecture Aliases
    "rest": "rest apis",
    "rest api": "rest apis",
    "rest apis": "rest apis",
    "restful": "rest apis",
    "restful api": "rest apis",
    "restful apis": "rest apis",
    "microservice": "microservices",
    "microservices": "microservices",
    "webhook": "webhooks",
    "webhooks": "webhooks",
    "async programming": "asynchronous programming",
    "async": "asynchronous programming",
    "asynchronous": "asynchronous programming",
    "retry mechanism": "retry mechanisms",
    "retry mechanisms": "retry mechanisms",
    "retries": "retry mechanisms",
    "error handling": "api error handling",
    "api error handling": "api error handling",
    "third party integrations": "third-party integrations",
    "third-party integration": "third-party integrations",
    "third party integration": "third-party integrations",
    "integrations": "third-party integrations",

    # Parallel Computing & High Performance Computing
    "open mp": "openmp",
    "breadth first search": "bfs",
    "breadth-first search": "bfs",
    "depth first search": "dfs",
    "depth-first search": "dfs",
    "triangle-counting": "triangle counting",
    "parallel algorithm": "parallel algorithms",
    "parallel computing": "parallel computing",
    "graph algorithm": "graph algorithms",
    "graph algorithms": "graph algorithms",
    "multi-threading": "multithreading",
    "multithreaded": "multithreading",
    "cache-coherence": "cache coherence",
    "cache coherency": "cache coherence",
    "cache-coherency": "cache coherence",
    "benchmarking": "performance benchmarking",
    "performance benchmark": "performance benchmarking",
    "performance benchmarks": "performance benchmarking",
    "performance benchmarking": "performance benchmarking",
    "performance profiling": "performance profiling",
    "profiling": "performance profiling",

    # Cloud & DevOps
    "k8s": "kubernetes",
    "amazon web services": "aws",
    "google cloud platform": "gcp",
    "google cloud": "gcp",
    "microsoft azure": "azure",
    "ci cd": "ci/cd",
    "ci-cd": "ci/cd",
    "continuous integration": "ci/cd",
    "github action": "github actions",
    "github actions": "github actions",
    "cloud computing": "cloud computing",
    "cloud deployment": "cloud deployment",

    # Automation & Integrations
    "slack api": "slack",
    "google workspace api": "google workspace",
    "oauth": "oauth 2.0",
    "oauth2": "oauth 2.0",
    "oauth 2": "oauth 2.0",
    "oauth2.0": "oauth 2.0",

    # Databases
    "postgres": "sql",
    "postgresql": "sql",
    "mysql": "sql",
    "pl/sql": "sql",
    "plsql": "sql",
    "relational database": "sql",
    "rdbms": "sql",
    "database design": "database design",
    "query optimization": "query optimization",

    # Business Intelligence & Analytics
    "powerbi": "power bi",

    # Domain Aliases
    "hr": "human resources",
    "qa": "quality control",
    "pr": "public relations",
    "ui": "ui/ux",
    "ux": "ui/ux",
}

# ============================================================
# SKILL DICTIONARY TAXONOMY
# ============================================================

SKILLS = [
    # --------------------------------------------------------
    # PROGRAMMING LANGUAGES
    # --------------------------------------------------------
    "python",
    "java",
    "c++",
    "c#",
    "c",
    "javascript",
    "typescript",
    "go",
    "rust",
    "ruby",
    "php",
    "swift",
    "kotlin",
    "scala",
    "r",
    "bash",
    "shell",

    # --------------------------------------------------------
    # FRONTEND DEVELOPMENT
    # --------------------------------------------------------
    "html",
    "css",
    "tailwind css",
    "bootstrap",
    "react",
    "next.js",
    "vue.js",
    "angular",

    # --------------------------------------------------------
    # BACKEND DEVELOPMENT & APIS
    # --------------------------------------------------------
    "node.js",
    "express",
    "django",
    "flask",
    "fastapi",
    "spring",
    "spring boot",
    "asp.net",
    "laravel",
    "rest apis",
    "graphql",
    "microservices",
    "webhooks",
    "asynchronous programming",
    "retry mechanisms",
    "fault tolerance",
    "api error handling",
    "third-party integrations",

    # --------------------------------------------------------
    # DATABASES & STORAGE
    # --------------------------------------------------------
    "sql",
    "mysql",
    "postgresql",
    "mongodb",
    "redis",
    "sqlite",
    "oracle",
    "dynamodb",
    "cassandra",
    "elasticsearch",
    "pinecone",
    "chromadb",
    "qdrant",
    "vector database",
    "nosql",
    "database design",
    "query optimization",

    # --------------------------------------------------------
    # CLOUD & DEVOPS
    # --------------------------------------------------------
    "aws",
    "ec2",
    "s3",
    "iam",
    "azure",
    "gcp",
    "cloud",
    "cloud computing",
    "cloud deployment",
    "docker",
    "kubernetes",
    "terraform",
    "helm",
    "ansible",
    "jenkins",
    "ci/cd",
    "github actions",
    "git",
    "github",
    "gitlab",
    "linux",
    "unix",
    "dns",
    "http",
    "monitoring",
    "logging",
    "networking",

    # --------------------------------------------------------
    # AI & MACHINE LEARNING
    # --------------------------------------------------------
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "natural language processing",
    "generative ai",
    "large language models",
    "rag",
    "langchain",
    "llamaindex",
    "transformers",
    "huggingface",
    "vllm",
    "tensorflow",
    "pytorch",
    "scikit-learn",
    "pandas",
    "numpy",
    "xgboost",
    "mlflow",
    "shap",
    "feature engineering",
    "model evaluation",
    "supervised learning",
    "unsupervised learning",
    "hyperparameter tuning",
    "opencv",
    "neural networks",
    "computer vision",

    # --------------------------------------------------------
    # PARALLEL COMPUTING & HIGH PERFORMANCE ALGORITHMS
    # --------------------------------------------------------
    "openmp",
    "cuda",
    "bfs",
    "dfs",
    "graph algorithms",
    "triangle counting",
    "parallel computing",
    "parallel algorithms",
    "multithreading",
    "synchronization",
    "shared memory",
    "cache coherence",
    "performance profiling",
    "performance benchmarking",

    # --------------------------------------------------------
    # AUTOMATION, AUTH & PROTOCOLS
    # --------------------------------------------------------
    "n8n",
    "oauth 2.0",
    "authentication",
    "authorization",
    "orm",
    "json",
    "slack",
    "google workspace",

    # --------------------------------------------------------
    # DATA & ANALYTICS
    # --------------------------------------------------------
    "data analysis",
    "data science",
    "tableau",
    "power bi",

    # --------------------------------------------------------
    # CORE DISCIPLINES
    # --------------------------------------------------------
    "programming",
    "web development",

    # --------------------------------------------------------
    # HUMAN RESOURCES
    # --------------------------------------------------------
    "recruitment",
    "talent acquisition",
    "employee relations",
    "talent management",
    "human resources",
    "hris",
    "onboarding",
    "performance management",
    "payroll",
    "employee engagement",
    "compensation",
    "benefits",
    "labor laws",
    "succession planning",
    "conflict resolution",
    "interviewing",
    "sourcing",

    # --------------------------------------------------------
    # FINANCE / ACCOUNTING / BANKING
    # --------------------------------------------------------
    "financial analysis",
    "accounting",
    "financial reporting",
    "financial planning",
    "auditing",
    "budgeting",
    "forecasting",
    "taxation",
    "bookkeeping",
    "risk management",
    "compliance",
    "treasury",
    "portfolio management",
    "cash flow",
    "quickbooks",
    "sap",
    "tally",
    "excel",
    "financial modeling",
    "investment banking",
    "wealth management",
    "credit analysis",
    "loans",
    "retail banking",
    "commercial banking",
    "reconciliation",
    "financial statements",

    # --------------------------------------------------------
    # HEALTHCARE
    # --------------------------------------------------------
    "patient care",
    "clinical",
    "nursing",
    "medical records",
    "emr",
    "ehr",
    "healthcare management",
    "triage",
    "phlebotomy",
    "diagnostics",
    "patient safety",
    "pharmacology",
    "icu",
    "cpr",
    "bls",
    "vital signs",
    "health information management",
    "patient assessment",

    # --------------------------------------------------------
    # SALES / BUSINESS / MARKETING / PR
    # --------------------------------------------------------
    "sales",
    "business development",
    "lead generation",
    "crm",
    "salesforce",
    "account management",
    "negotiation",
    "client relations",
    "b2b",
    "b2c",
    "cold calling",
    "digital marketing",
    "seo",
    "sem",
    "content marketing",
    "social media marketing",
    "google analytics",
    "copywriting",
    "public relations",
    "brand management",
    "media relations",
    "press releases",
    "campaign management",
    "market research",
    "marketing strategy",

    # --------------------------------------------------------
    # DESIGN / ARTS / APPAREL
    # --------------------------------------------------------
    "ui/ux",
    "graphic design",
    "photoshop",
    "illustrator",
    "figma",
    "adobe xd",
    "indesign",
    "wireframing",
    "prototyping",
    "user research",
    "fashion design",
    "textile design",
    "apparel design",
    "creative direction",
    "sketching",
    "adobe creative suite",

    # --------------------------------------------------------
    # OTHER DOMAINS
    # --------------------------------------------------------
    "aviation",
    "flight operations",
    "aircraft maintenance",
    "cabin crew",
    "air traffic control",

    "automotive engineering",
    "vehicle maintenance",
    "autocad",
    "cad",
    "quality control",

    "construction management",
    "site supervision",
    "civil engineering",
    "project planning",
    "building codes",

    "agronomy",
    "crop management",
    "soil science",
    "agricultural engineering",
    "irrigation",

    "culinary arts",
    "food preparation",
    "menu planning",
    "kitchen management",
    "food safety",
    "haccp",

    "bpo",
    "customer service",
    "call center",
    "technical support",
    "helpdesk",
    "ticket resolution",

    "legal research",
    "litigation",
    "contract drafting",
    "corporate law",
    "legal compliance",
    "legal advisory",

    "personal training",
    "fitness instruction",
    "nutrition",
    "wellness coaching",
    "strength training",

    "teaching",
    "curriculum development",
    "classroom management",
    "lesson planning",
    "educational leadership",

    "management consulting",
    "strategy",
    "process improvement",
    "business analysis",
    "change management",
]

# ============================================================
# NON-SKILL GENERIC TERMS
# ============================================================

GENERIC_TERMS = {
    "software engineer",
    "software development",
    "problem solving",
    "team player",
    "communication skills",
    "critical thinking",
    "good communication",
    "analytical skills",
}


# ============================================================
# ROLE-BASED RELEVANCE & RELATED SKILLS TAXONOMY
# ============================================================

ROLE_TAXONOMY = {
    "web_fullstack": {
        "javascript", "typescript", "react", "next.js", "vue.js", "angular",
        "html", "css", "tailwind css", "bootstrap", "node.js", "express",
        "fastapi", "django", "flask", "spring", "spring boot", "rest apis",
        "graphql", "microservices", "webhooks", "sql", "postgresql", "mysql",
        "mongodb", "redis", "docker", "ci/cd", "oauth 2.0", "authentication",
        "authorization", "orm", "json", "cloud deployment"
    },
    "ai_ml": {
        "python", "machine learning", "deep learning", "natural language processing",
        "generative ai", "large language models", "rag", "langchain", "llamaindex",
        "transformers", "huggingface", "tensorflow", "pytorch", "scikit-learn",
        "pandas", "numpy", "xgboost", "mlflow", "shap", "feature engineering",
        "model evaluation", "hyperparameter tuning", "neural networks", "computer vision",
        "opencv", "data science", "data analysis", "fastapi", "docker", "sql"
    },
    "devops_cloud": {
        "aws", "ec2", "s3", "iam", "azure", "gcp", "cloud", "cloud computing",
        "cloud deployment", "docker", "kubernetes", "terraform", "helm", "ansible",
        "jenkins", "ci/cd", "github actions", "git", "github", "gitlab", "linux",
        "unix", "bash", "shell", "dns", "http", "monitoring", "logging",
        "networking", "python", "go"
    },
    "systems_parallel": {
        "c", "c++", "openmp", "cuda", "bfs", "dfs", "graph algorithms",
        "triangle counting", "parallel computing", "parallel algorithms",
        "multithreading", "synchronization", "shared memory", "cache coherence",
        "performance profiling", "performance benchmarking", "linux", "unix"
    },
    "automation_integration": {
        "python", "javascript", "n8n", "rest apis", "webhooks",
        "third-party integrations", "slack", "google workspace",
        "asynchronous programming", "retry mechanisms", "fault tolerance",
        "oauth 2.0", "json", "api error handling", "docker", "git", "ci/cd"
    },
    "data_engineering": {
        "python", "sql", "postgresql", "mysql", "mongodb", "redis", "cassandra",
        "elasticsearch", "vector database", "pinecone", "chromadb", "qdrant",
        "database design", "query optimization", "data analysis", "pandas",
        "numpy", "tableau", "power bi"
    }
}


def get_relevant_skills_for_jd(jd_skills):
    """
    Given canonical skills required by a JD, determine which primary domains are activated
    and return the union set of technical skills deemed relevant for those roles.
    Uses proportional dominance (fraction of domain requirements matched) to avoid
    cross-domain leakage from generic utility skills like git or docker.
    """
    jd_skills_set = set(jd_skills)
    if not jd_skills_set:
        return set()

    # Calculate domain affinity scores based on fraction of JD skills in domain
    domain_scores = {}
    for role_name, domain_skills in ROLE_TAXONOMY.items():
        overlap = jd_skills_set & domain_skills
        if overlap:
            # Fraction of JD requirements covered by this domain
            score = len(overlap) / len(jd_skills_set)
            domain_scores[role_name] = (score, len(overlap), domain_skills)

    if not domain_scores:
        return set()

    max_score = max(s[0] for s in domain_scores.values())

    # Only activate domains that have at least 2 distinct domain skills or cover >= 30% of JD skills
    activated_skills = set()
    for role_name, (score, cnt, domain_skills) in domain_scores.items():
        if cnt >= 2 and score >= max_score * 0.5:
            activated_skills.update(domain_skills)

    return activated_skills

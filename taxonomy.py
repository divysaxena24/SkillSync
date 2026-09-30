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
    "csharp": "c#",

    # Machine Learning & AI
    "ml": "machine learning",
    "dl": "deep learning",
    "ai": "artificial intelligence",
    "genai": "generative ai",
    "llm": "large language models",
    "llms": "large language models",
    "nlp": "natural language processing",
    "sklearn": "scikit-learn",
    "tf": "tensorflow",

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

    # Databases
    "postgres": "sql",
    "postgresql": "sql",
    "mysql": "sql",
    "pl/sql": "sql",
    "plsql": "sql",

    # Cloud & DevOps
    "k8s": "kubernetes",
    "amazon web services": "aws",
    "google cloud platform": "gcp",
    "google cloud": "gcp",
    "microsoft azure": "azure",

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
    # IT / SOFTWARE / DATA SCIENCE / DEVOPS / AI
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

    "html",
    "css",
    "sql",

    "react",
    "next.js",
    "vue.js",
    "angular",

    "node.js",
    "express",

    "django",
    "flask",
    "fastapi",
    "spring boot",
    "spring",
    "asp.net",
    "laravel",
    "bootstrap",
    "tailwind",
    "tailwind css",

    "mysql",
    "postgresql",
    "postgres",
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

    "aws",
    "azure",
    "gcp",
    "cloud",
    "docker",
    "kubernetes",
    "terraform",
    "ansible",
    "jenkins",
    "ci/cd",

    "git",
    "github",
    "gitlab",

    "linux",
    "unix",
    "bash",
    "shell",

    "rest api",
    "restful api",
    "graphql",
    "microservices",

    "machine learning",
    "deep learning",
    "generative ai",
    "large language models",
    "rag",
    "langchain",
    "llamaindex",
    "transformers",
    "huggingface",
    "vllm",
    "natural language processing",
    "tensorflow",
    "pytorch",
    "scikit-learn",
    "pandas",
    "numpy",
    "opencv",

    "data analysis",
    "data science",
    "neural networks",
    "computer vision",
    "tableau",
    "power bi",

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
}

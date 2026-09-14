import re
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# NLTK RESOURCES
# ============================================================

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# ============================================================
# DATASET
# ============================================================

DATA_DIR = Path("data/resume-data/data/data")


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(pdf_path):
    """
    Extract text from a resume PDF.
    """

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text


# ============================================================
# NLP PREPROCESSING
# ============================================================

def preprocess_text(text):
    """
    Clean, tokenize and lemmatize text.
    """

    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    tokens = word_tokenize(text)

    cleaned_tokens = []

    for word in tokens:

        if word not in STOP_WORDS and len(word) > 1:

            lemma = lemmatizer.lemmatize(word)

            cleaned_tokens.append(lemma)

    return cleaned_tokens


def clean_text(text):
    """
    Return preprocessed tokens as a single string.
    """

    return " ".join(preprocess_text(text))


# ============================================================
# RESUME DATASET LOADING
# ============================================================

def load_resume_paths():
    """
    Load all resume PDF paths along with their categories.
    """

    resumes = []

    if not DATA_DIR.exists():
        return resumes

    for category_dir in DATA_DIR.iterdir():

        if not category_dir.is_dir():
            continue

        for pdf_path in category_dir.glob("*.pdf"):

            resumes.append(
                {
                    "category": category_dir.name,
                    "path": pdf_path,
                }
            )

    return resumes


# ============================================================
# SKILL DICTIONARY
# ============================================================

SKILLS = [

    # --------------------------------------------------------
    # IT / SOFTWARE / DATA SCIENCE / DEVOPS
    # --------------------------------------------------------

    "python",
    "java",
    "c++",
    "c#",
    "c",
    "javascript",
    "typescript",
    "go",
    "golang",
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
    "react.js",
    "next.js",
    "vue.js",
    "angular",

    "node.js",
    "express",
    "express.js",

    "django",
    "flask",
    "fastapi",
    "spring boot",
    "spring",
    "asp.net",
    "laravel",
    "bootstrap",
    "tailwind",

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

    "aws",
    "azure",
    "gcp",
    "cloud",
    "docker",
    "kubernetes",
    "k8s",
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
    "nlp",
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
# CANONICAL SKILL ALIASES
# ============================================================

SKILL_ALIASES = {

    # Programming
    "js": "javascript",
    "jscript": "javascript",

    "ts": "typescript",

    "py": "python",

    # Machine Learning
    "ml": "machine learning",
    "dl": "deep learning",

    # React
    "react.js": "react",
    "reactjs": "react",

    # Node
    "node": "node.js",
    "nodejs": "node.js",

    # Express
    "express.js": "express",
    "expressjs": "express",

    # PostgreSQL
    "postgres": "sql",
    "postgresql": "sql",

    # MySQL
    "mysql": "sql",

    # PL/SQL
    "pl/sql": "sql",

    # Kubernetes
    "k8s": "kubernetes",

    # HR
    "hr": "human resources",

    # QA
    "qa": "quality control",

    # PR
    "pr": "public relations",

    # UI/UX
    "ui": "ui/ux",
    "ux": "ui/ux",
}


# ============================================================
# NON-SKILL GENERIC TERMS
# ============================================================

GENERIC_TERMS = {
    "software engineer",
    "software development",
    "problem solving",
}


# ============================================================
# SKILL SEARCH NORMALIZATION
# ============================================================

def normalize_for_skill_search(text):
    """
    Normalize text while preserving technical characters.
    """

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\+\#\.\/\-\s]",
        " ",
        text,
    )

    return text


# ============================================================
# CANONICALIZATION
# ============================================================

def canonicalize_skill(skill):
    """
    Convert a detected skill into its canonical representation.

    Examples:
        React.js -> react
        ReactJS  -> react
        Express.js -> express
        PostgreSQL -> sql
        MySQL -> sql
    """

    skill = skill.lower().strip()

    return SKILL_ALIASES.get(
        skill,
        skill,
    )


# ============================================================
# SKILL EXTRACTION
# ============================================================

def extract_skills(text):
    """
    Extract skills and return only canonical skill names.

    Equivalent skills are merged so that aliases do not
    artificially inflate the skill-match score.
    """

    if not text:
        return []

    norm_text = normalize_for_skill_search(text)

    found_skills = set()

    # --------------------------------------------------------
    # Detect skills
    # --------------------------------------------------------

    for skill in SKILLS:

        skill_norm = skill.lower()

        # Ignore generic terms
        if skill_norm in GENERIC_TERMS:
            continue

        escaped = re.escape(skill_norm)

        pattern = (
            r"(?<![a-z0-9])"
            + escaped
            + r"(?![a-z0-9])"
        )

        if re.search(
            pattern,
            norm_text,
        ):

            canonical_skill = canonicalize_skill(
                skill_norm
            )

            if canonical_skill not in GENERIC_TERMS:
                found_skills.add(
                    canonical_skill
                )

    # --------------------------------------------------------
    # Detect aliases explicitly
    # --------------------------------------------------------

    for alias, canonical in SKILL_ALIASES.items():

        if alias in GENERIC_TERMS:
            continue

        escaped_alias = re.escape(alias)

        pattern = (
            r"(?<![a-z0-9])"
            + escaped_alias
            + r"(?![a-z0-9])"
        )

        if re.search(
            pattern,
            norm_text,
        ):

            if canonical not in GENERIC_TERMS:
                found_skills.add(canonical)

    return sorted(found_skills)


# ============================================================
# SKILL MATCHING
# ============================================================

def calculate_skill_match(
    resume_text,
    job_description,
):
    """
    Compare canonical skills present in the resume
    against canonical skills required by the JD.

    Aliases are normalized before calculating the score,
    preventing duplicate skills from inflating the result.
    """

    resume_skills = set(
        extract_skills(resume_text)
    )

    jd_skills = set(
        extract_skills(job_description)
    )

    matched_skills = sorted(
        resume_skills & jd_skills
    )

    missing_skills = sorted(
        jd_skills - resume_skills
    )

    if not jd_skills:

        skill_score = 0.0

    else:

        skill_score = (
            len(matched_skills)
            / len(jd_skills)
        ) * 100

    return {
        "resume_skills": sorted(resume_skills),
        "jd_skills": sorted(jd_skills),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "skill_score": skill_score,
    }


# ============================================================
# TF-IDF SIMILARITY
# ============================================================

def calculate_similarity(
    resume_text,
    job_description,
):
    """
    Calculate JD-focused TF-IDF similarity.

    Combines:
        70% JD term coverage
        30% traditional cosine similarity
    """

    resume_clean = clean_text(
        resume_text
    )

    jd_clean = clean_text(
        job_description
    )

    if (
        not resume_clean.strip()
        or not jd_clean.strip()
    ):
        return 0.0

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    vectors = vectorizer.fit_transform(
        [
            jd_clean,
            resume_clean,
        ]
    )

    jd_vector = vectors[0].toarray()[0]

    resume_vector = vectors[1].toarray()[0]

    jd_indices = [
        i
        for i, value in enumerate(jd_vector)
        if value > 0
    ]

    if not jd_indices:
        return 0.0

    # --------------------------------------------------------
    # JD coverage
    # --------------------------------------------------------

    matched_weight = sum(
        jd_vector[i]
        for i in jd_indices
        if resume_vector[i] > 0
    )

    total_weight = sum(
        jd_vector[i]
        for i in jd_indices
    )

    coverage_score = (
        matched_weight
        / total_weight
    ) * 100

    # --------------------------------------------------------
    # Cosine similarity
    # --------------------------------------------------------

    cosine_sim = (
        cosine_similarity(
            vectors[0:1],
            vectors[1:2],
        )[0][0]
        * 100
    )

    # --------------------------------------------------------
    # Combined TF-IDF score
    # --------------------------------------------------------

    similarity = (
        0.7 * coverage_score
        + 0.3 * cosine_sim
    )

    return similarity


# ============================================================
# FINAL SCORE
# ============================================================

def calculate_final_score(
    resume_text,
    job_description,
):
    """
    Calculate the final SkillSync score.

    Final weighting:

        70% Skill Match
        30% TF-IDF Similarity

    Skill matching receives higher weight because
    explicit technical skill compatibility is more
    important for resume-JD matching than raw wording overlap.
    """

    tfidf_score = calculate_similarity(
        resume_text,
        job_description,
    )

    skill_result = calculate_skill_match(
        resume_text,
        job_description,
    )

    skill_score = skill_result[
        "skill_score"
    ]

    # --------------------------------------------------------
    # Final weighted score
    # --------------------------------------------------------

    final_score = (
        0.70 * skill_score
        + 0.30 * tfidf_score
    )

    return {
        "tfidf_score": tfidf_score,
        "skill_score": skill_score,
        "final_score": final_score,
        "matched_skills": skill_result[
            "matched_skills"
        ],
        "missing_skills": skill_result[
            "missing_skills"
        ],
        "resume_skills": skill_result[
            "resume_skills"
        ],
        "jd_skills": skill_result[
            "jd_skills"
        ],
    }


# ============================================================
# VERDICT
# ============================================================

def get_verdict(score):
    """
    Convert final score into a human-readable verdict.
    """

    if score >= 75:
        return "Strong Match"

    elif score >= 50:
        return "Moderate Match"

    elif score >= 25:
        return "Fair Match"

    else:
        return "Low Match"
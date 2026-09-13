import re
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from pypdf import PdfReader
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Download required NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

DATA_DIR = Path("data/resume-data/data/data")


def extract_pdf_text(pdf_path):
    """
    Extract text from a resume PDF.
    """
    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text


def preprocess_text(text):
    """
    Clean, tokenize, and lemmatize text for NLP processing.
    """

    # Lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords, short tokens, and apply lemmatization
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


def load_resume_paths():
    """
    Load all resume PDF paths along with their categories.
    """

    resumes = []

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


# Comprehensive skill dictionary spanning all 24 resume dataset categories
SKILLS = [
    # Information Technology / Software Engineering / Data Science / DevOps
    "python", "java", "c++", "c#", "c", "javascript", "typescript", "go", "golang", "rust",
    "ruby", "php", "swift", "kotlin", "scala", "r", "html", "css", "sql", "pl/sql",
    "react", "react.js", "next.js", "vue.js", "angular", "node.js", "express", "express.js",
    "django", "flask", "fastapi", "spring boot", "spring", "asp.net", "laravel", "bootstrap", "tailwind",
    "mysql", "postgresql", "postgres", "mongodb", "redis", "sqlite", "oracle", "dynamodb", "cassandra", "elasticsearch",
    "aws", "azure", "gcp", "cloud", "docker", "kubernetes", "k8s", "terraform", "ansible", "jenkins", "ci/cd",
    "git", "github", "gitlab", "linux", "unix", "bash", "shell", "rest api", "restful api", "graphql", "microservices",
    "machine learning", "deep learning", "nlp", "natural language processing", "tensorflow", "pytorch",
    "scikit-learn", "pandas", "numpy", "opencv", "data analysis", "data science", "neural networks", "computer vision",
    "software development", "software engineer", "programming", "problem solving", "web development",

    # Human Resources (HR)
    "recruitment", "talent acquisition", "employee relations", "talent management", "human resources",
    "hris", "onboarding", "performance management", "payroll", "employee engagement", "compensation",
    "benefits", "labor laws", "succession planning", "conflict resolution", "interviewing", "sourcing",

    # Finance / Accounting / Banking
    "financial analysis", "accounting", "financial reporting", "financial planning", "auditing",
    "budgeting", "forecasting", "taxation", "bookkeeping", "risk management", "compliance",
    "treasury", "portfolio management", "cash flow", "quickbooks", "sap", "tally", "excel",
    "financial modeling", "investment banking", "wealth management", "credit analysis", "loans",
    "retail banking", "commercial banking", "reconciliation", "financial statements",

    # Healthcare / Nursing / Clinical
    "patient care", "clinical", "nursing", "medical records", "emr", "ehr", "healthcare management",
    "triage", "phlebotomy", "diagnostics", "patient safety", "pharmacology", "icu", "cpr", "bls",
    "vital signs", "health information management", "patient assessment",

    # Sales / Business Development / Marketing / Digital Media / PR
    "sales", "business development", "lead generation", "crm", "salesforce", "account management",
    "negotiation", "client relations", "b2b", "b2c", "cold calling", "digital marketing", "seo",
    "sem", "content marketing", "social media marketing", "google analytics", "copywriting",
    "public relations", "brand management", "media relations", "press releases", "campaign management",
    "market research", "marketing strategy",

    # Design / Arts / Apparel
    "ui/ux", "graphic design", "photoshop", "illustrator", "figma", "adobe xd", "indesign",
    "wireframing", "prototyping", "user research", "fashion design", "textile design", "apparel design",
    "creative direction", "sketching", "adobe creative suite",

    # Aviation / Automobile / Construction / Agriculture / Culinary / BPO / Advocate / Fitness / Teacher / Consultant
    "aviation", "flight operations", "aircraft maintenance", "cabin crew", "air traffic control",
    "automotive engineering", "vehicle maintenance", "autocad", "cad", "quality control",
    "construction management", "site supervision", "civil engineering", "project planning", "building codes",
    "agronomy", "crop management", "soil science", "agricultural engineering", "irrigation",
    "culinary arts", "food preparation", "menu planning", "kitchen management", "food safety", "haccp",
    "bpo", "customer service", "call center", "technical support", "helpdesk", "ticket resolution",
    "legal research", "litigation", "contract drafting", "corporate law", "legal compliance", "legal advisory",
    "personal training", "fitness instruction", "nutrition", "wellness coaching", "strength training",
    "teaching", "curriculum development", "classroom management", "lesson planning", "educational leadership",
    "management consulting", "strategy", "process improvement", "business analysis", "change management"
]

# Map common skill abbreviations/aliases to canonical forms for unified matching
SKILL_ALIASES = {
    "hr": "human resources",
    "ml": "machine learning",
    "dl": "deep learning",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "qa": "quality control",
    "pr": "public relations",
    "ui": "ui/ux",
    "ux": "ui/ux",
}


def normalize_for_skill_search(text):
    """
    Normalize text for skill search keeping special technical characters (+, #, ., /, -).
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\+\#\.\/\-\s]", " ", text)
    return text


def extract_skills(text):
    """
    Extract known technical and domain skills from text using boundary-aware regex matching.
    """
    if not text:
        return []

    norm_text = normalize_for_skill_search(text)
    found_skills = set()

    for skill in SKILLS:
        skill_norm = skill.lower()
        escaped = re.escape(skill_norm)
        pattern = r"(?<![a-z0-9])" + escaped + r"(?![a-z0-9])"

        if re.search(pattern, norm_text):
            found_skills.add(skill)

    # Check aliases
    for alias, canonical in SKILL_ALIASES.items():
        escaped_alias = re.escape(alias)
        alias_pattern = r"(?<![a-z0-9])" + escaped_alias + r"(?![a-z0-9])"
        if re.search(alias_pattern, norm_text):
            found_skills.add(canonical)

    return sorted(list(found_skills))


def calculate_skill_match(resume_text, job_description):
    """
    Compare skills present in the resume
    against skills required by the job description.
    """

    resume_skills = set(extract_skills(resume_text))
    jd_skills = set(extract_skills(job_description))

    matched_skills = sorted(resume_skills & jd_skills)
    missing_skills = sorted(jd_skills - resume_skills)

    if not jd_skills:
        skill_score = 0.0
    else:
        skill_score = (
            len(matched_skills) / len(jd_skills)
        ) * 100

    return {
        "resume_skills": sorted(resume_skills),
        "jd_skills": sorted(jd_skills),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "skill_score": skill_score,
    }


def calculate_similarity(resume_text, job_description):
    """
    Calculate TF-IDF JD coverage similarity between a resume and a job description.
    Measures the weighted proportion of Job Description terms and n-grams covered in the resume.
    """

    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(job_description)

    if not resume_clean.strip() or not jd_clean.strip():
        return 0.0

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    vectors = vectorizer.fit_transform([jd_clean, resume_clean])

    jd_vector = vectors[0].toarray()[0]
    resume_vector = vectors[1].toarray()[0]

    jd_indices = np.where(jd_vector > 0)[0]
    if len(jd_indices) == 0:
        return 0.0

    matched_weight = sum(jd_vector[i] for i in jd_indices if resume_vector[i] > 0)
    total_weight = sum(jd_vector[i] for i in jd_indices)

    coverage_score = (matched_weight / total_weight) * 100

    # Also compute traditional cosine similarity for blending
    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100

    # Combine coverage and cosine for optimal balance
    similarity = 0.7 * coverage_score + 0.3 * cosine_sim

    return similarity


def calculate_final_score(resume_text, job_description):
    """
    Calculate the final SkillSync score by combining
    TF-IDF similarity and skill matching.
    """

    tfidf_score = calculate_similarity(
        resume_text,
        job_description
    )

    skill_result = calculate_skill_match(
        resume_text,
        job_description
    )

    skill_score = skill_result["skill_score"]

    final_score = (
        0.5 * tfidf_score
        + 0.5 * skill_score
    )

    return {
        "tfidf_score": tfidf_score,
        "skill_score": skill_score,
        "final_score": final_score,
        "matched_skills": skill_result["matched_skills"],
        "missing_skills": skill_result["missing_skills"],
        "resume_skills": skill_result["resume_skills"],
        "jd_skills": skill_result["jd_skills"],
    }


def get_verdict(score):
    """
    Convert the final score into a human-readable verdict.
    """

    if score >= 75:
        return "Strong Match"

    elif score >= 50:
        return "Moderate Match"

    elif score >= 25:
        return "Fair Match"

    else:
        return "Low Match"
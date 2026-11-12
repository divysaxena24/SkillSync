import re
from functools import lru_cache
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# NLTK RESOURCES (Lazy / Safe Loading)
# ============================================================

for res_name in ("punkt", "punkt_tab", "stopwords", "wordnet"):
    try:
        nltk.data.find(f"tokenizers/{res_name}" if "punkt" in res_name else f"corpora/{res_name}")
    except LookupError:
        try:
            nltk.download(res_name, quiet=True)
        except Exception:
            pass

try:
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
        "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
        "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
        "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
        "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
        "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
        "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
        "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
        "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
        "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
    }

lemmatizer = WordNetLemmatizer()


@lru_cache(maxsize=20000)
def _cached_lemmatize(word):
    """
    Cached word lemmatization to prevent repetitive WordNet lookup overhead.
    """
    return lemmatizer.lemmatize(word)


# ============================================================
# DATASET
# ============================================================

DATA_DIR = Path("data/resume-data/data/data")


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(pdf_path):
    """
    Extract text from a resume PDF with OCR/font repair heuristics.
    """
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    # Repair common broken font/OCR patterns (e.g. 'T ypeScript' -> 'TypeScript')
    text = re.sub(r"(?<![a-zA-Z])T\s+ype([Ss]cript)?", r"Type\1", text)
    text = re.sub(r"(?<![a-zA-Z])NL\s+TK", "NLTK", text)
    text = re.sub(r"(?<![a-zA-Z])anXGBoost", "an XGBoost", text)
    text = re.sub(r"(?<![a-zA-Z])T\s+est(ing)?", r"Test\1", text)

    return text


# ============================================================
# NLP PREPROCESSING
# ============================================================

def preprocess_text(text):
    """
    Clean, tokenize and lemmatize text using cached WordNet lookups.
    """
    if not text:
        return []

    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    tokens = word_tokenize(text)

    cleaned_tokens = []
    for word in tokens:
        if word not in STOP_WORDS and len(word) > 1:
            lemma = _cached_lemmatize(word)
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


from taxonomy import SKILLS, SKILL_ALIASES, GENERIC_TERMS, get_relevant_skills_for_jd


# ============================================================
# SKILL SEARCH NORMALIZATION
# ============================================================

def normalize_for_skill_search(text):
    """
    Normalize text while preserving technical characters (+, #, ., /, -).
    """
    if not text:
        return ""

    text = text.lower()

    # Fix broken tokens
    text = re.sub(r"(?<![a-z0-9])t\s+ypescript(?![a-z0-9])", "typescript", text)
    text = re.sub(r"(?<![a-z0-9])nl\s+tk(?![a-z0-9])", "nltk", text)

    # Normalize punctuation while retaining technical indicators
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
    """
    skill = skill.lower().strip()
    return SKILL_ALIASES.get(
        skill,
        skill,
    )


# ============================================================
# COMPILED PATTERNS & HEURISTICS
# ============================================================

def _build_skill_pattern(term):
    """
    Build a precise regex boundary pattern for technical skill terms.
    """
    if term == "c":
        return re.compile(r"(?<![a-z0-9])c(?![a-z0-9\+\#])")
    if term == "c++":
        return re.compile(r"(?<![a-z0-9])c\+\+(?![a-z0-9\+])")
    if term == "c#":
        return re.compile(r"(?<![a-z0-9])c\#(?![a-z0-9])")
    if term in ("js",):
        return re.compile(r"(?<![a-z0-9\.])js(?![a-z0-9])")
    if term == "r":
        return re.compile(r"(?<![a-z0-9])r(?![a-z0-9\&\/\+\#])")
    if term == "oauth 2.0":
        return re.compile(r"(?<![a-z0-9])oauth(?:\s*2(?:\.0)?)?(?![a-z0-9])")
    if term == "ci/cd":
        return re.compile(r"(?<![a-z0-9])ci\s*[\/\-]?\s*cd(?![a-z0-9])")
    if term == "rest apis":
        return re.compile(r"(?<![a-z0-9])(?:restful|rest)\s*(?:apis?|endpoints?)?(?![a-z0-9])")
    if term == "microservices":
        return re.compile(r"(?<![a-z0-9])microservices?(?![a-z0-9])")
    if term == "webhooks":
        return re.compile(r"(?<![a-z0-9])webhooks?(?![a-z0-9])")
    if term == "retry mechanisms":
        return re.compile(r"(?<![a-z0-9])retry(?:\s*mechanisms?|\s*policies)?(?![a-z0-9])")
    if term == "api error handling":
        return re.compile(r"(?<![a-z0-9])(?:api\s*)?error\s*handling(?![a-z0-9])")
    if term == "third-party integrations":
        return re.compile(r"(?<![a-z0-9])third[\s\-]party(?:\s*integrations?|\s*saas)?(?![a-z0-9])")
    if term == "asynchronous programming":
        return re.compile(r"(?<![a-z0-9])(?:asynchronous\s*programming|async\s*programming|asynchronous|async)(?![a-z0-9])")
    if term == "parallel algorithms":
        return re.compile(r"(?<![a-z0-9])parallel\s*algorithms?(?![a-z0-9])")
    if term == "graph algorithms":
        return re.compile(r"(?<![a-z0-9])graph\s*algorithms?(?![a-z0-9])")
    if term == "performance benchmarking":
        return re.compile(r"(?<![a-z0-9])performance\s*benchmarks?(?:ing)?|benchmarking(?![a-z0-9])")
    if term == "cache coherence":
        return re.compile(r"(?<![a-z0-9])cache[\s\-]coheren(?:ce|cy)(?![a-z0-9])")
    if term == "cloud computing":
        return re.compile(r"(?<![a-z0-9])cloud\s*computing(?![a-z0-9])")
    if term == "cloud deployment":
        return re.compile(r"(?<![a-z0-9])(?:cloud\s*deployments?|automated\s*deployments?|deployments?)(?![a-z0-9])")

    escaped = re.escape(term)
    return re.compile(r"(?<![a-z0-9])" + escaped + r"(?![a-z0-9])")


# Precompile all skill & alias patterns once
_COMPILED_PATTERNS = []

for _skill in SKILLS:
    _s_norm = _skill.lower()
    if _s_norm not in GENERIC_TERMS:
        _COMPILED_PATTERNS.append((_s_norm, canonicalize_skill(_s_norm), _build_skill_pattern(_s_norm)))

for _alias, _can in SKILL_ALIASES.items():
    _a_norm = _alias.lower()
    if _a_norm not in GENERIC_TERMS:
        _COMPILED_PATTERNS.append((_a_norm, canonicalize_skill(_a_norm), _build_skill_pattern(_a_norm)))


def _is_valid_skill_context(skill_norm, raw_text):
    """
    Validate match context to eliminate false positives for ambiguous skills.
    """
    text_lower = raw_text.lower()

    # 1. Standalone 'c' vs 'c++' or 'c#'
    if skill_norm == "c":
        # Must have standalone 'c' NOT followed by + or #
        if not re.search(r"(?<![a-z0-9])c(?![a-z0-9\+\#])", text_lower):
            return False

    # 2. 'js' / 'javascript' false positive from '.js' suffix in framework names (e.g. React.js)
    if skill_norm in ("js", "javascript"):
        has_standalone_js = (
            re.search(r"(?<![a-z0-9\.])javascript(?![a-z0-9])", text_lower)
            or re.search(r"(?<![a-z0-9\.])js(?![a-z0-9])", text_lower)
        )
        if not has_standalone_js:
            return False

    # 3. 'r' language vs 'R&D' or 'R and D'
    if skill_norm == "r":
        clean_r_text = re.sub(r"(?<![a-z0-9])r\s*(\&|and|\/)\s*d(?![a-z0-9])", " ", text_lower)
        if not re.search(r"(?<![a-z0-9])r(?![a-z0-9\+\#])", clean_r_text):
            return False

    # 4. 'go' language vs verb usage ('go to', 'to go', 'on the go', 'going')
    if skill_norm == "go":
        has_tech_go = re.search(r"(?<![a-z0-9])(golang|go\s+(programming|developer|lang|backend|engineer))(?![a-z0-9])", text_lower)
        if not has_tech_go:
            if re.search(r"(?<![a-z0-9])(to\s+go|go\s+to|on\s+the\s+go)(?![a-z0-9])", text_lower):
                return False

    # 5. 'spring' framework vs season/quarter context
    if skill_norm == "spring":
        has_spring_tech = re.search(r"(?<![a-z0-9])spring\s*(boot|framework|mvc|security|data|cloud)(?![a-z0-9])", text_lower)
        if not has_spring_tech:
            if re.search(r"(?<![a-z0-9])spring\s*(20\d\d|quarter|semester|break|season|term)(?![a-z0-9])", text_lower):
                return False

    # 6. 'cloud' vs specific cloud vendors / general noise
    if skill_norm == "cloud":
        # Cloud as standalone skill
        if not re.search(r"(?<![a-z0-9])cloud(?![a-z0-9])", text_lower):
            return False

    return True


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

    for raw_term, canonical, pattern in _COMPILED_PATTERNS:
        if pattern.search(norm_text):
            if _is_valid_skill_context(raw_term, text):
                if canonical not in GENERIC_TERMS:
                    found_skills.add(canonical)

    return sorted(found_skills)


# ============================================================
# JD REQUIREMENT EXTRACTION
# ============================================================

def extract_jd_requirements(job_description):
    """
    Parse a Job Description to extract technical requirements reliably.
    Extracts individual skills from sentences, bullet points, and comma-separated lists,
    filtering out generic terms.
    """
    if not job_description:
        return []

    return extract_skills(job_description)


# ============================================================
# SKILL MATCHING
# ============================================================

MAX_ADDITIONAL_SKILLS = 5  # Reasonable cap to prevent skill-list gaming


def calculate_skill_match(
    resume_text,
    job_description,
):
    """
    Compare canonical skills present in the resume
    against canonical skills required by the JD.

    Also identifies additional relevant skills (skills in the resume
    that are not explicitly required by the JD, but belong to the relevant
    domain/role of the job).
    """
    resume_skills = set(extract_skills(resume_text))
    jd_skills = set(extract_jd_requirements(job_description))

    matched_skills = sorted(resume_skills & jd_skills)
    missing_skills = sorted(jd_skills - resume_skills)

    if not jd_skills:
        skill_score = 0.0
    else:
        skill_score = (len(matched_skills) / len(jd_skills)) * 100.0

    # Additional Relevant Skills:
    # 1. Skills present in resume but NOT in matched required skills
    extra_skills = resume_skills - jd_skills

    # 2. Filter extra skills by role/domain relevance based on JD requirements
    relevant_domain_skills = get_relevant_skills_for_jd(jd_skills)
    additional_relevant_skills = sorted(extra_skills & relevant_domain_skills)

    # 3. Calculate additional skill score normalized to 0-100 with cap
    additional_score = min(
        100.0,
        (len(additional_relevant_skills) / MAX_ADDITIONAL_SKILLS) * 100.0,
    ) if jd_skills else 0.0

    return {
        "resume_skills": sorted(resume_skills),
        "jd_skills": sorted(jd_skills),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "additional_skills": additional_relevant_skills,
        "skill_score": skill_score,
        "additional_score": additional_score,
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
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(job_description)

    if not resume_clean.strip() or not jd_clean.strip():
        return 0.0

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    try:
        vectors = vectorizer.fit_transform([jd_clean, resume_clean])
    except ValueError:
        return 0.0

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
        matched_weight / total_weight
    ) * 100.0

    # --------------------------------------------------------
    # Cosine similarity
    # --------------------------------------------------------
    cosine_sim = (
        cosine_similarity(
            vectors[0:1],
            vectors[1:2],
        )[0][0]
        * 100.0
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

    New weighted formula:
        Final Score = 0.70 * R + 0.10 * A + 0.20 * T

    Where:
        R (70%): Required Skill Match Score
        A (10%): Additional Relevant Skills Score
        T (20%): TF-IDF Similarity Score
    """
    tfidf_score = calculate_similarity(
        resume_text,
        job_description,
    )

    skill_result = calculate_skill_match(
        resume_text,
        job_description,
    )

    skill_score = skill_result["skill_score"]
    additional_score = skill_result["additional_score"]

    final_score = (
        0.70 * skill_score
        + 0.10 * additional_score
        + 0.20 * tfidf_score
    )

    return {
        "tfidf_score": tfidf_score,
        "skill_score": skill_score,
        "additional_score": additional_score,
        "final_score": final_score,
        "matched_skills": skill_result["matched_skills"],
        "missing_skills": skill_result["missing_skills"],
        "additional_skills": skill_result["additional_skills"],
        "resume_skills": skill_result["resume_skills"],
        "jd_skills": skill_result["jd_skills"],
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

"""
Unit tests for SkillSync NLP Engine (nlp.py).
Validates text preprocessing, PDF extraction, skill extraction, skill matching,
TF-IDF similarity, final score calculation, and verdict classification.
"""

from pathlib import Path
import pytest
from pypdf import PdfWriter

import nlp
from nlp import (
    preprocess_text,
    clean_text,
    normalize_for_skill_search,
    canonicalize_skill,
    extract_skills,
    calculate_skill_match,
    calculate_similarity,
    calculate_final_score,
    get_verdict,
    extract_pdf_text,
    load_resume_paths,
)


# ============================================================
# 1. BASIC FUNCTIONALITY & PREPROCESSING TESTS
# ============================================================

class TestPreprocessing:
    def test_preprocess_text_normal_english(self):
        text = "Developed web applications using Python 3 and Django framework."
        tokens = preprocess_text(text)
        assert isinstance(tokens, list)
        assert "python" in tokens
        assert "django" in tokens
        assert "framework" in tokens
        assert "and" not in tokens

    def test_preprocess_text_empty(self):
        assert preprocess_text("") == []
        assert preprocess_text("   ") == []

    def test_clean_text_normal_english(self):
        text = "Building scalable microservices with Python and FastAPI."
        cleaned = clean_text(text)
        assert isinstance(cleaned, str)
        assert "python" in cleaned
        assert "fastapi" in cleaned

    def test_clean_text_empty(self):
        assert clean_text("") == ""

    def test_normalize_for_skill_search(self):
        raw_text = "Proficient in C++, C#, Node.js, and React.js!"
        normalized = normalize_for_skill_search(raw_text)
        assert "c++" in normalized
        assert "c#" in normalized
        assert "node.js" in normalized
        assert "react.js" in normalized


# ============================================================
# 2. PDF & DATASET HELPER TESTS
# ============================================================

class TestHelperFunctions:
    def test_extract_pdf_text(self, tmp_path):
        pdf_file = tmp_path / "sample.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        with open(pdf_file, "wb") as f:
            writer.write(f)

        extracted = extract_pdf_text(pdf_file)
        assert isinstance(extracted, str)

    def test_load_resume_paths_nonexistent_dir(self, monkeypatch, tmp_path):
        nonexistent = tmp_path / "no_data_here"
        monkeypatch.setattr(nlp, "DATA_DIR", nonexistent)
        resumes = load_resume_paths()
        assert resumes == []

    def test_load_resume_paths_with_structure(self, monkeypatch, tmp_path):
        fake_data_dir = tmp_path / "data"
        cat_dir = fake_data_dir / "ENGINEERING"
        cat_dir.mkdir(parents=True)
        pdf_file = cat_dir / "resume_01.pdf"
        pdf_file.write_text("dummy content")

        monkeypatch.setattr(nlp, "DATA_DIR", fake_data_dir)
        resumes = load_resume_paths()
        assert len(resumes) == 1
        assert resumes[0]["category"] == "ENGINEERING"
        assert resumes[0]["path"] == pdf_file


# ============================================================
# 3. ALIAS & CANONICALIZATION TESTS
# ============================================================

class TestCanonicalization:
    @pytest.mark.parametrize(
        "raw_skill,expected_canonical",
        [
            ("React.js", "react"),
            ("ReactJS", "react"),
            ("react.js", "react"),
            ("PostgreSQL", "sql"),
            ("Postgres", "sql"),
            ("MySQL", "sql"),
            ("JS", "javascript"),
            ("jscript", "javascript"),
            ("TS", "typescript"),
            ("py", "python"),
            ("Node", "node.js"),
            ("Nodejs", "node.js"),
            ("Express.js", "express"),
            ("ExpressJS", "express"),
            ("k8s", "kubernetes"),
            ("hr", "human resources"),
            ("qa", "quality control"),
            ("pr", "public relations"),
            ("ui", "ui/ux"),
            ("ux", "ui/ux"),
            ("python", "python"),
            ("docker", "docker"),
        ],
    )
    def test_canonicalize_skill(self, raw_skill, expected_canonical):
        assert canonicalize_skill(raw_skill) == expected_canonical


# ============================================================
# 4. SKILL EXTRACTION TESTS
# ============================================================

class TestSkillExtraction:
    def test_extract_skills_target_skills(self):
        text = """
        Skilled in Python, Java, JavaScript, TypeScript, React, Docker, AWS, SQL,
        PostgreSQL, MySQL, Node.js, and Kubernetes.
        """
        extracted = extract_skills(text)
        expected_subset = {
            "python",
            "java",
            "javascript",
            "typescript",
            "react",
            "docker",
            "aws",
            "sql",
            "node.js",
            "kubernetes",
        }
        assert expected_subset.issubset(set(extracted))

    def test_extract_skills_aliases_normalized(self):
        text = "Hands-on experience with JS, TS, React.js, PostgreSQL, and MySQL."
        extracted = extract_skills(text)
        assert "javascript" in extracted
        assert "typescript" in extracted
        assert "react" in extracted
        assert "sql" in extracted

    def test_extract_skills_ignore_generic_terms(self):
        text = "Software Engineer with strong software development and problem solving skills."
        extracted = extract_skills(text)
        assert "software engineer" not in extracted
        assert "software development" not in extracted
        assert "problem solving" not in extracted

    def test_extract_skills_empty(self):
        assert extract_skills("") == []

    def test_extract_skills_deduplication(self):
        text = "Python Developer using python and PY with Python3."
        extracted = extract_skills(text)
        assert extracted.count("python") == 1


# ============================================================
# 5. SKILL MATCHING TESTS
# ============================================================

class TestSkillMatching:
    def test_strong_overlap(self):
        resume = "Expert in Python, Docker, AWS, and SQL databases."
        jd = "Looking for Python engineer with Docker, AWS, and SQL experience."
        match = calculate_skill_match(resume, jd)

        assert match["skill_score"] == pytest.approx(100.0)
        assert match["missing_skills"] == []
        assert set(match["matched_skills"]) == {"aws", "docker", "python", "sql"}

    def test_partial_overlap(self):
        resume = "Developer with Python and SQL experience."
        jd = "Requires Python, SQL, Docker, and AWS."
        match = calculate_skill_match(resume, jd)

        assert match["skill_score"] == pytest.approx(50.0)
        assert set(match["matched_skills"]) == {"python", "sql"}
        assert set(match["missing_skills"]) == {"aws", "docker"}

    def test_no_overlap(self):
        resume = "Experienced in Python, Django, and PostgreSQL."
        jd = "Nurse needed with patient care and CPR certification."
        match = calculate_skill_match(resume, jd)

        assert match["skill_score"] == pytest.approx(0.0)
        assert match["matched_skills"] == []

    def test_identical_skills(self):
        text = "Proficient in React, Node.js, and TypeScript."
        match = calculate_skill_match(text, text)
        assert match["skill_score"] == pytest.approx(100.0)
        assert match["missing_skills"] == []

    def test_empty_jd(self):
        resume = "Python, Java, C++ developer."
        jd = "General administrative role with no explicit technical stack listed."
        match = calculate_skill_match(resume, jd)
        assert match["skill_score"] == pytest.approx(0.0)
        assert match["jd_skills"] == []


# ============================================================
# 6. TF-IDF SIMILARITY TESTS
# ============================================================

class TestSimilarity:
    def test_related_text_produces_non_zero_similarity(self):
        resume = "Python software developer with machine learning experience building web backend models."
        jd = "Seeking a Python developer for building software applications and machine learning algorithms."
        similarity = calculate_similarity(resume, jd)

        assert isinstance(similarity, float)
        assert similarity > 0.0

    def test_unrelated_text_produces_lower_similarity(self):
        jd = "Senior Python engineer building microservices in Docker and AWS."
        related_resume = "Software engineer working with Python, Docker containers, and AWS cloud."
        unrelated_resume = "Experienced executive chef skilled in menu planning and commercial kitchen management."

        sim_related = calculate_similarity(related_resume, jd)
        sim_unrelated = calculate_similarity(unrelated_resume, jd)

        assert sim_related > sim_unrelated

    def test_identical_text_consistency(self):
        text = "Senior Fullstack Developer with React, Node.js, and PostgreSQL skills."
        similarity = calculate_similarity(text, text)
        assert similarity > 80.0

    def test_empty_input(self):
        assert calculate_similarity("", "Python developer") == 0.0
        assert calculate_similarity("Python developer", "") == 0.0
        assert calculate_similarity("", "") == 0.0


# ============================================================
# 7. FINAL SCORE TESTS
# ============================================================

class TestFinalScore:
    def test_score_structure_and_bounds(self):
        resume = "Python developer with SQL, Docker, and Git experience."
        jd = "Looking for a Python Developer proficient in SQL, Docker, Git, and AWS."
        result = calculate_final_score(resume, jd)

        assert "tfidf_score" in result
        assert "skill_score" in result
        assert "final_score" in result
        assert "matched_skills" in result
        assert "missing_skills" in result
        assert "resume_skills" in result
        assert "jd_skills" in result

        final_score = result["final_score"]
        assert isinstance(final_score, float)
        assert 0.0 <= final_score <= 100.0

    def test_weighting_formula(self):
        resume = "Python developer with SQL."
        jd = "Python developer with SQL and AWS."
        result = calculate_final_score(resume, jd)

        expected_final = 0.70 * result["skill_score"] + 0.30 * result["tfidf_score"]
        assert result["final_score"] == pytest.approx(expected_final)


# ============================================================
# 8. VERDICT TESTS
# ============================================================

class TestVerdict:
    @pytest.mark.parametrize(
        "score,expected_verdict",
        [
            (100.0, "Strong Match"),
            (75.0, "Strong Match"),
            (74.9, "Moderate Match"),
            (50.0, "Moderate Match"),
            (49.9, "Fair Match"),
            (25.0, "Fair Match"),
            (24.9, "Low Match"),
            (0.0, "Low Match"),
        ],
    )
    def test_verdict_thresholds(self, score, expected_verdict):
        assert get_verdict(score) == expected_verdict


# ============================================================
# 9. EDGE CASES
# ============================================================

class TestEdgeCases:
    def test_case_insensitivity(self):
        extracted_upper = extract_skills("PYTHON JAVA DOCKER AWS")
        extracted_lower = extract_skills("python java docker aws")
        assert extracted_upper == extracted_lower

    def test_punctuation_and_symbols(self):
        text = "Skills: [Python], (Java), <Docker>! AWS & SQL..."
        extracted = extract_skills(text)
        assert set(extracted) >= {"python", "java", "docker", "aws", "sql"}

    def test_unrelated_nonsense_text(self):
        result = calculate_final_score("asdfghjkl qwertyuiop zxcvbnm", "12345 67890 !@#$%^&*()")
        assert result["final_score"] == 0.0
        assert result["matched_skills"] == []

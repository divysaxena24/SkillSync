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
        assert "additional_score" in result
        assert "final_score" in result
        assert "matched_skills" in result
        assert "missing_skills" in result
        assert "additional_skills" in result
        assert "resume_skills" in result
        assert "jd_skills" in result

        final_score = result["final_score"]
        assert isinstance(final_score, float)
        assert 0.0 <= final_score <= 100.0

    def test_weighting_formula(self):
        resume = "Python developer with SQL."
        jd = "Python developer with SQL and AWS."
        result = calculate_final_score(resume, jd)

        expected_final = (
            0.70 * result["skill_score"]
            + 0.10 * result["additional_score"]
            + 0.20 * result["tfidf_score"]
        )
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


# ============================================================
# 10. REFINED BOUNDARY & FALSE POSITIVE ELIMINATION TESTS
# ============================================================

class TestRefinedBoundaryExtraction:
    def test_cpp_and_csharp_do_not_trigger_false_positive_c(self):
        text = "Experienced software engineer proficient in C++ and C# development."
        extracted = extract_skills(text)
        assert "c++" in extracted
        assert "c#" in extracted
        assert "c" not in extracted

    def test_standalone_c_triggers_c(self):
        text = "Proficient in C, C++, and Python."
        extracted = extract_skills(text)
        assert "c" in extracted
        assert "c++" in extracted

    def test_react_js_does_not_trigger_false_positive_javascript(self):
        text = "Built web applications using React.js and Node.js."
        extracted = extract_skills(text)
        assert "react" in extracted
        assert "node.js" in extracted
        assert "javascript" not in extracted

    def test_explicit_js_triggers_javascript(self):
        text = "Fullstack developer skilled in JS, React.js, and Node.js."
        extracted = extract_skills(text)
        assert "javascript" in extracted
        assert "react" in extracted
        assert "node.js" in extracted

    def test_r_and_d_does_not_trigger_false_positive_r_language(self):
        text = "Led R&D and R and D innovation projects in healthcare."
        extracted = extract_skills(text)
        assert "r" not in extracted

    def test_r_programming_triggers_r_language(self):
        text = "Statistical analysis using R programming and RStudio."
        extracted = extract_skills(text)
        assert "r" in extracted

    def test_go_verb_does_not_trigger_golang(self):
        text = "We go to office every day and have things to go."
        extracted = extract_skills(text)
        assert "go" not in extracted

    def test_golang_tech_triggers_go(self):
        text = "Backend services developed in Golang and Go programming."
        extracted = extract_skills(text)
        assert "go" in extracted

    def test_spring_season_does_not_trigger_spring_framework(self):
        text = "Graduated in Spring 2024 during the spring semester."
        extracted = extract_skills(text)
        assert "spring" not in extracted

    def test_spring_boot_tech_triggers_spring_framework(self):
        text = "Developed microservices with Spring Boot and Spring Framework."
        extracted = extract_skills(text)
        assert "spring" in extracted

    def test_modern_ai_ml_aliases(self):
        text = "Building Generative AI apps with RAG, LangChain, sklearn, and Amazon Web Services."
        extracted = extract_skills(text)
        assert "generative ai" in extracted
        assert "rag" in extracted
        assert "langchain" in extracted
        assert "scikit-learn" in extracted
        assert "aws" in extracted



# ============================================================
# 11. EXPANDED DOMAIN & ACCURACY REGRESSION TESTS
# ============================================================

class TestExpandedDomainAndAccuracy:
    def test_programming_languages_isolation(self):
        # C vs C++ vs C#
        text = "Expert in C++ and C# but never worked with another language."
        skills = extract_skills(text)
        assert "c++" in skills
        assert "c#" in skills
        assert "c" not in skills

    def test_node_and_next_js(self):
        text = "Built web apps with Node.js and Next.js and React."
        skills = extract_skills(text)
        assert "node.js" in skills
        assert "next.js" in skills
        assert "react" in skills

    def test_machine_learning_specialized_skills(self):
        text = "Trained XGBoost model with scikit-learn, tracked in MLflow, explained with SHAP, feature engineering applied."
        skills = extract_skills(text)
        assert "xgboost" in skills
        assert "scikit-learn" in skills
        assert "mlflow" in skills
        assert "shap" in skills
        assert "feature engineering" in skills

    def test_parallel_computing_specialized_skills(self):
        text = "Implemented BFS and DFS on graphs using OpenMP, Triangle Counting, parallel algorithms, synchronization, and cache coherence."
        skills = extract_skills(text)
        assert "openmp" in skills
        assert "bfs" in skills
        assert "dfs" in skills
        assert "triangle counting" in skills
        assert "parallel algorithms" in skills
        assert "synchronization" in skills
        assert "cache coherence" in skills

    def test_automation_and_integration_skills(self):
        text = "Configured n8n workflows with webhooks, REST APIs, OAuth 2.0, Slack, and Google Workspace integrations."
        skills = extract_skills(text)
        assert "n8n" in skills
        assert "webhooks" in skills
        assert "rest apis" in skills
        assert "oauth 2.0" in skills
        assert "slack" in skills
        assert "google workspace" in skills

    def test_devops_and_cloud_infrastructure(self):
        text = "Infrastructure with Terraform, Helm, Kubernetes, AWS, EC2, S3, IAM, DNS, HTTP, monitoring, and logging."
        skills = extract_skills(text)
        assert "terraform" in skills
        assert "helm" in skills
        assert "kubernetes" in skills
        assert "aws" in skills
        assert "ec2" in skills
        assert "s3" in skills
        assert "iam" in skills
        assert "dns" in skills
        assert "http" in skills
        assert "monitoring" in skills
        assert "logging" in skills

    def test_unrelated_resume_and_jd(self):
        chef_resume = "Experienced head chef specializing in pastry, culinary arts, menu design, kitchen management."
        dev_jd = "Senior Kubernetes and Terraform Cloud DevOps Engineer with AWS and CI/CD."
        result = calculate_final_score(chef_resume, dev_jd)
        assert result["skill_score"] == 0.0
        assert result["matched_skills"] == []
        assert result["final_score"] < 25.0
        assert get_verdict(result["final_score"]) == "Low Match"

    def test_empty_resume_and_jd_boundaries(self):
        res1 = calculate_final_score("", "Python and Docker")
        assert res1["final_score"] == 0.0
        assert res1["skill_score"] == 0.0

        res2 = calculate_final_score("Python and Docker", "")
        assert res2["final_score"] == 0.0
        assert res2["skill_score"] == 0.0

    def test_six_jds_benchmark_consistency(self):
        jd_ai_auto = (
            "Python, JavaScript, n8n, REST APIs, Webhooks, Third-party integrations, "
            "Slack, Google Workspace, Asynchronous Programming, Retry Mechanisms, "
            "Fault Tolerance, OAuth 2.0, JSON, API Error Handling, Docker, Git, CI/CD"
        )
        skills = extract_skills(jd_ai_auto)
        assert "n8n" in skills
        assert "webhooks" in skills
        assert "rest apis" in skills
        assert "oauth 2.0" in skills
        assert "retry mechanisms" in skills
        assert "fault tolerance" in skills
        assert "api error handling" in skills


# ============================================================
# 12. ADDITIONAL RELEVANT SKILLS & FORMULA TESTS
# ============================================================

class TestAdditionalRelevantSkills:
    def test_tailwind_and_tailwind_css_deduplication(self):
        # Tailwind and Tailwind CSS both normalize to 'tailwind css'
        text = "Frontend developer skilled in Tailwind and Tailwind CSS."
        skills = extract_skills(text)
        assert skills.count("tailwind css") == 1
        assert "tailwind" not in skills

    def test_rest_api_and_rest_apis_normalization(self):
        text = "Backend engineer creating REST API and REST APIs endpoints."
        skills = extract_skills(text)
        assert skills.count("rest apis") == 1

    def test_additional_relevant_skills_awarded_for_related_domain(self):
        resume = "Fullstack engineer proficient in React, TypeScript, Next.js, and FastAPI."
        jd = "Seeking a developer skilled in React and TypeScript."
        res = calculate_final_score(resume, jd)

        # R should be 100% (React, TypeScript matched)
        assert res["skill_score"] == 100.0
        assert set(res["matched_skills"]) == {"react", "typescript"}

        # Next.js and FastAPI should be rewarded as additional relevant skills
        assert "next.js" in res["additional_skills"]
        assert "fastapi" in res["additional_skills"]
        assert res["additional_score"] > 0.0

    def test_unrelated_skills_not_counted_in_additional(self):
        resume = "React and TypeScript developer with OpenMP, BFS, and C++ experience."
        jd = "Frontend Developer needed with React and TypeScript."
        res = calculate_final_score(resume, jd)

        # Systems/Parallel Computing skills must NOT be credited as additional for Frontend
        assert "openmp" not in res["additional_skills"]
        assert "c++" not in res["additional_skills"]
        assert "bfs" not in res["additional_skills"]

    def test_additional_skills_do_not_duplicate_required_matches(self):
        resume = "React and TypeScript developer."
        jd = "Frontend Developer needed with React and TypeScript."
        res = calculate_final_score(resume, jd)

        # Matched skills cannot appear in additional skills
        assert set(res["matched_skills"]).isdisjoint(set(res["additional_skills"]))

    def test_additional_skills_capped_at_max(self):
        # Even with many relevant skills, additional score cannot exceed 100%
        resume = "Fullstack expert: React, TypeScript, Next.js, FastAPI, Node.js, Express, Docker, CI/CD, MongoDB, Redis, GraphQL, SQL."
        jd = "React and TypeScript developer."
        res = calculate_final_score(resume, jd)
        assert res["additional_score"] <= 100.0

    def test_all_components_bounded_between_0_and_100(self):
        resume = "Python, Docker, AWS, React, Node.js developer."
        jd = "Python, Docker, Kubernetes developer."
        res = calculate_final_score(resume, jd)

        assert 0.0 <= res["skill_score"] <= 100.0
        assert 0.0 <= res["additional_score"] <= 100.0
        assert 0.0 <= res["tfidf_score"] <= 100.0
        assert 0.0 <= res["final_score"] <= 100.0

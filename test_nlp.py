import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# Ensure UTF-8 output encoding for Windows terminal compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from nlp import (
    load_resume_paths,
    extract_pdf_text,
    calculate_final_score,
    get_verdict,
)

# ============================================================
# EVALUATION CONFIGURATION
# ============================================================

SAMPLES_PER_CATEGORY = 10
RANDOM_SEED = 42

# Set random seed for reproducibility
random.seed(RANDOM_SEED)

# ============================================================
# JOB DESCRIPTIONS & CATEGORY MAPPING
# ============================================================

JOB_EVALUATION = {
    "Software / Python Developer": {
        "jd": """
        We are seeking a skilled Software / Python Developer to join our engineering team.
        The ideal candidate will have extensive hands-on experience developing scalable back-end
        applications using Python and Java. Key responsibilities include designing relational database schemas
        with SQL, containerizing microservices with Docker, deploying applications on AWS cloud infrastructure,
        and managing code repositories with Git. Strong expertise in software development principles,
        problem-solving, and machine learning models is highly desirable.
        """,
        "skills_expected": ["python", "java", "sql", "docker", "aws", "git", "machine learning", "software development"],
        "relevant_categories": [
            "INFORMATION-TECHNOLOGY",
            "ENGINEERING",
        ],
        "unrelated_categories": [
            "FINANCE",
            "HR",
            "TEACHER",
            "AVIATION",
        ],
    },

    "Frontend Developer": {
        "jd": """
        We are looking for a passionate Frontend Developer to build responsive, dynamic user interfaces.
        You will work closely with UI/UX designers to translate design mockups into high-performance web applications.
        Required technical skills include proficiency in JavaScript, TypeScript, and modern component frameworks like React.
        Solid mastery of semantic HTML, modern CSS styling, web development standards, and version control using Git
        is essential. Experience with cross-browser compatibility and frontend optimization is a plus.
        """,
        "skills_expected": ["javascript", "typescript", "react", "html", "css", "git", "web development"],
        "relevant_categories": [
            "INFORMATION-TECHNOLOGY",
            "DESIGNER",
            "DIGITAL-MEDIA",
        ],
        "unrelated_categories": [
            "FINANCE",
            "HR",
            "TEACHER",
        ],
    },

    "Finance Analyst": {
        "jd": """
        We are hiring a Finance Analyst to support corporate financial planning, budgeting, and forecasting processes.
        The candidate will prepare monthly financial reporting packages, analyze balance sheets, evaluate cash flows,
        and perform detailed financial modeling. Required skills include deep expertise in accounting principles,
        financial analysis, financial planning, advanced Microsoft Excel functions, and querying corporate databases using SQL.
        Strong analytical mindset and financial statement reconciliation experience are required.
        """,
        "skills_expected": ["accounting", "financial analysis", "financial reporting", "financial planning", "excel", "sql"],
        "relevant_categories": [
            "FINANCE",
            "BANKING",
            "ACCOUNTANT",
        ],
        "unrelated_categories": [
            "INFORMATION-TECHNOLOGY",
            "HR",
            "TEACHER",
        ],
    },

    "HR Manager": {
        "jd": """
        We are seeking an experienced HR Manager to oversee talent acquisition and human resources strategy.
        Responsibilities include leading full-cycle recruitment, building employee relations initiatives, developing
        talent management frameworks, and driving employee performance management programs. The successful candidate
        will have a proven background in human resources administration, labor law compliance, compensation & benefits,
        and conflict resolution. Outstanding interpersonal and organizational leadership skills are essential.
        """,
        "skills_expected": ["human resources", "recruitment", "talent acquisition", "employee relations", "talent management", "performance management"],
        "relevant_categories": [
            "HR",
        ],
        "unrelated_categories": [
            "FINANCE",
            "INFORMATION-TECHNOLOGY",
            "AVIATION",
            "TEACHER",
        ],
    },

    "Data Scientist": {
        "jd": """
        We are searching for a Data Scientist to transform large datasets into actionable business insights and predictive algorithms.
        Key duties include building machine learning models, statistical analysis, and feature engineering using Python data stacks.
        Core requirements include proficiency with Python, Pandas, NumPy, Scikit-learn, and querying databases using SQL.
        Experience in data science workflows, deep learning frameworks, data visualization, and exploratory data analysis
        is highly valued. Strong mathematical background and analytical problem solving are required.
        """,
        "skills_expected": ["python", "machine learning", "data science", "pandas", "numpy", "scikit-learn", "sql"],
        "relevant_categories": [
            "INFORMATION-TECHNOLOGY",
            "ENGINEERING",
        ],
        "unrelated_categories": [
            "FINANCE",
            "HR",
            "TEACHER",
            "AVIATION",
        ],
    },
}

# ============================================================
# EVALUATION ENGINE
# ============================================================

def run_evaluation():
    # 1. Load Dataset
    print("=" * 70)
    print("SKILLSYNC DATASET EVALUATION")
    print("=" * 70)

    resumes = load_resume_paths()
    total_resumes = len(resumes)
    print(f"\nDataset size           : {total_resumes}")
    print(f"Samples per category   : {SAMPLES_PER_CATEGORY}")
    print(f"Random seed            : {RANDOM_SEED}")

    # Group resumes by category
    by_category = defaultdict(list)
    for r in resumes:
        by_category[r["category"]].append(r)

    # 2. Sample Resumes per Category using fixed random seed
    sampled_by_category = {}
    total_sampled = 0
    for cat_name, cat_resumes in by_category.items():
        # Deterministic sample per category
        sample_count = min(SAMPLES_PER_CATEGORY, len(cat_resumes))
        sampled_by_category[cat_name] = random.sample(cat_resumes, sample_count)
        total_sampled += len(sampled_by_category[cat_name])

    print(f"Total categories found : {len(by_category)}")
    print(f"Total resumes sampled  : {total_sampled}")

    # Text Caching & Error Counters
    text_cache = {}
    successful_evaluations = 0
    failed_pdfs = 0
    failed_paths = []

    def get_pdf_text_cached(path):
        nonlocal successful_evaluations, failed_pdfs
        if path not in text_cache:
            try:
                extracted = extract_pdf_text(path)
                if not extracted or not extracted.strip():
                    print(f"  ⚠️ WARNING: Empty text extracted from {path.name}")
                    text_cache[path] = ""
                    failed_pdfs += 1
                    failed_paths.append(path)
                else:
                    text_cache[path] = extracted
                    successful_evaluations += 1
            except Exception as exc:
                print(f"  ⚠️ WARNING: Could not extract {path.name}: {exc}")
                text_cache[path] = ""
                failed_pdfs += 1
                failed_paths.append(path)
        return text_cache[path]

    # Overall Summary Storage
    overall_summary = []

    # 3. Process Each Job Description
    for job_name, config in JOB_EVALUATION.items():
        print("\n" + "=" * 70)
        print(f"JOB: {job_name.upper()}")
        print("=" * 70)

        jd_text = config["jd"].strip()
        relevant_cats = config["relevant_categories"]
        unrelated_cats = config["unrelated_categories"]

        all_target_cats = sorted(list(set(relevant_cats + unrelated_cats)))

        # Track results per category for this job
        cat_stats = {}
        all_job_candidates = []

        for cat_name in all_target_cats:
            cat_samples = sampled_by_category.get(cat_name, [])
            cat_results = []

            for r in cat_samples:
                text = get_pdf_text_cached(r["path"])
                if not text:
                    continue

                eval_result = calculate_final_score(text, jd_text)
                verdict = get_verdict(eval_result["final_score"])

                item = {
                    "category": cat_name,
                    "path": r["path"],
                    "filename": r["path"].name,
                    "tfidf_score": eval_result["tfidf_score"],
                    "skill_score": eval_result["skill_score"],
                    "final_score": eval_result["final_score"],
                    "verdict": verdict,
                    "matched_skills": eval_result["matched_skills"],
                    "missing_skills": eval_result["missing_skills"],
                    "is_relevant": cat_name in relevant_cats,
                }
                cat_results.append(item)
                all_job_candidates.append(item)

            # Compute category statistics
            if cat_results:
                tfidfs = [x["tfidf_score"] for x in cat_results]
                skills = [x["skill_score"] for x in cat_results]
                finals = [x["final_score"] for x in cat_results]

                cat_stats[cat_name] = {
                    "count": len(cat_results),
                    "avg_tfidf": statistics.mean(tfidfs),
                    "avg_skill": statistics.mean(skills),
                    "avg_final": statistics.mean(finals),
                    "min_final": min(finals),
                    "max_final": max(finals),
                    "stdev_final": statistics.stdev(finals) if len(finals) > 1 else 0.0,
                    "is_relevant": cat_name in relevant_cats,
                }

        # Display Category Performance Table
        print("\nCATEGORY PERFORMANCE")
        print("-" * 70)
        print(
            f"{'Category':<26}"
            f"{'Type':<12}"
            f"{'Samples':>8}"
            f"{'Avg TF-IDF':>12}"
            f"{'Avg Skill':>12}"
            f"{'Avg Final':>12}"
        )
        print("-" * 70)

        rel_finals = []
        unrel_finals = []

        for cat_name, stats in cat_stats.items():
            cat_type = "Relevant" if stats["is_relevant"] else "Unrelated"
            print(
                f"{cat_name[:25]:<26}"
                f"{cat_type:<12}"
                f"{stats['count']:>8}"
                f"{stats['avg_tfidf']:>11.2f}%"
                f"{stats['avg_skill']:>11.2f}%"
                f"{stats['avg_final']:>11.2f}%"
            )
            if stats["is_relevant"]:
                rel_finals.extend([x["final_score"] for x in all_job_candidates if x["category"] == cat_name])
            else:
                unrel_finals.extend([x["final_score"] for x in all_job_candidates if x["category"] == cat_name])

        rel_avg = statistics.mean(rel_finals) if rel_finals else 0.0
        unrel_avg = statistics.mean(unrel_finals) if unrel_finals else 0.0
        separation = rel_avg - unrel_avg

        print("-" * 70)
        print(f"Relevant categories average  : {rel_avg:6.2f}%")
        print(f"Unrelated categories average : {unrel_avg:6.2f}%")
        print(f"Separation                   : {separation:6.2f} percentage points")

        # Top 5 Candidates Ranking
        all_job_candidates.sort(key=lambda x: x["final_score"], reverse=True)
        top_5 = all_job_candidates[:5]

        print("\nTOP 5 CANDIDATES")
        print("-" * 70)
        print(
            f"{'Rank':<6}"
            f"{'Category':<25}"
            f"{'Resume File':<16}"
            f"{'TF-IDF':>10}"
            f"{'Skill':>10}"
            f"{'Final Score':>12}"
        )
        print("-" * 70)

        for rank, cand in enumerate(top_5, start=1):
            print(
                f"{rank:<6}"
                f"{cand['category'][:24]:<25}"
                f"{cand['filename']:<16}"
                f"{cand['tfidf_score']:>9.2f}%"
                f"{cand['skill_score']:>9.2f}%"
                f"{cand['final_score']:>11.2f}%"
            )

        overall_summary.append({
            "job": job_name,
            "rel_avg": rel_avg,
            "unrel_avg": unrel_avg,
            "separation": separation,
            "top_candidates": top_5,
        })

    # 4. Final Evaluation Summary
    print("\n\n")
    print("=" * 70)
    print("OVERALL EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"{'Job Description':<32}"
        f"{'Relevant Avg':>14}"
        f"{'Unrelated Avg':>15}"
        f"{'Separation':>14}"
    )
    print("-" * 70)

    for item in overall_summary:
        print(
            f"{item['job'][:30]:<32}"
            f"{item['rel_avg']:>13.2f}%"
            f"{item['unrel_avg']:>14.2f}%"
            f"{item['separation']:>13.2f} pts"
        )

    print("=" * 70)

    # Best & Weakest Separation calculation
    best_job = max(overall_summary, key=lambda x: x["separation"])
    weakest_job = min(overall_summary, key=lambda x: x["separation"])

    print(f"\nBest separation    : {best_job['job']} (+{best_job['separation']:.2f} points)")
    print(f"Weakest separation : {weakest_job['job']} (+{weakest_job['separation']:.2f} points)")

    # Overall dataset statistics
    avg_separation = statistics.mean([x["separation"] for x in overall_summary])

    print("\n" + "=" * 70)
    print("EVALUATION CONCLUSION")
    print("=" * 70)

    if avg_separation >= 15.0:
        print("Evaluation conclusion:")
        print(f"Relevant categories generally score significantly higher than unrelated categories")
        print(f"(Average separation: +{avg_separation:.2f} percentage points across all 5 job roles).")
    elif avg_separation >= 5.0:
        print("Evaluation conclusion:")
        print("Relevant categories score moderately higher than unrelated categories.")
        print(f"(Average separation: +{avg_separation:.2f} percentage points).")
    else:
        print("Evaluation conclusion:")
        print("The current matcher shows weak separation between relevant and unrelated categories.")
        print("Further NLP improvements may be required.")

    print(f"\nExecution Stats:")
    print(f"  - Total unique PDFs processed: {len(text_cache)}")
    print(f"  - Successful evaluations     : {successful_evaluations}")
    print(f"  - Failed PDFs                : {failed_pdfs}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_evaluation()
import streamlit as st

from nlp import (
    extract_pdf_text,
    calculate_final_score,
    get_verdict,
)


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="SkillSync",
    page_icon="🎯",
    layout="wide",
)


# -----------------------------
# Header
# -----------------------------

st.title("🎯 SkillSync")

st.subheader(
    "NLP-Powered Resume & Job Description Matching Platform"
)

st.write(
    "Upload your resume and paste a job description "
    "to analyze how well your profile matches the role."
)


# -----------------------------
# Input Section
# -----------------------------

col1, col2 = st.columns(2)

with col1:
    resume_file = st.file_uploader(
        "📄 Upload Resume",
        type=["pdf"],
    )

with col2:
    job_description = st.text_area(
        "💼 Paste Job Description",
        height=250,
        placeholder="Paste the complete job description here...",
    )


# -----------------------------
# Analyze Button
# -----------------------------

analyze = st.button(
    "🔍 Analyze Resume",
    type="primary",
    use_container_width=True,
)


if analyze:

    # Validate inputs
    if resume_file is None:
        st.warning("Please upload a resume PDF.")

    elif not job_description.strip():
        st.warning("Please paste a job description.")

    else:

        with st.spinner("Analyzing resume..."):

            # Extract resume text
            resume_text = extract_pdf_text(resume_file)

            if not resume_text.strip():
                st.error(
                    "Could not extract text from this PDF."
                )
                st.stop()

            # Run NLP matching engine
            result = calculate_final_score(
                resume_text,
                job_description,
            )

        # -----------------------------
        # Results
        # -----------------------------

        st.divider()

        st.header("📊 Matching Results")

        score = result["final_score"]
        verdict = get_verdict(score)

        # Score + Verdict
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Match Score",
                f"{score:.2f}%",
            )

        with col2:
            st.metric(
                "Verdict",
                verdict,
            )

        # -----------------------------
        # Skills
        # -----------------------------

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("✅ Matched Skills")

            matched = result["matched_skills"]

            if matched:
                for skill in matched:
                    st.success(skill)

            else:
                st.info("No matching skills detected.")

        with col2:

            st.subheader("❌ Missing Skills")

            missing = result["missing_skills"]

            if missing:
                for skill in missing:
                    st.error(skill)

            else:
                st.success(
                    "No missing skills detected!"
                )

        # -----------------------------
        # Detailed Scores
        # -----------------------------

        st.divider()

        st.subheader("📈 Score Breakdown")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "TF-IDF Score",
                f"{result['tfidf_score']:.2f}%",
            )

        with col2:

            st.metric(
                "Skill Match Score",
                f"{result['skill_score']:.2f}%",
            )

        # -----------------------------
        # Resume Skills
        # -----------------------------

        with st.expander("📋 View Detected Resume Skills"):

            resume_skills = result["resume_skills"]

            if resume_skills:
                st.write(", ".join(resume_skills))
            else:
                st.info(
                    "No known skills detected in the resume."
                )
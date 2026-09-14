"""
SkillSync — SaaS-style Streamlit Frontend
Light/Clean AI SaaS Theme matching reference design.

All NLP logic remains untouched in nlp.py.
"""

import html
import streamlit as st
from nlp import extract_pdf_text, calculate_final_score, get_verdict


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SkillSync - Resume Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# GLOBAL LIGHT SAAS STYLING (MATCHING REFERENCE DESIGN)
# ============================================================

st.markdown(
    """
    <style>
    /* ---------- Root & Canvas ---------- */
    :root {
        --bg: #f8fafc;
        --card-bg: #ffffff;
        --border: #e2e8f0;
        --border-hover: #cbd5e1;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --text-muted: #94a3b8;
        --primary: #6366f1;
        --primary-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        --accent-glow: radial-gradient(ellipse 80% 50% at 50% -20%, rgba(99, 102, 241, 0.14), rgba(168, 85, 247, 0.05), transparent);
        --success-bg: #dcfce7;
        --success-text: #15803d;
        --danger-bg: #ffe4e6;
        --danger-text: #be123c;
        --radius: 20px;
    }

    /* Hide standard Streamlit chrome */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    section[data-testid="stSidebar"] { display: none !important; }

    .stApp {
        background-color: var(--bg) !important;
        background-image: var(--accent-glow) !important;
        background-attachment: fixed !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }

    .main .block-container {
        max-width: 1060px;
        padding-top: 1.5rem;
        padding-bottom: 4rem;
    }

    /* ---------- Navigation Header ---------- */
    .nav-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.75rem 1.25rem;
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 999px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
        margin-bottom: 2.5rem;
    }

    .nav-brand {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        text-decoration: none;
    }

    .nav-logo {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: var(--primary-gradient);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 1.1rem;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);
    }

    .nav-title {
        font-size: 1.15rem;
        font-weight: 750;
        color: var(--text-primary);
        letter-spacing: -0.02em;
    }

    .nav-btn {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.45rem 0.95rem;
        border: 1px solid var(--border);
        border-radius: 999px;
        background: #ffffff;
        color: var(--text-primary);
        font-size: 0.85rem;
        font-weight: 600;
        text-decoration: none;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        transition: all 0.15s ease;
    }

    .nav-btn:hover {
        border-color: var(--border-hover);
        background: #f8fafc;
    }

    /* ---------- Hero Banner ---------- */
    .hero-section {
        text-align: center;
        margin-bottom: 2.2rem;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.35rem 0.85rem;
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--text-secondary);
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
        margin-bottom: 1.2rem;
    }

    .hero-badge-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #22c55e;
        display: inline-block;
    }

    .hero-title {
        font-size: clamp(2.3rem, 4.5vw, 3.4rem);
        font-weight: 800;
        color: var(--text-primary);
        line-height: 1.1;
        letter-spacing: -0.035em;
        margin: 0 0 1rem 0;
    }

    .hero-title .gradient-text {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        max-width: 680px;
        margin: 0 auto;
        font-size: 1.02rem;
        color: var(--text-secondary);
        line-height: 1.6;
        font-weight: 450;
    }

    /* ---------- Input Container Card ---------- */
    .input-wrapper {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.75rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.03), 0 1px 3px rgba(0, 0, 0, 0.02);
        margin-bottom: 2rem;
    }

    .samples-bar {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-bottom: 1.2rem;
    }

    .samples-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--text-secondary);
        margin-right: 0.2rem;
    }

    .sample-pill {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 500;
        color: var(--text-secondary);
    }

    /* ---------- OVERRIDE STREAMLIT FILE UPLOADER ---------- */
    div[data-testid="stFileUploader"],
    section[data-testid="stFileUploaderDropzone"],
    div[data-testid="stFileUploader"] section {
        background-color: #ffffff !important;
        background: #ffffff !important;
        border: 1.5px dashed #cbd5e1 !important;
        border-radius: 16px !important;
        color: #0f172a !important;
        padding: 1rem !important;
    }

    section[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #6366f1 !important;
        background-color: #f8fafc !important;
    }

    section[data-testid="stFileUploaderDropzone"] *,
    div[data-testid="stFileUploaderDropzoneInstructions"] *,
    div[data-testid="stFileUploader"] small,
    div[data-testid="stFileUploader"] span,
    div[data-testid="stFileUploader"] p,
    div[data-testid="stFileUploader"] label {
        color: #475569 !important;
    }

    div[data-testid="stFileUploader"] button,
    button[kind="secondary"],
    button[data-testid="stBaseButton-secondary"] {
        background-color: #ffffff !important;
        background: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    }

    div[data-testid="stFileUploader"] button:hover,
    button[kind="secondary"]:hover,
    button[data-testid="stBaseButton-secondary"]:hover {
        background-color: #f1f5f9 !important;
        border-color: #94a3b8 !important;
        color: #0f172a !important;
    }

    /* ---------- OVERRIDE STREAMLIT TEXTAREA ---------- */
    div[data-testid="stTextArea"] textarea {
        background-color: #ffffff !important;
        background: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 14px !important;
        color: #0f172a !important;
        font-size: 0.92rem !important;
        line-height: 1.5 !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3.5px rgba(99, 102, 241, 0.15) !important;
    }

    /* ---------- OVERRIDE STREAMLIT PRIMARY BUTTON ---------- */
    div.stButton > button,
    button[kind="primary"],
    button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        border: none !important;
        border-radius: 12px !important;
        min-height: 52px !important;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.35) !important;
    }

    div.stButton > button *,
    button[kind="primary"] *,
    button[data-testid="stBaseButton-primary"] * {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    div.stButton > button:hover,
    button[kind="primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        box-shadow: 0 6px 24px rgba(99, 102, 241, 0.48) !important;
        color: #ffffff !important;
        transform: translateY(-1px);
    }

    /* ---------- Empty State (No Analysis Yet) ---------- */
    .empty-card {
        background: #ffffff;
        border: 1px dashed #cbd5e1;
        border-radius: var(--radius);
        padding: 3rem 2rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.02);
        margin-bottom: 2.5rem;
    }

    .empty-icon-wrap {
        width: 56px;
        height: 56px;
        border-radius: 50%;
        background: #f3e8ff;
        color: #9333ea;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        margin: 0 auto 1.2rem;
    }

    .empty-title {
        font-size: 1.3rem;
        font-weight: 750;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }

    .empty-subtitle {
        max-width: 540px;
        margin: 0 auto 1.4rem;
        font-size: 0.92rem;
        color: var(--text-secondary);
        line-height: 1.55;
    }

    .empty-tags {
        display: flex;
        justify-content: center;
        gap: 0.6rem;
        flex-wrap: wrap;
    }

    .empty-tag {
        padding: 0.35rem 0.8rem;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 999px;
        font-size: 0.78rem;
        color: var(--text-secondary);
        font-weight: 500;
    }

    /* ---------- Feature Cards Row ---------- */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.2rem;
        margin-bottom: 2.5rem;
    }

    .feature-card {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.6rem 1.4rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.02);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.04);
    }

    .feature-icon {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        margin-bottom: 1.1rem;
    }

    .feature-icon.purple { background: #f3e8ff; color: #9333ea; }
    .feature-icon.blue { background: #e0f2fe; color: #0284c7; }
    .feature-icon.green { background: #dcfce7; color: #16a34a; }

    .feature-title {
        font-size: 1.05rem;
        font-weight: 750;
        color: var(--text-primary);
        margin-bottom: 0.4rem;
    }

    .feature-desc {
        font-size: 0.85rem;
        color: var(--text-secondary);
        line-height: 1.55;
        margin: 0;
    }

    /* ---------- Results Styling ---------- */
    .results-card {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.75rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.03);
        margin-bottom: 1.5rem;
    }

    .score-banner {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.5rem;
        padding: 1.5rem;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        margin-bottom: 1.5rem;
    }

    .score-big {
        font-size: 3rem;
        font-weight: 800;
        color: var(--primary);
        line-height: 1;
        letter-spacing: -0.04em;
    }

    .verdict-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    .verdict-strong { background: #dcfce7; color: #15803d; }
    .verdict-moderate { background: #fef3c7; color: #b45309; }
    .verdict-fair { background: #ffedd5; color: #c2410c; }
    .verdict-low { background: #ffe4e6; color: #be123c; }

    .progress-track-light {
        height: 8px;
        width: 100%;
        background: #e2e8f0;
        border-radius: 999px;
        overflow: hidden;
        margin-top: 0.6rem;
    }

    .progress-fill-gradient {
        height: 100%;
        border-radius: 999px;
        background: var(--primary-gradient);
    }

    .metrics-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.2rem;
        margin-bottom: 1.5rem;
    }

    .metric-box {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.25rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }

    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: var(--text-primary);
        margin: 0.4rem 0;
    }

    .stats-card {
        background: #f8fafc;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem;
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        text-align: center;
        margin-bottom: 1.5rem;
    }

    .stat-num {
        font-size: 1.3rem;
        font-weight: 750;
        color: var(--text-primary);
    }
    .stat-num.success { color: var(--success-text); }
    .stat-num.danger { color: var(--danger-text); }

    .stat-lbl {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .skills-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.2rem;
        margin-bottom: 1.5rem;
    }

    .skill-pill {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 8px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 0.2rem;
    }

    .skill-pill.matched {
        background: var(--success-bg);
        color: var(--success-text);
        border: 1px solid #bbf7d0;
    }

    .skill-pill.missing {
        background: var(--danger-bg);
        color: var(--danger-text);
        border: 1px solid #fecdd3;
    }

    .insight-box {
        background: #f5f3ff;
        border: 1px solid #e0e7ff;
        border-radius: 14px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
    }

    .footer {
        text-align: center;
        padding-top: 2rem;
        color: var(--text-muted);
        font-size: 0.82rem;
    }

    @media (max-width: 768px) {
        .features-grid, .metrics-row, .skills-grid {
            grid-template-columns: 1fr;
        }
        .score-banner {
            flex-direction: column;
            text-align: center;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS & RENDERING FUNCTIONS
# ============================================================

def esc(value):
    """Safely escape dynamic string values."""
    return html.escape(str(value))


def render_navbar():
    st.markdown(
        """
        <div class="nav-bar">
            <a href="#" class="nav-brand">
                <div class="nav-logo">🎯</div>
                <span class="nav-title">SkillSync</span>
            </a>
            <a href="https://github.com" target="_blank" class="nav-btn">
                <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.28.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
                </svg>
                GitHub
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <div class="hero-section">
            <div class="hero-badge">
                <span class="hero-badge-dot"></span>
                AI-powered Resume Intelligence ✨
            </div>
            <h1 class="hero-title">
                Match Your Resume <br>
                <span class="gradient-text">with AI</span>
            </h1>
            <p class="hero-subtitle">
                Analyze any job description and resume in seconds using Machine Learning and NLP.
                Discover skill coverage, match scores, and missing requirements.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_input_section():
    st.markdown(
        """
        <div class="input-wrapper">
            <div class="samples-bar">
                <span class="samples-label">Try a sample:</span>
                <span class="sample-pill">Software Engineer (SDE)</span>
                <span class="sample-pill">Python Backend Developer</span>
                <span class="sample-pill">Data Scientist</span>
                <span class="sample-pill">Fullstack Developer</span>
            </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("<p style='font-size:0.9rem; font-weight:750; color:#0f172a; margin-bottom:0.4rem;'>📄 Upload Resume (PDF)</p>", unsafe_allow_html=True)
        resume_file = st.file_uploader(
            "Upload Resume",
            type=["pdf"],
            help="Upload your PDF resume",
            label_visibility="collapsed",
        )
        if resume_file is not None:
            size_kb = resume_file.size / 1024
            st.markdown(
                f"<div style='margin-top:0.5rem; font-size:0.8rem; color:#15803d; font-weight:600;'>✓ Loaded: {esc(resume_file.name)} ({size_kb:.1f} KB)</div>",
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("<p style='font-size:0.9rem; font-weight:750; color:#0f172a; margin-bottom:0.4rem;'>💼 Paste Job Description</p>", unsafe_allow_html=True)
        job_description = st.text_area(
            "Job Description",
            height=200,
            placeholder="Paste a job description here...",
            label_visibility="collapsed",
        )
        char_count = len(job_description)
        st.markdown(
            f"<div style='text-align:right; font-size:0.75rem; color:#94a3b8; margin-top:0.2rem;'>{char_count} characters</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    clicked = st.button(
        "⚡ Analyze Job Match",
        use_container_width=True,
        help="Run NLP analysis to compare your resume with the job description.",
    )

    st.markdown("</div>", unsafe_allow_html=True)

    return resume_file, job_description, clicked


def render_empty_state():
    st.markdown(
        """
        <div class="empty-card">
            <div class="empty-icon-wrap">🎯</div>
            <div class="empty-title">No Analysis Yet</div>
            <div class="empty-subtitle">
                Upload a resume and paste a job description above to let our AI calculate your match score, TF-IDF similarity, and skill gaps.
            </div>
            <div class="empty-tags">
                <span class="empty-tag">Match score</span>
                <span class="empty-tag">Skill coverage</span>
                <span class="empty-tag">Top missing skills</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_features_grid():
    st.markdown(
        """
        <div class="features-grid">
            <div class="feature-card">
                <div class="feature-icon purple">🎯</div>
                <div class="feature-title">Pattern-based detection</div>
                <p class="feature-desc">
                    Extracts technical skills, qualifications, and industry terms directly from job postings and resumes.
                </p>
            </div>
            <div class="feature-card">
                <div class="feature-icon blue">⚡</div>
                <div class="feature-title">Fast & accurate</div>
                <p class="feature-desc">
                    Powered by optimized TF-IDF vectorization and Cosine Similarity delivering predictions in milliseconds.
                </p>
            </div>
            <div class="feature-card">
                <div class="feature-icon green">📊</div>
                <div class="feature-title">Explainable results</div>
                <p class="feature-desc">
                    Every score includes transparent explanations of required skills vs missing requirements.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_verdict_class(verdict):
    mapping = {
        "Strong Match": "verdict-strong",
        "Moderate Match": "verdict-moderate",
        "Fair Match": "verdict-fair",
        "Low Match": "verdict-low",
    }
    return mapping.get(verdict, "verdict-moderate")


def render_results(result):
    score = result["final_score"]
    verdict = get_verdict(score)
    tfidf_score = result["tfidf_score"]
    skill_score = result["skill_score"]

    matched_skills = result.get("matched_skills", [])
    missing_skills = result.get("missing_skills", [])
    resume_skills = result.get("resume_skills", [])

    required_cnt = len(matched_skills) + len(missing_skills)
    matched_cnt = len(matched_skills)
    missing_cnt = len(missing_skills)

    pct = max(0.0, min(100.0, float(score)))

    # Main score banner
    st.markdown(
        f"""
        <div class="results-card">
            <h3 style="margin:0 0 1rem 0; font-size:1.2rem; font-weight:750; color:#0f172a;">Resume Match Analysis</h3>
            <div class="score-banner">
                <div>
                    <span class="verdict-badge {get_verdict_class(verdict)}">{esc(verdict)}</span>
                    <div style="font-size:0.9rem; color:#64748b;">Overall compatibility score between your resume & role.</div>
                </div>
                <div style="text-align:right;">
                    <div class="score-big">{pct:.2f}%</div>
                    <div style="font-size:0.78rem; color:#94a3b8; font-weight:600; margin-top:0.2rem;">SKILL COVERAGE: {skill_score:.2f}%</div>
                </div>
            </div>
            <div class="progress-track-light">
                <div class="progress-fill-gradient" style="width:{pct:.2f}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metrics row
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div style="font-size:0.85rem; font-weight:700; color:#64748b;">TF-IDF Similarity</div>
                <div class="metric-val" style="color:#6366f1;">{tfidf_score:.2f}%</div>
                <div style="font-size:0.78rem; color:#94a3b8;">Measures textual similarity between resume and job description.</div>
                <div class="progress-track-light" style="margin-top:0.6rem;">
                    <div class="progress-fill-gradient" style="width:{tfidf_score:.2f}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-box">
                <div style="font-size:0.85rem; font-weight:700; color:#64748b;">Skill Match Score</div>
                <div class="metric-val" style="color:#8b5cf6;">{skill_score:.2f}%</div>
                <div style="font-size:0.78rem; color:#94a3b8;">Percentage of required job skills found in your resume.</div>
                <div class="progress-track-light" style="margin-top:0.6rem;">
                    <div class="progress-fill-gradient" style="width:{skill_score:.2f}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Summary Stats Row
    missing_color_cls = "danger" if missing_cnt > 0 else "success"
    st.markdown(
        f"""
        <div class="stats-card">
            <div>
                <div class="stat-lbl">Required Skills</div>
                <div class="stat-num">{required_cnt}</div>
            </div>
            <div>
                <div class="stat-lbl">Matched</div>
                <div class="stat-num success">{matched_cnt}</div>
            </div>
            <div>
                <div class="stat-lbl">Missing Gaps</div>
                <div class="stat-num {missing_color_cls}">{missing_cnt}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Matched vs Missing skills
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown(
            f"""
            <div class="results-card">
                <h4 style="margin:0 0 0.4rem 0; color:#0f172a; font-weight:750;">Matched Skills ({matched_cnt})</h4>
                <p style="font-size:0.8rem; color:#64748b; margin-bottom:0.8rem;">Skills from the job description detected in your resume.</p>
            """,
            unsafe_allow_html=True,
        )
        if matched_skills:
            badges = "".join([f'<span class="skill-pill matched">{esc(s)}</span>' for s in matched_skills])
            st.markdown(f"<div>{badges}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='font-size:0.8rem; color:#94a3b8;'>No matched skills detected.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            f"""
            <div class="results-card">
                <h4 style="margin:0 0 0.4rem 0; color:#0f172a; font-weight:750;">Skill Gaps ({missing_cnt})</h4>
                <p style="font-size:0.8rem; color:#64748b; margin-bottom:0.8rem;">Requirements from the job description missing in your resume.</p>
            """,
            unsafe_allow_html=True,
        )
        if missing_skills:
            badges = "".join([f'<span class="skill-pill missing">{esc(s)}</span>' for s in missing_skills])
            st.markdown(f"<div>{badges}</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                """
                <div style="background:#dcfce7; border:1px solid #bbf7d0; border-radius:10px; padding:0.8rem; text-align:center;">
                    <span style="color:#15803d; font-weight:700; font-size:0.85rem;">✓ No skill gaps detected!</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # Insight box
    explanations = {
        "Strong Match": "Your resume demonstrates strong alignment with the technical requirements of this role. Most of the required skills were detected.",
        "Moderate Match": "Your resume matches several key requirements, but there are a few skill gaps worth addressing before applying.",
        "Fair Match": "Your resume has partial alignment with this role. Strengthening missing skills will boost your match.",
        "Low Match": "This role has limited overlap with the detected skills in your resume.",
    }
    insight_text = explanations.get(verdict, "")

    st.markdown(
        f"""
        <div class="insight-box">
            <h4 style="margin:0 0 0.3rem 0; color:#4338ca; font-weight:750; font-size:0.95rem;">💡 Match Insights & Recommendation</h4>
            <p style="margin:0; font-size:0.85rem; color:#3730a3; line-height:1.5;">{esc(insight_text)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Detected skills expander
    with st.expander("View all detected skills in resume"):
        if resume_skills:
            badges = "".join([f'<span class="sample-pill" style="margin:0.2rem;">{esc(s)}</span>' for s in resume_skills])
            st.markdown(f"<div>{badges}</div>", unsafe_allow_html=True)
        else:
            st.info("No known skills were detected in the resume text.")


def render_footer():
    st.markdown(
        """
        <div class="footer">
            <p style="margin:0;">SkillSync · AI-Powered Resume Intelligence Platform</p>
            <p style="margin-top:0.3rem; font-size:0.75rem; color:#cbd5e1;">Built for smarter, data-driven job applications.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN APPLICATION FLOW
# ============================================================

render_navbar()
render_hero()

resume_file, job_description, analyze_clicked = render_input_section()

if analyze_clicked:
    if resume_file is None:
        st.warning("Please upload a PDF resume.")
    elif not job_description.strip():
        st.warning("Please paste a job description.")
    else:
        with st.spinner("Analyzing your resume against the job description..."):
            resume_text = extract_pdf_text(resume_file)

            if not resume_text.strip():
                st.error("Could not extract text from this PDF file.")
                st.stop()

            # Execute NLP scoring (unmodified backend logic)
            result = calculate_final_score(resume_text, job_description)

        st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
        render_results(result)
        render_features_grid()
        render_footer()
else:
    render_empty_state()
    render_features_grid()
    render_footer()

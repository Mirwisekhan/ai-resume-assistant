import os
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from pypdf import PdfReader
from docx import Document

load_dotenv()

st.set_page_config(
    page_title="Resume ATS Analyzer",
    page_icon="📄",
    layout="wide",
)

# -----------------------------
# Configuration
# -----------------------------
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

ATS_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "your", "you",
    "are", "was", "were", "will", "have", "has", "had", "our", "their",
    "they", "them", "his", "her", "its", "into", "about", "over", "under",
    "using", "use", "used", "job", "role", "work", "working", "years",
    "year", "team", "skills", "skill", "experience", "required", "preferred",
    "candidate", "position", "responsibilities", "responsibility",
}

SECTION_ALIASES = {
    "summary": ["summary", "professional summary", "profile", "objective"],
    "experience": ["experience", "work experience", "professional experience", "employment"],
    "education": ["education", "academic background", "qualifications"],
    "skills": ["skills", "technical skills", "core skills", "competencies"],
    "projects": ["projects", "personal projects", "academic projects"],
    "certifications": ["certifications", "certificates", "licenses"],
    "contact": ["contact", "contact information"],
}


# -----------------------------
# Helpers
# -----------------------------
def get_api_key() -> str:
    """Read Gemini API key from Streamlit secrets or environment."""
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secret_key = ""

    return secret_key or os.getenv("GEMINI_API_KEY", "")


def extract_pdf_text(file) -> str:
    reader = PdfReader(file)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def extract_docx_text(file) -> str:
    document = Document(file)
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]

    # Include text from tables, because some resumes use tables for content.
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                paragraphs.append(" | ".join(cells))

    return "\n".join(paragraphs).strip()


def extract_resume_text(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".pdf":
        return extract_pdf_text(uploaded_file)
    if suffix == ".docx":
        return extract_docx_text(uploaded_file)
    if suffix == ".txt":
        return uploaded_file.getvalue().decode("utf-8", errors="ignore").strip()

    raise ValueError("Unsupported file type. Please upload PDF, DOCX, or TXT.")


def normalize_words(text: str) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9+#.\-/]{1,}", text.lower())
    return words


def keyword_candidates(job_description: str) -> List[str]:
    """Extract useful single-word and common technical terms from a JD."""
    if not job_description.strip():
        return []

    words = normalize_words(job_description)
    counts = {}
    for word in words:
        if word in ATS_STOPWORDS or len(word) < 3:
            continue
        counts[word] = counts.get(word, 0) + 1

    # Preserve technical terms such as c++, c#, node.js, sql, etc.
    ranked = sorted(counts.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    return [word for word, _ in ranked[:60]]


def local_ats_checks(resume_text: str, job_description: str) -> Dict[str, Any]:
    text_lower = resume_text.lower()

    checks = []

    contact_patterns = {
        "email": r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "phone": r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)",
        "linkedin": r"linkedin\.com",
    }

    for name, pattern in contact_patterns.items():
        checks.append({
            "check": f"Contact: {name.title()}",
            "passed": bool(re.search(pattern, resume_text, re.I)),
            "impact": "medium",
        })

    for section, aliases in SECTION_ALIASES.items():
        passed = any(alias in text_lower for alias in aliases)
        checks.append({
            "check": f"Section: {section.title()}",
            "passed": passed,
            "impact": "medium" if section in {"experience", "skills", "education"} else "low",
        })

    checks.extend([
        {
            "check": "Resume is not extremely short",
            "passed": len(resume_text.split()) >= 180,
            "impact": "medium",
        },
        {
            "check": "Contains measurable results/numbers",
            "passed": bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:%|percent|k|m|million|thousand|users|customers|projects|years)\b", resume_text, re.I)),
            "impact": "medium",
        },
        {
            "check": "Contains action-oriented language",
            "passed": bool(re.search(
                r"\b(led|built|created|developed|implemented|improved|increased|reduced|"
                r"designed|managed|analyzed|automated|delivered|launched|optimized)\b",
                resume_text,
                re.I,
            )),
            "impact": "medium",
        },
    ])

    jd_keywords = keyword_candidates(job_description)
    resume_words = set(normalize_words(resume_text))
    matched = [kw for kw in jd_keywords if kw.lower() in resume_words]
    missing = [kw for kw in jd_keywords if kw.lower() not in resume_words]

    keyword_match = (len(matched) / len(jd_keywords) * 100) if jd_keywords else None

    return {
        "checks": checks,
        "keyword_match": keyword_match,
        "matched_keywords": matched,
        "missing_keywords": missing,
        "word_count": len(resume_text.split()),
    }


def configure_gemini() -> None:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "Gemini API key not found. Add GEMINI_API_KEY to Streamlit Secrets "
            "or your environment variables."
        )
    genai.configure(api_key=api_key)


def extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON even if Gemini wraps it in markdown fences."""
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise ValueError("Gemini did not return valid JSON.")
        return json.loads(match.group(0))


def analyze_with_gemini(
    resume_text: str,
    job_description: str,
    local_checks: Dict[str, Any],
) -> Dict[str, Any]:
    configure_gemini()

    model = genai.GenerativeModel(DEFAULT_MODEL)

    prompt = f"""
You are an expert resume reviewer and ATS optimization specialist.

Analyze the resume below against the job description.

IMPORTANT RULES:
1. Return ONLY valid JSON. No markdown and no explanation outside JSON.
2. The ATS score is an ESTIMATE, not a guarantee. ATS systems differ.
3. Never invent experience, education, skills, metrics, employers, projects, or achievements.
4. If a keyword is missing from the resume, recommend it only when it genuinely appears relevant
   to the job description. Do not tell the applicant to add a skill they do not have.
5. Separate formatting/parser issues from content/keyword issues.
6. Resume bullet rewrites must preserve the candidate's original facts. If a metric is unavailable,
   use [ADD METRIC] rather than inventing a number.

Return exactly this JSON structure:
{{
  "ats_score": 0,
  "score_breakdown": {{
    "keyword_match": 0,
    "format_parseability": 0,
    "experience_relevance": 0,
    "skills_alignment": 0,
    "section_completeness": 0
  }},
  "overall_verdict": "string",
  "top_strengths": ["string"],
  "priority_improvements": [
    {{
      "priority": "High|Medium|Low",
      "issue": "string",
      "recommendation": "string"
    }}
  ],
  "matched_keywords": ["string"],
  "missing_or_weak_keywords": ["string"],
  "section_feedback": [
    {{
      "section": "string",
      "status": "Strong|Needs improvement|Missing",
      "feedback": "string"
    }}
  ],
  "bullet_rewrites": [
    {{
      "original": "exact or close quote from resume",
      "improved": "fact-preserving improved bullet",
      "reason": "string"
    }}
  ],
  "formatting_checks": [
    {{
      "issue": "string",
      "severity": "High|Medium|Low",
      "fix": "string"
    }}
  ],
  "next_steps": ["string"]
}}

LOCAL PRE-CHECK DATA:
{json.dumps(local_checks, indent=2)}

JOB DESCRIPTION:
{job_description if job_description.strip() else "No job description provided. Evaluate general ATS readiness."}

RESUME:
{resume_text}
"""

    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
        },
    )

    return extract_json(response.text)


def clamp_score(value: Any) -> int:
    try:
        return max(0, min(100, int(float(value))))
    except (TypeError, ValueError):
        return 0


def display_results(result: Dict[str, Any], local: Dict[str, Any]) -> None:
    score = clamp_score(result.get("ats_score", 0))

    st.subheader("📊 ATS Score")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Estimated ATS Score", f"{score}/100")
    with col2:
        if local["keyword_match"] is not None:
            st.metric("Local Keyword Match", f"{local['keyword_match']:.0f}%")
        else:
            st.metric("Resume Words", f"{local['word_count']:,}")
    with col3:
        passed = sum(1 for item in local["checks"] if item["passed"])
        st.metric("Local Checks Passed", f"{passed}/{len(local['checks'])}")

    st.progress(score / 100)

    verdict = result.get("overall_verdict", "")
    if verdict:
        st.info(verdict)

    breakdown = result.get("score_breakdown", {})
    if breakdown:
        st.subheader("Score Breakdown")
        cols = st.columns(min(5, len(breakdown)))
        for idx, (key, value) in enumerate(breakdown.items()):
            with cols[idx % len(cols)]:
                st.metric(key.replace("_", " ").title(), f"{clamp_score(value)}/100")

    strengths = result.get("top_strengths", [])
    if strengths:
        st.subheader("✅ Strengths")
        for item in strengths:
            st.markdown(f"- {item}")

    improvements = result.get("priority_improvements", [])
    if improvements:
        st.subheader("⚠️ Priority Improvements")
        for item in improvements:
            priority = item.get("priority", "Medium")
            st.markdown(
                f"**{priority}: {item.get('issue', 'Issue')}**  \n"
                f"{item.get('recommendation', '')}"
            )

    matched = result.get("matched_keywords", [])
    missing = result.get("missing_or_weak_keywords", [])

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🔑 Matched Keywords")
        if matched:
            st.write(", ".join(matched))
        else:
            st.write("No strong matches identified.")

    with c2:
        st.subheader("🔎 Missing / Weak Keywords")
        if missing:
            st.write(", ".join(missing))
        else:
            st.write("No major missing keywords identified.")

    section_feedback = result.get("section_feedback", [])
    if section_feedback:
        st.subheader("📑 Section Feedback")
        for item in section_feedback:
            st.markdown(
                f"**{item.get('section', 'Section')} — {item.get('status', '')}**  \n"
                f"{item.get('feedback', '')}"
            )

    rewrites = result.get("bullet_rewrites", [])
    if rewrites:
        st.subheader("✍️ Bullet Rewrite Suggestions")
        for idx, item in enumerate(rewrites, 1):
            with st.expander(f"Suggestion {idx}"):
                st.markdown("**Original**")
                st.write(item.get("original", ""))
                st.markdown("**Improved**")
                st.write(item.get("improved", ""))
                st.caption(item.get("reason", ""))

    formatting = result.get("formatting_checks", [])
    if formatting:
        st.subheader("🧩 Formatting / ATS Checks")
        for item in formatting:
            st.markdown(
                f"**{item.get('severity', 'Medium')}: {item.get('issue', '')}**  \n"
                f"{item.get('fix', '')}"
            )

    local_failed = [item for item in local["checks"] if not item["passed"]]
    if local_failed:
        st.subheader("🛠️ Local Parser Checks")
        for item in local_failed:
            st.warning(f"{item['check']} — review this area.")

    next_steps = result.get("next_steps", [])
    if next_steps:
        st.subheader("🚀 Recommended Next Steps")
        for step in next_steps:
            st.markdown(f"- {step}")


# -----------------------------
# UI
# -----------------------------
st.title("📄 Resume ATS Analyzer")
st.write(
    "Upload a resume and optionally paste a job description. "
    "The app estimates ATS readiness and gives practical improvement suggestions."
)

with st.sidebar:
    st.header("Settings")
    st.caption(f"Gemini model: `{DEFAULT_MODEL}`")
    st.markdown(
        """
**Supported files**
- PDF
- DOCX
- TXT

**Privacy**
Your resume is sent to Gemini for analysis when you click **Analyze Resume**.
Do not upload documents containing information you do not want processed by a third-party AI service.
"""
    )

uploaded_file = st.file_uploader(
    "Upload your resume",
    type=["pdf", "docx", "txt"],
    help="PDF, DOCX, and TXT are supported.",
)

job_description = st.text_area(
    "Paste the Job Description (recommended)",
    height=260,
    placeholder="Paste the complete job description here...",
)

if uploaded_file:
    st.caption(f"Selected file: {uploaded_file.name}")

analyze = st.button("🔍 Analyze Resume", type="primary", use_container_width=True)

if analyze:
    if not uploaded_file:
        st.error("Please upload a resume first.")
        st.stop()

    with st.spinner("Extracting and analyzing your resume..."):
        try:
            resume_text = extract_resume_text(uploaded_file)

            if not resume_text:
                st.error(
                    "No readable text was found. If your PDF is scanned/image-only, "
                    "please use an OCR-enabled PDF or DOCX/TXT version."
                )
                st.stop()

            if len(resume_text.split()) < 50:
                st.warning(
                    "Very little text was extracted. The file may be image-based or poorly formatted."
                )

            local = local_ats_checks(resume_text, job_description)
            result = analyze_with_gemini(resume_text, job_description, local)

            st.session_state["analysis_result"] = result
            st.session_state["local_checks"] = local
            st.session_state["resume_text"] = resume_text

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.info(
                "Check that your Gemini API key is configured and that the uploaded "
                "resume is a readable PDF, DOCX, or TXT file."
            )

if "analysis_result" in st.session_state:
    st.divider()
    display_results(
        st.session_state["analysis_result"],
        st.session_state["local_checks"],
    )

    with st.expander("View extracted resume text"):
        st.text(st.session_state["resume_text"])

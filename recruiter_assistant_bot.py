import streamlit as st
import time
import fitz  # PyMuPDF
import docx
import pandas as pd
import re
import io
import spacy
import google.generativeai as genai

# ---------- Page ----------
st.set_page_config(page_title="Resume Matcher (Gemini)", layout="centered")
st.title("📌 Terrabit Consulting Talent Match System")
st.write("Upload a JD and multiple resumes. Get match scores, red flags, and follow-up messaging.")

# ---------- NLP ----------
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# ---------- Gemini (free API) ----------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

gemini_model = genai.GenerativeModel(
    "gemini-1.5-flash",  # fast & free; switch to "gemini-1.5-pro" if you need higher quality
    generation_config={
        "temperature": 0,
        "top_p": 0.1,
        "top_k": 1,
        "max_output_tokens": 1024
    }
)

def call_gemini(prompt: str) -> str:
    """Single LLM caller (Gemini). Returns clean text only."""
    try:
        resp = gemini_model.generate_content(prompt)
        text = (getattr(resp, "text", "") or "").strip()
        # Remove accidental code fences if present
        if text.startswith("```"):
            text = text.strip("`")
            lines = text.splitlines()
            if lines and lines[0].lower().startswith(("markdown", "txt", "json")):
                lines = lines[1:]
            text = "\n".join(lines).strip()
        return text
    except Exception as e:
        st.error(f"❌ Gemini API failed: {str(e)}")
        return "⚠️ Gemini processing failed."

# ---------- File reading ----------
def read_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def read_docx(file):
    doc = docx.Document(file)
    full_text = []

    for para in doc.paragraphs:
        full_text.append(para.text)

    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                full_text.append(cell.text)

    try:
        section = doc.sections[0]
        for para in section.footer.paragraphs:
            full_text.append(para.text)
    except Exception:
        pass

    return "\n".join(full_text)

def read_file(file):
    if file.type == "application/pdf":
        return read_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return read_docx(file)
    else:
        return file.read().decode("utf-8", errors="ignore")

# ---------- Extraction helpers ----------
def extract_email(text):
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match.group() if match else "Not found"

def extract_candidate_name_from_table(text):
    matches = re.findall(r"(?i)Candidate Name\s*[\t:–-]*\s*(.+)", text)
    for match in matches:
        name = match.strip().title()
        if 2 <= len(name.split()) <= 4:
            return name
    return None

def extract_candidate_name_from_footer(text):
    footer_match = re.search(r"(?i)Resume of\s+([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", text)
    if footer_match:
        return footer_match.group(1).strip().title()
    return None

def improved_extract_candidate_name(text, filename):
    try:
        trimmed_text = "\n".join(text.splitlines()[:50])
        prompt = f"""
You are a resume parser. Extract ONLY the candidate's full name.

Rules:
- Return ONLY the name text (no labels, no quotes, no punctuation).
- 2 to 4 words, each starting with a capital letter (e.g., "Amit Kumar", "Neha Reddy Varma").
- Prefer explicit cues: "Candidate Name", "Resume of <Name>", headers/footers.
- Ignore job titles, skills, technologies, locations, emails, phone numbers, company names.
- If no valid name, return exactly: Name Not Found

Text to analyze (first 50 lines):
{trimmed_text}

Output:
<name only or "Name Not Found">
"""
        name = call_gemini(prompt)
        suspicious_keywords = ["java", "python", "developer", "resume", "engineer", "servers"]
        if (
            not name or
            len(name.split()) > 5 or
            any(word in name.lower() for word in suspicious_keywords) or
            "@" in name or
            name.lower().startswith("name not found")
        ):
            return "Name Not Found"
        return name.strip().title()
    except Exception:
        return "Name Not Found"

def extract_candidate_name(text, filename):
    table_name = extract_candidate_name_from_table(text)
    if table_name:
        return table_name
    footer_name = extract_candidate_name_from_footer(text)
    if footer_name:
        return footer_name
    return improved_extract_candidate_name(text, filename)

# ---------- LLM tasks ----------
def compare_resume(jd_text, resume_text, candidate_name):
    prompt = f"""
You are a recruiter assistant. Compare the resume to the JD and produce STRICT Markdown in the EXACT format below.

Formatting rules (MUST FOLLOW):
- Use these headings and nothing else.
- Score must be an INTEGER 0–100.
- Keep bullets concise. No tables. No extra sections.

---BEGIN TEMPLATE---
**Name**: {candidate_name}
**Score**: [NN]%

**Reason**:
- Role Match: <one sentence>
- Skill Match: <matched & missing skills in 1–2 bullets>
- Major Gaps: <key gaps in 1–2 bullets>

Warning: <ONLY include this single line if Score < 70%; otherwise omit>
---END TEMPLATE---

Job Description:
{jd_text}

Resume:
{resume_text}
"""
    return call_gemini(prompt)

def generate_followup(jd_text, resume_text, candidate_name):
    prompt = f"""
You are a recruiter at Terrabit Consulting writing to a candidate named {candidate_name}.
All outputs must be addressed to the candidate (never to a recruiter) and must be short and actionable.

Goals:
- Invite {candidate_name} to a quick screening call for the best-fit role inferred from the JD.
- Personalize with 1–2 strengths found in the resume.
- Ask for availability and provide 2–3 time-slot options.
- Ask only for info that is missing/uncertain in the resume.
- Keep messages concise (4–6 lines each).

Strict format (use these exact section headings, no extras):

---BEGIN OUTPUT---
### WhatsApp Message to Candidate
<friendly, concise outreach to {candidate_name}; mention inferred role; 2–3 time slots; ask to confirm phone/email if missing>

### Email to Candidate
Subject: Quick chat about a {{<role>}} opportunity at Terrabit Consulting
Dear {candidate_name},
<3–5 lines: why we’re reaching out based on their resume, key fit, ask for 2–3 preferred slots this week, and a polite close with signature placeholder>

### Screening Questions (Tailored)
- <Q1 based on core skills in the JD; confirm years/level if unclear from resume>
- <Q2 about a missing or weak skill vs JD>
- <Q3 about domain/tools mentioned in JD>
- <Q4 optional about availability/notice period>
- <Q5 optional about location/remote preference if relevant>
---END OUTPUT---

Information to use:
Job Description:
{jd_text}

Resume:
{resume_text}

Rules:
- Do NOT say "my resume is attached" or write as if you are the candidate.
- Keep tone professional and positive.
"""
    return call_gemini(prompt)

# ---------- Session state ----------
if "results" not in st.session_state:
    st.session_state["results"] = []
if "processed_resumes" not in st.session_state:
    st.session_state["processed_resumes"] = set()
if "jd_text" not in st.session_state:
    st.session_state["jd_text"] = ""
if "jd_file" not in st.session_state:
    st.session_state["jd_file"] = None
if "summary" not in st.session_state:
    st.session_state["summary"] = []

# Reset
if st.button("🔁 Start New Matching Session"):
    st.session_state.clear()
    st.rerun()

# Uploaders
jd_file = st.file_uploader("📄 Upload Job Description", type=["txt", "pdf", "docx"], key="jd_uploader")
resume_files = st.file_uploader("📑 Upload Candidate Resumes", type=["txt", "pdf", "docx"], accept_multiple_files=True, key="resume_uploader")

# ✅ Store JD once
if jd_file and not st.session_state.get("jd_text"):
    jd_text = read_file(jd_file)
    st.session_state["jd_text"] = jd_text
    st.session_state["jd_file"] = jd_file.name

jd_text = st.session_state.get("jd_text", "")

# Run
if st.button("🚀 Run Matching") and jd_text and resume_files:
    for resume_file in resume_files:
        if resume_file.name in st.session_state["processed_resumes"]:
            continue

        resume_text = read_file(resume_file)
        correct_name = extract_candidate_name(resume_text, resume_file.name)
        correct_email = extract_email(resume_text)

        with st.spinner(f"🔎 Analyzing {correct_name}..."):
            result = compare_resume(jd_text, resume_text, correct_name)

        # Robust score extraction
        score_match = re.search(r"\*\*Score\*\*:\s*\[(\d{1,3})\]%", result)
        if not score_match:
            score_match = re.search(r"\*\*Score\*\*:\s*(\d{1,3})%", result)
        score = int(score_match.group(1)) if score_match else 0

        st.session_state["results"].append({
            "correct_name": correct_name,
            "email": correct_email,
            "score": score,
            "result": result,
            "resume_text": resume_text
        })
        st.session_state["processed_resumes"].add(resume_file.name)

        st.session_state["summary"].append({
            "Candidate Name": correct_name,
            "Email": correct_email,
            "Score": score
        })

# Results
for entry in st.session_state["results"]:
    st.markdown("---")
    st.subheader(f"📌 {entry['correct_name']}")
    st.markdown(f"📧 **Email**: {entry['email']}")
    st.markdown(entry["result"], unsafe_allow_html=True)

    score = entry["score"]
    if score < 50:
        st.error("❌ Not suitable – Major role mismatch")
    elif score < 70:
        st.warning("⚠️ Consider with caution – Lacks core skills")
    else:
        st.success("✅ Strong match – Good alignment with JD")

    if st.button(f"✉️ Generate Follow-up for {entry['correct_name']}", key=f"followup_{entry['correct_name']}"):
        with st.spinner("Generating messages..."):
            followup = generate_followup(jd_text, entry["resume_text"], entry["correct_name"])
            st.markdown("---")
            st.markdown(followup, unsafe_allow_html=True)

# Summary + download
if st.session_state["summary"]:
    st.markdown("### 📊 Summary of All Candidates")
    df_summary = pd.DataFrame(st.session_state["summary"]).sort_values(by="Score", ascending=False)
    st.dataframe(df_summary)

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df_summary.to_excel(writer, index=False)

    st.download_button(
        label="📥 Download Summary as Excel",
        data=excel_buffer.getvalue(),
        file_name="resume_match_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# resume_matcher_gemini_gpt_style_strict.py
import streamlit as st
import fitz  # PyMuPDF
import docx
import pandas as pd
import re, io
import google.generativeai as genai

# -------------------- Page --------------------
st.set_page_config(page_title="Resume Matcher (Gemini, GPT-style)", layout="centered")
st.title("📌 Terrabit Consulting Talent Match System")
st.write("Upload a JD and multiple resumes. Get match scores, red flags, and follow-up messaging.")

# -------------------- Gemini --------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Please add GEMINI_API_KEY to Streamlit secrets.")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
MODEL_NAME = st.secrets.get("GEMINI_MODEL", "gemini-1.5-flash")

gemini = genai.GenerativeModel(
    MODEL_NAME,
    generation_config={
        "temperature": 0.0,   # deterministic
        "top_p": 0.1,
        "top_k": 1,
        "max_output_tokens": 1400
    }
)

def call_gemini(prompt: str) -> str:
    """LLM caller with light code-fence cleanup."""
    try:
        resp = gemini.generate_content(prompt)
        text = (getattr(resp, "text", "") or "").strip()
        # Strip accidental code fences
        if text.startswith("```"):
            text = text.strip("`")
            lines = text.splitlines()
            # drop "markdown"/"txt" language hint line if present
            if lines and lines[0].lower().startswith(("markdown", "txt", "json")):
                lines = lines[1:]
            text = "\n".join(lines).strip()
        return text
    except Exception as e:
        st.error(f"❌ Gemini failed: {e}")
        return "⚠️ Gemini processing failed."

# -------------------- File reading --------------------
def read_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text")
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

# -------------------- Small helpers --------------------
def extract_email(text):
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return m.group() if m else "Not found"

def extract_candidate_name_from_table(text):
    matches = re.findall(r"(?i)Candidate Name\s*[\t:–-]*\s*(.+)", text)
    for match in matches:
        name = match.strip().title()
        if 2 <= len(name.split()) <= 4:
            return name
    return None

def extract_candidate_name_from_footer(text):
    m = re.search(r"(?i)Resume of\s+([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", text)
    if m:
        return m.group(1).strip().title()
    return None

def improved_extract_candidate_name(text, filename):
    try:
        trimmed = "\n".join(text.splitlines()[:50])
        prompt = f"""
You are a resume parser. Return ONLY the candidate's full name (2–4 words, capitalized).
If no valid name, return exactly: Name Not Found

Text (first 50 lines):
{trimmed}

Output:
"""
        name = call_gemini(prompt).strip()
        bad = ["java","python","developer","resume","engineer","servers","manager","tester"]
        if (not name) or len(name.split()) > 5 or any(w in name.lower() for w in bad) or "@" in name.lower():
            return "Name Not Found"
        return name.title()
    except Exception:
        return "Name Not Found"

def extract_candidate_name(text, filename):
    return (
        extract_candidate_name_from_table(text)
        or extract_candidate_name_from_footer(text)
        or improved_extract_candidate_name(text, filename)
    )

def enforce_markdown(md: str) -> str:
    """Ensure separate Name/Score lines and normalize score token."""
    if not md:
        return md
    # put Score on its own line if model merged it
    md = re.sub(r"(\*\*Name\*\*:[^\n]*?)\s+(?=\*\*Score\*\*:)", r"\1\n", md)
    # normalize score formats to **Score**: [NN]%
    md = re.sub(r"\*\*Score\*\*:\s*\[(\d{1,3})\]\s*%", r"**Score**: [\1]%", md)
    md = re.sub(r"\*\*Score\*\*:\s*(\d{1,3})\s*%", r"**Score**: [\1]%", md)
    return md.strip()

def extract_score(md: str) -> int:
    for pat in [
        r"\*\*Score\*\*:\s*\[(\d{1,3})\]\s*%",
        r"\*\*Score\*\*:\s*(\d{1,3})\s*%",
        r"Score\*\*:\s*(\d{1,3})\s*%"
    ]:
        m = re.search(pat, md)
        if m:
            try:
                return int(m.group(1))
            except:
                pass
    return 0

# -------------------- LLM tasks (GPT-style prompts) --------------------
RUBRIC = """
SCORING & CLASSIFICATION (be consistent, 0–100%):

Step 1 — Detect areas (decide from TITLES & RESPONSIBILITIES, not tools):
- Resume PRIMARY AREA (exactly one): dev | test | automation | ops | data | security | cloud
  * Use titles & verbs:
    - dev: developer/software engineer/backend/full-stack; build/design/implement/architect/features/services/APIs
    - test: tester/QA/quality assurance/manual testing; test cases/execution/defect triage
    - automation: SDET/test automation; builds automated test frameworks (Selenium/Playwright/etc.) as main role
    - ops: SRE/DevOps/production ops; incidents/runbooks/monitoring/release ops
    - data: data engineer/ETL/warehouse/pipelines
    - security: security engineer/AppSec/VAPT
    - cloud: cloud/infra architect/engineer (infra focus)
- JD TARGET AREA: infer from JD titles & responsibilities the same way.
IMPORTANT: Tools alone (Jenkins, SonarQube, Burp, Postman, JMeter, etc.) DO NOT make a resume “testing”.
If titles/responsibilities say "Software Engineer / Developer" and work is building software/services, classify as **dev** even if some testing or DevSecOps tools were used.

Step 2 — Scoring weights:
- Area alignment (PRIMARY AREA vs JD TARGET AREA) = 40%
  * Same area = 1.0; adjacent (dev vs automation) = 0.6; otherwise = 0.2
  * If areas differ, HARD CAP the total final score at **60%**.
- Skills/families match = 40%
  * Map tools→families: Jenkins/GitHub Actions/GitLab CI/Azure DevOps ⇒ CI/CD
    Selenium/Playwright/Cypress/Appium ⇒ Test Automation
    JMeter/k6/LoadRunner ⇒ Performance Testing
    Postman/REST Assured/SoapUI ⇒ API Testing
    Oracle/PL/SQL/SQL Server/Postgres/MySQL ⇒ SQL/Database
    Docker ⇒ Containers; Kubernetes/OpenShift ⇒ Kubernetes
    Cloud services imply provider (EC2/S3 ⇒ AWS; AKS ⇒ Azure; GKE/BigQuery ⇒ GCP)
- Years vs JD minimum (if mentioned) = 15%
- Domain/location fit (if mentioned) = 5%

OUTPUT FORMAT (STRICT):
**Name**: <candidate_name>
**Score**: [NN]%

**Reason**:
- **Role Match**: Primary area detected: <dev|test|automation|ops|data|security|cloud>. JD area: <...>. Alignment: <high|medium|low> with one-sentence justification based on titles/responsibilities.
- **Skill Match**: <1–2 bullets: matched & missing skills/families>
- **Major Gaps**: <1–2 bullets of critical gaps; if area mismatch, include “Profile is primarily <area>, JD is <area>”>

**Warning**: include ONLY if Score < 70%.
You must implement the HARD CAP when areas differ; do not exceed 60% in that case.
"""

def compare_resume(jd_text, resume_text, candidate_name):
    prompt = f"""
You are a Recruiter Assistant bot. Compare the resume to the JD and return STRICT Markdown exactly as specified.

{RUBRIC}

Job Description (JD):
{jd_text}

Resume:
{resume_text}

Return ONLY the Markdown described in OUTPUT FORMAT.
Candidate name to use: {candidate_name}
"""
    raw = call_gemini(prompt)
    return enforce_markdown(raw)

def generate_followup(jd_text, resume_text, candidate_name):
    prompt = f"""
You are a recruiter at Terrabit Consulting writing to {candidate_name}.
Create three sections ONLY:

### WhatsApp Message to Candidate
- Friendly, 4–6 lines
- Mention inferred role & 2–3 time slots
- Ask to confirm phone/email if missing

### Email to Candidate
Subject: Quick chat about a {{<role>}} opportunity at Terrabit Consulting
Dear {candidate_name},
<3–5 lines: why we’re reaching out based on resume, key fit, ask for 2–3 preferred slots this week, polite close with signature placeholder>

### Screening Questions (Tailored)
- Q1 (core JD skill; confirm years/level if unclear)
- Q2 (missing/weak skill vs JD)
- Q3 (domain/tools from JD)
- Q4 optional (availability/notice)
- Q5 optional (location/remote preference)

Information:
JD:
{jd_text}

Resume:
{resume_text}
"""
    return call_gemini(prompt)

# -------------------- Session state --------------------
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

# Reset session
if st.button("🔁 Start New Matching Session"):
    st.session_state.clear()
    st.rerun()

# Uploaders
jd_file = st.file_uploader("📄 Upload Job Description", type=["txt", "pdf", "docx"], key="jd_uploader")
resume_files = st.file_uploader("📑 Upload Candidate Resumes", type=["txt", "pdf", "docx"], accept_multiple_files=True, key="resume_uploader")

# Store JD once per run
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
            result = enforce_markdown(result)

        # Robust score extraction
        score = extract_score(result)

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
        st.warning("⚠️ Consider with caution – Lacks core skills / area alignment")
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

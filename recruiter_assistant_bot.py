# app_gemini_gpt_parity.py
import streamlit as st
import io, re, json, numpy as np
import pandas as pd
import fitz  # PyMuPDF
import docx
import spacy
from sentence_transformers import SentenceTransformer, util

# Optional (only for follow-ups; scoring does NOT use LLM)
import google.generativeai as genai

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Resume Matcher (Gemini – GPT-parity)", layout="centered")
st.title("📌 Terrabit Consulting Talent Match System")
st.caption("Deterministic scoring with skill taxonomy + semantic similarity (model-agnostic).")

# Weights for final score (sum ≈ 1.0)
WEIGHTS = {
    "skill_overlap": 0.55,   # Jaccard overlap of JD vs Resume skills
    "semantic_sim": 0.25,    # Embedding similarity of JD vs Resume
    "title_align":  0.10,    # QA/Tester vs Dev emphasis
    "years_fit":    0.10,    # Resume years vs JD min years (if any)
    # Optional location/domain can be added if you want
}

# Calibration to match your GPT scale (adjust if needed)
CALIB = {"scale": 1.03, "offset": 3}  # mild boost

# Use Gemini for follow-ups (set your key in secrets). If not present, we skip.
USE_GEMINI_FOLLOWUP = "GEMINI_API_KEY" in st.secrets

# -------------------- MODELS --------------------
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_sbert():
    # small, fast, accurate enough for similarity
    return SentenceTransformer("all-MiniLM-L6-v2")

nlp = load_spacy()
sbert = load_sbert()

if USE_GEMINI_FOLLOWUP:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    from google.generativeai import GenerativeModel
    gemini_text = GenerativeModel("gemini-1.5-flash",
        generation_config={"temperature": 0, "top_p": 0.1, "top_k": 1, "max_output_tokens": 1024}
    )

# -------------------- IO UTILS --------------------
def read_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text")
    return text

def read_docx(file):
    d = docx.Document(file)
    parts = []
    for p in d.paragraphs:
        parts.append(p.text)
    for table in d.tables:
        for row in table.rows:
            for cell in row.cells:
                parts.append(cell.text)
    try:
        section = d.sections[0]
        for para in section.footer.paragraphs:
            parts.append(para.text)
    except Exception:
        pass
    return "\n".join(parts)

def read_file(file):
    if file.type == "application/pdf":
        return read_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return read_docx(file)
    else:
        return file.read().decode("utf-8", errors="ignore")

def normalize_text(t: str, max_chars=60000) -> str:
    if not t: return ""
    t = t.replace("\x00", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()[:max_chars]

# -------------------- BASIC EXTRACTORS --------------------
def extract_email(text):
    m = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return m.group() if m else "Not found"

def extract_candidate_name(text, filename):
    # Table/label
    m = re.search(r"(?i)Candidate Name\s*[:–-]\s*(.+)", text)
    if m:
        nm = re.sub(r"[^A-Za-z \-']", " ", m.group(1)).strip()
        if 2 <= len(nm.split()) <= 4:
            return nm.title()
    # Footer-like
    m = re.search(r"(?i)Resume of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})", text)
    if m:
        return m.group(1).strip().title()
    # Top lines heuristic
    first = "\n".join(text.splitlines()[:15])
    m = re.search(r"(?m)^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*$", first)
    if m:
        return m.group(1).strip().title()
    # Filename fallback
    base = re.sub(r"\.(pdf|docx|txt)$", "", filename, flags=re.I)
    base = re.sub(r"[_\-]+", " ", base)
    cand = re.sub(r"[^A-Za-z \-']", " ", base).strip()
    if 2 <= len(cand.split()) <= 4:
        return cand.title()
    return "Name Not Found"

def extract_min_years_from_jd(jd):
    # Look for "X+ years", "at least X years", etc. Return max value found.
    nums = re.findall(r"(?i)(?:at least|minimum|min)?\s*(\d+)\s*\+?\s*(?:years|yrs)", jd)
    return max([int(n) for n in nums], default=0)

def extract_years_from_resume(resume):
    # Max years mentioned anywhere (rough heuristic)
    nums = re.findall(r"(?i)(\d+)\s*\+?\s*(?:years|yrs)", resume)
    return max([int(n) for n in nums], default=0)

# -------------------- SKILL TAXONOMY --------------------
# Canonical skill -> list of regex patterns (lowercase)
SKILL_MAP = {
    # QA/Testing core
    "qa": [r"\bqa\b", r"\bquality assurance\b"],
    "testing": [r"\btesting\b"],
    "manual testing": [r"\bmanual testing\b"],
    "functional testing": [r"\bfunctional testing\b", r"\bsystem testing\b"],
    "regression testing": [r"\bregression testing\b"],
    "performance testing": [r"\bperformance testing\b"],
    "automation": [r"\bautomation testing\b", r"\btest automation\b", r"\bsdet\b"],
    "selenium": [r"\bselenium\b", r"\bwebdriver\b"],
    "sit": [r"\bsit\b"],
    "uat": [r"\buat\b"],
    "pst": [r"\bpst\b"],

    # CI/CD & DevOps
    "ci/cd": [r"\bci/?cd\b", r"\bjenkins\b", r"\bazure devops\b", r"\bgitlab ci\b", r"\bansible\b", r"\bopenshift\b"],
    "docker": [r"\bdocker\b"],
    "kubernetes": [r"\bkubernetes\b"],

    # DB / backend
    "sql": [r"\bsql\b", r"\bpl\s*sql\b", r"\boracle sql\b"],
    "oracle": [r"\boracle\b"],
    "linux": [r"\blinux\b", r"\bunix\b", r"\bbash\b"],
    "rest": [r"\brest\b"],
    "java": [r"\bjava\b", r"\bspring boot\b"],
    "microservices": [r"\bmicroservices?\b"],

    # Security tools
    "security tools": [r"\bsonarqube\b", r"\bfortify\b", r"\bburp suite\b"],
}

def find_skills(text: str):
    t = text.lower()
    found = []
    for canon, patterns in SKILL_MAP.items():
        for p in patterns:
            if re.search(p, t):
                found.append(canon)
                break
    # de-dupe keep order
    seen, res = set(), []
    for s in found:
        if s not in seen:
            seen.add(s); res.append(s)
    return res

# -------------------- SEMANTIC SIMILARITY --------------------
def chunk_text(text, chunk_chars=1200, overlap=150):
    text = re.sub(r"\s+", " ", text)
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+chunk_chars])
        i += max(chunk_chars - overlap, 1)
    return chunks or [""]

@st.cache_resource
def _embed_cache(_model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(_model_name)

def semantic_similarity(jd_text, resume_text, topk=3):
    # chunk resume → compute cosine sim against JD; avg top-k
    jd_emb = sbert.encode([jd_text], normalize_embeddings=True)
    chunks = chunk_text(resume_text)
    res_emb = sbert.encode(chunks, normalize_embeddings=True)
    sims = util.cos_sim(jd_emb, res_emb).cpu().numpy()[0]  # shape (n_chunks,)
    if sims.size == 0:
        return 0.0
    sims.sort()  # ascending
    top = sims[-topk:] if sims.size >= topk else sims
    return float(np.mean(top))  # 0..1

# -------------------- SCORING --------------------
def jaccard(a, b):
    A, B = set(a), set(b)
    if not A and not B: return 0.0
    return len(A & B) / len(A | B)

def title_alignment_score(jd_text, resume_text):
    jd = jd_text.lower(); cv = resume_text.lower()
    qa_like = any(k in jd for k in ["qa","testing","tester","sdet"])
    qa_hit  = any(k in cv for k in ["qa","testing","tester","sdet"])
    dev_like = any(k in jd for k in ["developer","software engineer"])
    dev_hit  = any(k in cv for k in ["developer","software engineer"])
    # Favor QA/test alignment; penalize pure-dev for QA JDs
    if qa_like and qa_hit: return 1.0
    if qa_like and dev_hit and not qa_hit: return 0.2
    if dev_like and dev_hit: return 0.7
    return 0.5  # neutral

def years_fit_score(jd_min, cv_years):
    if jd_min <= 0:
        return 1.0  # no requirement
    if cv_years <= 0:
        return 0.0
    ratio = cv_years / max(jd_min, 1)
    return float(min(ratio, 1.0))  # cap at 1

def compute_score(jd_text, resume_text):
    jd_sk = find_skills(jd_text)
    cv_sk = find_skills(resume_text)

    skill_overlap = jaccard(jd_sk, cv_sk)                       # 0..1
    sim            = semantic_similarity(jd_text, resume_text)   # 0..1
    title_align    = title_alignment_score(jd_text, resume_text) # 0..1
    jd_min_years   = extract_min_years_from_jd(jd_text)
    cv_years       = extract_years_from_resume(resume_text)
    years_ok       = years_fit_score(jd_min_years, cv_years)     # 0..1

    # Weighted sum
    raw = (
        WEIGHTS["skill_overlap"] * skill_overlap +
        WEIGHTS["semantic_sim"]  * sim +
        WEIGHTS["title_align"]   * title_align +
        WEIGHTS["years_fit"]     * years_ok
    )
    # Calibration & clamp
    score = raw * 100.0
    score = score * CALIB["scale"] + CALIB["offset"]
    return int(max(0, min(100, round(score)))), jd_sk, cv_sk, jd_min_years, cv_years

# -------------------- FOLLOW-UPS (LLM OPTIONAL) --------------------
def generate_followup(jd_text, resume_text, candidate_name):
    if not USE_GEMINI_FOLLOWUP:
        return "_Follow-up generation disabled (no GEMINI_API_KEY)._"
    prompt = f"""
You are a recruiter at Terrabit Consulting writing to a candidate named {candidate_name}.
Return exactly three sections:

### WhatsApp Message to Candidate
<friendly outreach; mention inferred role; 2–3 slots; confirm phone/email if missing>

### Email to Candidate
Subject: Quick chat about a {{<role>}} opportunity at Terrabit Consulting
Dear {candidate_name},
<3–5 lines about fit based on their resume; ask for 2–3 preferred slots this week; polite close with signature placeholder>

### Screening Questions (Tailored)
- <Q1 about JD core>
- <Q2 about a missing/weak skill>
- <Q3 about domain/tools in JD>
- <Q4 optional availability/notice>
- <Q5 optional location/remote>

Information to use:
JD:
{jd_text}

Resume:
{resume_text}
"""
    try:
        resp = gemini_text.generate_content(prompt)
        return (getattr(resp, "text", "") or "").strip()
    except Exception as e:
        return f"⚠️ Gemini follow-up failed: {e}"

# -------------------- UI STATE --------------------
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

if st.button("🔁 Start New Matching Session"):
    st.session_state.clear()
    st.rerun()

jd_file = st.file_uploader("📄 Upload Job Description", type=["txt", "pdf", "docx"], key="jd_uploader")
resume_files = st.file_uploader("📑 Upload Candidate Resumes", type=["txt", "pdf", "docx"], accept_multiple_files=True, key="resume_uploader")

# Store JD once
if jd_file and not st.session_state.get("jd_text"):
    st.session_state["jd_text"] = normalize_text(read_file(jd_file))
    st.session_state["jd_file"] = jd_file.name

jd_text = st.session_state.get("jd_text", "")

# -------------------- RUN --------------------
if st.button("🚀 Run Matching") and jd_text and resume_files:
    for resume_file in resume_files:
        if resume_file.name in st.session_state["processed_resumes"]:
            continue

        resume_text = normalize_text(read_file(resume_file))
        name = extract_candidate_name(resume_text, resume_file.name)
        email = extract_email(resume_text)

        with st.spinner(f"🔎 Analyzing {name}..."):
            score, jd_sk, cv_sk, jd_min, cv_years = compute_score(jd_text, resume_text)

        # Reason lines
        matched = sorted(set(jd_sk) & set(cv_sk))
        missing = sorted(set(jd_sk) - set(cv_sk))
        role_line = "Experience overlaps with JD focus" if "testing" in " ".join(cv_sk) or "qa" in " ".join(cv_sk) else "Resume emphasizes development more than testing"
        gaps_line = []
        if score < 70:
            if jd_min and cv_years < jd_min:
                gaps_line.append(f"Insufficient years (JD: {jd_min}+ yrs, Resume: {cv_years} yrs)")
            if missing:
                gaps_line.append(f"Missing core skills: {', '.join(missing)}")
        gaps_line = "; ".join(gaps_line) if gaps_line else ("—" if score >= 70 else "Missing required tools or insufficient years")

        # Markdown like GPT
        warn = "\n\n**Warning**: Score below 70% – candidate may not meet core testing specialization." if score < 70 else ""
        result_md = f"""**Name**: {name}
**Score**: [{score}]%

**Reason**:
- **Role Match**: {role_line}
- **Skill Match**: Matched skills: {', '.join(matched) if matched else 'None'}. Missing skills: {', '.join(missing) if missing else 'None'}.
- **Major Gaps**: {gaps_line}{warn}
"""

        st.session_state["results"].append({
            "correct_name": name,
            "email": email,
            "score": score,
            "result": result_md,
            "resume_text": resume_text
        })
        st.session_state["processed_resumes"].add(resume_file.name)
        st.session_state["summary"].append({"Candidate Name": name, "Email": email, "Score": score})

# -------------------- RESULTS --------------------
for entry in st.session_state["results"]:
    st.markdown("---")
    st.subheader(f"📌 {entry['correct_name']}")
    st.markdown(f"📧 **Email**: {entry['email']}")
    st.markdown(entry["result"], unsafe_allow_html=True)

    s = entry["score"]
    if s < 50:
        st.error("❌ Not suitable – Major role mismatch")
    elif s < 70:
        st.warning("⚠️ Consider with caution – Lacks core skills")
    else:
        st.success("✅ Strong match – Good alignment with JD")

    if st.button(f"✉️ Generate Follow-up for {entry['correct_name']}", key=f"followup_{entry['correct_name']}"):
        with st.spinner("Generating messages..."):
            followup = generate_followup(jd_text, entry["resume_text"], entry["correct_name"])
        st.markdown("---")
        st.markdown(followup, unsafe_allow_html=True)

# -------------------- SUMMARY --------------------
if st.session_state["summary"]:
    st.markdown("### 📊 Summary of All Candidates")
    df_summary = pd.DataFrame(st.session_state["summary"]).sort_values(by="Score", ascending=False)
    st.dataframe(df_summary)
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df_summary.to_excel(writer, index=False)
    st.download_button("📥 Download Summary as Excel",
                       data=excel_buffer.getvalue(),
                       file_name="resume_match_summary.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -------------------- DEBUG --------------------
with st.expander("🔍 Debug (skills & years)"):
    if st.session_state.get("jd_text"):
        jd_sk_dbg = find_skills(st.session_state["jd_text"])
        st.write("JD skills:", jd_sk_dbg)
        st.write("JD min years (regex):", extract_min_years_from_jd(st.session_state["jd_text"]))
    if st.session_state.get("results"):
        last = st.session_state["results"][-1]
        st.write("Last candidate raw years (regex):", extract_years_from_resume(last["resume_text"]))

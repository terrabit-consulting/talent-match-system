import streamlit as st
import time, io, re, json
import fitz  # PyMuPDF
import docx
import pandas as pd
import spacy
import google.generativeai as genai
from collections import Counter

# ---------- Page ----------
st.set_page_config(page_title="Resume Matcher (Gemini)", layout="centered")
st.title("📌 Terrabit Consulting Talent Match System")
st.write("Upload a JD and multiple resumes. Get match scores, red flags, and follow-up messaging.")

# ---------- NLP ----------
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")
nlp = load_spacy_model()

# ---------- Gemini ----------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

gemini_json = genai.GenerativeModel(
    "gemini-1.5-flash",
    generation_config={
        "temperature": 0,
        "top_p": 0.1,
        "top_k": 1,
        "max_output_tokens": 2048,
        "response_mime_type": "application/json",  # force JSON
    }
)

gemini_text = genai.GenerativeModel(
    "gemini-1.5-flash",
    generation_config={
        "temperature": 0,
        "top_p": 0.1,
        "top_k": 1,
        "max_output_tokens": 1024
    }
)

def call_json(prompt: str, schema_hint: dict) -> dict:
    """Ask Gemini for JSON only; return parsed dict or {}."""
    try:
        # Lightly steer with a minimal schema hint inside the prompt
        resp = gemini_json.generate_content(
            f"You must return valid JSON only. Follow this json schema shape (keys only, types implied):\n{json.dumps(schema_hint, indent=2)}\n\n---\n{prompt}"
        )
        text = getattr(resp, "text", "") or ""
        return json.loads(text)
    except Exception as e:
        st.error(f"❌ Gemini JSON failed: {e}")
        return {}

def call_text(prompt: str) -> str:
    try:
        resp = gemini_text.generate_content(prompt)
        return (getattr(resp, "text", "") or "").strip()
    except Exception as e:
        st.error(f"❌ Gemini text failed: {e}")
        return ""

# ---------- Normalization ----------
def normalize_text(t: str, max_chars=40000) -> str:
    if not t: return ""
    t = t.replace("\x00", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = t.strip()
    return t[:max_chars]

# ---------- File reading ----------
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
            parts.extend([c.text for c in row.cells])
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

# ---------- Name/Email ----------
def extract_email(text):
    m = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return m.group() if m else "Not found"

def extract_candidate_name_from_table(text):
    matches = re.findall(r"(?i)Candidate Name\s*[\t:–-]*\s*(.+)", text)
    for match in matches:
        nm = match.strip()
        nm = re.sub(r"[^A-Za-z \-']", "", nm)
        if 2 <= len(nm.split()) <= 4:
            return nm.title()
    return None

def extract_candidate_name_from_footer(text):
    m = re.search(r"(?i)Resume of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})", text)
    return m.group(1).strip().title() if m else None

def extract_candidate_name(text, filename):
    # 1) explicit markers
    for fn in (extract_candidate_name_from_table, extract_candidate_name_from_footer):
        nm = fn(text)
        if nm: return nm
    # 2) heuristic: first lines likely name
    first = "\n".join(text.splitlines()[:15])
    m = re.search(r"(?m)^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*$", first)
    if m: return m.group(1).strip().title()
    # 3) filename fallback
    base = re.sub(r"\.(pdf|docx|txt)$", "", filename, flags=re.I)
    base = re.sub(r"[_\-]+", " ", base)
    cand = re.sub(r"[^A-Za-z \-']", " ", base).strip()
    if 2 <= len(cand.split()) <= 4:
        return cand.title()
    # 4) final LLM fallback (strict)
    prompt = f"""Return ONLY the full name from the text below.
Rules: 2-4 words, capitalized words only. If not found, return exactly 'Name Not Found'.
Text (first 60 lines):
{first}
"""
    nm = call_text(prompt).strip()
    if not nm or "not found" in nm.lower() or len(nm.split()) > 5:
        return "Name Not Found"
    return re.sub(r"[^A-Za-z \-']", "", nm).strip().title()

# ---------- LLM extraction (JSON) ----------
JD_SCHEMA = {
  "role_titles": ["list of role titles from JD"],
  "must_have_skills": ["list of hard skills/tools/frameworks"],
  "nice_to_have_skills": ["list of optional skills"],
  "min_years_core_stack": "integer years if specified else 0",
  "domain_keywords": ["e.g., fintech, telecom, ecommerce"],
  "location_keywords": ["city/country/remote"]
}

RESUME_SCHEMA = {
  "role_titles": ["candidate past/current titles"],
  "skills": ["hard skills/tools/frameworks"],
  "years_overall": "integer estimate",
  "years_core_stack": "integer estimate on JD core skills if possible",
  "domain_keywords": ["industries mentioned"],
  "location_keywords": ["locations mentioned"]
}

def extract_jd_facts(jd_text: str) -> dict:
    prompt = f"""Extract JD facts as JSON. Keep lists short and canonicalized (e.g., 'Java', 'Selenium', 'Oracle', 'CI/CD').
JD:
{jd_text}"""
    data = call_json(prompt, JD_SCHEMA)
    return data or {k: ([] if isinstance(v, list) else 0) for k,v in JD_SCHEMA.items()}

def extract_resume_facts(resume_text: str) -> dict:
    prompt = f"""Extract candidate facts as JSON. Canonicalize skills (e.g., 'Java', 'Selenium', 'Oracle', 'CI/CD').
Resume:
{resume_text}"""
    data = call_json(prompt, RESUME_SCHEMA)
    return data or {k: ([] if isinstance(v, list) else 0) for k,v in RESUME_SCHEMA.items()}

# ---------- Scoring ----------
def jaccard_weighted(a, b):
    A, B = set([s.lower() for s in a]), set([s.lower() for s in b])
    if not A and not B: return 0.0
    inter = len(A & B); union = len(A | B)
    return inter / union if union else 0.0

def title_alignment(jd_titles, cv_titles):
    jd = " ".join(jd_titles).lower()
    cv = " ".join(cv_titles).lower()
    keywords = ["qa", "quality", "tester", "testing", "sdet", "automation", "developer", "engineer"]
    hits = sum(1 for k in keywords if k in jd and k in cv)
    return min(hits / 3.0, 1.0)  # cap at 1

def keyword_hit(jd_kw, cv_kw):
    return 1.0 if set([x.lower() for x in jd_kw]) & set([y.lower() for y in cv_kw]) else 0.0

def compute_score(jd, cv):
    # Components
    hard = jaccard_weighted(jd.get("must_have_skills", []), cv.get("skills", []))
    exp_core = cv.get("years_core_stack", 0); min_core = jd.get("min_years_core_stack", 0)
    exp_ratio = 1.0 if min_core == 0 else min(exp_core / max(min_core, 1), 1.25)  # small reward for exceeding
    exp_ratio = min(exp_ratio, 1.0)

    title = title_alignment(jd.get("role_titles", []), cv.get("role_titles", []))
    domain = keyword_hit(jd.get("domain_keywords", []), cv.get("domain_keywords", []))
    location = keyword_hit(jd.get("location_keywords", []), cv.get("location_keywords", []))

    # Weights
    score = (
        0.50 * hard +
        0.20 * exp_ratio +
        0.10 * title +
        0.10 * domain +
        0.10 * location
    )
    return round(score * 100)

def gaps_and_matches(jd, cv):
    jd_must = set([s.lower() for s in jd.get("must_have_skills", [])])
    cv_sk = set([s.lower() for s in cv.get("skills", [])])
    matched = sorted([s for s in jd_must & cv_sk])
    missing = sorted([s for s in jd_must - cv_sk])
    return matched, missing

# ---------- Rendering ----------
def render_markdown(candidate_name, score, matched, missing, analysis_reason):
    warn = "\n\n**Warning**: Score below 70% – candidate may not meet core testing specialization." if score < 70 else ""
    md = f"""**Name**: {candidate_name}
**Score**: [{score}]%

**Reason**:
- **Role Match**: {analysis_reason.get('role_match', 'Role alignment estimated from titles and responsibilities.')}
- **Skill Match**: Matched skills: {', '.join(matched) if matched else 'None'}. Missing skills: {', '.join(missing) if missing else 'None'}.
- **Major Gaps**: {analysis_reason.get('gaps', 'Missing required tools or insufficient years on core stack.' if score < 70 else 'No major gaps relative to JD.')}{warn}
"""
    return md.strip()

def build_reason_text(jd, cv):
    rm = "Experience overlaps with JD focus" if title_alignment(jd.get("role_titles", []), cv.get("role_titles", [])) >= 0.5 else "Resume focus differs from JD emphasis"
    gaps = []
    if compute_score(jd, cv) < 70:
        if jd.get("min_years_core_stack", 0) > cv.get("years_core_stack", 0):
            gaps.append("Insufficient years on core stack")
        if not keyword_hit(jd.get("domain_keywords", []), cv.get("domain_keywords", [])):
            gaps.append("Domain exposure not explicit")
    return {"role_match": rm, "gaps": "; ".join(gaps) if gaps else "—"}

# ---------- Follow-up ----------
def generate_followup(jd_text, resume_text, candidate_name):
    prompt = f"""
You are a recruiter at Terrabit Consulting writing to a candidate named {candidate_name}.
All outputs must be addressed to the candidate and concise.

---BEGIN OUTPUT---
### WhatsApp Message to Candidate
<friendly outreach; mention inferred role; 2–3 slots; ask to confirm phone/email if missing>

### Email to Candidate
Subject: Quick chat about a {{<role>}} opportunity at Terrabit Consulting
Dear {candidate_name},
<3–5 lines: why we’re reaching out based on their resume, key fit, ask for 2–3 preferred slots this week, and a polite close with signature placeholder>

### Screening Questions (Tailored)
- <Q1 about JD core>
- <Q2 about a missing/weak skill>
- <Q3 about domain/tools in JD>
- <Q4 optional availability/notice>
- <Q5 optional location/remote>
---END OUTPUT---

Information:
JD:
{jd_text}

Resume:
{resume_text}
"""
    return call_text(prompt)

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

# Store JD
if jd_file and not st.session_state.get("jd_text"):
    st.session_state["jd_text"] = normalize_text(read_file(jd_file))
    st.session_state["jd_file"] = jd_file.name

jd_text = st.session_state.get("jd_text", "")

# Run
if st.button("🚀 Run Matching") and jd_text and resume_files:
    jd_facts = extract_jd_facts(jd_text)

    for resume_file in resume_files:
        if resume_file.name in st.session_state["processed_resumes"]:
            continue

        resume_text_raw = read_file(resume_file)
        resume_text = normalize_text(resume_text_raw)

        name = extract_candidate_name(resume_text, resume_file.name)
        email = extract_email(resume_text)

        with st.spinner(f"🔎 Analyzing {name}..."):
            cv_facts = extract_resume_facts(resume_text)
            score = compute_score(jd_facts, cv_facts)
            matched, missing = gaps_and_matches(jd_facts, cv_facts)
            reason = build_reason_text(jd_facts, cv_facts)
            result_md = render_markdown(name, score, matched, missing, reason)

        st.session_state["results"].append({
            "correct_name": name,
            "email": email,
            "score": score,
            "result": result_md,
            "resume_text": resume_text
        })
        st.session_state["processed_resumes"].add(resume_file.name)
        st.session_state["summary"].append({
            "Candidate Name": name,
            "Email": email,
            "Score": score
        })

# Results
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

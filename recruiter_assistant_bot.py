# recruiter_resume_matcher_gemini.py
import streamlit as st
import time, io, re, json
import fitz  # PyMuPDF
import docx
import pandas as pd
import spacy
import google.generativeai as genai

# -------------------- Page --------------------
st.set_page_config(page_title="Resume Matcher (Gemini)", layout="centered")
st.title("📌 Terrabit Consulting Talent Match System")
st.write("Upload a JD and multiple resumes. Get match scores, red flags, and follow-up messaging.")

# -------------------- NLP ---------------------
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")
nlp = load_spacy_model()

# -------------------- Gemini ------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# JSON extractor (force JSON)
gemini_json = genai.GenerativeModel(
    "gemini-1.5-flash",
    generation_config={
        "temperature": 0,
        "top_p": 0.1,
        "top_k": 1,
        "max_output_tokens": 2048,
        "response_mime_type": "application/json"
    }
)
# Text generator (for follow-ups)
gemini_text = genai.GenerativeModel(
    "gemini-1.5-flash",
    generation_config={"temperature": 0, "top_p": 0.1, "top_k": 1, "max_output_tokens": 1024}
)

# -------------------- IO utils ----------------
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

def normalize_text(t: str, max_chars=40000) -> str:
    if not t: return ""
    t = t.replace("\x00", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()[:max_chars]

# ---------------- Email/Name ------------------
def extract_email(text):
    m = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return m.group() if m else "Not found"

def extract_candidate_name_from_table(text):
    matches = re.findall(r"(?i)Candidate Name\s*[\t:–-]*\s*(.+)", text)
    for match in matches:
        nm = re.sub(r"[^A-Za-z \-']", "", match).strip()
        if 2 <= len(nm.split()) <= 4:
            return nm.title()
    return None

def extract_candidate_name_from_footer(text):
    m = re.search(r"(?i)Resume of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})", text)
    return m.group(1).strip().title() if m else None

def extract_candidate_name(text, filename):
    for fn in (extract_candidate_name_from_table, extract_candidate_name_from_footer):
        nm = fn(text)
        if nm: return nm
    first = "\n".join(text.splitlines()[:15])
    m = re.search(r"(?m)^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*$", first)
    if m: return m.group(1).strip().title()
    base = re.sub(r"\.(pdf|docx|txt)$", "", filename, flags=re.I)
    base = re.sub(r"[_\-]+", " ", base)
    cand = re.sub(r"[^A-Za-z \-']", " ", base).strip()
    if 2 <= len(cand.split()) <= 4:
        return cand.title()
    return "Name Not Found"

# --------------- Extraction (JSON) ------------
JD_SCHEMA = {
  "role_titles": ["list of JD role titles"],
  "must_have_skills": ["hard skills/tools/frameworks"],
  "nice_to_have_skills": ["optional skills"],
  "min_years_core_stack": "integer years if specified else 0",
  "domain_keywords": ["e.g., fintech, telecom, ecommerce"],
  "location_keywords": ["city/country/remote/hybrid"]
}
RESUME_SCHEMA = {
  "role_titles": ["candidate past/current titles"],
  "skills": ["hard skills/tools/frameworks"],
  "years_overall": "integer estimate",
  "years_core_stack": "integer estimate for JD core skills",
  "domain_keywords": ["industries mentioned"],
  "location_keywords": ["locations mentioned"]
}

def call_json(prompt: str, schema_hint: dict) -> dict:
    try:
        resp = gemini_json.generate_content(
            f"Return valid JSON only. Follow this shape (keys only, types implied):\n"
            f"{json.dumps(schema_hint, indent=2)}\n\n---\n{prompt}"
        )
        txt = getattr(resp, "text", "") or "{}"
        return json.loads(txt)
    except Exception as e:
        st.error(f"❌ Gemini JSON failed: {e}")
        return {}

# --------------- Sanitizers & synonyms --------
def to_years(x) -> int:
    if x is None: return 0
    if isinstance(x, (int, float)): return int(x)
    if isinstance(x, list): 
        vals = [to_years(i) for i in x]
        return max(vals) if vals else 0
    nums = re.findall(r"\d+", str(x))
    return max(int(n) for n in nums) if nums else 0

def to_list_str(x):
    if x is None: return []
    if isinstance(x, list): return [str(i).strip() for i in x if str(i).strip()]
    parts = re.split(r"[;,|\n]+", str(x))
    return [p.strip() for p in parts if p.strip()]

# alias map collapses synonyms to a canonical token used in scoring
ALIASES = {
    # testing
    "selenium": "selenium", "selenium webdriver": "selenium", "webdriver": "selenium",
    "automation testing": "automation", "test automation": "automation", "sdet": "automation",
    "manual testing": "manual testing", "functional testing": "functional testing",
    "regression testing": "regression testing", "performance testing": "performance testing",
    "sit": "sit", "uat": "uat", "pst": "pst", "system testing": "functional testing",
    # ci/cd
    "ci/cd": "ci/cd", "jenkins": "ci/cd", "azure devops": "ci/cd", "gitlab ci": "ci/cd",
    "openshift": "ci/cd", "ansible": "ci/cd",
    # db
    "sql": "sql", "oracle": "oracle", "oracle sql": "sql", "plsql": "sql",
    # os
    "linux": "linux", "unix": "linux", "bash": "linux",
    # dev
    "java": "java", "spring boot": "java", "rest": "rest",
    # tools
    "sonarqube": "security tools", "fortify": "security tools", "burp suite": "security tools"
}

def canonicalize(skill: str) -> str:
    s = re.sub(r"[^a-z0-9 +/.-]", " ", skill.lower()).strip()
    s = re.sub(r"\s+", " ", s)
    return ALIASES.get(s, s)

def canon_list(items):
    out = []
    for it in items:
        if not it: continue
        c = canonicalize(it)
        # split tokens like "java / spring boot"
        for part in re.split(r"[+/]", c):
            p = part.strip()
            if p: out.append(p)
    # de-dupe while preserving order
    seen, res = set(), []
    for x in out:
        if x not in seen:
            seen.add(x); res.append(x)
    return res

def sanitize_jd_facts(d: dict) -> dict:
    return {
        "role_titles": to_list_str(d.get("role_titles")),
        "must_have_skills": canon_list(to_list_str(d.get("must_have_skills"))),
        "nice_to_have_skills": canon_list(to_list_str(d.get("nice_to_have_skills"))),
        "min_years_core_stack": to_years(d.get("min_years_core_stack")),
        "domain_keywords": to_list_str(d.get("domain_keywords")),
        "location_keywords": to_list_str(d.get("location_keywords")),
    }

def sanitize_cv_facts(d: dict) -> dict:
    return {
        "role_titles": to_list_str(d.get("role_titles")),
        "skills": canon_list(to_list_str(d.get("skills"))),
        "years_overall": to_years(d.get("years_overall")),
        "years_core_stack": to_years(d.get("years_core_stack")),
        "domain_keywords": to_list_str(d.get("domain_keywords")),
        "location_keywords": to_list_str(d.get("location_keywords")),
    }

def extract_jd_facts(jd_text: str) -> dict:
    prompt = f"""Extract JD facts as JSON. Canonicalize concise skill names (e.g., 'Java', 'Selenium', 'CI/CD', 'Oracle', 'Linux').
JD:
{jd_text}"""
    return sanitize_jd_facts(call_json(prompt, JD_SCHEMA))

def extract_resume_facts(resume_text: str) -> dict:
    prompt = f"""Extract resume facts as JSON. Canonicalize concise skill names (e.g., 'Java', 'Selenium', 'CI/CD', 'Oracle', 'Linux').
Resume:
{resume_text}"""
    return sanitize_cv_facts(call_json(prompt, RESUME_SCHEMA))

# ---------------- Scoring ---------------------
def jaccard(a, b):
    A, B = set(a), set(b)
    if not A and not B: return 0.0
    return len(A & B) / len(A | B)

def title_alignment(jd_titles, cv_titles):
    jd = " ".join(jd_titles).lower()
    cv = " ".join(cv_titles).lower()
    keys = ["qa","quality","tester","testing","sdet","automation","developer","engineer"]
    hits = sum(1 for k in keys if k in jd and k in cv)
    return min(hits/3.0, 1.0)

def keyword_hit(jd_kw, cv_kw):
    return 1.0 if set([x.lower() for x in jd_kw]) & set([y.lower() for y in cv_kw]) else 0.0

def compute_score(jd, cv):
    hard = jaccard(jd.get("must_have_skills", []), cv.get("skills", []))
    exp_core = cv.get("years_core_stack", 0)
    min_core = jd.get("min_years_core_stack", 0)
    exp_ratio = 1.0 if min_core == 0 else min(exp_core / max(min_core,1), 1.0)
    title = title_alignment(jd.get("role_titles", []), cv.get("role_titles", []))
    domain = keyword_hit(jd.get("domain_keywords", []), cv.get("domain_keywords", []))
    location = keyword_hit(jd.get("location_keywords", []), cv.get("location_keywords", []))
    score = 0.50*hard + 0.20*exp_ratio + 0.10*title + 0.10*domain + 0.10*location
    return round(score*100)

def gaps_and_matches(jd, cv):
    must = set(jd.get("must_have_skills", []))
    cv_sk = set(cv.get("skills", []))
    return sorted(must & cv_sk), sorted(must - cv_sk)

# -------------- Rendering (GPT-like) ----------
def render_markdown(candidate_name, score, matched, missing, role_line, gaps_line):
    warn = "\n\n**Warning**: Score below 70% – candidate may not meet core testing specialization." if score < 70 else ""
    md = f"""**Name**: {candidate_name}
**Score**: [{score}]%

**Reason**:
- **Role Match**: {role_line}
- **Skill Match**: Matched skills: {', '.join(matched) if matched else 'None'}. Missing skills: {', '.join(missing) if missing else 'None'}.
- **Major Gaps**: {gaps_line if gaps_line else ('No major gaps relative to JD.' if score >= 70 else 'Missing required tools or insufficient years on core stack.')}{warn}
"""
    return md.strip()

def build_reason_lines(jd, cv, score):
    role_line = "Experience overlaps with JD focus" if title_alignment(jd.get("role_titles", []), cv.get("role_titles", [])) >= 0.5 else "Resume focus differs from JD emphasis"
    gaps = []
    if score < 70:
        if jd.get("min_years_core_stack", 0) > cv.get("years_core_stack", 0):
            gaps.append("Insufficient years on core stack")
        if not keyword_hit(jd.get("domain_keywords", []), cv.get("domain_keywords", [])):
            gaps.append("Domain exposure not explicit")
    return role_line, "; ".join(gaps)

# -------------- Follow-up ---------------------
def generate_followup(jd_text, resume_text, candidate_name):
    prompt = f"""
You are a recruiter at Terrabit Consulting writing to a candidate named {candidate_name}.
Return exactly three sections:

### WhatsApp Message to Candidate
<friendly outreach; mention inferred role; 2–3 slots; confirm phone/email if missing>

### Email to Candidate
Subject: Quick chat about a {{<role>}} opportunity at Terrabit Consulting
Dear {candidate_name},
<3–5 lines about fit based on resume; ask for 2–3 preferred slots this week; polite close with signature placeholder>

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
        st.error(f"❌ Gemini text failed: {e}")
        return "⚠️ Gemini processing failed."

# ---------------- Session State ---------------
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

# Store JD once
if jd_file and not st.session_state.get("jd_text"):
    st.session_state["jd_text"] = normalize_text(read_file(jd_file))
    st.session_state["jd_file"] = jd_file.name

jd_text = st.session_state.get("jd_text", "")

# ---------------- Run Matching ----------------
if st.button("🚀 Run Matching") and jd_text and resume_files:
    jd_facts = extract_jd_facts(jd_text)

    for resume_file in resume_files:
        if resume_file.name in st.session_state["processed_resumes"]:
            continue

        resume_text = normalize_text(read_file(resume_file))
        correct_name = extract_candidate_name(resume_text, resume_file.name)
        correct_email = extract_email(resume_text)

        with st.spinner(f"🔎 Analyzing {correct_name}..."):
            cv_facts = extract_resume_facts(resume_text)
            score = compute_score(jd_facts, cv_facts)
            matched, missing = gaps_and_matches(jd_facts, cv_facts)
            role_line, gaps_line = build_reason_lines(jd_facts, cv_facts, score)
            result_md = render_markdown(correct_name, score, matched, missing, role_line, gaps_line)

        st.session_state["results"].append({
            "correct_name": correct_name,
            "email": correct_email,
            "score": score,
            "result": result_md,
            "resume_text": resume_text
        })
        st.session_state["processed_resumes"].add(resume_file.name)
        st.session_state["summary"].append({
            "Candidate Name": correct_name,
            "Email": correct_email,
            "Score": score
        })

# ---------------- Results ---------------------
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

# ---------------- Summary + download ----------
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

# --------------- Debug (optional) -------------
with st.expander("🔍 Debug: Extracted facts from last run"):
    try:
        st.write("JD facts:"); st.json(jd_facts)   # shown after a run
    except Exception:
        st.caption("Run the matcher to see JD facts.")

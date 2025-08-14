# recruiter_matcher_minimal_ai.py
import streamlit as st
import re, io, json
import fitz  # PyMuPDF
import docx
import pandas as pd
import google.generativeai as genai

# =================== UI ===================
st.set_page_config(page_title="Recruiter Matcher (Minimal AI)", layout="centered")
st.title("📌 Terrabit Consulting – Recruiter Matcher (Minimal AI)")
st.write("Upload **one JD** and **multiple resumes**. Model maps tools ⇒ families & areas; code scores with a small rubric.")

# =================== Config ===================
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Please set GEMINI_API_KEY in Streamlit secrets.")
    st.stop()
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

gem_json = genai.GenerativeModel(
    "gemini-1.5-flash",
    generation_config={
        "temperature": 0,
        "top_p": 0.1,
        "top_k": 1,
        "max_output_tokens": 2000,
        "response_mime_type": "application/json",
    },
)
gem_text = genai.GenerativeModel(
    "gemini-1.5-flash",
    generation_config={"temperature": 0, "top_p": 0.1, "top_k": 1, "max_output_tokens": 900},
)

# =================== I/O ===================
def read_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for p in doc:
            text += p.get_text("text")
    return text

def read_docx(file):
    d = docx.Document(file)
    parts = []
    for p in d.paragraphs: parts.append(p.text)
    for t in d.tables:
        for r in t.rows:
            for c in r.cells: parts.append(c.text)
    try:
        for p in d.sections[0].footer.paragraphs: parts.append(p.text)
    except Exception:
        pass
    return "\n".join(parts)

def read_file(file):
    if file.type == "application/pdf": return read_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document": return read_docx(file)
    else: return file.read().decode("utf-8", errors="ignore")

def normalize_text(t, max_chars=60000):
    if not t: return ""
    t = t.replace("\x00", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()[:max_chars]

def extract_email(text):
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return m.group() if m else "Not found"

def extract_name(text, filename):
    m = re.search(r"(?i)Candidate Name\s*[:–-]\s*(.+)", text)
    if m:
        nm = re.sub(r"[^A-Za-z \-']", " ", m.group(1)).strip()
        if 2 <= len(nm.split()) <= 4: return nm.title()
    m = re.search(r"(?i)Resume of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})", text)
    if m: return m.group(1).strip().title()
    first = "\n".join(text.splitlines()[:15])
    m = re.search(r"(?m)^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*$", first)
    if m: return m.group(1).strip().title()
    base = re.sub(r"\.(pdf|docx|txt)$", "", filename, flags=re.I)
    base = re.sub(r"[_\-]+", " ", base)
    cand = re.sub(r"[^A-Za-z \-']", " ", base).strip()
    if 2 <= len(cand.split()) <= 4: return cand.title()
    return "Name Not Found"

def extract_years(text):
    nums = re.findall(r"(?i)(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)", text)
    return int(float(max(nums))) if nums else 0

# =================== Taxonomy (prompt-only) ===================
# We keep taxonomy SMALL in code; the model is responsible for mapping.
ALLOWED_FAMILIES = [
    # delivery / devops
    "cicd","scm","build","iac","containers","kubernetes",
    # db / data
    "db-sql","db-nosql","dw-snowflake","dw-redshift","dw-bigquery",
    "etl-airflow","etl-dbt","stream-kafka","batch-spark",
    # cloud (provider-level)
    "aws","azure","gcp",
    # observability
    "monitoring","logging","tracing",
    # testing
    "testing-functional","testing-automation","testing-api","testing-performance",
    # languages / frameworks (kept short)
    "java","python","dotnet",".net","javascript","typescript","spring","django","react","angular",
    # os/shell
    "linux","bash","windows","powershell",
    # misc
    "security-testing","webserver-nginx","webserver-apache"
]
ALLOWED_AREAS = ["dev","test","automation","ops","data","security","cloud"]

# Few-shot normalization examples in the prompt (no code aliasing):
FEW_SHOT = """
Map tools to families **only from ALLOWED_FAMILIES**. Examples:
- Jenkins, GitHub Actions, GitLab CI, Azure DevOps, TeamCity, Bamboo, CircleCI ⇒ "cicd"
- Selenium, Playwright, Cypress, Appium ⇒ "testing-automation"
- JMeter, k6, LoadRunner ⇒ "testing-performance"
- Postman, REST Assured, SoapUI ⇒ "testing-api"
- Oracle, PL/SQL, SQL Server, PostgreSQL/MySQL ⇒ "db-sql"
- MongoDB, Cassandra, DynamoDB ⇒ "db-nosql"
- Docker ⇒ "containers"; Kubernetes/OpenShift ⇒ "kubernetes"
- AWS services (EC2, S3, RDS, Lambda) ⇒ also map "aws"
- Azure services (AKS, Functions, Blob) ⇒ also map "azure"
- GCP services (GKE, BigQuery, Pub/Sub) ⇒ also map "gcp"
- Prometheus/Grafana ⇒ "monitoring"; ELK/Splunk ⇒ "logging"; Jaeger/Zipkin/OTel ⇒ "tracing"
- Linux/Unix ⇒ "linux"; Bash ⇒ "bash"
- Java ⇒ "java"; Spring Boot ⇒ "spring"
Return **families only from ALLOWED_FAMILIES**. If a tool is unknown, choose the nearest family or omit.
"""

# =================== Gemini calls ===================
def call_json(prompt: str) -> dict:
    try:
        resp = gem_json.generate_content(prompt)
        text = getattr(resp, "text", "") or "{}"
        return json.loads(text)
    except Exception as e:
        st.error(f"Gemini JSON error: {e}")
        return {}

def jd_to_json(jd_text: str) -> dict:
    prompt = f"""
You are an ATS skill normalizer.

ALLOWED_FAMILIES = {json.dumps(ALLOWED_FAMILIES)}
ALLOWED_AREAS = {json.dumps(ALLOWED_AREAS)}
{FEW_SHOT}

TASK: From the Job Description (JD) text, output JSON with:
{{
  "required_families": ["family from ALLOWED_FAMILIES", ...],   // max 12, most critical
  "optional_families": ["family from ALLOWED_FAMILIES", ...],   // optional or nice-to-have
  "target_area": "one of {ALLOWED_AREAS}",                      // dev | test | automation | ops | data | security | cloud
  "role_titles": ["short titles found"],
  "min_years": <integer years if specified else 0>
}}

Rules:
- Families must be chosen **only** from ALLOWED_FAMILIES.
- Prefer families (e.g., "cicd") over tool names (e.g., "jenkins").
- If CI/CD is present by tools (Jenkins, ADO, GitHub Actions), include "cicd".
- Testing types (UAT/SIT/regression/functional) ⇒ "testing-functional".
- Selenium/Playwright/Appium ⇒ "testing-automation".
- API testing ⇒ "testing-api"; JMeter/k6 ⇒ "testing-performance".
- Oracle/SQL/PLSQL/SQL Server/Postgres/MySQL ⇒ "db-sql".

JD TEXT:
{jd_text}
"""
    return call_json(prompt)

def resume_to_json(resume_text: str) -> dict:
    prompt = f"""
You are an ATS skill normalizer.

ALLOWED_FAMILIES = {json.dumps(ALLOWED_FAMILIES)}
ALLOWED_AREAS = {json.dumps(ALLOWED_AREAS)}
{FEW_SHOT}

TASK: From the Resume text, output JSON with:
{{
  "skills_families": ["family from ALLOWED_FAMILIES", ...],     // deduped; only families
  "areas_present": ["subset of {ALLOWED_AREAS}"],               // which areas the candidate shows
  "role_titles": ["short titles"],
  "years_overall": <integer best estimate>
}}

Rules:
- Families must be chosen **only** from ALLOWED_FAMILIES.
- Prefer families (e.g., "cicd") over tool names (e.g., "jenkins").
- If CI/CD tools present, include "cicd".
- Selenium/Playwright/Appium ⇒ "testing-automation".
- Functional/UAT/SIT/regression/system testing ⇒ "testing-functional".
- Oracle/SQL/PLSQL/SQL Server/Postgres/MySQL ⇒ "db-sql".
- If unsure about years, return 0.
RESUME TEXT:
{resume_text}
"""
    return call_json(prompt)

# =================== Scoring (small & deterministic) ===================
WEIGHTS = {
    "required_coverage": 0.70,  # how many required families are satisfied by resume families
    "area_alignment":   0.20,   # target_area ∈ areas_present
    "years_fit":        0.10,   # resume years vs JD min
}
CALIB = {"scale": 1.02, "offset": 2}  # gentle lift to match recruiter intuition

def ratio_hits(required, got):
    if not required: return 1.0
    R, G = set(required), set(got)
    return len(R & G) / len(R)

def area_align(target_area, areas_present):
    if not target_area: return 0.6   # neutral
    return 1.0 if target_area in set(areas_present or []) else 0.3

def years_ok(min_years, overall):
    if not min_years or min_years <= 0: return 1.0
    if overall <= 0: return 0.0
    return min(overall / min_years, 1.0)

def score_candidate(jd_json: dict, cv_json: dict) -> int:
    req  = jd_json.get("required_families", [])
    got  = cv_json.get("skills_families", [])
    area = jd_json.get("target_area", "")
    areas= cv_json.get("areas_present", [])
    miny = int(jd_json.get("min_years", 0) or 0)
    yrs  = int(cv_json.get("years_overall", 0) or 0)

    cov = ratio_hits(req, got)
    aa  = area_align(area, areas)
    yf  = years_ok(miny, yrs)

    raw = WEIGHTS["required_coverage"]*cov + WEIGHTS["area_alignment"]*aa + WEIGHTS["years_fit"]*yf
    sc  = int(max(0, min(100, round(raw*100*CALIB["scale"] + CALIB["offset"]))))
    return sc

# =================== Session State ===================
if "results" not in st.session_state: st.session_state["results"] = []
if "processed_resumes" not in st.session_state: st.session_state["processed_resumes"] = set()
if "jd_text" not in st.session_state: st.session_state["jd_text"] = ""
if "jd_file" not in st.session_state: st.session_state["jd_file"] = None
if "jd_json" not in st.session_state: st.session_state["jd_json"] = None
if "summary" not in st.session_state: st.session_state["summary"] = []

if st.button("🔁 Start New Matching Session"):
    st.session_state.clear()
    st.rerun()

# =================== Uploaders ===================
jd_file = st.file_uploader("📄 Upload Job Description", type=["txt","pdf","docx"], key="jd_uploader")
resume_files = st.file_uploader("📑 Upload Candidate Resumes", type=["txt","pdf","docx"], accept_multiple_files=True, key="resume_uploader")

if jd_file and not st.session_state.get("jd_text"):
    st.session_state["jd_text"] = normalize_text(read_file(jd_file))
    st.session_state["jd_file"] = jd_file.name

jd_text = st.session_state.get("jd_text","")

# =================== Run ===================
if st.button("🚀 Run Matching") and jd_text and resume_files:
    # Extract JD canonical families/area/years once
    jd_json = jd_to_json(jd_text) or {}
    # Safety clamps
    jd_json["required_families"] = [f for f in jd_json.get("required_families", []) if f in ALLOWED_FAMILIES][:12]
    jd_json["optional_families"] = [f for f in jd_json.get("optional_families", []) if f in ALLOWED_FAMILIES][:12]
    jd_json["target_area"] = jd_json.get("target_area", "") if jd_json.get("target_area","") in ALLOWED_AREAS else ""
    try:
        jd_json["min_years"] = int(jd_json.get("min_years", 0) or 0)
    except:
        jd_json["min_years"] = 0
    st.session_state["jd_json"] = jd_json

    for resume_file in resume_files:
        if resume_file.name in st.session_state["processed_resumes"]:
            continue

        resume_text = normalize_text(read_file(resume_file))
        name = extract_name(resume_text, resume_file.name)
        email = extract_email(resume_text)

        with st.spinner(f"🔎 Analyzing {name}..."):
            cv_json = resume_to_json(resume_text) or {}
            cv_json["skills_families"] = [f for f in cv_json.get("skills_families", []) if f in ALLOWED_FAMILIES]
            cv_json["areas_present"] = [a for a in cv_json.get("areas_present", []) if a in ALLOWED_AREAS]
            try:
                cv_json["years_overall"] = int(cv_json.get("years_overall", 0) or 0)
            except:
                cv_json["years_overall"] = extract_years(resume_text)  # fallback

            score = score_candidate(jd_json, cv_json)

        # Reasoning blurbs
        req = jd_json.get("required_families", [])
        got = cv_json.get("skills_families", [])
        matched = sorted(set(req) & set(got))
        missing = sorted(set(req) - set(got))
        role_line = f"Target area: **{jd_json.get('target_area','—')}**; candidate shows: **{', '.join(cv_json.get('areas_present', []) or ['—'])}**"
        gaps = []
        if score < 70:
            if jd_json.get("min_years", 0) > (cv_json.get("years_overall", 0) or 0):
                gaps.append(f"Insufficient years (JD {jd_json['min_years']}+, Resume {cv_json.get('years_overall',0)})")
            if missing:
                gaps.append(f"Missing required families: {', '.join(missing)}")
        gaps_text = "; ".join(gaps) if gaps else ("—" if score >= 70 else "Missing core families or years")

        result_md = f"""**Name**: {name}
**Score**: [{score}]%

**Reason**:
- **Role Match**: {role_line}
- **Skill Match**: Matched families: {', '.join(matched) if matched else 'None'}. Missing: {', '.join(missing) if missing else 'None'}.
- **Major Gaps**: {gaps_text}{("\n\n**Warning**: Score below 70% – review gaps before shortlisting." if score < 70 else "")}
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

# =================== Results ===================
for entry in st.session_state["results"]:
    st.markdown("---")
    st.subheader(f"📌 {entry['correct_name']}")
    st.markdown(f"📧 **Email**: {entry['email']}")
    st.markdown(entry["result"], unsafe_allow_html=True)

    s = entry["score"]
    if s < 50: st.error("❌ Not suitable – Major role mismatch")
    elif s < 70: st.warning("⚠️ Consider with caution – Lacks core families")
    else: st.success("✅ Strong match – Good alignment with JD")

    if st.button(f"✉️ Generate Follow-up for {entry['correct_name']}", key=f"followup_{entry['correct_name']}"):
        follow = gem_text.generate_content(f"""
Create:
1) WhatsApp message (friendly, 4–6 lines) inviting {entry['correct_name']} to a quick screening.
2) Email (formal, 4–6 lines).
3) 3 tailored screening questions based on JD and resume contents below.

JD:
{st.session_state.get('jd_text','')}

Resume:
{entry['resume_text']}
""")
        st.markdown("---")
        st.markdown((getattr(follow, "text", "") or "").strip(), unsafe_allow_html=True)

# =================== Summary ===================
if st.session_state["summary"]:
    st.markdown("### 📊 Summary of All Candidates")
    df = pd.DataFrame(st.session_state["summary"]).sort_values(by="Score", ascending=False)
    st.dataframe(df)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w: df.to_excel(w, index=False)
    st.download_button("📥 Download Summary as Excel", data=buf.getvalue(),
                       file_name="resume_match_summary.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

import openai
import streamlit as st
import time
import fitz  # PyMuPDF
import docx
import pandas as pd
import re
import io
import spacy

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def call_gpt_with_fallback(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"❌ GPT-4o failed. {str(e)}")
        return "⚠️ GPT processing failed."

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

def extract_email(text):
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}", text)
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
You are a resume parser assistant.

Extract the candidate's **full name only** from the following resume text.

✅ Look for patterns like:
- Candidate Name:
- Resume of <Name>
- Table headers or footers
- A standalone name at the top (2–4 words, capitalized)

❌ Do NOT return:
- Job titles (e.g., Developer, Manager)
- Technical terms (e.g., DB Servers, Azure, Python)
- Locations (e.g., Bangalore, India)
- Email addresses or phone numbers

If no valid name is found, respond only with: Name Not Found

Resume:
{trimmed_text}

Return only the name.
"""
        name = call_gpt_with_fallback(prompt)
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

def compare_resume(jd_text, resume_text, candidate_name):
    prompt = f"""
You are a Recruiter Assistant bot.

Compare the following resume to the job description and return the result in the following format:

**Name**: {candidate_name}
**Score**: [Match Score]%

**Reason**:
- Role Match: (Brief explanation)
- Skill Match: (Matched or missing skills)
- Major Gaps: (What is completely missing or irrelevant)

Warning: Add only if score < 70%

Job Description:
{jd_text}

Resume:
{resume_text}
"""
    return call_gpt_with_fallback(prompt)

def generate_followup(jd_text, resume_text):
    prompt = f"""
Based on the resume and job description below, generate:
1. WhatsApp message (casual)
2. Email message (formal)
3. Screening questions (3-5)

Job Description:
{jd_text}

Resume:
{resume_text}
"""
    return call_gpt_with_fallback(prompt)

# Streamlit UI
st.set_page_config(page_title="Resume Matcher GPT", layout="centered")
st.title("📌 Terrabit Consulting Talent Match System")
st.write("Upload a JD and multiple resumes. Get match scores, red flags, and follow-up messaging.")

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

# ✅ Store JD only once and reuse for every resume
if jd_file and not st.session_state.get("jd_text"):
    jd_text = read_file(jd_file)
    st.session_state["jd_text"] = jd_text
    st.session_state["jd_file"] = jd_file.name

jd_text = st.session_state.get("jd_text", "")

if st.button("🚀 Run Matching") and jd_text and resume_files:
    for resume_file in resume_files:
        if resume_file.name in st.session_state["processed_resumes"]:
            continue

        resume_text = read_file(resume_file)
        correct_name = extract_candidate_name(resume_text, resume_file.name)
        correct_email = extract_email(resume_text)

        with st.spinner(f"🔎 Analyzing {correct_name}..."):
            result = compare_resume(jd_text, resume_text, correct_name)

        score_match = re.search(r"Score\*\*: ?([0-9]+)%", result)
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
            followup = generate_followup(jd_text, entry["resume_text"])
            st.markdown("---")
            st.markdown(followup, unsafe_allow_html=True)

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
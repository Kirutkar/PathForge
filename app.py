import streamlit as st
import fitz
import joblib
import numpy as np
import openai
import pandas as pd
from fpdf import FPDF
import tempfile
import os 

openai.api_key=os.getenv("OPENAI_API_KEY")

linkedin_rag_df=pd.read_csv("final_linkedin_post_ideas.csv")

# Load model + vectorizer
model = joblib.load("xgb_resume_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Thresholds from earlier training
q_low = 0.5166355336146575
q_high = 2.831921823997124


# Weighted scoring dict (add yours here)
weighted_keywords = {
    # 🔹 Advanced AI / Technical
    'llm': 3.5, 'langchain': 3.5, 'rag': 3.5, 'rag pipeline': 3.5,
    'vector db': 3.5, 'weaviate': 3, 'chromadb': 3, 'pinecone': 3,
    'agent': 3, 'langchain agents': 3.5, 'autonomous agent': 3,
    'fine-tuning': 3, 'embedding': 3, 'semantic search': 3,
    'transformers': 3, 'huggingface': 3, 'openai': 3,
    'streamlit': 2.5, 'flask': 2.5, 'gradio': 2.5,
    'pytorch': 2.5, 'tensorflow': 2.5,
    'sql': 2, 'power bi': 2, 'pandas': 2, 'numpy': 2,
    'data analysis': 2,

    # 🔸 Business / Management
    'project management': 3.5, 'agile': 3, 'stakeholder': 2.5,
    'scrum': 3, 'planning': 2, 'budgeting': 2,
    'strategic partnerships': 3, 'gtm': 2.5, 'account planning': 2.5,
    'market share': 2.5, 'revenue growth': 3, 'client relationships': 2.5,

    # 🟢 Sales / CRM
    'crm': 3, 'channel sales': 3, 'business development': 3,
    'partner engagement': 3, 'sales forecasting': 2.5,
    'campaign': 2.5, 'salesforce': 2.5, 'leads': 2, 'market research': 2.5,
    'negotiation': 2, 'presentation': 2,

    # 🟠 Education / Teaching
    'curriculum': 3, 'lesson planning': 2.5, 'teaching': 3,
    'student engagement': 2.5, 'learning outcomes': 2.5, 'training': 2.5,
    'academic institutions': 2.5, 'ministry of education': 3,
    'educational partnerships': 2.5, 'demo': 2, 'workshop': 2,

    # 🔵 HR / Support
    'recruitment': 3, 'employee engagement': 2.5, 'onboarding': 2.5,
    'conflict resolution': 2, 'policy': 2, 'human resources': 3,

    # 🔴 Security / Law Enforcement
    'security guard': 3, 'loss prevention': 3, 'cctv': 2.5, 'access control': 2.5,
    'conflict de-escalation': 2, 'law enforcement': 2.5, 'threat assessment': 2,
    'certified protection': 3, 'crisis intervention': 2, 'surveillance': 2
}

# Score function
def weighted_score(text):
    text = text.lower()
    return sum(weight for kw, weight in weighted_keywords.items() if kw in text)


# GPT Job Role Prediction
roles = ["AI Engineer", "Data Scientist", "Project Manager", "Sales Executive", "Teacher", "HR Specialist", "Security Officer"]

def gpt_predict_role(resume_text):
    prompt = f"""
You are a job role classification expert. You will be given a resume summary and skills.
From the list below, identify the **single most appropriate job role** this candidate fits into.
Do not guess or create new titles. Choose **only from the list**.
Roles:
{', '.join(roles)}
Resume:
{resume_text}
Answer only with one of the roles from the list.
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"

# GPT Resume Feedback
def gpt_resume_feedback(resume_text):
    prompt = f"""
You are an expert resume reviewer.
Analyze the resume text below and provide **structured, clear, and reader-friendly** markdown feedback under these sections:
## 🔧 Resume Improvement Suggestions
**Clarity & Formatting**
- (Short bullet points)
**Missing Sections**
- (Mention if Skills, Certifications, Projects, etc. are missing)
**Projects**
- (How to describe them better or where to move them)
---
## 🧠 Missing Keywords
List any specific tools, technologies, or keywords that are missing for the predicted role.
---
## 🚀 Quick Wins
**Certifications**
- Recommend top 2 certifications to improve resume strength
**Free Courses**
- Suggest 1-2 free courses from platforms like Coursera, edX, or YouTube
**Small Edits**
- Easy improvements: better formatting, quantifiable achievements, etc.
Be professional, clean, encouraging and use proper markdown formatting:
- Use **bold** for subheadings
- Use `-` for bullet points
- If suggesting any certifications or courses, format them as **clickable links** using markdown like:
  `[Course Title](https://example.com)`
Resume:
{resume_text}
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Main app logic
def process_resume(file):
    doc = fitz.open(file.name)
    resume_text = " ".join([page.get_text() for page in doc]).strip()

    # ML prediction
    X_input = vectorizer.transform([resume_text])
    predicted_strength = model.predict(X_input)[0]

    # Hybrid logic
    resume_score = weighted_score(resume_text)
    normalized_score = resume_score / np.log(len(resume_text.split()) + 1)

    if predicted_strength == 'Average' and normalized_score >= q_high:
        predicted_strength = 'Strong'
    elif predicted_strength == 'Average' and normalized_score < q_low:
        predicted_strength = 'Weak'

    # GPT feedback + role
    role = gpt_predict_role(resume_text)
    tips = gpt_resume_feedback(resume_text)

    return predicted_strength, role, tips

#Linkedin Enhancement
def generate_linkedin_feedback(about_text, file, role):
    try:
        doc = fitz.open(file.name)
        resume_text = " ".join([page.get_text() for page in doc]).strip()
    except:
        resume_text = ""

    # Get RAG tips
    # Check if the role exists in the dataframe before accessing .values
    if role in linkedin_rag_df['role'].values:
        rag_tip = linkedin_rag_df[linkedin_rag_df['role'] == role]['tips'].values
        rag_tip_text = rag_tip[0] if len(rag_tip) > 0 else ""
    else:
        rag_tip_text = "" # Provide a default or empty tip if role not found


    prompt = f"""
You are a career branding expert helping people improve their LinkedIn.
Based on the resume and predicted role, generate structured LinkedIn content guidance.
1. 🔧 **Improve "About Me"**
If About Me is provided: suggest improvements.
If no About Me is given, generate a new one in a confident, first-person tone — as if the user is speaking directly to their network without mentioning that nothing was provided. Avoid formal third-person voice. Use warm, natural language suitable for LinkedIn.
2. ✍ **Suggest 3 LinkedIn post ideas**
Inspire posts relevant to their role. Include tips from this RAG input:\n{rag_tip_text}
3. 🔖 **Offer engagement tips**
How to grow visibility (e.g., comment, hashtag use, follow-up posts)
Format your reply with markdown bullets and emojis. Be concise and encouraging.
Resume: {resume_text}
About Section: {about_text}
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"

#Job Match
def match_resume_with_jd(resume_file, jd_text):
    try:
        doc = fitz.open(resume_file.name)
        resume_text = " ".join([page.get_text() for page in doc]).strip()
    except:
        return "❌ Unable to read resume."

    prompt = f"""
You are a helpful and ethical career assistant.
Compare the candidate's resume and the job description below. Do these 3 things:
1. **Match Score**: Estimate how well the resume matches the JD (0–100%) with clear reasoning.
2. **Missing Keywords**: Identify only the important keywords or skills that are *actually not found* in the resume.
3. **Suggestions to Improve**: Based ONLY on the content present in the resume, suggest realistic ways the candidate can:
- Rephrase existing experience to better match the job
- Emphasize transferrable skills (like mentoring, public speaking, teamwork)
- Avoid fabricating roles or experiences not present
Never invent teaching experience, tools, or certifications that are not mentioned.
Resume:
{resume_text}
Job Description:
{jd_text}
Respond in markdown format with bold section headings.
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ GPT Error: {str(e)}"

# Job Explorer
def generate_job_explorer_output(resume_file):
    import fitz
    from urllib.parse import quote

    # Step 1: Extract text from PDF
    try:
        doc = fitz.open(resume_file.name)
        resume_text = " ".join([page.get_text() for page in doc]).strip()
    except:
        return "❌ Unable to read resume. Please upload a valid PDF."

    # Step 2: Use GPT to detect experience level + suggest roles
    prompt = f"""
You are an AI career coach.
Read the resume below and do the following:
1. Predict the user's experience level: Entry / Mid / Senior
   - Consider total years of work **and** how recent their last full-time job was.
   - If they had a long break or are doing a training/residency now, treat them as Entry-Level.
2. Suggest 3–4 job roles the candidate is likely to be a good fit for (avoid duplicates)
Respond in this markdown format:
**Experience Level**: Entry
**Suggested Roles**:
- Data Analyst
- Junior BI Developer
- Reporting Analyst
Resume:
{resume_text}
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        result = response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error from GPT: {str(e)}"

    # Step 3: Generate Indeed links based on experience level
    final_output = result + "\n\n**🔗 Explore Jobs on Indeed UAE:**\n"

    # Extract experience level
    experience_level = "entry"
    for line in result.splitlines():
        if "Experience Level" in line:
            experience_level = line.split(":")[-1].strip().lower()

    # Experience level filters for Indeed
    experience_filters = {
        "entry": "&explvl=entry_level",
        "mid": "&explvl=mid_level",
        "senior": "&explvl=senior_level"
    }
    exp_filter = experience_filters.get(experience_level, "")

    # Create links for each suggested role
    for line in result.splitlines():
        if "- " in line and "Suggested Roles" not in line:
            role = line.strip("- ").strip()
            query = quote(role)
            indeed_url = f"https://ae.indeed.com/jobs?q={query}&l=United+Arab+Emirates{exp_filter}"
            final_output += f"- [{role} Jobs in UAE]({indeed_url})\n"

    # Final tip
    final_output += "\n💡 _Tip: You can also search the same job titles on LinkedIn or Bayt for more options._"

    return final_output

# 🧠 Conversational career agent
def chat_with_career_agent(history, user_message, resume_file):
    try:
        doc = fitz.open(resume_file.name)
        resume_text = " ".join([page.get_text() for page in doc]).strip()
    except:
        return history + [{"role": "user", "content": user_message},
                          {"role": "assistant", "content": "❌ Unable to read resume."}]

    prompt = f"""
You are a warm and friendly AI career coach.
ONLY answer questions related to:
- Resume review, improvement
- LinkedIn profile enhancement
- Role suitability or job matching
- Career growth plans (e.g., certifications, skill roadmaps)
- Interview tips, career clarity
Ignore personal, unrelated questions (like recipes, coding help, travel).
Resume:
{resume_text}
User asked:
{user_message}
If the query is valid (even if slightly unclear), ask a clarifying question and help warmly.
If it's off-topic (not career/job related), reply:
"I'm here to support your career and resume journey only 😊"
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a career guidance assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        reply = f"❌ Error: {str(e)}"

    return history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": reply}
    ]

#Download PDF
#for main tab
def rewrite_resume_main(resume_file, strength, role, tips):
    resume_text = extract_resume_text(resume_file)

    prompt = f"""
You are a professional resume rewriter. Rewrite the following resume to improve its strength, based on:
- Strength: {strength}
- Predicted Role: {role}
- AI Feedback: {tips}
Generate a clean, ATS-friendly version with proper formatting in sections like:
1. **Summary**
2. **Skills**
3. **Experience**
4. **Projects**
5. **Certifications**
Keep the language warm, confident, and professional. Do NOT mention the words 'suggestion' or 'AI'.
Resume to rewrite:
{resume_text}
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        rewritten = response.choices[0].message.content.strip()
    except Exception as e:
        return None, f"❌ GPT Error: {str(e)}"

    pdf_path = generate_pdf(rewritten)
    print("✅ PDF saved at:", pdf_path)
    return pdf_path, "✅ Resume rewritten successfully!"

#for jdmatch tab
def rewrite_resume_for_jd(resume_file, jd_text):
    resume_text = extract_resume_text(resume_file)

    prompt = f"""
You are an AI resume enhancer.
Rewrite this resume to best match the following job description (JD) while being honest and using only real information found in the resume.
Resume:
{resume_text}
JD:
{jd_text}
Structure it with proper headings: Summary, Skills, Experience, Projects, and Certifications.
Do not add false experiences. Use persuasive language to reframe existing experience in a way that aligns with the JD.
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        rewritten = response.choices[0].message.content.strip()
    except Exception as e:
        return None, f"❌ GPT Error: {str(e)}"

    pdf_path = generate_pdf(rewritten)
    return pdf_path, "✅ Resume rewritten for JD match!"

def generate_pdf(resume_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    for line in resume_text.split("\n"):
        if line.strip() == "":
            pdf.ln()
        else:
            pdf.multi_cell(0, 10, line)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)
    return temp_file.name

# ------------------- SESSION CLEAR FUNCTIONS -------------------
def clear_all_main_tab():
    st.session_state["main_resume"] = None
    st.session_state["about_text"] = ""
    st.session_state["strength"] = ""
    st.session_state["role"] = ""
    st.session_state["tips"] = ""

def clear_all_jd_tab():
    st.session_state["shared_resume"] = None
    st.session_state["jd_text"] = ""
    
# ----------------- STREAMLIT UI -----------------
st.set_page_config(page_title="PathForge AI", layout="wide")
st.title("🚀 PathForge AI")
st.caption("Your AI-powered resume & LinkedIn career coach")

tab1, tab2 = st.tabs(["🏠 Resume & LinkedIn", "🎯 JD Match & Explorer"])

# ---------------------- TAB 1 ----------------------
with tab1:
    st.subheader("📄 Upload Your Resume")
    resume_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"], key="main_resume")

    if resume_file:
        if st.button("🔍 Analyze My Resume"):
            with st.spinner("⏳ Analyzing your resume... Please wait."):
                strength, role, tips = process_resume(resume_file)
                st.session_state["strength"] = strength
                st.session_state["role"] = role
                st.session_state["tips"] = tips

                st.success(f"💪 Resume Strength: {strength}")
                st.info(f"🧩 Predicted Role: {role}")
                st.markdown("### 🛠️ Resume Feedback")
                st.markdown(tips, unsafe_allow_html=True)

        st.markdown("### 📥 Download AI-Enhanced Resume")
        if st.button("📥 Generate Enhanced Resume PDF"):
            with st.spinner("⏳ Generating your resume..."):
                pdf_path, msg = rewrite_resume_main(resume_file, st.session_state.get("strength", ""), st.session_state.get("role", ""), st.session_state.get("tips", ""))
                if pdf_path:
                    with open(pdf_path, "rb") as f:
                        st.download_button("📥 Download PDF", f, file_name="Enhanced_Resume.pdf")
                else:
                    st.error(msg)

    st.markdown("---")
    st.subheader("💼 LinkedIn Enhancer")
    about_text = st.text_area("🔗 Paste your LinkedIn 'About Me' (Optional)", height=150, key="about_text")
    if st.button("✨ Improve My LinkedIn"):
        if resume_file:
            with st.spinner("⏳ Generating your LinkedIn suggestions..."):
                feedback = generate_linkedin_feedback(about_text, resume_file, st.session_state.get("role", ""))
                st.markdown(feedback, unsafe_allow_html=True)
        else:
            st.warning("Please upload your resume first.")

    # 🔴 Clear All button for Tab 1
    if st.button("🗑️ Clear All (Main Tab)"):
        clear_all_main_tab()
        st.experimental_rerun()


# ---------------------- TAB 2 ----------------------
with tab2:
    st.subheader("📌 Match Resume with Job Description")
    shared_resume = st.file_uploader("Upload resume for matching (PDF)", type="pdf", key="shared_resume")

    jd_text = st.text_area("📋 Paste Job Description Here", height=200, key="jd_text")
    if st.button("🔍 Match with JD"):
        if shared_resume and jd_text:
            with st.spinner("⏳ Matching your resume with JD..."):
                jd_result = match_resume_with_jd(shared_resume, jd_text)
                st.markdown(jd_result, unsafe_allow_html=True)
        else:
            st.warning("Please upload resume and enter job description.")

    if st.button("📥 Download JD-Tailored Resume"):
        if shared_resume and jd_text:
            with st.spinner("⏳ Generating JD-tailored resume..."):
                jd_pdf_path, jd_msg = rewrite_resume_for_jd(shared_resume, jd_text)
                if jd_pdf_path:
                    with open(jd_pdf_path, "rb") as f:
                        st.download_button("📥 Download JD Resume", f, file_name="JD_Tailored_Resume.pdf")
                else:
                    st.error(jd_msg)

    st.markdown("---")
    st.subheader("🌐 Job Explorer")
    if st.button("🔍 Explore Job Suggestions"):
        if shared_resume:
            with st.spinner("⏳ Finding jobs based on your resume..."):
                explore_text = generate_job_explorer_output(shared_resume)
                st.markdown(explore_text, unsafe_allow_html=True)
        else:
            st.warning("Please upload resume to explore jobs.")

    # 🔴 Clear All button for Tab 2
    if st.button("🗑️ Clear All (JD Tab)"):
        clear_all_jd_tab()
        st.experimental_rerun()


# Optional footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("✨ Built with 💻 ML + GPT | Made for the AI Challenge", unsafe_allow_html=True)



import gradio as gr
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


def clear_fields():
    return None, "", "","","","",None,"",""

def show_loading_linkedin():
    return "⏳ Generating your LinkedIn suggestions... Please wait."

def hide_loading_linkedin():
    return ""

def show_main_loading():
    return "⏳ Preparing your resume... Please wait."

def show_main_file(file_path):
    return gr.update(value=file_path, visible=True)

def hide_main_loading():
    return ""  # Clears the status message

def show_loading_jd():
    return "⏳ Matching in progress..."

def hide_loading_jd():
    return ""

def show_loading():
    return "⏳ Looking for jobs based on your resume…"

def hide_loading():
    return " "  # Clears the status message


def clear_jd_fields():
    # Clears the JD Tab fields
    return None, "", ""," ",None  # Corresponds to shared_resume_file, jd_text_input, jd_output

def clear_explore_fields():
    # Clears the Job Explorer Tab fields
     return None, "" , " "# Corresponds to shared_resume_file, explore_output

def extract_resume_text(resume_file):
    try:
        doc = fitz.open(resume_file.name)
        return " ".join([page.get_text() for page in doc]).strip()
    except:
        return ""
def show_chat_ui():
    return gr.update(visible=True)

with gr.Blocks(css="""
/* ✅ Set consistent light background */
body {
    background-color: #f7fafc !important;
}

.gradio-container {
    font-family: 'Segoe UI', sans-serif;
    max-width: 960px;
    margin: auto;
    padding: 30px;
    background-color: #ffffff;
    border-radius: 16px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
}

/* ✅ Tab Styling */
.tabs, .tab-nav, .tabitem, .tab-nav button {
    background-color: #eaf4fc !important;
    color: #222 !important;
}
.tab-nav button.selected {
    background-color: #d0ebff !important;
    border-radius: 6px 6px 0 0 !important;
    margin-right: 8px !important;
    font-weight: bold;
}

/* ✅ Textareas and Inputs */
textarea, input {
    border-radius: 8px !important;
    padding: 10px;
    font-size: 15px;
    background-color: #f9f9f9 !important;
    border: 1px solid #ccc !important;
    color: #333 !important;
}

/* ✅ Upload Box Styling */
.gr-file-upload, .gr-file, div[data-testid="file"], .wrap.svelte-1ipelgc, .wrap.svelte-1u7sq69, .wrap.svelte-pc1qv7 {
    background-color: #ffffff !important;
    border: 2px dashed #90caf9 !important;
    border-radius: 10px !important;
    color: #999 !important;
    padding: 12px !important;
    font-size: 15px !important;
    min-height: 80px !important;
    box-shadow: none !important;
}

/* ✅ Compact Upload Size — override spacing if needed */
.compact-upload {
    background-color: #ffffff !important;
    border: 2px dashed #90caf9 !important;
    border-radius: 10px !important;
    padding: 12px !important;
    min-height: 80px !important;
    font-size: 15px !important;
    color: #666 !important;
}

/* ✅ Remove "Drop file here" and other unwanted visuals */
.gr-file-upload * span,
.gr-file-upload .icon,
.gr-file-upload .file-preview,
.gr-file-upload .upload-box,
.gr-file-upload svg,
.gr-file-upload label,
.gr-file-upload .upload-box__drag,
.gr-file-upload .upload-box__label {
    display: none !important;
}

/* ✅ Resume Output Boxes */
textarea[aria-label*="Resume Strength"],
textarea[aria-label*="Predicted Job Role"] {
    background-color: #fff !important;
    border: 1px solid #ddd !important;
    color: #111 !important;
}

/* ✅ Button Styling */
.gr-button-row {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    margin-top: 12px;
}
button {
    background-color: #007acc !important;
    color: white !important;
    font-weight: 600;
    border-radius: 8px !important;
    padding: 10px 16px !important;
    flex: 1;
}
button.secondary {
    background-color: #cbd5e0 !important;
    color: #2d3748 !important;
}

/* ✅ Markdown Styling */
.gr-markdown {
    font-size: 16px;
    color: #1a202c;
}


""") as demo:




    gr.Markdown("## <span style='font-size:28px'>🚀 <b>PATH FORGE AI</b></span>")
    gr.Markdown("<p style='text-align:center'>Your personal AI-powered resume coach: analyze, improve, and grow.</p>")

    with gr.Tabs():
        # 🔹 Tab 1: Resume + LinkedIn
        with gr.Tab("🏠 Main"):

            resume_file = gr.File(label="📄 Upload Resume", file_types=[".pdf"], elem_classes="compact-upload")

            with gr.Row(elem_classes="gr-button-row"):
                submit_btn = gr.Button("🔍 Analyze My Resume")
                clear_btn = gr.Button("🗑️ Clear All", variant="secondary")
                download_main_btn = gr.Button("📥 Download AI-Enhanced Resume")
                status_text_main = gr.Markdown(" ", visible=True)
            main_pdf_file = gr.File(visible=True,elem_classes="compact-upload")



            gr.Markdown("---")

            with gr.Row():
                with gr.Column(scale=1):
                    strength_output = gr.Textbox(label="💪 Resume Strength")
                with gr.Column(scale=1):
                    role_output = gr.Textbox(label="🧩 Predicted Job Role")

            tips_output = gr.Markdown(label="🛠️ AI Resume Feedback")

            gr.Markdown("---")
            gr.Markdown("### 💼 LinkedIn Enhancer", elem_id="linkedin-header")

            about_input = gr.Textbox(label="🔗 Paste your LinkedIn 'About Me' (Optional)", lines=4, placeholder="Paste it here (or leave blank)")
            linkedin_btn = gr.Button("✨ Improve My LinkedIn Presence")
            linkedin_output = gr.Markdown(label="🧠 LinkedIn Suggestions")
            status_text = gr.Markdown(" ", visible=True)


        # 🔹 Tab 2: JD Match + Job Explorer Unified

        with gr.Tab("🎯 Job Fit & Role Explorer"):

            # One resume upload for both sections
            gr.Markdown("## 📄 Upload Your Resume")
            shared_resume_file = gr.File(label="📄 Upload Resume ", file_types=[".pdf"], elem_classes="compact-upload")

            # -------- JD Matching Section --------
            gr.Markdown("## 📌 Match Resume with Job Description")


            jd_text_input = gr.Textbox(
                label="📋 Paste Job Description",
                lines=6,
                placeholder="Paste the full job description here..."
            )
            jd_status = gr.Markdown(" ", visible=True) # Status specific to JD Match
            jd_output = gr.Markdown(label="🧠 JD Match Insights")


            with gr.Row():
                jd_match_btn = gr.Button("🔍 Match Resume with JD")
                jd_clear_btn = gr.Button("🗑️ Clear JD Fields", variant="secondary") # Changed text
                download_jd_btn = gr.Button("📥 Download JD-Tailored Resume")



            jd_pdf_file = gr.File(visible=True,elem_classes="compact-upload")  # Keep outside row


            # -------- Job Explorer Section --------
            gr.Markdown("## 🌐 Job Explorer – Find Your Best Fit")
            explore_status = gr.Markdown(" ", visible=True) # Status specific to Job Explorer
            explore_output = gr.Markdown(label="🧭 Suggested Roles + Job Links")


            with gr.Row():
                explore_btn = gr.Button("🔍 Explore Job Roles")
                clear_explore_btn = gr.Button("🗑️ Clear Explorer Fields", variant="secondary") # Added Clear button for Explorer

            # ✅ UI: Career Chat Agent
            gr.Markdown("## 🧠 Career Agent Support")
            career_chat_btn = gr.Button("💬 Chat with Career Agent")

            chat_section = gr.Column(visible=False)  # Hidden by default

            with chat_section:
                career_chatbot = gr.Chatbot(label="Your AI Career Guide",type="messages")
                text_input = gr.Textbox(label="💬 Ask your question")
                send_btn = gr.Button("🧠 Send")




    # Define button clicks *inside* the gr.Blocks context
    submit_btn.click(
        fn=process_resume,
        inputs=resume_file,
        outputs=[strength_output, role_output, tips_output]
    )

    clear_btn.click(
        fn=clear_fields,
        inputs=[],
        outputs=[resume_file, strength_output, role_output, tips_output, about_input, linkedin_output,main_pdf_file, status_text_main, status_text]
    )

    linkedin_btn.click(
        fn=show_loading_linkedin,
        inputs=[],
        outputs=[status_text]
    ).then(
        fn=generate_linkedin_feedback,
        inputs=[about_input, resume_file, role_output], # Use resume_file from Tab 1
        outputs=[linkedin_output]
    ).then(
        fn=hide_loading_linkedin,
        inputs=[],
        outputs=[status_text]
    )

    def rewrite_main_flow(resume_file, strength, role, tips):
         path, msg = rewrite_resume_main(resume_file, strength, role, tips)
         print("✅ PDF Path:", path)
         if path:
          return gr.update(value=path, visible=True, interactive=True), msg
         else:
          return gr.update(visible=False), msg  # Handle error case

    download_main_btn.click(
        fn=show_main_loading,
        inputs=[],
        outputs=[status_text_main]
    ).then(
       fn=rewrite_main_flow,
       inputs=[resume_file, strength_output, role_output, tips_output],
       outputs=[main_pdf_file, status_text_main]
    ).then(
       fn=hide_main_loading,
       inputs=[],
       outputs=[status_text_main]
    )



    # JD Match Clicks
    jd_match_btn.click(
        fn=show_loading_jd,
        inputs=[],
        outputs=[jd_status] # Update JD status
    ).then(
        fn=match_resume_with_jd,
        inputs=[shared_resume_file, jd_text_input], # Use shared_resume_file from Tab 2
        outputs=[jd_output]
    ).then(
        fn=hide_loading_jd,  # hide status
        inputs=[],
        outputs=[jd_status] # Update JD status
    )
    jd_clear_btn.click(
       fn=clear_jd_fields,
       inputs=[],
       outputs=[shared_resume_file, jd_text_input, jd_output, jd_status, jd_pdf_file]
   )



    # Processing flow
    explore_btn.click(
     fn=show_loading,
     inputs=[],
     outputs=[explore_status]
   ).then(
      fn=generate_job_explorer_output,
      inputs=[shared_resume_file],
      outputs=[explore_output]
   ).then(
      fn=hide_loading,
      inputs=[],
      outputs=[explore_status]
   )
    clear_explore_btn.click(
      fn=clear_explore_fields,
      inputs=[],
      outputs=[shared_resume_file, explore_output, explore_status]
   )
    def rewrite_jd_flow(resume_file, jd_text_input):
       path, msg = rewrite_resume_for_jd(resume_file, jd_text_input)
       print("✅ JD PDF Path:", path)
       return gr.update(value=path, visible=True, interactive=True), msg

    download_jd_btn.click(
      fn=lambda: "⏳ Preparing tailored resume for JD... please wait.",
      inputs=[],
      outputs=[jd_status]
   ).then(
      fn=rewrite_jd_flow,
      inputs=[shared_resume_file, jd_text_input],
      outputs=[jd_pdf_file, jd_status]
   ).then(
      fn=lambda: "",  # hide status
      inputs=[],
      outputs=[jd_status]
   )

    career_chat_btn.click(
     fn=show_chat_ui,
     inputs=[],
     outputs=[chat_section]
   )
    send_btn.click(
      fn=chat_with_career_agent,
      inputs=[career_chatbot, text_input, shared_resume_file],  # shared_resume_file from JD tab
      outputs=career_chatbot
   ).then(
      fn=lambda: "",  # clear the text input
      inputs=[],
      outputs=[text_input]
   )




    gr.Markdown(
        "<p style='text-align:center; font-size: 14px;'>✨ Built with 💻 ML + GPT | Made for the AI Challenge</p>")

demo.launch()

if __name__ == "__main__":
    demo.launch()

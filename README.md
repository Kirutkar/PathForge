# PathForge
It is an AI powered career assistant that analyzes resumes, predict roles and offers personalized growth suggesstions.

# 🚀 PathForge AI – Your Personalized Resume & Career Coach

PathForge AI is an intelligent career assistant that helps users:
- Analyze their resume strength
- Predict suitable job roles
- Get structured AI feedback for improvement
- Rewrite resumes using GPT
- Match their resume with job descriptions
- Improve LinkedIn branding
- Explore job opportunities based on their experience level

---

## ✨ Key Features

- 📄 **Resume Analyzer** using ML (XGBoost)
- 🧠 **Role Prediction & AI Feedback** via OpenAI GPT-4o
- 📌 **JD Matching** with GPT-guided suggestions
- 💼 **LinkedIn Enhancer** with RAG-powered content tips
- 🌐 **Job Explorer** – suggests roles + real job links (Indeed UAE)
- 💬 **Conversational Career Agent** powered by GPT-4o
- 📥 **PDF Resume Rewriter** (main + JD-matched versions)

---

## 🔍 Technologies Used

| Component              | Technology        |
|------------------------|-------------------|
| Resume Strength Model  | `XGBoost`         |
| Resume Vectorization   | `TF-IDF`          |
| Resume Feedback & NLP  | `OpenAI GPT-4o`   |
| UI Framework           | `Gradio`          |
| Resume Parsing         | `PyMuPDF (fitz)`  |
| PDF Generation         | `FPDF`            |
| Dataset Handling       | `Pandas`, `NumPy` |

---

## 🤖 ML Model Details

- Model: `XGBoost Classifier`
- Accuracy: `~92%`
- Labels: Strong, Average, Weak
- Preprocessing:
  - Real resume data from Kaggle
  - Extended with synthetic resumes for non-technical roles
  - Balanced across domains like AI, Sales, Education, HR, Law
  - Feature extraction using `TF-IDF` and keyword-based logic

---

## 🧠 LLM Capabilities (GPT-4o)

- **Predicts job roles** from resume text
- **Gives resume improvement suggestions** in markdown
- **Identifies missing keywords**, certifications, formatting issues
- **Rewrites resumes** cleanly (main + job-specific versions)
- **Improves LinkedIn ‘About Me’**
- **Generates post ideas + visibility tips**
- **Answers career growth queries** via chatbot interface

---

## 🚀 Demo

Try it on [Hugging Face Spaces]https://huggingface.co/spaces/Kiruthikaramalingam/PathForgeAI
_(replace with your live URL)_

---

## 🛠️ How to Run Locally

1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/pathforge-ai.git
   cd pathforge-ai
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   python app.py
   ```

4. (Optional) Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY=your-key-here
   ```

---

## 📁 Folder Structure

```
├── app.py                  # Main Gradio app
├── xgb_resume_model.pkl    # ML model
├── tfidf_vectorizer.pkl    # Vectorizer for resumes
├── resume_dataset.csv      # Training data
├── FinalLinkedInPostIdeas.csv  # RAG data for LinkedIn suggestions
├── requirements.txt
└── README.md
```

---

## 🧑‍💼 Author

Built with ❤️ by Kiruthika Ramalingam
Connect with me on [LinkedIn]www.linkedin.com/in/kiruthika-ramalingam

---

## 📢 Note

This project was built as part of the **Decoding Data Science AI Challenge**.  
It’s a showcase of how ML + LLMs can assist real-world career growth in an ethical, supportive way.

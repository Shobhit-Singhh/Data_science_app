from pathlib import Path
import streamlit as st
import pandas as pd
from PIL import Image
import os

st.set_page_config(layout="wide",initial_sidebar_state="auto")


# --- GENERAL SETTINGS ---
PAGE_TITLE = "Your Digital CV"
PAGE_ICON = ":wave:"
NAME = "Shobhit Kumar Singh"


EMAIL = "shobhit22iit@gmail.com"

SOCIAL_MEDIA = {
    "LinkedIn": "https://www.linkedin.com/in/shobhit-singhh/",
    "GitHub": "https://github.com/Shobhit-Singhh",
    "Twitter": "https://twitter.com/SinghhShob84657",
}

PROJECTS = {
    "🏆 Data Analysis Tool: No code All in one Data Analysis and Machine Learning Tool" : " ",
    "🏆 Advance Data Analysis Tool: Correlation and Hypothesis Testing": " ",
    "🏆 Web Scraping for Keyword Extraction and Sentiment Analysis": "https://github.com/Shobhit-Singhh/scraper",
    "🏆 NLP Project: AI-based Question Generation": "https://github.com/Shobhit-Singhh/NLP_ques_gen",
    "🏆 Querying SQL Database Using Generative AI": " "
}

# --- LOAD PROFILE PIC ---
profile_pic = Image.open(os.path.join('shobhit.png'))
banner = Image.open(os.path.join('iitg.jpg'))

# --- HERO SECTION ---
col1, col2 = st.columns(2, gap="small")
st.image(profile_pic, width=530)


st.title(NAME)
st.write(
    """
-    Machine Learning and Data Science Student
-    Seeking Full-Time ML ops / Data Science Job
-    Love to Teach and Share Knowledge in domain of
-    Machine learning, Deep Learning, Data Analysis, NLP, Python, C++, SQL
    """
)
st.write("📫", EMAIL)

# --- SOCIAL LINKS ---
st.write('\n')
cols = st.columns(len(SOCIAL_MEDIA))
for platform, link in SOCIAL_MEDIA.items():
    cols[0].write(f"[{platform}]({link})")

# --- EXPERIENCE & QUALIFICATIONS ---
st.write('\n')
st.subheader("Experience & Qulifications")
st.write(
    """
- ✔️ Expereinced in extracting actionable insights from data
- ✔️ Strong hands on experience and knowledge in Python, SQL and C/C++
- ✔️ Good understanding of statistical principles and Hypothesis Testing 
- ✔️ Closly worked in the field of Machine Learning and Natural Language Processing
- ✔️ Excellent team-player and displaying strong sense of initiative on tasks
- ✔️ Excels in clear communication and teaching, simplifying complex concepts effortlessly.
"""
)


# --- SKILLS ---
st.write('\n')
st.subheader("Hard Skills")
st.write(
    """
- 👩‍💻 Programming: Python (Scikit-learn, Pandas, TesnorFlow, OpenCV, SpaCy), SQL, C++ 
- 📊 Data Visulization: PowerBi, Tableau, Plotly
- 📚 Modeling: Logistic regression, linear regression, SVM, KNN, decition trees, ensemble techniques
- 📈 Statistical Analysis: Hypothesis Testing, A/B Testing
- 🧑🏻‍💻 Deployment: Streamlit, Mlops, GitHub Action, Docker
"""
)

# --- Projects & Accomplishments ---
st.write('\n')
st.subheader("Projects & Accomplishments")
st.write("---")
for project, link in PROJECTS.items():
    st.write(f"[{project}]({link})")

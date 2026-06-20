
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import numpy as np
import joblib
import pdfplumber
import io
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

model = SentenceTransformer("all-MiniLM-L6-v2")

classifier = joblib.load("best_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

job_embeddings = np.load("job_embeddings.npy")

jobs_df = pd.read_csv("jobs_unique.csv")

SKILLS = [
    "python","java","javascript","html","css",
    "react","nodejs","sql","mongodb",
    "machine learning","deep learning",
    "nlp","aws","docker","git",
    "communication","leadership",
    "management","teamwork"
]

def clean_text(text):
    text = re.sub(r'http\\S+','',str(text))
    text = re.sub(r'\\S+@\\S+','',text)
    text = re.sub(r'[^a-zA-Z\\s]',' ',text)
    return re.sub(r'\\s+',' ',text).lower()

def extract_skills(text):
    text = text.lower()
    return [s for s in SKILLS if s in text]

@app.get("/")
def root():
    return {"message":"AI Resume Analyzer Running"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    pdf_bytes = await file.read()

    text = ""

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""

    cleaned = clean_text(text)

    skills = extract_skills(cleaned)

    embedding = model.encode([cleaned[:500]])

    pred = classifier.predict(embedding)[0]

    category = label_encoder.inverse_transform([pred])[0]

    scores = cosine_similarity(
        embedding,
        job_embeddings
    )[0]

    top5 = scores.argsort()[::-1][:5]

    matches = []

    for idx in top5:
        matches.append({
            "job_title":
            jobs_df.iloc[idx]["job_title"],

            "score":
            round(float(scores[idx])*100,2)
        })

    return {
        "predicted_category": category,
        "skills": skills,
        "matches": matches
    }

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import re
import google.generativeai as genai
from utils import extract_text_from_pdf
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import uuid, json, os

 
app = FastAPI()
 
# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
 
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384
category_indexes = {}  # category: (faiss index, chunk store)
 
# Gemini API setup
genai.configure(api_key="AIzaSyDzf2NKO7x3ff28z542P_fwQvqOwgTgjB4")
model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
 
def get_or_create_index(category):
    if category not in category_indexes:
        category_indexes[category] = (faiss.IndexFlatL2(dimension), [])
    return category_indexes[category]
 
def chunk_and_index(text, category, chunk_size=500, overlap=100):
    index, chunk_store = get_or_create_index(category)
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size].strip().replace("\n", " ")
        if chunk:
            vector = embedding_model.encode([chunk])
            index.add(np.array(vector).astype("float32"))
            chunk_store.append(chunk)
 
@app.on_event("startup")
async def load_documents_on_startup():
    for root, _, files in os.walk(UPLOAD_DIR):
        for file in files:
            if file.endswith(".pdf"):
                file_path = Path(root) / file
                relative_category = Path(root).relative_to(UPLOAD_DIR).as_posix()

                try:
                    text = extract_text_from_pdf(str(file_path))
                    if text.strip():
                        chunk_and_index(text, relative_category)
                        print(f"✅ Indexed: {relative_category}/{file}")
                    else:
                        print(f"⚠️ Empty: {relative_category}/{file}")
                except Exception as e:
                    print(f"❌ Failed: {relative_category}/{file} - {e}")

 
# ✅ Upload Endpoint
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), category: str = Form(...)):
    try:
        # Create the nested folder if it doesn't exist
        category_path = UPLOAD_DIR / category
        category_path.mkdir(parents=True, exist_ok=True)

        # Save the file
        file_path = category_path / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # ✅ Immediately index the uploaded PDF
        text = extract_text_from_pdf(str(file_path))
        if text.strip():
            chunk_and_index(text, category)
            print(f"✅ Indexed uploaded file: {category}/{file.filename}")
        else:
            print(f"⚠️ Uploaded file is empty: {category}/{file.filename}")

        return JSONResponse(content={"message": f"File uploaded to {category}."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Upload failed: {str(e)}"})

    
# ✅ Serve Uploaded Files
@app.get("/files/{category:path}/{filename}")
async def get_uploaded_file(category: str, filename: str):
    file_path = UPLOAD_DIR / category / filename
    if not file_path.exists():
        return JSONResponse(status_code=404, content={"message": "File not found."})
    return FileResponse(file_path)
 
@app.post("/ask/")
async def ask_question(question: str = Form(...), category: str = Form(...)):
    if category not in category_indexes:
        return {"answer": "No documents indexed for this category. Please upload a PDF."}
 
    index, chunk_store = category_indexes[category]
    question_vec = embedding_model.encode([question]).astype("float32")
    D, I = index.search(question_vec, 5)
    relevant_chunks = [chunk_store[i] for i in I[0] if i < len(chunk_store)]
    context = "\n\n".join(relevant_chunks)
 
    prompt = f"""
You are a helpful assistant. Use only the following document content to answer the question below.
 
Answer clearly and cleanly using:
- Bullet points with "•"
- No Markdown (no *, **, or symbols like ` or >)
- No formatting characters, just plain readable text
 
### DOCUMENT START
{context}
### DOCUMENT END
 
Question: {question}
Answer:
"""
 
    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
 
        cleaned = re.sub(r"\*+", "", raw)
        cleaned = re.sub(r"`+", "", cleaned)
        cleaned = re.sub(r"_{2,}", "", cleaned)
        cleaned = re.sub(r"•\s*•", "•", cleaned)
        cleaned = cleaned.replace("- ", "• ")  # make all bullets consistent
        cleaned = re.sub(r"\n\s*\n", "\n\n", cleaned)  # spacing between paragraphs
        cleaned = re.sub(r"^\s*•", "•", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*:\s*", ": ", cleaned)
        cleaned = re.sub(r"\s*•\s*", "\n• ", cleaned)  # force each bullet to new line
        cleaned = re.sub(r"(Answer:)", r"\1\n\n", cleaned)
 
 
        cleaned = cleaned.strip()
 
 
        return {"answer": cleaned}
    except Exception as e:
        return {"answer": f"Error calling Gemini: {e}"}
 
@app.post("/generate_question/")
async def generate_question(category: str = Form(...)):
    if category not in category_indexes:
        return {"question": "No documents indexed for this category. Please upload a PDF."}
    _, chunk_store = category_indexes[category]
    combined_text = "\n\n".join(chunk_store)[-3000:]
    prompt = f"""
You are an educational AI tutor.
Generate **one** standalone quiz question that tests understanding of the following material.
Do NOT start with 'Based on the document...' or similar.
=== CONTENT START ===
{combined_text}
=== CONTENT END ===
Question:"""
    try:
        response = model.generate_content(prompt)
        return {"question": response.text.strip()}
    except Exception as e:
        return {"question": f"Error generating question: {e}"}
 
user_answers = []
 
@app.post("/submit_answer/")
async def submit_answer(question: str = Form(...), answer: str = Form(...)):
    user_answers.append({"question": question, "answer": answer})
    return {"status": "saved"}
 
@app.post("/evaluate_answer/")
async def evaluate_answer(question: str = Form(...), answer: str = Form(...), category: str = Form(...)):
    if category not in category_indexes:
        return {"evaluation": "No documents indexed for this category."}
    _, chunk_store = category_indexes[category]
    combined_text = "\n\n".join(chunk_store)[-3000:]
    prompt = f"""
You are a teacher assistant. Evaluate the user's answer to the question below, using the provided content as the source of truth.
=== STUDY MATERIAL ===
{combined_text}
=== END ===
Question: {question}
User's Answer: {answer}
Feedback:
"""
    try:
        response = model.generate_content(prompt)
        return {"evaluation": response.text.strip()}
    except Exception as e:
        return {"evaluation": f"Error evaluating: {e}"}
 
# ✅ List Uploaded Files (Group by class/subject)
@app.get("/uploaded_files/")
async def list_uploaded_files():
    files_by_category = {}
    for root, dirs, files in os.walk("uploads"):
        rel_path = os.path.relpath(root, "uploads")
        if rel_path == ".":
            continue
        key = rel_path.replace("\\", "/")
        files_by_category[key] = files
    return {"files_by_category": files_by_category}





from fastapi.responses import StreamingResponse
from fpdf import FPDF
import io
 
# --- Utility Functions ---
def clean_extracted_text(text: str) -> str:
    text = re.sub(r"/[A-Z]+\d+", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text

def sanitize_text(text: str) -> str:
    return text.encode("latin-1", "replace").decode("latin-1")

# --- Main Endpoint ---
@app.post("/generate_pdf")
async def generate_pdf(data: dict):
    category = data["category"]
    file_name = data["file"]
    level = data.get("level", "medium")

    counts = {
        "mcq": int(data.get("mcq", 0)),
        "fill": int(data.get("fill", 0)),
        "short": int(data.get("short", 0)),
        "long": int(data.get("long", 0)),
    }

    file_path = UPLOAD_DIR / category / file_name
    if not file_path.exists():
        return JSONResponse(status_code=404, content={"error": "File not found."})

    raw_text = extract_text_from_pdf(str(file_path))
    if not raw_text.strip():
        return JSONResponse(status_code=400, content={"error": "No extractable content in PDF."})

    # Clean and skip non-topic sections
    text = clean_extracted_text(raw_text)
    lines = text.split('\n')
    skip_phrases = ["acknowledgment", "copyright", "preface", "author",
                    "published", "isbn", "editor", "introduction", "dedication"]

    content_lines = [line.strip() for line in lines
                     if len(line.strip()) > 20 and not any(skip in line.lower() for skip in skip_phrases)]

    cleaned_text = "\n".join(content_lines)
    trimmed_text = cleaned_text[1000:4000] if len(cleaned_text) > 4000 else cleaned_text

    all_questions = []
    generated_questions = set()

    for qtype, count in counts.items():
        for i in range(count):
            # === Prompt per question type ===
            if qtype.lower() == "fill":
                prompt = f"""
You are an exam question generator.

TASK:
Generate a meaningful FILL IN THE BLANKS question #{i+1} at {level.upper()} difficulty.
Insert exactly one blank using '__________' in place of a key term or concept.

Avoid ending the sentence unnaturally. Keep it educational and focused on topic content.

CONTENT:
---------------------
{trimmed_text}
---------------------
Only return the question text. No answers or hints.
"""
            elif qtype.lower() == "long":
                prompt = f"""
You are an exam question generator.

TASK:
Generate a LONG ANSWER QUESTION #{i+1} at {level.upper()} difficulty.
- Make it unique.
- Limit it to **2 lines maximum**.
- Avoid repeating earlier questions.

CONTENT:
---------------------
{trimmed_text}
---------------------
Only return the question text.
"""
            elif qtype.lower() == "short":
                prompt = f"""
You are an exam question generator.

TASK:
Generate a SHORT ANSWER QUESTION #{i+1} at {level.upper()} difficulty.
Keep the question brief and clear (1–2 lines).

CONTENT:
---------------------
{trimmed_text}
---------------------
Only return the question.
"""
            elif qtype.lower() == "mcq":
                prompt = f"""
            You are an exam question generator.

            TASK:
            Generate one multiple-choice question at {level.upper()} difficulty from the content below.

            The output should be structured like this (exactly):

            Question: <your question text here>
            A. Option A
            B. Option B
            C. Option C
            D. Option D

            ⚠️ Rules:
            - Do NOT include the correct answer
            - Do NOT explain the question or options
            - Keep question short and relevant to the content

            CONTENT:
            ---------------------
            {trimmed_text}
            ---------------------
            Return only the question and 4 options in the exact format shown.
            """
            else:
                prompt = f"Generate a unique question from this content:\n{trimmed_text}"

            # === Generate Question and Avoid Duplicates ===
            try:
                response = model.generate_content(prompt)
                question = response.text.strip()

                # Add blank if missing for FIB
                if qtype.lower() == "fill" and "____" not in question:
                    if len(question.split()) > 5:
                        question = " ".join(question.split()[:-1]) + " __________"
                    else:
                        question += " __________"

                # Retry if duplicate
                attempt = 0
                while question in generated_questions and attempt < 2:
                    response = model.generate_content(prompt)
                    question = response.text.strip()
                    attempt += 1

                if not question or question in generated_questions:
                    question = f"[No unique question returned for {qtype.upper()} #{i+1}]"
                else:
                    generated_questions.add(question)

            except Exception as e:
                question = f"[Error generating {qtype.upper()} #{i+1}]: {e}"

            all_questions.append((qtype.upper(), question))

    # === Build PDF ===
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for idx, (qtype, question) in enumerate(all_questions, 1):
        safe_text = sanitize_text(f"{idx}. [{qtype}] {question}")
        pdf.multi_cell(0, 10, safe_text)
        pdf.ln(2)

    pdf_output = pdf.output(dest='S').encode('latin1')
    pdf_bytes = io.BytesIO(pdf_output)

    return StreamingResponse(pdf_bytes, media_type="application/pdf", headers={
        "Content-Disposition": f"attachment; filename=Question_Paper_{category}_{level}.pdf"
    })


 
from fastapi import Query
 
# ✅ Delete Uploaded File
@app.delete("/delete_file")
async def delete_file(filename: str, category: str):
    file_path = UPLOAD_DIR / category / filename
    if not file_path.exists():
        return JSONResponse(status_code=404, content={"message": "File not found."})
    try:
        file_path.unlink()
        return JSONResponse(content={"message": f"{filename} deleted from {category}."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error deleting file: {str(e)}"})
 


# Lesson Plan Builder
 
from fastapi import Request
from fastapi.responses import StreamingResponse
from fpdf import FPDF
import io
 
@app.post("/generate_lesson_plan")
async def generate_lesson_plan(data: dict):
    subject = data.get("subject", "")
    grade = data.get("grade", "")
    standard = data.get("standard", "")
    objective = data.get("objective", "")
    activities = data.get("activities", "")
    assessment = data.get("assessment", "")
    template = data.get("template", "detailed")
 
    prompt = f"""
You are an educational expert. Create a {template} lesson plan using the following information.
 
Subject: {subject}
Grade Level: {grade}
Curriculum Standard: {standard}
Learning Objective: {objective}
Teaching Activities: {activities}
Assessment Methods: {assessment}
 
The lesson plan should include labeled sections and use clear formatting.
"""
 
    try:
        response = model.generate_content(prompt)
        return {"plan": response.text.strip()}
    except Exception as e:
        return {"plan": f"Error: {e}"}
 
 
@app.post("/generate_lesson_plan_pdf")
async def generate_lesson_plan_pdf(request: Request):
    try:
        data = await request.json()
        plan_text = data.get("plan", "").strip()
 
        if not plan_text:
            return JSONResponse(status_code=400, content={"error": "No content to generate."})
 
        class LessonPlanPDF(FPDF):
            def header(self):
                self.set_font("Arial", "B", 14)
                self.cell(0, 10, "Lesson Plan", ln=True, align="C")
                self.ln(5)
 
            def chapter_title(self, title):
                clean_title = re.sub(r"[*\-•]+", "", title).strip()
                self.set_font("Arial", "B", 12)
                self.multi_cell(0, 10, clean_title)
                self.ln(2)

            def chapter_body(self, body):
                clean_body = re.sub(r"[*\-•]+", "", body).strip()
                safe_text = clean_body.encode("latin-1", "replace").decode("latin-1")
                self.set_font("Arial", "", 11)
                self.multi_cell(0, 8, safe_text)
                self.ln(3)

        pdf = LessonPlanPDF()
        pdf.add_page()

         # Clean up markdown formatting from the plan text
        plan_text = re.sub(r"\*\*(.*?)\*\*", r"\1", plan_text)  # remove bold markers
        plan_text = re.sub(r"\*(.*?)\*", r"\1", plan_text)      # remove italics
        plan_text = re.sub(r"__([^_]+)__", r"\1", plan_text)     # remove __ underline
        plan_text = re.sub(r"`([^`]+)`", r"\1", plan_text)       # remove code backticks
 
        for section in plan_text.split("\n\n"):
            if ":" in section:
                title, content = section.split(":", 1)
                pdf.chapter_title(title)
                pdf.chapter_body(content)
            else:
                pdf.chapter_body(section)
 
        # ✅ Fix: use `dest='S'` to get PDF output as bytes
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        buffer = io.BytesIO(pdf_bytes)
 
        return StreamingResponse(buffer, media_type="application/pdf", headers={
            "Content-Disposition": "attachment; filename=lesson_plan.pdf"
        })
 
    except Exception as e:
        print("❌ PDF generation error:", e)
        return JSONResponse(status_code=500, content={"error": f"PDF generation failed: {str(e)}"})



import requests
from fastapi import Query
 
# 🔐 Set your YouTube API key here
YOUTUBE_API_KEY = "AIzaSyBShu90YRJ-VT7ox_LyXyO19EqpGdkgHcY"
 
@app.get("/real_youtube_links/")
def get_real_youtube_links(q: str = Query(...), max_results: int = 7):
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": q,
        "key": YOUTUBE_API_KEY,
        "maxResults": max_results,
        "type": "video",
        "safeSearch": "strict"
    }
 
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json().get("items", [])
 
        links = [
            {
                "title": item["snippet"]["title"],
                "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}"
            }
            for item in results
        ]
        return {"results": links}
    except Exception as e:
        return {"results": [], "error": str(e)}
 
from youtube_transcript_api import YouTubeTranscriptApi
import re
 
def extract_video_id(url):
    """
    Extracts the YouTube video ID from a standard YouTube URL.
    """
    match = re.search(r"(?:v=|youtu\.be/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None
 
@app.post("/summarize_youtube")
async def summarize_youtube(data: dict):
    link = data.get("link", "")
    video_id = extract_video_id(link)
 
    if not video_id:
        return {"summary": "❌ Invalid YouTube link format."}
 
    try:
        # ✅ Attempt to get transcript
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([item["text"] for item in transcript_data])[:3500]  # limit for Gemini
 
        # ✅ Summarization prompt
        prompt = f"""
You are an AI tutor. Summarize the key educational points from the transcript below.
 
=== BEGIN TRANSCRIPT ===
{transcript_text}
=== END TRANSCRIPT ===
 
Provide a clear, student-friendly summary (1-2 paragraphs):
"""
 
        response = model.generate_content(prompt)
        return {"summary": response.text.strip()}
 
    except Exception as e:
        return {"summary": f"⚠️ Unable to summarize video. Error: {str(e)}"}
 
 
@app.post("/summarize_text")
async def summarize_text(data: dict):
    input_text = data.get("text", "").strip()
    if not input_text:
        return {"summary": "⚠️ No input text provided."}
 
    prompt = f"""
You are a helpful assistant. Please summarize the following content in a simple, clear, and student-friendly way.
 
=== INPUT ===
{input_text}
=== END ===
 
Summary:
"""
    try:
        response = model.generate_content(prompt)
        return {"summary": response.text.strip()}
    except Exception as e:
        return {"summary": f"❌ Error during summarization: {str(e)}"}
 
 
from fastapi import UploadFile, File, Form
from PIL import Image
import io
 
@app.post("/analyze_image")
async def analyze_image(image: UploadFile = File(...), prompt: str = Form(...)):
    try:
        image_data = await image.read()
        image_pil = Image.open(io.BytesIO(image_data))
 
        # Combine user prompt with image
        full_prompt = prompt if prompt.strip() else "Describe this image in simple terms for a student."
 
        response = model.generate_content([full_prompt, image_pil])
        return {"description": response.text.strip()}
 
    except Exception as e:
        return {"description": f"❌ Error analyzing image: {str(e)}"}
 
from fastapi import FastAPI, UploadFile, File, Form
import json
from pathlib import Path
from passlib.hash import bcrypt
from fastapi.responses import JSONResponse
 
TEACHER_FILE = Path("teachers.json")
 
def load_teachers():
    if not TEACHER_FILE.exists():
        return []
    with open(TEACHER_FILE, "r") as f:
        content = f.read().strip()
        return json.loads(content) if content else []
 
@app.post("/register_teacher/")
async def register_teacher(
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    grade: str = Form(...),
    subject: str = Form(...)
):
    teachers = load_teachers()
 
    if any(t["email"] == email for t in teachers):
        return JSONResponse(status_code=400, content={"message": "Email already registered."})
 
    hashed_password = bcrypt.hash(password)
 
    new_teacher = {
        "name": name,
        "email": email,
        "password": hashed_password,
        "grade": grade,
        "subject": subject
    }
 
    teachers.append(new_teacher)
    with open(TEACHER_FILE, "w") as f:
        json.dump(teachers, f, indent=2)
 
    return {"message": f"Teacher {name} registered successfully!"}
 
@app.post("/login_teacher/")
async def login_teacher(email: str = Form(...), password: str = Form(...)):
    teachers = load_teachers()
 
    user = next((t for t in teachers if t["email"] == email), None)
 
    if user and bcrypt.verify(password, user["password"]):
        return {
            "message": f"Welcome back, {user['name']}!",
            "name": user["name"],
            "email": user["email"],
            "grade": user["grade"],
            "subject": user["subject"]
        }
 
    return JSONResponse(status_code=400, content={"message": "Invalid credentials."})
 
@app.post("/update_profile/")
def update_profile(
    email: str = Form(...),
    name: str = Form(...),
    grade: str = Form(...),
    subject: str = Form(...),
    current_password: str = Form(...),
    new_password: str = Form(...),
):
    teachers = load_teachers()
    for teacher in teachers:
        if teacher["email"].lower() == email.lower():
            if not bcrypt.verify(current_password, teacher["password"]):
                return JSONResponse(status_code=401, content={"message": "❌ Current password is incorrect."})
 
            teacher["name"] = name
            teacher["grade"] = grade
            teacher["subject"] = subject
 
            if new_password:  # only hash and set if new password is provided
                teacher["password"] = bcrypt.hash(new_password)
 
            with open(TEACHER_FILE, "w") as f:
                json.dump(teachers, f, indent=2)
 
            print("✅ Profile updated for:", email)
            return {"message": "Profile updated successfully."}
 
    return JSONResponse(status_code=404, content={"message": "❌ Teacher not found."})
 
# === Ensure profile photo directory exists ===
Path("profile_photos").mkdir(exist_ok=True)
 
# === Mount static folder ===
app.mount("/profile_photos", StaticFiles(directory="profile_photos"), name="profile_photos")
 
@app.post("/upload_profile_photo/")
async def upload_profile_photo(email: str = Form(...), photo: UploadFile = File(...)):
    photo_dir = Path("profile_photos")
    filename = email.replace("@", "_").replace(".", "_") + ".jpg"
    file_path = photo_dir / filename
 
    with open(file_path, "wb") as f:
        shutil.copyfileobj(photo.file, f)
 
    return {"url": f"/profile_photos/{filename}"}
 


@app.post("/chat/")
async def general_chat(question: str = Form(...)):
    prompt = f"You are a helpful, friendly AI tutor. Answer the following question in a clear, student-friendly way:\n\n{question}\n\nAnswer:"

    try:
        response = model.generate_content(prompt)
        return {"answer": response.text.strip()}
    except Exception as e:
        return {"answer": f"Error: {e}"}




@app.post("/generate_text")
async def generate_text(data: dict):
    category = data["category"]
    file_name = data["file"]
    level = data.get("level", "medium")

    counts = {
        "mcq": int(data.get("mcq", 0)),
        "fill": int(data.get("fill", 0)),
        "short": int(data.get("short", 0)),
        "long": int(data.get("long", 0)),
    }

    file_path = UPLOAD_DIR / category / file_name
    if not file_path.exists():
        return JSONResponse(status_code=404, content={"error": "File not found."})

    raw_text = extract_text_from_pdf(str(file_path))
    if not raw_text.strip():
        return JSONResponse(status_code=400, content={"error": "No extractable content in PDF."})

    # Clean and process the text (same logic as PDF)
    text = clean_extracted_text(raw_text)
    lines = text.split('\n')
    skip_phrases = ["acknowledgment", "copyright", "preface", "author",
                    "published", "isbn", "editor", "introduction", "dedication"]

    content_lines = [line.strip() for line in lines
                     if len(line.strip()) > 20 and not any(skip in line.lower() for skip in skip_phrases)]

    cleaned_text = "\n".join(content_lines)
    trimmed_text = cleaned_text[1000:4000] if len(cleaned_text) > 4000 else cleaned_text

    all_questions = []
    generated_questions = set()

    for qtype, count in counts.items():
        for i in range(count):
            if qtype.lower() == "fill":
                prompt = f"""
You are an exam question generator.

TASK:
Generate a meaningful FILL IN THE BLANKS question #{i+1} at {level.upper()} difficulty.
Insert exactly one blank using '__________' in place of a key term or concept.

Avoid ending the sentence unnaturally. Keep it educational and focused on topic content.

CONTENT:
---------------------
{trimmed_text}
---------------------
Only return the question text. No answers or hints.
"""
            elif qtype.lower() == "long":
                prompt = f"""
You are an exam question generator.

TASK:
Generate a LONG ANSWER QUESTION #{i+1} at {level.upper()} difficulty.
- Make it unique.
- Limit it to **2 lines maximum**.
- Avoid repeating earlier questions.

CONTENT:
---------------------
{trimmed_text}
---------------------
Only return the question text.
"""
            elif qtype.lower() == "short":
                prompt = f"""
You are an exam question generator.

TASK:
Generate a SHORT ANSWER QUESTION #{i+1} at {level.upper()} difficulty.
Keep the question brief and clear (1–2 lines).

CONTENT:
---------------------
{trimmed_text}
---------------------
Only return the question.
"""
            elif qtype.lower() == "mcq":
                prompt = f"""
You are an exam question generator.

TASK:
Generate one multiple-choice question at {level.upper()} difficulty from the content below.

The output should be structured like this:

Question: <your question>
A. Option A
B. Option B
C. Option C
D. Option D

CONTENT:
---------------------
{trimmed_text}
---------------------
"""
            else:
                prompt = f"Generate a unique question from this content:\n{trimmed_text}"

            try:
                response = model.generate_content(prompt)
                question = response.text.strip()

                if qtype.lower() == "fill" and "____" not in question:
                    question += " __________"

                attempt = 0
                while question in generated_questions and attempt < 2:
                    response = model.generate_content(prompt)
                    question = response.text.strip()
                    attempt += 1

                if not question or question in generated_questions:
                    question = f"[No unique question returned for {qtype.upper()} #{i+1}]"
                else:
                    generated_questions.add(question)

            except Exception as e:
                question = f"[Error generating {qtype.upper()} #{i+1}]: {e}"

            all_questions.append(f"[{qtype.upper()}] {question}")

    # Join all questions into plain text
    text_output = "\n\n".join(f"{i+1}. {q}" for i, q in enumerate(all_questions))

    return JSONResponse(content={"text": text_output})




 


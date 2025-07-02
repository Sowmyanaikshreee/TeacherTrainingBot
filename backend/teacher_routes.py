from fastapi import APIRouter, Form, UploadFile, File
from fastapi.responses import JSONResponse
from passlib.hash import bcrypt
from pathlib import Path
import shutil
from db import db

router = APIRouter()
Path("profile_photos").mkdir(exist_ok=True)

@router.post("/register_teacher/")
async def register_teacher(
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    grade: str = Form(...),
    subject: str = Form(...)
):
    existing = await db.teachers.find_one({"email": email})
    if existing:
        return JSONResponse(status_code=400, content={"message": "Email already registered."})

    hashed_password = bcrypt.hash(password)
    teacher = {
        "name": name,
        "email": email,
        "password": hashed_password,
        "grade": grade,
        "subject": subject
    }
    await db.teachers.insert_one(teacher)
    return {"message": f"Teacher {name} registered successfully!"}

@router.post("/login_teacher/")
async def login_teacher(email: str = Form(...), password: str = Form(...)):
    user = await db.teachers.find_one({"email": email})
    if user and bcrypt.verify(password, user["password"]):
        return {
            "message": f"Welcome back, {user['name']}!",
            "name": user["name"],
            "email": user["email"],
            "grade": user["grade"],
            "subject": user["subject"]
        }
    return JSONResponse(status_code=400, content={"message": "Invalid credentials."})

@router.post("/upload_profile_photo/")
async def upload_profile_photo(email: str = Form(...), photo: UploadFile = File(...)):
    filename = email.replace("@", "_").replace(".", "_") + ".jpg"
    file_path = Path("profile_photos") / filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(photo.file, f)
    return {"url": f"/profile_photos/{filename}"}

@router.post("/update_profile/")
async def update_teacher_profile(
    name: str = Form(...),
    email: str = Form(...),
    grade: str = Form(...),
    subject: str = Form(...),
    current_password: str = Form(...),
    new_password: str = Form("")
):
    user = await db.teachers.find_one({"email": email})
    if not user or not bcrypt.verify(current_password, user["password"]):
        return JSONResponse(status_code=400, content={"message": "❌ Current password is incorrect."})

    update_data = {
        "name": name,
        "grade": grade,
        "subject": subject,
    }

    if new_password.strip():
        update_data["password"] = bcrypt.hash(new_password)

    await db.teachers.update_one({"email": email}, {"$set": update_data})
    return {"message": "✅ Profile updated successfully."}


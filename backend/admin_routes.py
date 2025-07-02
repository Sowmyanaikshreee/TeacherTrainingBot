from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from passlib.hash import bcrypt
from db import db

router = APIRouter()

@router.post("/register_admin/")
async def register_admin(name: str = Form(...), email: str = Form(...), password: str = Form(...)):
    existing = await db.admins.find_one({"email": email})
    if existing:
        return JSONResponse(status_code=400, content={"message": "Email already registered."})

    hashed_password = bcrypt.hash(password)
    admin = {"name": name, "email": email, "password": hashed_password}
    await db.admins.insert_one(admin)
    return {"message": f"Admin {name} registered successfully!"}

@router.post("/login_admin/")
async def login_admin(email: str = Form(...), password: str = Form(...)):
    user = await db.admins.find_one({"email": email})
    if user and bcrypt.verify(password, user["password"]):
        return {
            "message": f"Welcome back, {user['name']}!",
            "name": user["name"],
            "email": user["email"]
        }
    return JSONResponse(status_code=400, content={"message": "Invalid credentials."})

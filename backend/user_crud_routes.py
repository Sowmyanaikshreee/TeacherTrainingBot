from fastapi import APIRouter, Request, HTTPException, Form
from fastapi.responses import JSONResponse
from bson import ObjectId
from db import db

router = APIRouter()

def serialize_user(doc, role):
    return {
        "id": str(doc.get("_id")),
        "role": role,
        "name": doc.get("name"),
        "email": doc.get("email"),
        "grade": doc.get("grade", ""),
        "subject": doc.get("subject", "")
    }

@router.get("/list_users")
async def list_users(role: str):
    if role == "teacher":
        users = await db.teachers.find().to_list(100)
        return [serialize_user(u, "teacher") for u in users]
    elif role == "admin":
        users = await db.admins.find().to_list(100)
        return [serialize_user(u, "admin") for u in users]
    raise HTTPException(status_code=400, detail="Invalid role")

@router.post("/delete_user/")
async def delete_user(id: str = Form(...)):
    for collection in ["teachers", "admins"]:
        result = await db[collection].delete_one({"_id": ObjectId(id)})
        if result.deleted_count:
            return {"message": "User deleted successfully."}
    raise HTTPException(status_code=404, detail="User not found")

@router.post("/update_user/")
async def update_user(request: Request):
    data = await request.json()
    id = data.get("id")
    metadata = data.get("metadata")

    if not id or not metadata:
        raise HTTPException(status_code=400, detail="Missing data")

    role = metadata.get("role")
    updates = {k: v for k, v in metadata.items() if k not in ["role"]}

    if role == "teacher":
        result = await db.teachers.update_one({"_id": ObjectId(id)}, {"$set": updates})
    elif role == "admin":
        result = await db.admins.update_one({"_id": ObjectId(id)}, {"$set": updates})
    else:
        raise HTTPException(status_code=400, detail="Invalid role")

    if result.modified_count:
        return {"message": "User updated successfully."}
    else:
        return {"message": "No changes made."}

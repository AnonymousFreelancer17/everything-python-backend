from fastapi import FastAPI, Request, HTTPException
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, EmailStr
from utils.supabase_client import supabase


app = FastAPI()


class RegisterUser(BaseModel):
    email: EmailStr
    password: str

class LoginUser(BaseModel):
    email: EmailStr
    password: str

@app.get("/")
def root():
    return {
        "message": "hello from auth-services"
    }

@app.post("/signup")
def signup(user: RegisterUser):
    try:
        response = supabase.auth.sign_up({
            "email": user.email,
            "password": user.password
        })
        return {"message": "Signup successful. Check your email to confirm.", "data": response}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/login")
def login(user: LoginUser):
    try:
        session = supabase.auth.sign_in_with_password({
            "email": user.email,
            "password": user.password
        })
        return {"access_token": session.session.access_token}
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))


@app.get("/oauth/{provider}")
def oauth_login(provider: str):
    try:
        res = supabase.auth.sign_in_with_oauth({
            "provider": provider
        })
        return {"url": res.url}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
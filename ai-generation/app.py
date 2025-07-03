from fastapi import FastAPI, Request
import httpx

app = FastAPI()

@app.get("/")
async def root():
	return {"message": "Hello from ai-generation service"}


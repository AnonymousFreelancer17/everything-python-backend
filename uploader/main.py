from fastapi import FastAPI, HTTPException
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET = "aidatabucket"

# Connect to Supabase
try:
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ Supabase connection was successfully created.")
    # Optional: try listing to validate bucket
    test = client.storage.from_(BUCKET).list()
    print(f"🪣 Bucket '{BUCKET}' is accessible.")
except Exception as e:
    print("❌ Failed to connect to Supabase or access bucket:", str(e))

app = FastAPI()

@app.get("/list")
def list_files():
    try:
        response = client.storage.from_(BUCKET).list()
        if not response:
            return {"files": []}
        return {"files": [file['name'] for file in response]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")

@app.get("/file/{file_path:path}")
def get_file(file_path: str):
    try:
        url = client.storage.from_(BUCKET).get_public_url(file_path)
        if not url:
            raise HTTPException(status_code=404, detail="File not found or URL invalid")
        return {"url": url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching file URL: {str(e)}")

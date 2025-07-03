from supabase import create_client
import os
from pathlib import Path

# ==== CONFIG ====
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-public-or-service-key"
BUCKET = "aidatabucket"
FOLDER_PATH = "C:\Users\adity\.cache\kagglehub\datasets\chetankv\dogs-cats-images\versions\1\dataset" 

def upload_folder(folder_path):
    base = Path(folder_path)

    print("📡 Connecting to Supabase...")
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Connection to Supabase was successful.")
    except Exception as e:
        print("❌ Failed to connect to Supabase:", str(e))
        return

    for root, _, files in os.walk(base):
        for file in files:
            full_path = Path(root) / file
            key = str(full_path.relative_to(base))

            print(f"⬆️ Uploading: {key} ... ", end="")
            try:
                with open(full_path, "rb") as f:
                    result = client.storage.from_(BUCKET).upload(key, f, {"upsert": True})
                if result:
                    print("✅ Success")
                else:
                    print("⚠️ Unknown response")
            except Exception as e:
                print(f"❌ Failed - {str(e)}")

if __name__ == "__main__":
    upload_folder(FOLDER_PATH)

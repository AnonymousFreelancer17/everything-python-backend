from supabase import create_client, Client
import os


SUPABASE_URL = os.environ()
SUPABASE_KEY = "your-public-or-service-role-key"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_to_supabase(file_path: str, bucket: str = "videos") -> str:
    with open(file_path, "rb") as f:
        file_data = f.read()
    file_name = os.path.basename(file_path)
    supabase.storage.from_(bucket).upload(file_name, file_data)
    public_url = supabase.storage.from_(bucket).get_public_url(file_name)
    return public_url

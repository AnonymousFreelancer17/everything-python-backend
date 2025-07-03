from urllib.parse import urljoin
import requests
import hashlib
from PIL import Image
from io import BytesIO
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np


BLOCKED_KEYWORDS = ["avatars", "ozzmodz_badges_badge", "badge", "logo", "sprite", "icon", "emojis"]
HREF_SKIP_KEYWORDS = [
    "logout", "signup", "login", "share", "facebook", "twitter",
    "instagram", "mailto:", "tel:", "pinterest", "linkedin", "youtube",
    "whatsapp", "rss", "javascript", "#"
]
CACHE = {} 
seen_hashes = set()

def cache_key(url, start, end):
    return hashlib.md5(f"{url}_{start}_{end}".encode()).hexdigest()

def is_valid_image(src, collected):
    return src and not src.startswith("data:image") and not any(k in src for k in BLOCKED_KEYWORDS) and src not in collected

def normalize_url(src, base_url):
    if src.startswith('//'):
        return 'https:' + src
    elif src.startswith('/'):
        return urljoin(base_url, src)
    elif not src.startswith('http'):
        return urljoin(base_url, src)
    return src

def is_duplicate_by_ssim(img_url, seen_images_cv, threshold=0.92):
    try:
        response = requests.get(img_url, timeout=5)
        img = Image.open(BytesIO(response.content))
        if img.format == 'GIF':
            img.seek(0)
        img = img.convert("RGB")
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        img_cv = cv2.resize(img_cv, (128, 128))

        for existing in seen_images_cv:
            score = ssim(img_cv, existing)
            if score > threshold:
                return True

        seen_images_cv.append(img_cv)
        return False
    except Exception as e:
        print(f"[INFO] Ignored image due to SSIM error: {img_url}")
        return True


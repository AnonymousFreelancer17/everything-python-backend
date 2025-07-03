# from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from bs4 import BeautifulSoup
# from urllib.parse import urljoin
# import os
# from flask_cors import CORS
# import requests
# import time
# import hashlib
# from PIL import Image
# import imagehash
# from io import BytesIO
# from flask_cors import CORS
# from skimage.metrics import structural_similarity as ssim
# import cv2
# import numpy as np
# from datetime import datetime
# import json
# from selenium.webdriver import DesiredCapabilities
# from webdriver_manager.chrome import ChromeDriverManager
# from selenium.webdriver.chrome.service import Service


# app = Flask(__name__)
# CORS(app,  resources={r"/api/*": {"origins": "http://localhost:3000"}})
# app.secret_key = 'supersecretkey'
# app.config['DOWNLOAD_FOLDER'] = os.path.join('static', 'downloads')
# os.makedirs(app.config['DOWNLOAD_FOLDER'], exist_ok=True)
# HISTORY_FILE = "scrape_history.json"

# BLOCKED_KEYWORDS = ["avatars", "ozzmodz_badges_badge", "badge", "logo", "sprite", "icon", "emojis"]
# HREF_SKIP_KEYWORDS = [
#     "logout", "signup", "login", "share", "facebook", "twitter",
#     "instagram", "mailto:", "tel:", "pinterest", "linkedin", "youtube",
#     "whatsapp", "rss", "javascript", "#"
# ]
# CACHE = {} 
# seen_hashes = set()

# def cache_key(url, start, end):
#     return hashlib.md5(f"{url}_{start}_{end}".encode()).hexdigest()

# def is_valid_image(src, collected):
#     return src and not src.startswith("data:image") and not any(k in src for k in BLOCKED_KEYWORDS) and src not in collected

# def normalize_url(src, base_url):
#     if src.startswith('//'):
#         return 'https:' + src
#     elif src.startswith('/'):
#         return urljoin(base_url, src)
#     elif not src.startswith('http'):
#         return urljoin(base_url, src)
#     return src



# def create_chrome_driver():
#     options = Options()
#     options.headless = True
#     options.add_argument("--incognito")
#     options.add_argument("--disable-blink-features=AutomationControlled")
#     options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
#     options.add_argument("--no-sandbox")
#     options.add_argument("--disable-dev-shm-usage")

#     # Set page load strategy
#     options.set_capability("pageLoadStrategy", "eager")

#     # Use Service object from Selenium
#     service = Service(ChromeDriverManager().install())
#     driver = webdriver.Chrome(service=service, options=options)
#     driver.set_page_load_timeout(30)
#     driver.set_script_timeout(30)
#     return driver


# def log_scrape_event(url, images):
#     try:
#         if os.path.exists(HISTORY_FILE):
#             with open(HISTORY_FILE, "r") as f:
#                 history = json.load(f)
#         else:
#             history = []

#         event = {
#             "timestamp": datetime.utcnow().isoformat(),
#             "url": url,
#             "image_count": len(images),
#             "images": images[:10]
#         }

#         history.insert(0, event)
#         with open(HISTORY_FILE, "w") as f:
#             json.dump(history[:100], f, indent=2)
#     except Exception as e:
#         print(f"Failed to log history: {e}")

# def is_duplicate_by_ssim(img_url, seen_images_cv, threshold=0.92):
#     try:
#         response = requests.get(img_url, timeout=5)
#         img = Image.open(BytesIO(response.content))
#         if img.format == 'GIF':
#             img.seek(0)
#         img = img.convert("RGB")
#         img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
#         img_cv = cv2.resize(img_cv, (128, 128))

#         for existing in seen_images_cv:
#             score = ssim(img_cv, existing)
#             if score > threshold:
#                 return True

#         seen_images_cv.append(img_cv)
#         return False
#     except Exception as e:
#         print(f"[INFO] Ignored image due to SSIM error: {img_url}")
#         return True

# def scrape_images_single_page(url, driver, collected, seen_images_cv=None):
#     if seen_images_cv is None:
#         seen_images_cv = []

#     try:
#         for attempt in range(3):
#             try:
#                 start_time = time.time()
#                 driver.get(url)
#                 print(f"Loaded {url} in {round(time.time() - start_time, 2)}s")
#                 break
#             except Exception as e:
#                 print(f"[Attempt {attempt+1}] Error loading {url}: {e}")
#                 if attempt == 2:
#                     raise
#                 time.sleep(3)

#         soup = BeautifulSoup(driver.page_source, 'html.parser')
#         images = []
#         href_links = set()

#         for img in soup.find_all('img'):
#             src = img.get('src')
#             if not is_valid_image(src, collected):
#                 continue
#             full_img_url = normalize_url(src, url)
#             if is_duplicate_by_ssim(full_img_url, seen_images_cv):
#                 continue
#             collected.add(full_img_url)
#             parent_link = img.find_parent('a')
#             href = parent_link.get('href') if parent_link else None
#             full_href = normalize_url(href, url) if href else None
#             if full_href and not any(k in full_href.lower() for k in HREF_SKIP_KEYWORDS):
#                 href_links.add(full_href)
#             images.append({"img_url": full_img_url, "linked_href": full_href})

#         for a_tag in soup.find_all('a', href=True):
#             raw_href = a_tag['href'].strip()
#             full_href = normalize_url(raw_href, url)
#             if full_href and not any(k in full_href.lower() for k in HREF_SKIP_KEYWORDS):
#                 href_links.add(full_href)

#         print(f"images: {images}, hrefs: {href_links}")
#         return {"images": images, "hrefs": list(href_links)}

#     except Exception as e:
#         print(f"Error scraping {url}: {e}")
#         return {"images": [], "hrefs": []}

# def extract_base_url(url):
#     return url.rsplit("/page-", 1)[0] if "/page-" in url else url

# def scrape_images_multi_page(base_url, driver, page_start, page_end):
#     collected = set()
#     all_images = []
#     seen_images_cv = []
#     for page in range(page_start, page_end + 1):
#         if "/page-" in base_url:
#             page_url = base_url.rsplit("/page-", 1)[0] + f"/page-{page}"
#         elif base_url.endswith('/'):
#             page_url = base_url + f"page-{page}"
#         else:
#             page_url = base_url + f"/page-{page}"
#         result = scrape_images_single_page(page_url, driver, collected, seen_images_cv)
#         all_images.extend(result['images'])
#     return all_images

# def scrape_images_auto(url, page_start=1, page_end=1):
#     key = cache_key(url, page_start, page_end)
#     if key in CACHE:
#         print(f"[CACHE] Using cached result for {url} [{page_start}-{page_end}]")
#         return {"images": CACHE[key], "duration": 0.0, "from_cache": True}

#     driver = create_chrome_driver()
#     start_time = time.time()
#     try:
#         base_url = extract_base_url(url)
#         images = (
#             scrape_images_multi_page(base_url, driver, page_start, page_end)
#             if page_start != page_end or "/page-" in url
#             else scrape_images_single_page(url, driver, set())["images"]
#         )
#         duration = time.time() - start_time
#         log_scrape_event(url, images)
#         CACHE[key] = images
#         return {"images": images, "duration": duration, "from_cache": False}
#     except Exception as e:
#         print(f"Scraping failed: {e}")
#         return {"images": [], "duration": time.time() - start_time, "from_cache": False}
#     finally:
#         driver.quit()


# @app.route("/" ,methods=['GET'] )
# async def root():
# 	return {"message": "Hello World"}

# @app.route('/api/scrape', methods=['POST'])
# def scrape_images():
#     data = request.json
#     url = data.get('url', '').strip()
#     try:
#         page_start = int(data.get('start_page', 1))
#         page_end = int(data.get('end_page') or page_start)
#     except ValueError:
#         return jsonify({'error': 'Page numbers must be valid integers.'}), 400

#     if page_start > page_end:
#         return jsonify({'error': 'Start page cannot be greater than end page.'}), 400

#     result = scrape_images_auto(url, page_start, page_end)
#     return jsonify({
#         'images': result["images"],
#         'duration': round(result["duration"], 2),
#         'from_cache': result.get("from_cache", False),
#         'page_range': [page_start, page_end]
#     })

# @app.route('/api/download', methods=['POST'])
# def download_images():
#     data = request.json
#     selected = data.get('selected_images', [])
#     if not selected:
#         return jsonify({'error': 'No images selected.'}), 400

#     downloaded = []
#     failed = []
#     for img_url in selected:
#         try:
#             img_data = requests.get(img_url, timeout=10).content
#             filename = os.path.basename(img_url.split("?")[0])
#             save_path = os.path.join(app.config['DOWNLOAD_FOLDER'], filename)
#             with open(save_path, 'wb') as f:
#                 f.write(img_data)
#             downloaded.append(filename)
#         except Exception as e:
#             print(f"Failed: {img_url}", e)
#             failed.append(img_url)

#     return jsonify({
#         'downloaded': downloaded,
#         'failed': failed,
#         'message': 'Download complete with some failures' if failed else 'All images downloaded.'
#     })

# @app.route('/api/history', methods=['GET'])
# def get_scrape_history():
#     if os.path.exists(HISTORY_FILE):
#         with open(HISTORY_FILE, "r") as f:
#             history = json.load(f)
#         return jsonify(history)
#     return jsonify([])

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5002)



from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def scrape():
    return {"status": "Hello from Scrapper"}
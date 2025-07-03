from bs4 import BeautifulSoup
from scrape_utils.driver import create_chrome_driver
from scrape_utils.image_filter import normalize_url, is_valid_image, is_duplicate_by_ssim
import hashlib
import time
 


CACHE = {} 
seen_hashes = set()


BLOCKED_KEYWORDS = ["avatars", "ozzmodz_badges_badge", "badge", "logo", "sprite", "icon", "emojis"]
HREF_SKIP_KEYWORDS = [
    "logout", "signup", "login", "share", "facebook", "twitter",
    "instagram", "mailto:", "tel:", "pinterest", "linkedin", "youtube",
    "whatsapp", "rss", "javascript", "#"
]


def cache_key(url, start, end):
    return hashlib.md5(f"{url}_{start}_{end}".encode()).hexdigest()

def scrape_images_single_page(url, driver, collected, seen_images_cv=None):
    if seen_images_cv is None:
        seen_images_cv = []

    try:
        for attempt in range(3):
            try:
                start_time = time.time()
                driver.get(url)
                print(f"Loaded {url} in {round(time.time() - start_time, 2)}s")
                break
            except Exception as e:
                print(f"[Attempt {attempt+1}] Error loading {url}: {e}")
                if attempt == 2:
                    raise
                time.sleep(3)

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        images = []
        href_links = set()

        for img in soup.find_all('img'):
            src = img.get('src')
            if not is_valid_image(src, collected):
                continue
            full_img_url = normalize_url(src, url)
            if is_duplicate_by_ssim(full_img_url, seen_images_cv):
                continue
            collected.add(full_img_url)
            parent_link = img.find_parent('a')
            href = parent_link.get('href') if parent_link else None
            full_href = normalize_url(href, url) if href else None
            if full_href and not any(k in full_href.lower() for k in HREF_SKIP_KEYWORDS):
                href_links.add(full_href)
            images.append({"img_url": full_img_url, "linked_href": full_href})

        for a_tag in soup.find_all('a', href=True):
            raw_href = a_tag['href'].strip()
            full_href = normalize_url(raw_href, url)
            if full_href and not any(k in full_href.lower() for k in HREF_SKIP_KEYWORDS):
                href_links.add(full_href)

        print(f"images: {images}, hrefs: {href_links}")
        return {"images": images, "hrefs": list(href_links)}

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return {"images": [], "hrefs": []}

def extract_base_url(url):
    return url.rsplit("/page-", 1)[0] if "/page-" in url else url

def scrape_images_multi_page(base_url, driver, page_start, page_end):
    collected = set()
    all_images = []
    seen_images_cv = []
    for page in range(page_start, page_end + 1):
        if "/page-" in base_url:
            page_url = base_url.rsplit("/page-", 1)[0] + f"/page-{page}"
        elif base_url.endswith('/'):
            page_url = base_url + f"page-{page}"
        else:
            page_url = base_url + f"/page-{page}"
        result = scrape_images_single_page(page_url, driver, collected, seen_images_cv)
        all_images.extend(result['images'])
    return all_images

def scrape_images_auto(url, page_start=1, page_end=1):
    key = cache_key(url, page_start, page_end)
    if key in CACHE:
        print(f"[CACHE] Using cached result for {url} [{page_start}-{page_end}]")
        return {"images": CACHE[key], "duration": 0.0, "from_cache": True}

    driver = create_chrome_driver()
    start_time = time.time()
    try:
        base_url = extract_base_url(url)
        images = (
            scrape_images_multi_page(base_url, driver, page_start, page_end)
            if page_start != page_end or "/page-" in url
            else scrape_images_single_page(url, driver, set())["images"]
        )
        duration = time.time() - start_time
        log_scrape_event(url, images)
        CACHE[key] = images
        return {"images": images, "duration": duration, "from_cache": False}
    except Exception as e:
        print(f"Scraping failed: {e}")
        return {"images": [], "duration": time.time() - start_time, "from_cache": False}
    finally:
        driver.quit()
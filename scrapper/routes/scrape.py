from flask import Blueprint, request, jsonify
from scrape_utils.scrape_logic import scrape_images_auto
from services.history_service import log_scrape_event
import time

scrape_bp = Blueprint("scrape", __name__)

@scrape_bp.route("/", methods=["POST"])
def scrape():
    data = request.json
    url = data.get("url", "").strip()
    start_page = int(data.get("start_page", 1))
    end_page = int(data.get("end_page") or start_page)

    start_time = time.time()
    images = scrape_images_auto(url, start_page, end_page)
    elapsed = round(time.time() - start_time, 2)

    log_scrape_event(url, images, elapsed)

    return jsonify({"images": images, "time_taken": elapsed})

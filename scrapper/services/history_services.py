from config import db
from datetime import datetime

def log_scrape_event(url, images, time_taken):
    db.history.insert_one({
        "url": url,
        "timestamp": datetime.utcnow(),
        "image_count": len(images),
        "sample": images[:10],
        "time_taken": time_taken
    })

def get_scrape_history():
    return list(db.history.find().sort("timestamp", -1).limit(100))
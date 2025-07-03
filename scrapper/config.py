import os
from pymongo import MongoClient

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = "image_scraper"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

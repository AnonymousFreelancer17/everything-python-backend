from config import db
from werkzeug.security import generate_password_hash

class User:
    @staticmethod
    def create(username, password):
        hashed_password = generate_password_hash(password)
        user_data = {"username": username, "password": hashed_password}
        db.users.insert_one(user_data)
        return user_data

    @staticmethod
    def find_by_username(username):
        return db.users.find_one({"username": username})

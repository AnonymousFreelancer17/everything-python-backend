from models.user import User
from werkzeug.security import check_password_hash

class AuthService:
    @staticmethod
    def login(username, password):
        user = User.find_by_username(username)
        if user and check_password_hash(user["password"], password):
            return user
        return None

    @staticmethod
    def register(username, password):
        if User.find_by_username(username):
            return {"error": "User already exists"}
        return User.create(username, password)

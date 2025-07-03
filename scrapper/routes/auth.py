from flask import Blueprint, request, jsonify, session
from models.user import User
from werkzeug.security import check_password_hash

auth_bp = Blueprint("auth", __name__)

@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.json
    user = User.find_by_username(data["username"])
    if user and check_password_hash(user["password"], data["password"]):
        session["user"] = user["username"]
        return jsonify({"message": "Login successful"})
    return jsonify({"error": "Invalid credentials"}), 401

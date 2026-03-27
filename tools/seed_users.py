"""
Seed script — uploads two test users to Firestore.
Run from the project root:
    python tools/seed_users.py

Test credentials:
    Admin     → username: admin     | password: Admin@1234
    Forensic  → username: forensic  | password: Forensic@1234
"""
import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import firebase_admin
from firebase_admin import credentials, firestore
import bcrypt
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── Firebase init (from env vars — no JSON file needed) ───────────────────────
cert_dict = {
    "type": "service_account",
    "project_id": os.getenv("FIREBASE_PROJECT_ID"),
    "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
    "private_key": os.getenv("FIREBASE_PRIVATE_KEY", "").replace("\\n", "\n"),
    "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
    "client_id": os.getenv("FIREBASE_CLIENT_ID"),
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_CERT_URL"),
    "universe_domain": "googleapis.com",
}
cred = credentials.Certificate(cert_dict)
firebase_admin.initialize_app(cred)
db = firestore.client()

# ── Password hashing ───────────────────────────────────────────────────────────
def _hash(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# ── Test users ─────────────────────────────────────────────────────────────────
TEST_USERS = [
    {
        "id": "ADM01",
        "email": "admin@argus.io",
        "username": "admin",
        "full_name": "ARGUS Administrator",
        "password": "Admin@1234",
        "role": "admin",
    },
    {
        "id": "FOR01",
        "email": "forensic@argus.io",
        "username": "forensic",
        "full_name": "ARGUS Forensic Analyst",
        "password": "Forensic@1234",
        "role": "forensic",
    },
]


def seed():
    print("\n🚀 Seeding test users to Firestore...\n")

    for user in TEST_USERS:
        doc_id = user["id"]
        payload = {
            "email": user["email"],
            "username": user["username"],
            "full_name": user["full_name"],
            "hashed_password": _hash(user["password"]),
            "role": user["role"],
            "is_active": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        db.collection("users").document(doc_id).set(payload)
        print(f"  ✅ {user['role'].upper():10} | ID: {doc_id} | username: {user['username']:10} | password: {user['password']}")

    print("\n✨ Done! Both users are now live in Firestore.")


if __name__ == "__main__":
    seed()

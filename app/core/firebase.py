"""
Firebase Admin SDK Initialization

Builds credentials from environment variables (via config.py) so no
service account JSON file needs to ship with the codebase.
"""
import logging
import firebase_admin
from firebase_admin import credentials, firestore as firebase_firestore

from .config import settings

logger = logging.getLogger(__name__)

_db = None


def _build_credentials() -> credentials.Certificate:
    """Construct Firebase credentials from env vars (no JSON file needed)."""
    cert_dict = {
        "type": "service_account",
        "project_id": settings.firebase_project_id,
        "private_key_id": settings.firebase_private_key_id,
        # .env stores \n as literal backslash-n — convert back to real newlines
        "private_key": settings.firebase_private_key.replace("\\n", "\n"),
        "client_email": settings.firebase_client_email,
        "client_id": settings.firebase_client_id,
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": settings.firebase_client_cert_url,
        "universe_domain": "googleapis.com",
    }
    return credentials.Certificate(cert_dict)


def initialize_firebase() -> None:
    """Initialize Firebase Admin SDK and Firestore client."""
    global _db
    if not firebase_admin._apps:
        cred = _build_credentials()
        firebase_admin.initialize_app(cred)
        logger.info("Firebase Admin SDK initialized (project: %s)", settings.firebase_project_id)
    _db = firebase_firestore.client()
    logger.info("Firestore client ready")


def get_firestore():
    """Get the Firestore client instance, initializing on first call."""
    global _db
    if _db is None:
        initialize_firebase()
    return _db

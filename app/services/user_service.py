"""
User service — Firestore CRUD operations for the User collection.
All I/O is wrapped in asyncio.to_thread() to keep FastAPI non-blocking.
"""
import asyncio
import logging
from typing import Optional, List

from app.core.firebase import get_firestore
from app.core.security import hash_password
from app.models.user import User, UserRole, generate_user_id

logger = logging.getLogger(__name__)

USERS_COLLECTION = "users"


# ── Internal helpers ───────────────────────────────────────────────────────────

def _get_user_doc(user_id: str):
    return get_firestore().collection(USERS_COLLECTION).document(user_id).get()


def _query_users(field: str, value: str):
    return list(
        get_firestore()
        .collection(USERS_COLLECTION)
        .where(field, "==", value)
        .limit(1)
        .stream()
    )


def _list_all_users():
    return list(get_firestore().collection(USERS_COLLECTION).stream())


def _write_user(user: User):
    get_firestore().collection(USERS_COLLECTION).document(user.id).set(
        user.to_firestore()
    )


# ── Public API ─────────────────────────────────────────────────────────────────

async def create_user(
    email: str,
    username: str,
    full_name: str,
    password: str,
    role: UserRole = UserRole.FORENSIC,
) -> User:
    """Create and persist a new user in Firestore."""
    user = User(
        id=generate_user_id(),
        email=email,
        username=username,
        full_name=full_name,
        hashed_password=hash_password(password),
        role=role,
    )
    await asyncio.to_thread(_write_user, user)
    logger.info("Created user '%s' (ID: %s, role: %s)", user.username, user.id, user.role)
    return user


async def get_user_by_id(user_id: str) -> Optional[User]:
    """Fetch a user by their 5-char ID."""
    doc = await asyncio.to_thread(_get_user_doc, user_id)
    if not doc.exists:
        return None
    return User.from_firestore(doc.id, doc.to_dict())


async def get_user_by_username(username: str) -> Optional[User]:
    """Fetch a user by username (used for auth)."""
    docs = await asyncio.to_thread(_query_users, "username", username)
    if not docs:
        return None
    return User.from_firestore(docs[0].id, docs[0].to_dict())


async def get_user_by_email(email: str) -> Optional[User]:
    """Fetch a user by email."""
    docs = await asyncio.to_thread(_query_users, "email", email)
    if not docs:
        return None
    return User.from_firestore(docs[0].id, docs[0].to_dict())


async def list_users() -> List[User]:
    """Return all users (admin only)."""
    docs = await asyncio.to_thread(_list_all_users)
    return [User.from_firestore(doc.id, doc.to_dict()) for doc in docs]


async def update_user(user_id: str, updates: dict) -> Optional[User]:
    """Partial-update a user document in Firestore."""
    db = get_firestore()

    def _update():
        db.collection(USERS_COLLECTION).document(user_id).update(updates)

    await asyncio.to_thread(_update)
    return await get_user_by_id(user_id)


async def delete_user(user_id: str) -> bool:
    """Delete a user document from Firestore."""
    db = get_firestore()

    def _delete():
        db.collection(USERS_COLLECTION).document(user_id).delete()

    await asyncio.to_thread(_delete)
    logger.info("Deleted user ID: %s", user_id)
    return True

"""
User domain model.
"""
import random
import string
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class UserRole(str, Enum):
    ADMIN = "admin"
    FORENSIC = "forensic"


def generate_user_id() -> str:
    """Generate a unique 5-character alphanumeric user ID (uppercase + digits)."""
    chars = string.ascii_uppercase + string.digits
    return "".join(random.choices(chars, k=5))


class User(BaseModel):
    id: str = Field(
        default_factory=generate_user_id,
        min_length=5,
        max_length=5,
        description="5-character alphanumeric user ID",
    )
    email: str
    username: str
    full_name: str
    hashed_password: str
    role: UserRole
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_firestore(self) -> dict:
        """Serialize to a Firestore-compatible dict (no 'id' field — stored as doc ID)."""
        data = self.model_dump()
        data.pop("id", None)
        data["role"] = self.role.value
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_firestore(cls, doc_id: str, data: dict) -> "User":
        """Reconstruct a User from a Firestore document."""
        data = dict(data)
        data["id"] = doc_id
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)

    class Config:
        use_enum_values = True

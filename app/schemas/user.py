"""
User and Auth request/response Pydantic schemas for the API layer.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field

from app.models.user import UserRole


# ── Requests ──────────────────────────────────────────────────────────────────


class UserCreate(BaseModel):
    email: str = Field(..., description="User email address")
    username: str = Field(
        ..., min_length=3, max_length=30, description="Unique username"
    )
    full_name: str = Field(..., description="User full name")
    password: str = Field(
        ..., min_length=6, description="Plain-text password (will be hashed)"
    )
    role: UserRole = Field(default=UserRole.FORENSIC, description="User role")


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    is_active: Optional[bool] = None
    role: Optional[UserRole] = None


# ── Responses ─────────────────────────────────────────────────────────────────


class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    full_name: str
    role: UserRole
    is_active: bool
    created_at: datetime

    class Config:
        use_enum_values = True


# ── Auth ──────────────────────────────────────────────────────────────────────


class LoginRequest(BaseModel):
    user_id: str = Field(
        ...,
        min_length=5,
        max_length=5,
        pattern=r"^[A-Za-z0-9]{5}$",
        description="5-character alphanumeric user ID (e.g. ADM01, FOR01)",
    )
    password: str = Field(..., description="Plain-text password")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Token lifetime in seconds")
    user: UserResponse

"""
Auth API routes — login and current-user endpoints.
"""
import logging
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.config import settings
from app.core.security import create_access_token, verify_password
from app.api.deps import get_current_user
from app.models.user import User
from app.schemas.user import LoginRequest, TokenResponse, UserResponse
from app.services.user_service import get_user_by_username, get_user_by_email

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


# ── POST /api/v1/auth/login ───────────────────────────────────────────────────

@router.post("/login", response_model=TokenResponse, summary="Obtain a JWT access token")
async def login(payload: LoginRequest) -> TokenResponse:
    """
    Authenticate with **username** (or email) and **password**.

    Returns a signed JWT bearer token valid for `jwt_expire_minutes`.
    """
    # Support login by username OR email
    user: User | None = await get_user_by_username(payload.username)
    if user is None:
        user = await get_user_by_email(payload.username)

    # Unified message to avoid username enumeration
    _invalid = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if user is None:
        raise _invalid

    if not verify_password(payload.password, user.hashed_password):
        raise _invalid

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled",
        )

    # Build token payload
    token_data = {
        "sub": user.id,
        "username": user.username,
        "role": user.role,
    }
    expires_in = settings.jwt_expire_minutes * 60  # seconds
    access_token = create_access_token(
        data=token_data,
        expires_delta=timedelta(seconds=expires_in),
    )

    logger.info("User '%s' (%s) logged in", user.username, user.role)

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=expires_in,
        user=UserResponse(
            id=user.id,
            email=user.email,
            username=user.username,
            full_name=user.full_name,
            role=user.role,
            is_active=user.is_active,
            created_at=user.created_at,
        ),
    )


# ── GET /api/v1/auth/me ────────────────────────────────────────────────────────

@router.get("/me", response_model=UserResponse, summary="Get the current authenticated user")
async def get_me(current_user: User = Depends(get_current_user)) -> UserResponse:
    """Return the profile of the currently authenticated user."""
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        username=current_user.username,
        full_name=current_user.full_name,
        role=current_user.role,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
    )

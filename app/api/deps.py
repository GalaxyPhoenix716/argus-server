"""
FastAPI dependency injection — extracts & validates the JWT bearer token,
returns the authenticated User, and provides role-based access guards.
"""
import logging
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import jwt

from app.core.security import decode_access_token
from app.models.user import User, UserRole
from app.services.user_service import get_user_by_id

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)

# ── Core auth dependency ───────────────────────────────────────────────────────

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> User:
    """
    Extract Bearer token → decode JWT → return authenticated User.
    Raises HTTP 401 for any invalid / expired / missing token.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    try:
        payload = decode_access_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id: str = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token payload missing subject",
        )

    user = await get_user_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled",
        )

    return user


# ── Role-based guards ──────────────────────────────────────────────────────────

async def require_admin(
    current_user: User = Depends(get_current_user),
) -> User:
    """Only ADMIN users may proceed."""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user


async def require_forensic_or_above(
    current_user: User = Depends(get_current_user),
) -> User:
    """ADMIN and FORENSIC users may proceed (any authenticated role)."""
    if current_user.role not in (UserRole.ADMIN, UserRole.FORENSIC):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions",
        )
    return current_user

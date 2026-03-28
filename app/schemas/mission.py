"""
Mission request/response Pydantic schemas for the API layer.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.models.mission import MissionStatus


# ── Requests ───────────────────────────────────────────────────────────────────


class MissionCreate(BaseModel):
    """Payload to create a new mission."""

    mission_name: str = Field(
        ...,
        min_length=3,
        max_length=100,
        description="Human-readable mission name",
    )
    channel_id: str = Field(
        ...,
        description="Simulator channel identifier that will stream data (e.g. 'SIM-1')",
    )


class MissionUpdate(BaseModel):
    """Payload for partial mission updates (status transitions, rename)."""

    mission_name: Optional[str] = Field(None, min_length=3, max_length=100)
    status: Optional[MissionStatus] = None
    channel_id: Optional[str] = None


# ── Audit log sub-schema ───────────────────────────────────────────────────────


class AuditLogEntryResponse(BaseModel):
    """A single audit log entry as returned by the API."""

    firestore_ref   : str
    tx_hash         : Optional[str]  = None
    log_index       : Optional[int]  = None
    event_type      : str
    data_hash       : str
    storage_backend : str
    timestamp       : datetime

    class Config:
        use_enum_values = True


# ── Responses ──────────────────────────────────────────────────────────────────


class MissionResponse(BaseModel):
    """Full mission representation returned by the API (no live telemetry)."""

    id          : str
    mission_name: str
    created_by  : str
    status      : MissionStatus
    channel_id  : str
    created_at  : datetime
    updated_at  : datetime

    class Config:
        use_enum_values = True


class MissionDetailResponse(MissionResponse):
    """
    Extended mission response that also includes the audit log.
    Returned by GET /missions/{mission_id}.
    """

    audit_log: list[AuditLogEntryResponse] = Field(default_factory=list)


class MissionListResponse(BaseModel):
    """Paginated list of missions."""

    missions : list[MissionResponse]
    total    : int
    limit    : int
    offset   : int

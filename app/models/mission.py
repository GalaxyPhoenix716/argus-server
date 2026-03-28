"""
Mission domain model.

A Mission is the top-level operational unit in ARGUS. It is stored in Firestore
(minus live telemetry) and has two sub-documents:

  • mission_telemetry — a TelemetryFrame instance that is populated in-memory
                        from the real-time WebSocket stream.  NOT persisted.

  • audit_log         — a list of AuditLogEntry objects that reference both:
                          - the Firestore document ID of the encrypted log
                          - the on-chain tx_hash (blockchain anchor)

Firestore document layout  (collection: "missions"):
  doc_id = mission_id
  {
    mission_name   : str,
    created_by     : str  (user_id),
    status         : str  (MissionStatus enum),
    created_at     : ISO-8601 string,
    updated_at     : ISO-8601 string,
    channel_id     : str  (simulator channel, e.g. "SIM-1"),
    -- audit_log entries are stored as a sub-collection "audit_logs" --
  }

Telemetry frames are NEVER written to Firestore.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.models.telemetry import TelemetryFrame
from app.services.ids import generate_mission_id


# ── Enums ──────────────────────────────────────────────────────────────────────


class MissionStatus(str, Enum):
    PENDING   = "pending"     # created, simulator not yet connected
    ACTIVE    = "active"      # simulator streaming live data
    COMPLETED = "completed"   # mission ended normally
    ABORTED   = "aborted"     # mission terminated early


# ── Audit Log Entry ────────────────────────────────────────────────────────────


class AuditLogEntry(BaseModel):
    """
    A single immutable audit event linked to both Firestore and the blockchain.

    firestore_ref   : Firestore document ID inside the global 'audit_logs' collection
    tx_hash         : Ethereum transaction hash (None until the chain write settles)
    log_index       : Smart-contract log index (None until the chain write settles)
    event_type      : Human-readable event label  (e.g. "THREAT_DETECTED")
    data_hash       : SHA-256 of the encrypted payload — the tamper-proof anchor
    storage_backend : "firestore" | "ipfs"
    timestamp       : When this entry was created (UTC)
    """

    firestore_ref   : str            = Field(..., description="Firestore doc ID of the encrypted log")
    tx_hash         : Optional[str]  = Field(None, description="Blockchain transaction hash")
    log_index       : Optional[int]  = Field(None, description="On-chain log index")
    event_type      : str            = Field(..., description="Event type label")
    data_hash       : str            = Field(..., description="SHA-256 of encrypted payload")
    storage_backend : str            = Field("firestore", description="'firestore' or 'ipfs'")
    timestamp       : datetime       = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_firestore(self) -> dict:
        return {
            "firestore_ref":   self.firestore_ref,
            "tx_hash":         self.tx_hash,
            "log_index":       self.log_index,
            "event_type":      self.event_type,
            "data_hash":       self.data_hash,
            "storage_backend": self.storage_backend,
            "timestamp":       self.timestamp.isoformat(),
        }

    @classmethod
    def from_firestore(cls, data: dict) -> "AuditLogEntry":
        data = dict(data)
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


# ── Mission ────────────────────────────────────────────────────────────────────


class Mission(BaseModel):
    """
    Core mission entity.

    Fields stored in Firestore:
      id, mission_name, created_by, status, created_at, updated_at, channel_id

    Fields that are runtime-only (NOT stored):
      mission_telemetry  — latest TelemetryFrame received from the simulator
      audit_log          — list of AuditLogEntry; individual entries are stored
                           in the global 'audit_logs' Firestore collection and
                           referenced from here by firestore_ref.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    id: str = Field(
        default_factory=generate_mission_id,
        min_length=7,
        max_length=7,
        description="7-character alphanumeric mission ID",
    )
    mission_name: str = Field(..., description="Human-readable mission name")

    # ── Ownership & Status ────────────────────────────────────────────────────
    created_by: str = Field(..., description="User ID of the mission creator")
    status: MissionStatus = Field(default=MissionStatus.PENDING)

    # ── Simulator linkage ─────────────────────────────────────────────────────
    channel_id: str = Field(
        ...,
        description="Simulator channel identifier (e.g. 'SIM-1')",
    )

    # ── Timestamps ────────────────────────────────────────────────────────────
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # ── Runtime-only — NOT persisted to Firestore ─────────────────────────────
    mission_telemetry: Optional[TelemetryFrame] = Field(
        default=None,
        description="Latest live telemetry frame (in-memory only, not stored)",
    )
    audit_log: list[AuditLogEntry] = Field(
        default_factory=list,
        description="Audit entries linking Firestore logs to blockchain anchors",
    )

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_firestore(self) -> dict:
        """Serialize to a Firestore-compatible dict (excludes runtime-only fields)."""
        return {
            "mission_name": self.mission_name,
            "created_by":   self.created_by,
            "status":       self.status.value,
            "channel_id":   self.channel_id,
            "created_at":   self.created_at.isoformat(),
            "updated_at":   self.updated_at.isoformat(),
        }

    @classmethod
    def from_firestore(cls, doc_id: str, data: dict) -> "Mission":
        """Reconstruct a Mission from a Firestore document (audit_log loaded separately)."""
        data = dict(data)
        data["id"] = doc_id
        for ts_field in ("created_at", "updated_at"):
            if isinstance(data.get(ts_field), str):
                data[ts_field] = datetime.fromisoformat(data[ts_field])
        # telemetry and audit_log are not stored — keep defaults
        data.setdefault("mission_telemetry", None)
        data.setdefault("audit_log", [])
        return cls(**data)

    def touch(self) -> None:
        """Update the updated_at timestamp (call before any Firestore write)."""
        self.updated_at = datetime.now(timezone.utc)

    class Config:
        use_enum_values = False   # keep enum objects in memory

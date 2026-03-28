"""
Audit Storage Backend — abstract interface + implementations.

Active backend:  Firestore  (current)
Future backend:  IPFS       (swap by setting IPFS_GATEWAY in .env)

Switching to IPFS requires no changes to services.py or blockchain.py —
just set IPFS_GATEWAY and the factory returns the IPFS implementation.
"""
import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

AUDIT_LOGS_COLLECTION = "audit_logs"


# ── Abstract Interface ─────────────────────────────────────────────────────────

class AuditStorageBackend(ABC):

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Identifier string: 'firestore' | 'ipfs'"""
        ...

    @abstractmethod
    async def store(self, encrypted_data: str, metadata: dict) -> str:
        """
        Persist encrypted audit data.
        Returns a reference string (Firestore doc_id  OR  IPFS CID).
        """
        ...

    @abstractmethod
    async def retrieve(self, ref: str) -> Optional[dict]:
        """Fetch a full log document by its reference."""
        ...

    @abstractmethod
    async def list_logs(
        self,
        mission_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Return recent audit logs, optionally scoped to a mission."""
        ...

    @abstractmethod
    async def update_tx_hash(self, ref: str, tx_hash: str, log_index: int) -> None:
        """Back-fill the blockchain tx_hash and log_index after the chain write."""
        ...


# ── Firestore Implementation ───────────────────────────────────────────────────

class FirestoreAuditStorage(AuditStorageBackend):
    """Stores encrypted audit logs in Firestore as the primary off-chain backend."""

    @property
    def backend_name(self) -> str:
        return "firestore"

    def _col(self):
        from app.core.firebase import get_firestore
        return get_firestore().collection(AUDIT_LOGS_COLLECTION)

    async def store(self, encrypted_data: str, metadata: dict) -> str:
        """Write log to Firestore; returns the document ID."""
        doc_id = uuid.uuid4().hex  # 32-char hex string

        document = {
            "encrypted_data": encrypted_data,
            "storage_backend": self.backend_name,
            "tx_hash": None,        # filled in after blockchain write
            "log_index": None,      # filled in after blockchain write
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metadata,
        }

        def _write():
            self._col().document(doc_id).set(document)

        await asyncio.to_thread(_write)
        logger.info("Audit log stored in Firestore [%s]", doc_id)
        return doc_id

    async def retrieve(self, ref: str) -> Optional[dict]:
        def _get():
            return self._col().document(ref).get()

        doc = await asyncio.to_thread(_get)
        if not doc.exists:
            return None
        data = doc.to_dict()
        data["id"] = doc.id
        return data

    async def list_logs(
        self,
        mission_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        def _query():
            ref = self._col()
            if mission_id:
                # Simple equality filter — no composite index needed
                ref = ref.where("mission_id", "==", mission_id)
            else:
                ref = ref.order_by("timestamp", direction="DESCENDING")
            return list(ref.limit(limit).stream())

        docs = await asyncio.to_thread(_query)
        result = []
        for doc in docs:
            data = doc.to_dict()
            data["id"] = doc.id
            data.pop("encrypted_data", None)   # never expose raw cipher in listings
            result.append(data)

        # Sort in Python when mission_id filter is applied (avoids composite index)
        if mission_id:
            result.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return result

    async def update_tx_hash(self, ref: str, tx_hash: str, log_index: int) -> None:
        def _update():
            self._col().document(ref).update({
                "tx_hash": tx_hash,
                "log_index": log_index,
            })

        await asyncio.to_thread(_update)
        logger.info("Firestore log [%s] linked to tx %s (index %d)", ref, tx_hash, log_index)


# ── IPFS Implementation (stub — ready for future activation) ──────────────────

class IPFSAuditStorage(AuditStorageBackend):
    """
    IPFS backend — to be activated by setting IPFS_GATEWAY in .env.

    Key difference from Firestore:
      • The CID returned by IPFS IS the content hash.
      • No separate SHA-256 step needed; pass the CID as both
        storage_ref AND data_hash when calling the blockchain.
    """

    def __init__(self, gateway_url: str):
        self.gateway_url = gateway_url

    @property
    def backend_name(self) -> str:
        return "ipfs"

    async def store(self, encrypted_data: str, metadata: dict) -> str:
        # TODO: pip install ipfshttpclient
        # import ipfshttpclient
        # async with ipfshttpclient.connect(self.gateway_url) as client:
        #     result = await client.add_json({"encrypted_data": encrypted_data, **metadata})
        #     return result["Hash"]  # CID
        raise NotImplementedError("IPFS backend — set IPFS_GATEWAY in .env to activate")

    async def retrieve(self, ref: str) -> Optional[dict]:
        raise NotImplementedError("IPFS retrieve not yet implemented")

    async def list_logs(self, mission_id: Optional[str] = None, limit: int = 100) -> list[dict]:
        raise NotImplementedError("IPFS list requires an external index layer")

    async def update_tx_hash(self, ref: str, tx_hash: str, log_index: int) -> None:
        # IPFS objects are immutable — tx_hash is stored elsewhere (e.g. a thin Firestore index)
        logger.info("IPFS: tx_hash %s for CID %s noted (no mutation needed)", tx_hash, ref)


# ── Factory ────────────────────────────────────────────────────────────────────

def get_storage_backend() -> AuditStorageBackend:
    """Return the active audit storage backend based on current config."""
    from app.core.config import settings
    ipfs_gw = getattr(settings, "ipfs_gateway", "")
    if ipfs_gw:
        logger.info("Using IPFS audit storage backend (%s)", ipfs_gw)
        return IPFSAuditStorage(ipfs_gw)
    return FirestoreAuditStorage()

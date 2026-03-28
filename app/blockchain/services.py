"""
Audit Log Service — the single entry point for all audit logging.

<<<<<<< HEAD
# All functions
def _receipt_field(receipt, key: str):
    """Read a field from web3 receipt objects or plain dicts."""
    if isinstance(receipt, dict):
        return receipt.get(key)
    return getattr(receipt, key, None)


def create_log(event_type: str, data: str):
    data_hash = create_hash(data)
    encrypted = encrypt(data)

    receipt = add_log(event_type, encrypted)
    tx_hash = _receipt_field(receipt, "transactionHash")
    block_hash = _receipt_field(receipt, "blockHash")
    block_number = _receipt_field(receipt, "blockNumber")

    return {
        "tx": tx_hash.hex() if tx_hash is not None else None,
        "block_hash": block_hash.hex() if block_hash is not None else None,
        "block_number": int(block_number) if block_number is not None else None,
        "hash": data_hash,
=======
Full hybrid flow:
    1. Serialize event payload → canonical JSON
    2. SHA-256 hash  →  data_hash
    3. AES-Fernet encrypt  →  encrypted_data
    4. Store encrypted_data in active backend (Firestore now, IPFS later)
       returns storage_ref (doc_id or CID)
    5. Write { event_type, data_hash, storage_ref, backend } to blockchain
       returns { tx_hash, log_index }
    6. Back-fill tx_hash + log_index into the storage document

Verification:
    retrieve(ref) → decrypt → hash → compare with on-chain data_hash  ✅ / ❌
"""
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from app.blockchain.audit_storage import get_storage_backend
from app.blockchain.crypto_utils import hash_and_encrypt, decrypt_and_verify
from app.blockchain import blockchain as chain

logger = logging.getLogger(__name__)


# ── Event Types ────────────────────────────────────────────────────────────────

class AuditEventType(str, Enum):
    ANOMALY_DETECTED   = "ANOMALY_DETECTED"
    THREAT_CLASSIFIED  = "THREAT_CLASSIFIED"
    DEFENSE_ACTIVATED  = "DEFENSE_ACTIVATED"


# ── Core service functions ─────────────────────────────────────────────────────

async def create_audit_log(
    event_type: AuditEventType,
    payload: dict,
    mission_id: Optional[str] = None,
) -> dict:
    """
    Create a fully hybrid audit log entry.

    Args:
        event_type : One of AuditEventType
        payload    : Arbitrary dict (anomaly data, threat info, defense action, etc.)
        mission_id : Optional mission context

    Returns:
        {
            storage_ref, storage_backend,
            data_hash,
            tx_hash, log_index,
            timestamp
        }
    """
    storage = get_storage_backend()
    timestamp = datetime.now(timezone.utc).isoformat()

    # Enrich payload with standard fields
    enriched = {
        "event_type": event_type.value,
        "mission_id": mission_id,
        "timestamp": timestamp,
        **payload,
    }

    # Step 1: Hash + Encrypt
    data_hash, encrypted_data = hash_and_encrypt(enriched)

    # Step 2: Store encrypted data off-chain
    metadata = {
        "event_type": event_type.value,
        "data_hash": data_hash,
        "mission_id": mission_id,
    }
    storage_ref = await storage.store(encrypted_data, metadata)

    # Step 3: Anchor hash to blockchain
    chain_result = {"tx_hash": None, "log_index": None}
    try:
        chain_result = await chain.add_log(
            event_type.value,
            data_hash,
            storage_ref,
            storage.backend_name,
        )
        logger.info(
            "Blockchain anchor written — event: %s | tx: %s | index: %d",
            event_type.value,
            chain_result["tx_hash"],
            chain_result["log_index"],
        )
    except Exception as exc:
        # Blockchain unavailable — log is still safely in Firestore
        logger.warning("Blockchain write failed (log still in storage): %s", exc)

    # Step 4: Back-fill tx_hash into storage document
    if chain_result.get("tx_hash"):
        await storage.update_tx_hash(
            storage_ref,
            chain_result["tx_hash"],
            chain_result["log_index"],
        )

    return {
        "storage_ref":     storage_ref,
        "storage_backend": storage.backend_name,
        "data_hash":       data_hash,
        "tx_hash":         chain_result.get("tx_hash"),
        "log_index":       chain_result.get("log_index"),
        "timestamp":       timestamp,
>>>>>>> df753935eb55f49dbdfaddfb494e4f0fb43994db
    }


async def get_audit_log(storage_ref: str) -> Optional[dict]:
    """
    Retrieve and decrypt a single audit log, verifying against the blockchain hash.

    Returns the decrypted payload with an `is_authentic` field.
    """
    storage = get_storage_backend()
    doc = await storage.retrieve(storage_ref)
    if doc is None:
        return None

    encrypted_data = doc.get("encrypted_data")
    expected_hash  = doc.get("data_hash")

    if not encrypted_data or not expected_hash:
        logger.error("Log [%s] is missing encrypted_data or data_hash", storage_ref)
        return None

    payload, is_authentic = decrypt_and_verify(encrypted_data, expected_hash)

    return {
        "id":              storage_ref,
        "event_type":      doc.get("event_type"),
        "storage_backend": doc.get("storage_backend"),
        "tx_hash":         doc.get("tx_hash"),
        "log_index":       doc.get("log_index"),
        "timestamp":       doc.get("timestamp"),
        "mission_id":      doc.get("mission_id"),
        "data":            payload,
        "is_authentic":    is_authentic,
    }

<<<<<<< HEAD
def get_log_count():
    return {
        "logCount": no_of_logs()
    }
=======

async def list_audit_logs(
    mission_id: Optional[str] = None,
    limit: int = 100,
) -> list[dict]:
    """
    List audit log summaries (no decrypted data, no encrypted_data).
    Suitable for displaying event history in the frontend.
    """
    storage = get_storage_backend()
    return await storage.list_logs(mission_id=mission_id, limit=limit)


async def get_blockchain_log(index: int) -> dict:
    """Read a raw on-chain entry by its log_index."""
    return await chain.get_log(index)


async def get_blockchain_log_count() -> int:
    """Return the total number of anchored entries on-chain."""
    return await chain.get_log_count()
>>>>>>> df753935eb55f49dbdfaddfb494e4f0fb43994db

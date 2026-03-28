import hashlib
import json
import time
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["blockchain"])

try:
    from app.blockchain.services import create_log, read_log, get_log_count
    _blockchain_import_error: Optional[str] = None
except Exception as e:  # pragma: no cover - runtime environment dependent
    create_log = None
    read_log = None
    get_log_count = None
    _blockchain_import_error = str(e)
    logger.warning("Blockchain services unavailable, using local fallback only: %s", e)


class ThreatLogRequest(BaseModel):
    event_type: str = "threat"
    payload: Dict[str, Any]


_local_logs: list[Dict[str, Any]] = []


def _stable_hash(payload: Dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _append_local_log(event_type: str, payload: Dict[str, Any], payload_hash: str) -> Dict[str, Any]:
    now = time.time()
    entry = {
        "eventType": event_type,
        "data": payload,
        "hash": payload_hash,
        "timestamp": now,
    }
    _local_logs.append(entry)
    return {
        "verified": False,
        "block_hash": None,
        "tx_hash": None,
        "verification_time": now,
        "integrity_score": 0.65,
        "payload_hash": payload_hash,
        "storage_mode": "local",
    }


def record_threat_log(event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Record a threat payload hash on chain when available.
    Falls back to local in-memory log if blockchain deps/provider are unavailable.
    """
    payload_hash = _stable_hash(payload)

    if create_log is None:
        return _append_local_log(event_type, payload, payload_hash)

    try:
        result = create_log(event_type, json.dumps(payload, sort_keys=True))
        tx_hash = result.get("tx")
        block_hash = result.get("block_hash")
        return {
            "verified": bool(tx_hash),
            "block_hash": block_hash,
            "tx_hash": tx_hash,
            "verification_time": time.time(),
            "integrity_score": 1.0 if tx_hash else 0.8,
            "payload_hash": result.get("hash", payload_hash),
            "storage_mode": "onchain" if tx_hash else "local",
        }
    except Exception as e:
        logger.warning("On-chain log write failed, using local fallback: %s", e)
        return _append_local_log(event_type, payload, payload_hash)


@router.get("/api/v1/blockchain/status")
async def blockchain_status():
    return {
        "enabled": create_log is not None,
        "import_error": _blockchain_import_error,
        "local_log_count": len(_local_logs),
    }


@router.post("/api/v1/blockchain/logs")
async def create_blockchain_log(request: ThreatLogRequest):
    return record_threat_log(request.event_type, request.payload)


@router.get("/api/v1/blockchain/logs/count")
async def blockchain_log_count():
    if get_log_count is not None:
        try:
            return get_log_count()
        except Exception as e:
            logger.warning("get_log_count failed, returning local count: %s", e)
    return {"logCount": len(_local_logs)}


@router.get("/api/v1/blockchain/logs/{index}")
async def blockchain_log_at(index: int):
    if index < 0:
        raise HTTPException(status_code=400, detail="index must be >= 0")

    if read_log is not None:
        try:
            return read_log(index)
        except Exception as e:
            logger.warning("read_log failed for index %s: %s", index, e)

    if index >= len(_local_logs):
        raise HTTPException(status_code=404, detail="log index out of range")
    return _local_logs[index]

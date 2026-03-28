"""
Comprehensive blockchain implementation test — ASCII output (Windows compatible).
Tests every layer: crypto -> storage -> blockchain -> verification.

Run from project root:
    python tools/test_blockchain.py
"""
import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

PASS = "[PASS]"
FAIL = "[FAIL]"
SEP  = "-" * 60

_failures = []


def header(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def check(label: str, condition: bool, detail: str = ""):
    icon = PASS if condition else FAIL
    msg  = f"  {icon}  {label}"
    if detail:
        msg += f"\n         -> {detail}"
    print(msg)
    if not condition:
        _failures.append(label)


# ============================================================
# LAYER 1: Crypto Utils
# ============================================================

def test_crypto():
    header("LAYER 1 -- Crypto Utilities")
    from app.blockchain.crypto_utils import (
        encrypt, decrypt, create_hash, hash_and_encrypt, decrypt_and_verify
    )

    payload = {"event": "ANOMALY_DETECTED", "score": 0.97, "channel": "P-1"}

    # Encrypt / Decrypt roundtrip
    encrypted = encrypt(json.dumps(payload))
    decrypted  = json.loads(decrypt(encrypted))
    check("Encrypt -> Decrypt roundtrip", decrypted == payload)

    # Hash determinism
    h1 = create_hash(json.dumps(payload, sort_keys=True))
    h2 = create_hash(json.dumps(payload, sort_keys=True))
    check("SHA-256 is deterministic", h1 == h2, f"hash: {h1[:32]}...")

    # hash_and_encrypt + decrypt_and_verify
    data_hash, encrypted_data = hash_and_encrypt(payload)
    restored, is_authentic = decrypt_and_verify(encrypted_data, data_hash)
    check("hash_and_encrypt + decrypt_and_verify (authentic)", is_authentic)
    check("Restored payload matches original", restored == payload)

    # Tamper detection
    tampered = encrypted_data[:-4] + "XXXX"
    try:
        _, is_auth = decrypt_and_verify(tampered, data_hash)
        check("Tamper detected (bad cipher -> not authentic)", not is_auth)
    except Exception:
        check("Tamper detected (bad cipher -> exception raised)", True)

    print(f"\n  Encryption key prefix: {os.getenv('ARGUS_ENCRYPTION_KEY', '')[:8]}...")


# ============================================================
# LAYER 2: Blockchain (Web3 + Smart Contract)
# ============================================================

async def test_blockchain():
    header("LAYER 2 -- Blockchain (Web3 + Smart Contract)")
    from app.blockchain import blockchain as chain
    from web3 import Web3
    from app.core.config import settings

    w3 = Web3(Web3.HTTPProvider(settings.blockchain_rpc_url))
    check("Connected to Hardhat node", w3.is_connected(),
          f"chain_id={w3.eth.chain_id}, rpc={settings.blockchain_rpc_url}")

    count_before = await chain.get_log_count()
    check("getLogCount() returns int", isinstance(count_before, int),
          f"count={count_before}")

    result = await chain.add_log(
        "ANOMALY_DETECTED",
        "deadbeef" * 8,          # fake 64-char hash
        "firestore_test_doc_001",
        "firestore",
    )
    check("addLog() returns tx_hash", bool(result.get("tx_hash")),
          f"tx={result['tx_hash'][:32]}...")
    check("addLog() returns log_index", result.get("log_index") is not None,
          f"index={result['log_index']}")

    count_after = await chain.get_log_count()
    check("Log count incremented by 1", count_after == count_before + 1,
          f"{count_before} -> {count_after}")

    log = await chain.get_log(result["log_index"])
    check("getLog event_type correct",     log["event_type"]      == "ANOMALY_DETECTED")
    check("getLog data_hash correct",      log["data_hash"]        == "deadbeef" * 8)
    check("getLog storage_ref correct",    log["storage_ref"]      == "firestore_test_doc_001")
    check("getLog storage_backend correct",log["storage_backend"]  == "firestore")
    check("getLog has timestamp",          log["timestamp"]        > 0,
          f"timestamp={log['timestamp']}")


# ============================================================
# LAYER 3: Full Hybrid Audit Service
# ============================================================

async def test_audit_service():
    header("LAYER 3 -- Hybrid Audit Service (Firestore + Blockchain)")
    from app.blockchain.services import (
        create_audit_log, get_audit_log, list_audit_logs,
        get_blockchain_log_count, AuditEventType
    )

    events = [
        (AuditEventType.ANOMALY_DETECTED,  {"channel": "P-1",   "score": 0.97,  "threshold": 0.85}),
        (AuditEventType.THREAT_CLASSIFIED, {"threat": "GPS_SPOOFING", "confidence": 0.94}),
        (AuditEventType.DEFENSE_ACTIVATED, {"action": "IMU_FAILOVER", "target_channel": "P-1"}),
    ]

    created = []
    for event_type, payload in events:
        result = await create_audit_log(
            event_type=event_type,
            payload=payload,
            mission_id="MSN_TEST_01",
        )
        check(
            f"create_audit_log [{event_type.value}]",
            bool(result.get("storage_ref")) and bool(result.get("tx_hash")),
            f"ref={result['storage_ref'][:16]}... tx={result['tx_hash'][:16]}..."
        )
        created.append(result)

    # Retrieve + verify each log
    for i, result in enumerate(created):
        log = await get_audit_log(result["storage_ref"])
        event_name = events[i][0].value
        check(f"get_audit_log is_authentic [{event_name}]",
              log is not None and log["is_authentic"])
        check(f"Decrypted event_type matches [{event_name}]",
              log["data"]["event_type"] == event_name)
        check(f"tx_hash back-filled in Firestore [{event_name}]",
              bool(log.get("tx_hash")))

    # List by mission
    logs = await list_audit_logs(mission_id="MSN_TEST_01")
    check("list_audit_logs by mission_id returns >= 3 logs", len(logs) >= 3,
          f"found={len(logs)}")
    check("encrypted_data NOT exposed in listing",
          all("encrypted_data" not in l for l in logs))

    # On-chain total
    on_chain = await get_blockchain_log_count()
    check("On-chain log count >= 3 (cumulative)", on_chain >= 3,
          f"total on chain={on_chain}")


# ============================================================
# LAYER 4: Tamper Detection
# ============================================================

async def test_tamper_detection():
    header("LAYER 4 -- Tamper Detection")
    from app.blockchain.services import create_audit_log, AuditEventType
    from app.blockchain.crypto_utils import decrypt_and_verify
    from app.core.firebase import get_firestore
    import asyncio as _asyncio

    result = await create_audit_log(
        event_type=AuditEventType.THREAT_CLASSIFIED,
        payload={"threat": "JAMMING", "confidence": 0.91},
        mission_id="MSN_TAMPER_TEST",
    )
    ref       = result["storage_ref"]
    data_hash = result["data_hash"]

    db = get_firestore()
    def _get():
        return db.collection("audit_logs").document(ref).get()
    doc = await _asyncio.to_thread(_get)
    enc = doc.to_dict()["encrypted_data"]

    _, is_authentic = decrypt_and_verify(enc, data_hash)
    check("Original data authenticates against hash", is_authentic)

    # Corrupt the ciphertext
    corrupted = enc[:-8] + "TAMPERED"
    try:
        _, is_auth_after = decrypt_and_verify(corrupted, data_hash)
        check("Corrupted data detected as NOT authentic", not is_auth_after)
    except Exception as e:
        check("Corrupted data raises exception (tamper caught)", True,
              f"ex={type(e).__name__}")


# ============================================================
# Main
# ============================================================

async def main():
    print("\n" + "=" * 60)
    print("  ARGUS Blockchain Implementation -- Full Test Suite")
    print("=" * 60)

    test_crypto()
    await test_blockchain()
    await test_audit_service()
    await test_tamper_detection()

    print(f"\n{'=' * 60}")
    if _failures:
        print(f"  {len(_failures)} test(s) FAILED:")
        for f in _failures:
            print(f"    {FAIL} {f}")
        sys.exit(1)
    else:
        print(f"  {PASS}  All tests passed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

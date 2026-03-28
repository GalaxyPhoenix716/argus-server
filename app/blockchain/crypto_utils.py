"""
Cryptographic utilities — AES-Fernet encryption and SHA-256 hashing.

The encryption key MUST be set as ARGUS_ENCRYPTION_KEY in .env.
Generate a fresh key with:
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

WARNING: Losing this key makes ALL encrypted audit logs permanently unreadable.
"""
import hashlib
import json
import logging
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

_fernet: Fernet | None = None


def _get_fernet() -> Fernet:
    global _fernet
    if _fernet is None:
        from app.core.config import settings
        key = settings.argus_encryption_key
        if not key:
            raise RuntimeError(
                "ARGUS_ENCRYPTION_KEY is not set in .env. "
                "Generate one: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
            )
        _fernet = Fernet(key.encode())
    return _fernet


def encrypt(data: str) -> str:
    """Encrypt a UTF-8 string using AES-128-CBC via Fernet."""
    return _get_fernet().encrypt(data.encode()).decode()


def decrypt(encrypted_data: str) -> str:
    """Decrypt a Fernet-encrypted string back to plaintext."""
    return _get_fernet().decrypt(encrypted_data.encode()).decode()


def create_hash(data: str) -> str:
    """Return the SHA-256 hex digest of a string."""
    return hashlib.sha256(data.encode()).hexdigest()


def hash_and_encrypt(payload: dict) -> tuple[str, str]:
    """
    Deterministically serialize, hash, then encrypt a dict payload.

    Returns:
        (data_hash, encrypted_data)
        - data_hash     : SHA-256 of the canonical JSON string  →  goes on blockchain
        - encrypted_data: Fernet cipher blob                    →  goes in storage backend
    """
    canonical = json.dumps(payload, sort_keys=True, default=str)
    data_hash = create_hash(canonical)
    encrypted = encrypt(canonical)
    return data_hash, encrypted


def decrypt_and_verify(encrypted_data: str, expected_hash: str) -> tuple[dict, bool]:
    """
    Decrypt encrypted_data and verify its integrity against expected_hash.

    Returns:
        (payload_dict, is_authentic)
        - is_authentic: True if the decrypted data's hash matches expected_hash
    """
    canonical = decrypt(encrypted_data)
    computed_hash = create_hash(canonical)
    is_authentic = computed_hash == expected_hash
    if not is_authentic:
        logger.warning("Hash mismatch! Data may have been tampered with.")
    return json.loads(canonical), is_authentic
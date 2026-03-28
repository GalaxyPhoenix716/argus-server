from .crypto_utils import encrypt, decrypt, create_hash
from .blockchain import add_log, get_log, no_of_logs

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
    }


def read_log(index: int):
    log = get_log(index)

    event_type = log[0]
    encrypted_data = log[1]
    timestamp = log[2]

    decrypted = decrypt(encrypted_data)

    return {
        "eventType": event_type,
        "data": decrypted,
        "timestamp": timestamp
    }

def get_log_count():
    return {
        "logCount": no_of_logs()
    }

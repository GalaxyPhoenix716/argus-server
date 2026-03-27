from crypto_utils import encrypt, decrypt, create_hash
from blockchain import add_log, get_log, no_of_logs

#All functions
def create_log(event_type: str, data: str):
    data_hash = create_hash(data)
    encrypted = encrypt(data)

    receipt = add_log(event_type, encrypted)

    return {
        "tx": receipt.transactionHash.hex(),
        "hash": data_hash
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
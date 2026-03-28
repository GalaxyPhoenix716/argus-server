"""
Web3 / Blockchain layer — connects to the Hardhat local node and calls
the ArgusAuditLog smart contract.

The contract stores only: eventType, dataHash, storageRef, storageBackend.
Full encrypted data lives in the off-chain storage backend (Firestore → IPFS).
"""
import asyncio
import logging
from web3 import Web3

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── ABI (must match ArgusAuditLog.sol) ────────────────────────────────────────
CONTRACT_ABI = [
    {
        "inputs": [
            {"internalType": "string", "name": "eventType",      "type": "string"},
            {"internalType": "string", "name": "dataHash",       "type": "string"},
            {"internalType": "string", "name": "storageRef",     "type": "string"},
            {"internalType": "string", "name": "storageBackend", "type": "string"},
        ],
        "name": "addLog",
        "outputs": [{"internalType": "uint256", "name": "logIndex", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "index", "type": "uint256"}],
        "name": "getLog",
        "outputs": [
            {"internalType": "string",  "name": "eventType",      "type": "string"},
            {"internalType": "string",  "name": "dataHash",       "type": "string"},
            {"internalType": "string",  "name": "storageRef",     "type": "string"},
            {"internalType": "string",  "name": "storageBackend", "type": "string"},
            {"internalType": "uint256", "name": "timestamp",      "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "getLogCount",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True,  "internalType": "uint256", "name": "logIndex",      "type": "uint256"},
            {"indexed": False, "internalType": "string",  "name": "eventType",     "type": "string"},
            {"indexed": False, "internalType": "string",  "name": "dataHash",      "type": "string"},
            {"indexed": False, "internalType": "string",  "name": "storageRef",    "type": "string"},
            {"indexed": False, "internalType": "string",  "name": "storageBackend","type": "string"},
            {"indexed": False, "internalType": "uint256", "name": "timestamp",     "type": "uint256"},
        ],
        "name": "LogCreated",
        "type": "event",
    },
]

# ── Web3 connection ────────────────────────────────────────────────────────────
_w3: Web3 | None = None
_contract = None


def _get_w3() -> Web3:
    global _w3
    if _w3 is None:
        _w3 = Web3(Web3.HTTPProvider(settings.blockchain_rpc_url))
        if not _w3.is_connected():
            logger.warning(
                "Blockchain node not reachable at %s — logs will be skipped",
                settings.blockchain_rpc_url,
            )
    return _w3


def _get_contract():
    global _contract
    if _contract is None:
        w3 = _get_w3()
        address = w3.to_checksum_address(settings.contract_address)
        _contract = w3.eth.contract(address=address, abi=CONTRACT_ABI)
    return _contract


# ── Public API (sync — wrapped in asyncio.to_thread by services.py) ───────────

def add_log_sync(event_type: str, data_hash: str, storage_ref: str, storage_backend: str) -> dict:
    """
    Write an audit anchor to the blockchain.
    Returns { tx_hash, log_index, block_number }.
    """
    w3 = _get_w3()
    if not w3.is_connected():
        raise ConnectionError("Blockchain node is not available")

    contract = _get_contract()
    tx_hash = contract.functions.addLog(
        event_type, data_hash, storage_ref, storage_backend
    ).transact({"from": w3.eth.accounts[0]})

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    # Decode the LogCreated event to get log_index
    logs = contract.events.LogCreated().process_receipt(receipt)
    log_index = logs[0]["args"]["logIndex"] if logs else None

    return {
        "tx_hash": receipt.transactionHash.hex(),
        "log_index": log_index,
        "block_number": receipt.blockNumber,
    }


def get_log_sync(index: int) -> dict:
    """Read a log entry from the blockchain by index."""
    result = _get_contract().functions.getLog(index).call()
    return {
        "event_type":      result[0],
        "data_hash":       result[1],
        "storage_ref":     result[2],
        "storage_backend": result[3],
        "timestamp":       result[4],
    }


def get_log_count_sync() -> int:
    """Return the total number of on-chain log entries."""
    return _get_contract().functions.getLogCount().call()


# ── Async wrappers ─────────────────────────────────────────────────────────────

async def add_log(event_type: str, data_hash: str, storage_ref: str, storage_backend: str) -> dict:
    return await asyncio.to_thread(add_log_sync, event_type, data_hash, storage_ref, storage_backend)


async def get_log(index: int) -> dict:
    return await asyncio.to_thread(get_log_sync, index)


async def get_log_count() -> int:
    return await asyncio.to_thread(get_log_count_sync)

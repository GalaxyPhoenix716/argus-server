"""
Deploy ArgusAuditLog smart contract to the local Hardhat node.

Prerequisites:
    1. Hardhat node running:  npx hardhat node  (in app/blockchain/)
    2. Contract compiled:     npx hardhat compile

Usage (from project root):
    python tools/deploy_contract.py

On success, copies the deployed address to your clipboard and prints the
line to add to .env.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from web3 import Web3

# ── Config ─────────────────────────────────────────────────────────────────────
ARTIFACT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "app", "blockchain", "artifacts",
    "contracts", "ArgusAuditLog.sol",
    "ArgusAuditLog.json",
)
RPC_URL = os.getenv("BLOCKCHAIN_RPC_URL", "http://127.0.0.1:8545")


def deploy():
    print(f"\n🔗 Connecting to Hardhat node at {RPC_URL}…")
    w3 = Web3(Web3.HTTPProvider(RPC_URL))

    if not w3.is_connected():
        print("❌  Cannot connect to Hardhat node.")
        print("    Make sure it's running:  cd app/blockchain && npx hardhat node")
        sys.exit(1)

    print(f"✅  Connected  |  chain ID: {w3.eth.chain_id}")

    # Load compiled artifact
    with open(ARTIFACT_PATH) as f:
        artifact = json.load(f)

    abi      = artifact["abi"]
    bytecode = artifact["bytecode"]

    # Use the first Hardhat test account as deployer
    deployer = w3.eth.accounts[0]
    print(f"🚀  Deploying from account: {deployer}")

    # Deploy
    Contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    tx_hash  = Contract.constructor().transact({"from": deployer})
    receipt  = w3.eth.wait_for_transaction_receipt(tx_hash)

    address = receipt.contractAddress
    print(f"\n✨  ArgusAuditLog deployed!")
    print(f"    Address   : {address}")
    print(f"    Tx hash   : {receipt.transactionHash.hex()}")
    print(f"    Block     : {receipt.blockNumber}")
    print(f"\n📋  Add this to your .env:")
    print(f"    CONTRACT_ADDRESS={address}")

    # Auto-update .env
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    _update_env(env_path, "CONTRACT_ADDRESS", address)
    print(f"\n✅  .env updated automatically with new contract address.")


def _update_env(env_path: str, key: str, value: str):
    """In-place replace or append a key=value pair in .env."""
    with open(env_path, "r") as f:
        lines = f.readlines()

    updated = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            updated = True
            break

    if not updated:
        lines.append(f"{key}={value}\n")

    with open(env_path, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    deploy()

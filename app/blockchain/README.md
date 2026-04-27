# ARGUS Blockchain Module Documentation

This folder implements the ARGUS audit integrity layer: encrypted off-chain storage plus on-chain hash anchoring.

## Purpose

1. Preserve tamper evidence for critical security events.
2. Keep sensitive payload content off-chain (encrypted storage backend).
3. Store minimal immutable anchor metadata on-chain.

## Architecture

The module follows a hybrid pattern:

1. Event payload is serialized and hashed.
2. Payload is encrypted before off-chain storage.
3. Off-chain reference and hash are written to smart contract.
4. Dashboard/API receives verification status and chain metadata.

Core pieces:

1. `contracts/ArgusAuditLog.sol` - smart contract schema and storage
2. `blockchain.py` - Web3 calls (`addLog`, `getLog`, `getLogCount`)
3. `crypto_utils.py` - hash + encryption + verification helpers
4. `audit_storage.py` - storage backend abstraction (`firestore` now, `ipfs` stub)
5. `services.py` - higher-level service facade (currently has unresolved merge markers)

## API Route Integration

Primary route bridge: `app/api/routes/blockchain.py`.

1. `GET /api/v1/blockchain/status`
2. `POST /api/v1/blockchain/logs`
3. `GET /api/v1/blockchain/logs/count`
4. `GET /api/v1/blockchain/logs/{index}`

If blockchain service import fails, the route layer falls back to local in-memory logging so the app remains functional.

## Smart Contract Data Model

`ArgusAuditLog.sol` stores:

1. `eventType`
2. `dataHash`
3. `storageRef`
4. `storageBackend`
5. `timestamp`

This keeps chain state compact and avoids storing full event bodies on-chain.

## Configuration Inputs

Configured through `app/core/config.py` and `.env`:

1. `ARGUS_ENCRYPTION_KEY`
2. `BLOCKCHAIN_RPC_URL`
3. `CONTRACT_ADDRESS`
4. `PRIVATE_KEY`
5. `IPFS_GATEWAY` (optional, for future backend switch)

## Local Hardhat Notes

Hardhat project files are colocated in this folder:

1. `hardhat.config.ts`
2. `package.json`
3. `contracts/ArgusAuditLog.sol`

Typical local workflow:

1. `npm install`
2. `npx hardhat node`
3. Deploy `ArgusAuditLog.sol` to local chain
4. Update backend `.env` with deployed `CONTRACT_ADDRESS`

## Verification Flow

1. Retrieve encrypted payload from storage backend.
2. Decrypt payload.
3. Recompute hash.
4. Compare computed hash with anchored hash.
5. Match indicates integrity (`authentic`).

## Known Issues (Current Repo State)

1. `services.py` contains unresolved Git merge markers (`<<<<<<<`, `=======`, `>>>>>>>`), which can break service import.
2. When service import fails, route layer intentionally switches to local fallback mode.
3. `IPFS` backend is scaffolded but not implemented yet in `audit_storage.py`.

## Recommended Improvements

1. Resolve merge conflict in `services.py`.
2. Add deployment script and ABI/artifact loading instead of hardcoded ABI in Python.
3. Add automated tests for route fallback and end-to-end verification path.

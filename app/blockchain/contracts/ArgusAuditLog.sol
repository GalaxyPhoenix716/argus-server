// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/**
 * @title ArgusAuditLog
 * @notice Immutable on-chain anchor for ARGUS audit events.
 *
 * Only the data_hash and a storage reference are stored on-chain.
 * The full encrypted payload lives in an off-chain backend (Firestore → IPFS).
 *
 * Verification flow:
 *   1. Fetch encrypted_data from off-chain storage using storageRef
 *   2. Decrypt → serialize → SHA-256 → computed_hash
 *   3. Assert computed_hash == dataHash stored here  ✅ authentic
 */
contract ArgusAuditLog {

    struct LogEntry {
        string eventType;       // "ANOMALY_DETECTED" | "THREAT_CLASSIFIED" | "DEFENSE_ACTIVATED"
        string dataHash;        // SHA-256 of the encrypted payload (or IPFS CID when on IPFS)
        string storageRef;      // Firestore doc_id  or  IPFS CID
        string storageBackend;  // "firestore" | "ipfs"
        uint256 timestamp;      // block.timestamp (unix seconds, immutable)
    }

    LogEntry[] private _logs;
    address public immutable owner;

    event LogCreated(
        uint256 indexed logIndex,
        string eventType,
        string dataHash,
        string storageRef,
        string storageBackend,
        uint256 timestamp
    );

    constructor() {
        owner = msg.sender;
    }

    /**
     * @notice Append a new audit log entry.
     * @return logIndex The index of the newly created entry.
     */
    function addLog(
        string calldata eventType,
        string calldata dataHash,
        string calldata storageRef,
        string calldata storageBackend
    ) external returns (uint256 logIndex) {
        logIndex = _logs.length;
        _logs.push(LogEntry({
            eventType:      eventType,
            dataHash:       dataHash,
            storageRef:     storageRef,
            storageBackend: storageBackend,
            timestamp:      block.timestamp
        }));
        emit LogCreated(logIndex, eventType, dataHash, storageRef, storageBackend, block.timestamp);
    }

    /**
     * @notice Read a log entry by index.
     */
    function getLog(uint256 index) external view returns (
        string memory eventType,
        string memory dataHash,
        string memory storageRef,
        string memory storageBackend,
        uint256 timestamp
    ) {
        require(index < _logs.length, "ArgusAuditLog: index out of bounds");
        LogEntry storage entry = _logs[index];
        return (entry.eventType, entry.dataHash, entry.storageRef, entry.storageBackend, entry.timestamp);
    }

    /**
     * @notice Total number of log entries.
     */
    function getLogCount() external view returns (uint256) {
        return _logs.length;
    }
}

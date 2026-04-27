# ARGUS Server (FastAPI + AI Pipeline)

Backend service for the ARGUS platform. It ingests telemetry from the simulator, runs anomaly/threat analysis, and streams events to the dashboard.

## Responsibilities

1. Accept telemetry stream from simulator via WebSocket
2. Broadcast telemetry and events to dashboard clients
3. Run AI-driven anomaly + threat classification pipeline
4. Expose auth, mission, simulation-control, and blockchain log APIs

## Tech Stack

1. FastAPI + Uvicorn
2. NumPy / scikit-learn / TensorFlow / PyTorch (AI dependencies)
3. Firebase Admin SDK (mission and user data)
4. Web3 + Hardhat support for blockchain logging

## Getting Started

### Prerequisites

1. Python 3.11+ (recommended)
2. pip

### Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Run

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

1. `GET http://localhost:8000/health`

Interactive docs:

1. `http://localhost:8000/docs`

## API and Stream Endpoints

### Streaming

1. `WS /stream` - dashboard event stream
2. `WS /ws/sim` - simulator uplink

### Simulation control

1. `POST /api/v1/simulation/control`
2. `GET /api/v1/simulation/status`
3. `GET /api/v1/telemetry/latest`

### Auth

1. `POST /api/v1/auth/login`
2. `GET /api/v1/auth/me`

### Mission

1. `GET /api/v1/mission/`

### Blockchain

1. `GET /api/v1/blockchain/status`
2. `POST /api/v1/blockchain/logs`
3. `GET /api/v1/blockchain/logs/count`
4. `GET /api/v1/blockchain/logs/{index}`

## Configuration

Settings are loaded from `.env` via `pydantic-settings` in [`app/core/config.py`](app/core/config.py).

### Core variables

1. `JWT_SECRET`
2. `JWT_ALGORITHM`
3. `JWT_EXPIRE_MINUTES`
4. `FIREBASE_PROJECT_ID`
5. `FIREBASE_PRIVATE_KEY_ID`
6. `FIREBASE_PRIVATE_KEY`
7. `FIREBASE_CLIENT_EMAIL`
8. `FIREBASE_CLIENT_ID`
9. `FIREBASE_CLIENT_CERT_URL`
10. `ARGUS_ENCRYPTION_KEY`
11. `BLOCKCHAIN_RPC_URL`
12. `CONTRACT_ADDRESS`
13. `PRIVATE_KEY`
14. `IPFS_GATEWAY`

If not set, defaults in config are used, allowing local demo startup.

## AI Layer

Telemetry ingestion and AI processing flow is implemented mainly in:

1. [`app/api/routes/sim_receiver.py`](app/api/routes/sim_receiver.py)
2. [`app/ai/AnomalyDetectionEngine`](app/ai/AnomalyDetectionEngine)
3. [`app/ai/ThreatClassificationEngine`](app/ai/ThreatClassificationEngine)

Detailed docs:

1. [AI_LAYER_DOCUMENTATION.md](app/ai/AI_LAYER_DOCUMENTATION.md)
2. [app/ai/README.md](app/ai/README.md)

Blockchain module docs:

1. [app/blockchain/README.md](app/blockchain/README.md)

## Important Implementation Notes

1. Simulation mode changes are informational; threat emission is AI-pipeline-driven.
2. Blockchain API gracefully falls back to local in-memory logging when on-chain services are unavailable.
3. Additional route modules exist (for example threat/explainability), but only routers included in [`app/main.py`](app/main.py) are currently active.
4. `app/blockchain/services.py` currently contains unresolved merge markers, so blockchain service import may fail and trigger fallback mode.

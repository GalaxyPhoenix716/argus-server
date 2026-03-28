"""
ARGUS Server — FastAPI application entry point.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.firebase import initialize_firebase
from app.api.routes import auth
from app.api.routes import streaming
from app.api.routes import sim_receiver
from app.api.routes import blockchain
from app.api.routes import mission

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────────────
    logger.info("Starting ARGUS server…")
    initialize_firebase()
    logger.info("All services initialized ✅")
    yield
    # ── Shutdown ───────────────────────────────────────────────────────────────
    logger.info("ARGUS server shutting down")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ARGUS Mission Security API",
    description="Real-time satellite telemetry monitoring, anomaly detection, and threat classification.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(streaming.router)      # /stream  (dashboard WebSocket)
app.include_router(sim_receiver.router)   # /ws/sim  (simulation WebSocket) + REST endpoints
app.include_router(blockchain.router)     # /api/v1/blockchain/*
app.include_router(mission.router)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
async def health():
    return {"status": "ok", "service": "argus-server"}

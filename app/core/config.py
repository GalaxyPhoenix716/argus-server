from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── JWT ───────────────────────────────────────────────────────────────────
    jwt_secret: str = "demo-secret-key-12345"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440  # 24 hours

    # ── Firebase Admin SDK ────────────────────────────────────────────────────
    firebase_project_id: str = "demo-project"
    firebase_private_key_id: str = "demo-key-id"
    firebase_private_key: str = "demo-key"
    firebase_client_email: str = "demo@example.com"
    firebase_client_id: str = "demo-client-id"
    firebase_client_cert_url: str = "demo-url"

    # ── Blockchain ────────────────────────────────────────────────────────────
    argus_encryption_key: str = ""
    blockchain_rpc_url: str = "http://127.0.0.1:8545"
    contract_address: str = "0x5FbDB2315678afecb367f032d93F642f64180aa3"
    private_key: str = ""

    # ── Storage ───────────────────────────────────────────────────────────────
    ipfs_gateway: str = ""   # set to activate IPFS backend, e.g. /ip4/127.0.0.1/tcp/5001

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

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
    private_key: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── JWT ───────────────────────────────────────────────────────────────────
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440  # 24 hours

    # ── Firebase Admin SDK ────────────────────────────────────────────────────
    firebase_project_id: str
    firebase_private_key_id: str
    firebase_private_key: str           # Full PEM key (newlines as \n in .env)
    firebase_client_email: str
    firebase_client_id: str
    firebase_client_cert_url: str

    # ── Blockchain ────────────────────────────────────────────────────────────
    argus_encryption_key: str = ""
    blockchain_rpc_url: str = "http://127.0.0.1:8545"
    private_key: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

from cryptography.fernet import Fernet
from dotenv import load_dotenv
import hashlib
import os

load_dotenv()
key = os.getenv("ARGUS_ENCRYPTION_KEY", Fernet.generate_key().decode()).encode()
cipher = Fernet(key)

def encrypt(data: str) -> str:
    return cipher.encrypt(data.encode()).decode()

def decrypt(data: str) -> str:
    return cipher.decrypt(data.encode()).decode()

def create_hash(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()
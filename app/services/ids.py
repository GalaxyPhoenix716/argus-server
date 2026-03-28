"""
Centralised ID generation for all ARGUS entities.

User ID  : 5 chars  alphanumeric uppercase  e.g. ADM01, FOR3X
Mission ID: 7 chars  alphanumeric uppercase  e.g. MSN0001, MSN_A3B
"""
import random
import string

_CHARS = string.ascii_uppercase + string.digits


def generate_user_id() -> str:
    """Generate a unique 5-character alphanumeric user ID (A-Z, 0-9)."""
    return "".join(random.choices(_CHARS, k=5))


def generate_mission_id() -> str:
    """Generate a unique 7-character alphanumeric mission ID (A-Z, 0-9)."""
    return "".join(random.choices(_CHARS, k=7))


# ── Validators (for use in Pydantic Field / validators) ───────────────────────

def is_valid_user_id(value: str) -> bool:
    return len(value) == 5 and value.isalnum()


def is_valid_mission_id(value: str) -> bool:
    return len(value) == 7 and value.isalnum()

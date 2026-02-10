from cryptography.fernet import Fernet
import os
from typing import Optional

ENCRYPTION_KEY_ENV = "ENCRYPTION_KEY"


def _get_fernet() -> Fernet:
    key = os.environ.get(ENCRYPTION_KEY_ENV)
    if not key:
        raise EnvironmentError(
            "ENCRYPTION_KEY environment variable not set. Generate one with `Fernet.generate_key()` and set it."
        )
    # Key is already a string (base64 encoded), convert to bytes if needed
    if isinstance(key, str):
        try:
            # Try to use it directly (should be base64 encoded by Fernet)
            key_bytes = key.encode() if isinstance(key, str) else key
            f = Fernet(key_bytes)
            return f
        except Exception as e:
            raise EnvironmentError(f"Invalid ENCRYPTION_KEY format: {e}")
    else:
        return Fernet(key)


def encrypt_bytes(data: bytes) -> bytes:
    """Encrypt raw bytes using Fernet and return token bytes."""
    f = _get_fernet()
    return f.encrypt(data)


def decrypt_bytes(token: bytes) -> bytes:
    """Decrypt a Fernet token and return original bytes."""
    f = _get_fernet()
    return f.decrypt(token)

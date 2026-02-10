import os
from cryptography.fernet import Fernet

from utils.encryption import encrypt_bytes, decrypt_bytes


def test_encrypt_decrypt_roundtrip(monkeypatch):
    key = Fernet.generate_key()
    monkeypatch.setenv('ENCRYPTION_KEY', key.decode())

    data = b"this is a test image bytes" 
    token = encrypt_bytes(data)
    assert isinstance(token, (bytes, bytearray))

    out = decrypt_bytes(token)
    assert out == data

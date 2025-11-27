from cryptography.fernet import Fernet
import os

KEY_PATH = "security/keys/encryption.key"

def load_or_create_key():
    if not os.path.exists(KEY_PATH):
        key = Fernet.generate_key()
        with open(KEY_PATH, "wb") as f:
            f.write(key)
    else:
        key = open(KEY_PATH, "rb").read()
    return Fernet(key)

fernet = load_or_create_key()

def encrypt_image(image_bytes, filename):
    encrypted_data = fernet.encrypt(image_bytes)
    path = f"security/encrypted_images/{filename}.enc"
    with open(path, "wb") as f:
        f.write(encrypted_data)
    return path


import hashlib

def compute_sha256(image_bytes):
    hash_obj = hashlib.sha256()
    hash_obj.update(image_bytes)
    return hash_obj.hexdigest()


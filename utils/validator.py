import imghdr

def validate_image(file_bytes):
    file_type = imghdr.what(None, h=file_bytes)
    if file_type not in ["jpeg", "png", "jpg"]:
        return False, "Unsupported file type."
    if len(file_bytes) > 2 * 1024 * 1024:
        return False, "File too large (max 2MB)."
    return True, "OK"


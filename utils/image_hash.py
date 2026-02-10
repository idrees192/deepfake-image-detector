"""
Image Hashing Utility
Generates hash values for images to detect duplicates.
"""

import hashlib
from PIL import Image
import io

def calculate_image_hash(image_file) -> str:
    """
    Calculates SHA256 hash of an image file.
    
    Args:
        image_file: File-like object or bytes of the image
    
    Returns:
        str: SHA256 hash in hexadecimal format
    """
    # Reset file pointer if it's a file object
    if hasattr(image_file, 'seek'):
        image_file.seek(0)
    
    # Read image bytes
    if hasattr(image_file, 'read'):
        image_bytes = image_file.read()
    else:
        image_bytes = image_file
    
    # Calculate SHA256 hash
    hash_object = hashlib.sha256(image_bytes)
    hash_hex = hash_object.hexdigest()
    
    # Reset file pointer for potential reuse
    if hasattr(image_file, 'seek'):
        image_file.seek(0)
    
    return hash_hex

def get_image_size(image_file) -> int:
    """
    Gets the size of an image file in bytes.
    
    Args:
        image_file: File-like object
    
    Returns:
        int: File size in bytes
    """
    if hasattr(image_file, 'seek'):
        current_pos = image_file.tell()
        image_file.seek(0, 2)  # Seek to end
        size = image_file.tell()
        image_file.seek(current_pos)  # Restore position
        return size
    elif hasattr(image_file, '__len__'):
        return len(image_file)
    return 0

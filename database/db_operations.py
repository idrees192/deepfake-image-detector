"""
Database Operations Module
Handles all database operations for the deepfake detection system.
"""

from datetime import datetime
from typing import Optional, Dict, List, Any
from database.db_connection import get_collection
from database.models import create_test_result

def _import_bson():
    try:
        from bson import ObjectId
        from bson.binary import Binary
        return ObjectId, Binary
    except Exception:
        raise ModuleNotFoundError(
            "BSON (part of PyMongo) is required. Install it with `pip install pymongo[srv]`."
        )


def save_test_result(
    image_hash: str,
    prediction: str,
    confidence_score: float,
    confidence_percentage: float,
    image_filename: str,
    image_size: Optional[int] = None,
    image_bytes: Optional[bytes] = None,
    is_duplicate: bool = False,
    original_test_id: Optional[str] = None
) -> Optional[str]:
    """
    Saves a test result to the database.
    
    Args:
        image_hash: SHA256 hash of the image
        prediction: "REAL" or "FAKE"
        confidence_score: Raw model score
        confidence_percentage: Confidence as percentage
        image_filename: Original filename
        image_size: Image file size in bytes
        is_duplicate: Whether this is a duplicate
        original_test_id: ID of original test if duplicate
    
    Returns:
        str: Inserted document ID, or None if failed
    """
    collection = get_collection()
    if collection is None:
        return None
    
    try:
        ObjectId, Binary = _import_bson()
        from utils.encryption import encrypt_bytes
        
        encrypted_blob = None
        if image_bytes is not None:
            try:
                print(f"Encrypting image bytes... (size: {len(image_bytes)} bytes)")
                encrypted_blob = encrypt_bytes(image_bytes)
                print(f"✅ Image encrypted successfully! (encrypted size: {len(encrypted_blob)} bytes)")
            except Exception as e:
                print(f"❌ Encryption error: {e}")
                encrypted_blob = None

        document = create_test_result(
            image_hash=image_hash,
            prediction=prediction,
            confidence_score=confidence_score,
            confidence_percentage=confidence_percentage,
            image_filename=image_filename,
            image_size=image_size,
            image_encrypted=Binary(encrypted_blob) if encrypted_blob is not None else None,
            is_duplicate=is_duplicate,
            original_test_id=original_test_id
        )
        
        result = collection.insert_one(document)
        return str(result.inserted_id)
    except Exception as e:
        print(f"Error saving test result: {e}")
        return None

def check_duplicate_image(image_hash: str) -> Optional[Dict[str, Any]]:
    """
    Checks if an image with the given hash has been tested before.
    
    Args:
        image_hash: SHA256 hash of the image
    
    Returns:
        dict: Previous test result if duplicate found, None otherwise
    """
    collection = get_collection()
    if collection is None:
        return None
    
    try:
        previous_result = collection.find_one(
            {"image_hash": image_hash},
            sort=[("timestamp", -1)]  # Get most recent
        )
        return previous_result
    except Exception as e:
        print(f"Error checking duplicate: {e}")
        return None

def get_statistics() -> Dict[str, Any]:
    """
    Retrieves statistics about all tests.
    
    Returns:
        dict: Statistics including total tests, real/fake counts, etc.
    """
    collection = get_collection()
    if collection is None:
        return {
            "total_tests": 0,
            "real_count": 0,
            "fake_count": 0,
            "unique_images": 0,
            "duplicate_tests": 0,
            "last_updated": None
        }
    
    try:
        # Total tests
        total_tests = collection.count_documents({})
        
        # Real and fake counts
        real_count = collection.count_documents({"prediction": "REAL"})
        fake_count = collection.count_documents({"prediction": "FAKE"})
        
        # Unique images (distinct hashes)
        unique_images = len(collection.distinct("image_hash"))
        
        # Duplicate tests
        duplicate_tests = collection.count_documents({"is_duplicate": True})
        
        return {
            "total_tests": total_tests,
            "real_count": real_count,
            "fake_count": fake_count,
            "unique_images": unique_images,
            "duplicate_tests": duplicate_tests,
            "last_updated": datetime.utcnow()
        }
    except Exception as e:
        print(f"Error getting statistics: {e}")
        return {
            "total_tests": 0,
            "real_count": 0,
            "fake_count": 0,
            "unique_images": 0,
            "duplicate_tests": 0,
            "last_updated": None
        }

def get_recent_tests(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Retrieves recent test results.
    
    Args:
        limit: Maximum number of results to return
    
    Returns:
        list: List of recent test results
    """
    collection = get_collection()
    if collection is None:
        return []
    
    try:
        results = collection.find().sort("timestamp", -1).limit(limit)
        # Convert ObjectId to string for JSON serialization
        return [
            {**doc, "_id": str(doc["_id"])} 
            for doc in results
        ]
    except Exception as e:
        print(f"Error getting recent tests: {e}")
        return []

def get_duplicate_images() -> List[Dict[str, Any]]:
    """
    Retrieves all duplicate image tests.
    
    Returns:
        list: List of duplicate test results
    """
    collection = get_collection()
    if collection is None:
        return []
    
    try:
        results = collection.find({"is_duplicate": True}).sort("timestamp", -1)
        return [
            {**doc, "_id": str(doc["_id"])} 
            for doc in results
        ]
    except Exception as e:
        print(f"Error getting duplicate images: {e}")
        return []

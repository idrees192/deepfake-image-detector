"""
Database Models for Deepfake Detection System
Defines data structures for storing test results and statistics.
"""

from datetime import datetime
from typing import Optional, Dict, Any

def create_test_result(
    image_hash: str,
    prediction: str,
    confidence_score: float,
    confidence_percentage: float,
    image_filename: str,
    image_size: Optional[int] = None,
    image_encrypted: Optional[bytes] = None,
    is_duplicate: bool = False,
    original_test_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Creates a test result document for MongoDB.
    
    Args:
        image_hash: SHA256 hash of the image
        prediction: "REAL" or "FAKE"
        confidence_score: Raw model score (0.0 to 1.0)
        confidence_percentage: Confidence as percentage
        image_filename: Original filename
        image_size: Image file size in bytes (optional)
        is_duplicate: Whether this is a duplicate image
        original_test_id: ID of original test if duplicate
    
    Returns:
        dict: Document ready for MongoDB insertion
    """
    return {
        "image_hash": image_hash,
        "prediction": prediction,
        "confidence_score": confidence_score,
        "confidence_percentage": confidence_percentage,
        "image_filename": image_filename,
        "image_size": image_size,
        "image_encrypted": image_encrypted,
        "is_duplicate": is_duplicate,
        "original_test_id": original_test_id,
        "timestamp": datetime.utcnow(),
        "created_at": datetime.utcnow()
    }

def create_statistics_document() -> Dict[str, Any]:
    """
    Creates a statistics document structure.
    
    Returns:
        dict: Statistics document structure
    """
    return {
        "total_tests": 0,
        "real_count": 0,
        "fake_count": 0,
        "unique_images": 0,
        "duplicate_tests": 0,
        "last_updated": datetime.utcnow()
    }

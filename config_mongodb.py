"""
MongoDB Configuration
Set your MongoDB Atlas connection string here or use environment variables.
"""

import os

# Load MongoDB connection string from environment variable for security.
# Do NOT keep plaintext credentials in this file. Set the `MONGODB_URI`
# environment variable instead (or use config_mongodb.example.py as a template).

MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    print("Warning: MONGODB_URI not set. Set the MONGODB_URI environment variable before running the app.")

# Database and Collection Names (can be overridden via env vars)
DATABASE_NAME = os.getenv("MONGODB_DATABASE", "deepfake_detection")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION", "test_results")

# Admin Credentials (Change these in production!)
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

"""
MongoDB Atlas Database Connection Module
Handles connection to MongoDB Atlas and database operations.
"""

import os
import sys

def _import_pymongo():
    try:
        from pymongo import MongoClient
        from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
        return MongoClient, ConnectionFailure, ServerSelectionTimeoutError
    except Exception:
        raise ModuleNotFoundError(
            "PyMongo is required but not installed. Install it with `pip install pymongo[srv]`."
        )

def _import_streamlit():
    try:
        import streamlit as st
        return st
    except Exception:
        raise ModuleNotFoundError(
            "Streamlit is required but not installed. Install it with `pip install streamlit`."
        )

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------
# SECURITY FIX: Load credentials from config file or env vars
# ---------------------------------------------------------
try:
    # Try to import from local config file (config_mongodb.py)
    from config_mongodb import MONGO_URI, DATABASE_NAME, COLLECTION_NAME
except ImportError:
    # If file not found (e.g., in cloud deployment), use Environment Variables
    # DO NOT put hardcoded passwords here!
    MONGO_URI = os.getenv("MONGO_URI")
    DATABASE_NAME = os.getenv("DATABASE_NAME", "deepfake_detection")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "test_results")

# Final check to ensure we have a URI
if not MONGO_URI:
    # Use Streamlit secrets if available (for Streamlit Cloud deployment)
    st = _import_streamlit()
    if "MONGO_URI" in st.secrets:
        MONGO_URI = st.secrets["MONGO_URI"]
    else:
        # Stop execution if no URI is found
        raise ValueError("❌ Error: MONGO_URI not found. Please set it in config_mongodb.py or as an Environment Variable.")

def get_database():
    """
    Establishes connection to MongoDB Atlas and returns database instance.
    Uses Streamlit cache to maintain connection across reruns.
    """
    MongoClient, ConnectionFailure, ServerSelectionTimeoutError = _import_pymongo()
    st = _import_streamlit()
    
    # Apply streamlit cache decorator
    @st.cache_resource
    def _connect():
        try:
            # Mask the URI in logs for security
            masked_uri = MONGO_URI[:15] + "..." + MONGO_URI[-10:] if MONGO_URI else "None"
            print(f"Attempting to connect to MongoDB with URI: {masked_uri}")
            
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)
            
            # Test the connection
            client.admin.command('ping')
            db = client[DATABASE_NAME]
            print(f"✅ Connected to MongoDB! Database: {DATABASE_NAME}")
            return db
        except Exception as e:
            error_msg = str(e)
            print(f"❌ MongoDB Connection Error: {error_msg}")
            
            if "authentication failed" in error_msg.lower() or "bad auth" in error_msg.lower():
                st.error("""
                ❌ **Authentication Failed**
                
                Please check your `config_mongodb.py` file to ensure your 
                username and password are correct.
                """)
            else:
                st.error(f"❌ Database Connection Error: {error_msg}")
                st.info("""
                **Troubleshooting:**
                1. Verify internet connection
                2. Check MongoDB Atlas cluster is running
                3. Allow your IP in Network Access settings
                """)
            return None
    
    return _connect()

def get_collection():
    """
    Returns the test results collection.
    """
    db = get_database()
    if db is None:
        return None
    return db[COLLECTION_NAME]

def test_connection():
    """
    Tests the MongoDB connection.
    """
    try:
        db = get_database()
        if db is None:
            return False
        # Try to access the database
        db.list_collection_names()
        return True
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False

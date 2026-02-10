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

try:
    from config_mongodb import MONGODB_URI, DATABASE_NAME, COLLECTION_NAME
except ImportError:
    # Try environment variable first
    MONGODB_URI = os.getenv("MONGODB_URI")
    if not MONGODB_URI:
        # Provide a helpful message
        MONGODB_URI = "mongodb+srv://khanmidrees693_db_user:IdreesMongo123@cluster0.aav7zwl.mongodb.net/deepfake_detection?retryWrites=true&w=majority"
    DATABASE_NAME = "deepfake_detection"
    COLLECTION_NAME = "test_results"

def get_database():
    """
    Establishes connection to MongoDB Atlas and returns database instance.
    Uses Streamlit cache to maintain connection across reruns.
    
    Returns:
        pymongo.database.Database: MongoDB database instance
    """
    MongoClient, ConnectionFailure, ServerSelectionTimeoutError = _import_pymongo()
    st = _import_streamlit()
    
    # Apply streamlit cache decorator
    @st.cache_resource
    def _connect():
        try:
            print(f"Attempting to connect to MongoDB with URI: {MONGODB_URI[:50]}...")
            client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10000)
            # Test the connection
            client.admin.command('ping')
            db = client[DATABASE_NAME]
            print(f"✅ Connected to MongoDB! Database: {DATABASE_NAME}")
            return db
        except Exception as e:
            error_msg = str(e)
            print(f"❌ MongoDB Connection Error: {error_msg}")
            
            if "authentication failed" in error_msg.lower() or "bad auth" in error_msg.lower():
                st.error(f"""
                ❌ **Authentication Failed**
                
                Check your MongoDB credentials:
                - Username: `khanmidrees693_db_user`
                - Password: `Mongo12345`
                - Cluster: `cluster0.aav7zwl.mongodb.net`
                
                **To fix:**
                1. Go to [MongoDB Atlas](https://cloud.mongodb.com)
                2. Click "Database Access"
                3. Verify username and password match
                4. Click "Network Access" and allow your IP
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
    
    Returns:
        pymongo.collection.Collection: MongoDB collection instance
    """
    db = get_database()
    if db is None:
        return None
    return db[COLLECTION_NAME]

def test_connection():
    """
    Tests the MongoDB connection.
    
    Returns:
        bool: True if connection successful, False otherwise
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

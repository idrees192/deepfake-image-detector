"""
Example MongoDB configuration file. DO NOT commit real credentials.

Copy this file to `config_mongodb.py` or set the environment variable
`MONGODB_URI` before running the application.

Replace the placeholder below with your MongoDB connection string.
Format example:
mongodb+srv://<username>:<password>@cluster0.example.mongodb.net/<dbname>?retryWrites=true&w=majority
"""

MONGODB_URI = "mongodb+srv://<username>:<password>@cluster0.example.mongodb.net/<dbname>?retryWrites=true&w=majority"

DATABASE_NAME = "deepfake_detection"
COLLECTION_NAME = "test_results"

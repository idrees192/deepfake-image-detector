"""
Test History Page
Displays all tested images and their complete data.
"""

import streamlit as st
import sys
import os
from datetime import datetime
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from database.db_operations import get_recent_tests, get_statistics
from database.db_connection import get_collection, test_connection

# Page configuration
st.set_page_config(
    page_title="Test History",
    page_icon="📋",
    layout="wide"
)

st.title("📋 Test History - All Tested Images")
st.markdown("---")

# Database connection status
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("### Database Status")
with col2:
    if test_connection():
        st.success("✅ Connected")
    else:
        st.error("❌ Not Connected")
        st.info("Please configure MongoDB Atlas connection. See MONGODB_SETUP.md for instructions.")

st.markdown("---")

# Filters and search
col1, col2, col3 = st.columns(3)

with col1:
    filter_prediction = st.selectbox(
        "Filter by Prediction",
        ["All", "REAL", "FAKE"],
        help="Filter results by prediction type"
    )

with col2:
    filter_duplicate = st.selectbox(
        "Filter by Duplicate Status",
        ["All", "Unique Only", "Duplicates Only"],
        help="Filter by duplicate status"
    )

with col3:
    sort_order = st.selectbox(
        "Sort Order",
        ["Newest First", "Oldest First", "Highest Confidence", "Lowest Confidence"],
        help="Sort results"
    )

# Search by hash or filename
search_term = st.text_input(
    "🔍 Search by Image Hash or Filename",
    placeholder="Enter hash or filename to search...",
    help="Search for specific images by hash or filename"
)

st.markdown("---")

# Get all tests
try:
    collection = get_collection()
    
    if collection is None:
        st.error("❌ Cannot connect to database. Please check your MongoDB Atlas connection.")
        st.stop()
    
    # Build query
    query = {}
    
    if filter_prediction != "All":
        query["prediction"] = filter_prediction
    
    if filter_duplicate == "Unique Only":
        query["is_duplicate"] = False
    elif filter_duplicate == "Duplicates Only":
        query["is_duplicate"] = True
    
    if search_term:
        query["$or"] = [
            {"image_hash": {"$regex": search_term, "$options": "i"}},
            {"image_filename": {"$regex": search_term, "$options": "i"}}
        ]
    
    # Get all matching documents
    all_tests = list(collection.find(query))
    
    # Sort results
    if sort_order == "Newest First":
        all_tests.sort(key=lambda x: x.get("timestamp", datetime.min), reverse=True)
    elif sort_order == "Oldest First":
        all_tests.sort(key=lambda x: x.get("timestamp", datetime.min))
    elif sort_order == "Highest Confidence":
        all_tests.sort(key=lambda x: x.get("confidence_percentage", 0), reverse=True)
    elif sort_order == "Lowest Confidence":
        all_tests.sort(key=lambda x: x.get("confidence_percentage", 0))
    
    # Convert ObjectId to string
    for test in all_tests:
        test["_id"] = str(test["_id"])
    
    # Display statistics
    total_found = len(all_tests)
    stats = get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Results", total_found)
    with col2:
        st.metric("Total Tests", stats["total_tests"])
    with col3:
        st.metric("Unique Images", stats["unique_images"])
    with col4:
        st.metric("Duplicates", stats["duplicate_tests"])
    
    st.markdown("---")
    
    if total_found == 0:
        st.info("No test results found matching your criteria.")
    else:
        # Display results
        st.markdown(f"### 📊 Showing {total_found} Test Result(s)")
        
        # Create expandable sections for each test
        for idx, test in enumerate(all_tests, 1):
            with st.expander(
                f"Test #{idx}: {test.get('image_filename', 'Unknown')} | "
                f"{test.get('prediction', 'N/A')} ({test.get('confidence_percentage', 0):.2f}%) | "
                f"{test.get('timestamp', 'Unknown')}",
                expanded=False
            ):
                # Create columns for better layout
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📝 Test Information")
                    st.write(f"**Test ID**: `{test.get('_id', 'N/A')}`")
                    st.write(f"**Filename**: {test.get('image_filename', 'N/A')}")
                    st.write(f"**Image Hash**: `{test.get('image_hash', 'N/A')}`")
                    st.write(f"**File Size**: {test.get('image_size', 'N/A')} bytes" if test.get('image_size') else "**File Size**: N/A")
                    st.write(f"**Timestamp**: {test.get('timestamp', 'N/A')}")
                
                with col2:
                    st.markdown("#### 🔍 Prediction Results")
                    prediction = test.get('prediction', 'N/A')
                    confidence = test.get('confidence_percentage', 0)
                    
                    if prediction == "REAL":
                        st.success(f"**Prediction**: ✅ {prediction}")
                    else:
                        st.error(f"**Prediction**: 🚨 {prediction}")
                    
                    st.write(f"**Confidence**: {confidence:.2f}%")
                    st.write(f"**Raw Score**: {test.get('confidence_score', 0):.4f}")
                    
                    # Duplicate status
                    if test.get('is_duplicate', False):
                        st.warning("⚠️ **Duplicate Image**")
                        if test.get('original_test_id'):
                            st.write(f"**Original Test ID**: `{test.get('original_test_id')}`")
                    else:
                        st.info("✅ **Unique Image**")
                
                # Additional details
                st.markdown("---")
                st.markdown("#### 📋 Complete Data")
                st.json({
                    "test_id": test.get('_id'),
                    "image_hash": test.get('image_hash'),
                    "image_filename": test.get('image_filename'),
                    "image_size": test.get('image_size'),
                    "prediction": test.get('prediction'),
                    "confidence_score": test.get('confidence_score'),
                    "confidence_percentage": test.get('confidence_percentage'),
                    "is_duplicate": test.get('is_duplicate'),
                    "original_test_id": test.get('original_test_id'),
                    "timestamp": str(test.get('timestamp')),
                    "created_at": str(test.get('created_at'))
                })
        
        # Summary table
        st.markdown("---")
        st.markdown("### 📈 Summary Table")
        
        # Prepare data for table
        table_data = []
        for test in all_tests:
            table_data.append({
                "Test ID": test.get('_id', 'N/A')[:8] + "...",
                "Filename": test.get('image_filename', 'N/A'),
                "Hash (Short)": test.get('image_hash', 'N/A')[:16] + "...",
                "Prediction": test.get('prediction', 'N/A'),
                "Confidence %": f"{test.get('confidence_percentage', 0):.2f}",
                "Duplicate": "Yes" if test.get('is_duplicate', False) else "No",
                "Timestamp": str(test.get('timestamp', 'N/A'))[:19] if test.get('timestamp') else 'N/A'
            })
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Download option
        st.markdown("---")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"test_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

except Exception as e:
    st.error(f"❌ Error loading test history: {e}")
    with st.expander("🔍 Technical Details"):
        import traceback
        st.code(traceback.format_exc())

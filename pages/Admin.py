"""
Admin Dashboard
Admin-only interface for viewing statistics and test history.
"""

import streamlit as st
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from database.db_operations import get_statistics, get_recent_tests, get_duplicate_images
from database.db_connection import test_connection

# Page configuration
st.set_page_config(
    page_title="Admin Dashboard",
    page_icon="🔐",
    layout="wide"
)

# Admin authentication
def check_admin_access():
    """Simple admin authentication using session state."""
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False
    
    if not st.session_state.admin_authenticated:
        st.title("🔐 Admin Login")
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.form("admin_login"):
                st.markdown("### Enter Admin Credentials")
                username = st.text_input("Username", type="default")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login", type="primary")
                
                # Admin credentials from config
                try:
                    from config_mongodb import ADMIN_USERNAME, ADMIN_PASSWORD
                except ImportError:
                    ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
                    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
                
                if submit:
                    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                        st.session_state.admin_authenticated = True
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials. Please try again.")
        
        st.stop()
    
    return True

# Check admin access
check_admin_access()

# Main admin dashboard
st.title("🔐 Admin Dashboard")
st.markdown("---")

# Navigation
with st.sidebar:
    st.markdown("### 🔗 Navigation")
    st.markdown("""
    - 🏠 [Main App](/)
    - 📋 [Test History](/Test_History)
    - 🔐 [Admin Dashboard](#)
    """)
    st.markdown("---")

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

st.markdown("---")

# Statistics section
st.markdown("## 📊 Statistics Overview")

try:
    stats = get_statistics()
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Tests",
            value=stats["total_tests"],
            help="Total number of images tested"
        )
    
    with col2:
        st.metric(
            label="Unique Images",
            value=stats["unique_images"],
            help="Number of unique images tested"
        )
    
    with col3:
        st.metric(
            label="Real Detections",
            value=stats["real_count"],
            help="Number of images classified as REAL"
        )
    
    with col4:
        st.metric(
            label="Fake Detections",
            value=stats["fake_count"],
            help="Number of images classified as FAKE"
        )
    
    # Additional metrics
    col5, col6 = st.columns(2)
    
    with col5:
        duplicate_count = stats["duplicate_tests"]
        st.metric(
            label="Duplicate Tests",
            value=duplicate_count,
            help="Number of times duplicate images were tested"
        )
    
    with col6:
        if stats["total_tests"] > 0:
            duplicate_percentage = (duplicate_count / stats["total_tests"]) * 100
            st.metric(
                label="Duplicate Rate",
                value=f"{duplicate_percentage:.1f}%",
                help="Percentage of tests that were duplicates"
            )
        else:
            st.metric(
                label="Duplicate Rate",
                value="0%"
            )
    
    # Charts
    st.markdown("---")
    st.markdown("## 📈 Visualizations")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        if stats["total_tests"] > 0:
            import pandas as pd
            chart_data = pd.DataFrame({
                "Category": ["REAL", "FAKE"],
                "Count": [stats["real_count"], stats["fake_count"]]
            })
            st.bar_chart(chart_data.set_index("Category"))
            st.caption("Real vs Fake Detections")
    
    with col_chart2:
        if stats["total_tests"] > 0:
            chart_data2 = pd.DataFrame({
                "Category": ["Unique", "Duplicates"],
                "Count": [stats["unique_images"], duplicate_count]
            })
            st.bar_chart(chart_data2.set_index("Category"))
            st.caption("Unique vs Duplicate Images")
    
except Exception as e:
    st.error(f"Error loading statistics: {e}")

# Recent tests section
st.markdown("---")
st.markdown("## 📋 Recent Test History")

try:
    recent_tests = get_recent_tests(limit=20)
    
    if recent_tests:
        # Create DataFrame for display
        import pandas as pd
        df_data = []
        for test in recent_tests:
            df_data.append({
                "Timestamp": test.get("timestamp", "N/A"),
                "Filename": test.get("image_filename", "N/A"),
                "Hash": test.get("image_hash", "N/A")[:16] + "...",  # Show first 16 chars
                "Prediction": test.get("prediction", "N/A"),
                "Confidence": f"{test.get('confidence_percentage', 0):.2f}%",
                "Duplicate": "Yes" if test.get("is_duplicate", False) else "No"
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No test results found.")
        
except Exception as e:
    st.error(f"Error loading recent tests: {e}")

# Duplicate images section
st.markdown("---")
st.markdown("## 🔄 Duplicate Image Tests")

try:
    duplicates = get_duplicate_images()
    
    if duplicates:
        st.info(f"Found {len(duplicates)} duplicate test entries.")
        
        # Display duplicates in expandable sections
        for dup in duplicates[:10]:  # Show first 10
            with st.expander(f"Hash: {dup.get('image_hash', 'N/A')[:32]}... | Tested: {dup.get('timestamp', 'N/A')}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Filename**: {dup.get('image_filename', 'N/A')}")
                    st.write(f"**Prediction**: {dup.get('prediction', 'N/A')}")
                    st.write(f"**Confidence**: {dup.get('confidence_percentage', 0):.2f}%")
                with col2:
                    st.write(f"**Test ID**: {dup.get('_id', 'N/A')}")
                    st.write(f"**Original Test ID**: {dup.get('original_test_id', 'N/A')}")
                    st.write(f"**Timestamp**: {dup.get('timestamp', 'N/A')}")
    else:
        st.info("No duplicate images found.")
        
except Exception as e:
    st.error(f"Error loading duplicate images: {e}")

# Logout button
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.admin_authenticated = False
        st.rerun()

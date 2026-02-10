"""
Deepfake Image Detection System - Condensed Single Page
A Streamlit web application for detecting deepfake images using deep learning.
"""

import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Page configuration
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Title
st.title("🔍 Deepfake Image Detection System")

# Import functions
try:
    from utils.preprocessing import preprocess_image
    from model.deepfake_model import predict
    from utils.image_hash import calculate_image_hash, get_image_size
    from database.db_operations import save_test_result, check_duplicate_image
    imports_successful = True
except Exception as e:
    st.error(f"❌ Error importing modules: {e}")
    imports_successful = False

# Main layout: upload + instructions side by side
col_upload, col_info = st.columns([2, 1])

with col_upload:
    st.subheader("📁 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose image (JPG, PNG, JPEG, WEBP)",
        type=["jpg", "png", "jpeg", "webp"],
        label_visibility="collapsed"
    )

with col_info:
    st.subheader("ℹ️ Quick Start")
    st.markdown("""
    1. Upload image
    2. Click Analyze
    3. View results
    
    **Model**: EfficientNet-B0
    **Input**: 224×224 px
    **Output**: Real/Fake
    """)

st.markdown("---")

# If file uploaded, show preview and analyze button
if uploaded_file is not None and imports_successful:
    col_img, col_result = st.columns([1, 2])
    
    with col_img:
        st.image(uploaded_file, caption="Preview", width=280)
        if st.button("🔍 Analyze Image", use_container_width=True, type="primary"):
            try:
                # Calculate hash & check duplicate
                image_hash = calculate_image_hash(uploaded_file)
                image_size = get_image_size(uploaded_file)
                duplicate_info = check_duplicate_image(image_hash)
                is_duplicate = duplicate_info is not None
                
                with st.spinner("Analyzing image..."):
                    # Preprocess and predict
                    processed_img = preprocess_image(uploaded_file)
                    confidence_score = predict(processed_img)
                    
                    # Determine result
                    is_real = confidence_score > 0.50
                    label = "✅ REAL" if is_real else "🚨 FAKE"
                    confidence = confidence_score * 100 if is_real else (1 - confidence_score) * 100
                    
                    # Save to database
                    test_id = save_test_result(
                        image_hash=image_hash,
                        prediction="REAL" if is_real else "FAKE",
                        confidence_score=confidence_score,
                        confidence_percentage=confidence,
                        image_filename=uploaded_file.name,
                        image_size=image_size,
                        image_bytes=uploaded_file.getvalue() if hasattr(uploaded_file, 'getvalue') else None,
                        is_duplicate=is_duplicate,
                        original_test_id=str(duplicate_info.get('_id', '')) if duplicate_info else None
                    )
                    
                    # Store result in session state for display
                    st.session_state['analysis_result'] = {
                        'label': label,
                        'confidence': confidence,
                        'test_id': test_id,
                        'is_duplicate': is_duplicate,
                        'duplicate_info': duplicate_info
                    }
                    
                    if is_real:
                        st.balloons()
                    
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
    
    with col_result:
        if 'analysis_result' in st.session_state:
            result = st.session_state['analysis_result']
            
            st.subheader("📊 Analysis Results")
            
            # Show prediction
            if "REAL" in result['label']:
                st.success(f"### {result['label']}\n\n**Confidence**: {result['confidence']:.1f}%\n\nImage appears **authentic**.")
            else:
                st.error(f"### {result['label']}\n\n**Confidence**: {result['confidence']:.1f}%\n\nImage may be **AI-generated**.")
            
            # Show duplicate info
            if result['is_duplicate'] and result['duplicate_info']:
                st.info(f"📌 **Duplicate Detected** - Previously tested on {result['duplicate_info'].get('timestamp', 'N/A')}")
            
            # Show saved confirmation
            if result['test_id']:
                st.success(f"✅ Result saved to database (ID: {result['test_id'][:8]}...)")
            
            # Additional details
            with st.expander("📋 Raw Details"):
                st.write(f"**Confidence Score**: {result['confidence']:.4f}")
                st.write(f"**Duplicate Test**: {result['is_duplicate']}")
        else:
            st.info("👈 Click **Analyze Image** to process the uploaded image.")

elif uploaded_file is not None and not imports_successful:
    st.error("⚠️ Please fix import errors before proceeding.")

else:
    st.info("👈 Upload an image to get started.")

# Sidebar with nav
with st.sidebar:
    st.markdown("### 🔗 Navigation")
    st.markdown("[📊 Main](#) | [📋 History](/Test_History) | [🔐 Admin](/Admin)")
    st.markdown("---")
    st.markdown("""
    **Deepfake Detection v1.0**
    
    Secure image analysis with encrypted storage in MongoDB Atlas.
    """)


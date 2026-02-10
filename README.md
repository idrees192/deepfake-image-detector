# 🔍 Deepfake Image Detection System

A web-based deepfake detection application built with Streamlit and TensorFlow. This project uses a deep learning model to classify images as either real or AI-generated (deepfake).

## 📚 Project Information

- **Course**: Information Systems and Artificial Intelligence
- **Institution**: National University of Sciences & Technology
- **Technology Stack**: Python, TensorFlow, Streamlit, EfficientNet-B0
- **Project Type**: Academic/Student Project

## ✨ Features

- 🖼️ **Image Upload**: Support for JPG, PNG, JPEG, and WEBP formats
- 🤖 **AI-Powered Detection**: Uses EfficientNet-B0 deep learning model
- 📊 **Confidence Scores**: Provides percentage confidence for predictions
- 🎨 **User-Friendly Interface**: Clean and intuitive web interface
- ⚡ **Real-Time Analysis**: Fast image processing and analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- 4GB+ RAM recommended

### Installation Steps

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd "project 2 IS and AI"
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   
   **Windows:**
   ```bash
   venv\Scripts\activate
   ```
   
   **macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

4. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in the terminal

## 📁 Project Structure

```
project 2 IS and AI/
│
├── app.py                    # Main Streamlit application
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── LICENSE                   # MIT License
│
├── model/                    # Model files
│   ├── deepfake_model.py     # Model loading and prediction
│   ├── deepfake_universal_patched.keras  # Trained model
│   └── model.weights.h5      # Model weights
│
├── utils/                    # Utility functions
│   └── preprocessing.py      # Image preprocessing
│
└── tests/                    # Test scripts
    ├── test_import.py        # Import tests
    └── test_model_load.py    # Model loading tests
```

## 🧪 Usage

1. **Upload an Image**
   - Click on the file uploader or drag and drop an image
   - Supported formats: JPG, PNG, JPEG, WEBP

2. **Analyze**
   - Click the "Analyze Image" button
   - Wait for the model to process (usually 2-5 seconds)

3. **View Results**
   - The app displays:
     - **Prediction**: REAL or FAKE
     - **Confidence Score**: Percentage indicating prediction certainty

## 🔧 Configuration

### Application Configuration
Edit `config.py` to customize:
- Model file path
- Allowed image file extensions
- Image input size

### MongoDB Atlas Setup
The application uses MongoDB Atlas for storing test results and statistics.

**Setup Steps:**
1. Create a free MongoDB Atlas account at [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)
2. Create a cluster and database user
3. Get your connection string
4. Set the `MONGODB_URI` environment variable or edit `config_mongodb.py`

**For detailed setup instructions, see:** [MONGODB_SETUP.md](MONGODB_SETUP.md)

**Default Admin Credentials:**
- Username: `admin`
- Password: `admin123`
- **⚠️ Change these in production!**

## 📦 Dependencies

- **streamlit** >= 1.52.0 - Web application framework
- **tensorflow** >= 2.20.0 - Deep learning framework
- **numpy** >= 1.24.0 - Numerical computing
- **Pillow** >= 10.0.0 - Image processing library
- **h5py** >= 3.15.0 - HDF5 file format support
- **pymongo** >= 4.6.0 - MongoDB driver for database operations
- **pandas** >= 2.0.0 - Data analysis and statistics
 - **cryptography** >= 41.0.0 - Encryption utilities for secure image storage

## 🔐 Image Encryption

All uploaded images are encrypted before being stored in MongoDB. The application uses Fernet (symmetric authenticated encryption).

Setup:
- Generate a key in Python:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

- Set the key in your environment before running the app:

**Windows (PowerShell):**
```powershell
$env:ENCRYPTION_KEY = 'your-generated-key-here'
```

**macOS / Linux:**
```bash
export ENCRYPTION_KEY='your-generated-key-here'
```

The app will raise an error if `ENCRYPTION_KEY` is not set.

## 🧰 Repository Cleanup & Publishing

- Sensitive credentials should not be committed. Set the following environment variables instead of storing secrets in files:
   - `MONGODB_URI` — MongoDB Atlas connection string
   - `ENCRYPTION_KEY` — Fernet key used to encrypt images before storage

- Large model files have been removed from the repository. To obtain model weights, add them to your local `model/` directory or use Git LFS for distribution.

- To remove local caches and optional virtualenv folders before committing, run:

```bash
python scripts/cleanup_repo.py --remove-venv
```

- Recommended git workflow before publishing:
   1. Run the cleanup script.
   2. Inspect `git status` and `git diff`.
   3. Commit changes and push to a remote repository.


## 🧠 Model Architecture

The detection model uses:

- **Base Model**: EfficientNet-B0 (pre-trained on ImageNet)
  - Transfer learning approach
  - Base model is frozen during training
  
- **Custom Layers**:
  - Global Average Pooling 2D
  - Batch Normalization
  - Dense layer (256 units, ReLU activation)
  - Dropout (0.5 rate)
  - Output layer (1 unit, Sigmoid activation)

- **Input**: 224x224 RGB images
- **Output**: Binary classification (0 = Fake, 1 = Real)

## 📊 How It Works

1. **Image Preprocessing**:
   - Convert to RGB format
   - Resize to 224x224 pixels
   - Normalize pixel values

2. **Model Prediction**:
   - Feed preprocessed image to EfficientNet-B0
   - Extract features through custom layers
   - Generate prediction score (0.0 to 1.0)

3. **Result Interpretation**:
   - Score > 0.5: Classified as REAL
   - Score ≤ 0.5: Classified as FAKE
   - Confidence = |score - 0.5| × 200%

## ⚠️ Important Notes

- This is an **academic project** for educational purposes
- Model accuracy may vary depending on image quality and type
- Results should be interpreted with caution
- The model is trained on specific datasets and may not generalize to all image types
- For production use, additional validation and testing would be required

## 🔐 Admin Dashboard

The application includes an admin-only dashboard for viewing statistics and test history.

**Access the Admin Dashboard:**
1. Navigate to `http://localhost:8501/Admin` (or click "Admin Dashboard" in the sidebar)
2. Login with admin credentials (default: admin/admin123)
3. View:
   - Total tests performed
   - Real vs Fake detection statistics
   - Unique images tested
   - Duplicate image detection
   - Recent test history
   - Detailed duplicate image reports

**Features:**
- 📊 Real-time statistics and metrics
- 📈 Visual charts and graphs
- 📋 Complete test history
- 🔄 Duplicate image tracking
- 🔍 Detailed test information

## 🧪 Testing

Run test scripts to verify installation:

```bash
# Test imports
python tests/test_import.py

# Test model loading
python tests/test_model_load.py
```

## 💾 Database Features

- **Image Hashing**: All images are hashed (SHA256) for duplicate detection
- **Result Storage**: All test results are stored in MongoDB Atlas
- **Duplicate Detection**: Automatically detects if an image has been tested before
- **Statistics Tracking**: Comprehensive statistics on all tests performed
- **Admin Access**: Secure admin dashboard for viewing all data

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Student Name**  
National University of Sciences & Technology  
Course: Information Systems and Artificial Intelligence

## 🙏 Acknowledgments

- **EfficientNet** architecture by Google Research
- **TensorFlow** team for the deep learning framework
- **Streamlit** for the web application framework
- Course instructors and teaching assistants

## 📧 Contact

For questions or issues, please open an issue on the repository or contact the project author.

---

**Note**: This project is part of an academic course and is intended for educational purposes.

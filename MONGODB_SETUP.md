# MongoDB Atlas Setup Guide

This guide will help you set up MongoDB Atlas for the Deepfake Detection System.

## Step 1: Create MongoDB Atlas Account

1. Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Sign up for a free account (or sign in if you already have one)
3. The free tier (M0) is sufficient for this project

## Step 2: Create a Cluster

1. Click "Create" or "Build a Database"
2. Choose the **FREE** (M0) tier
3. Select a cloud provider and region (choose closest to you)
4. Give your cluster a name (e.g., "deepfake-cluster")
5. Click "Create Cluster"
6. Wait 3-5 minutes for the cluster to be created

## Step 3: Create Database User

1. Go to "Database Access" in the left sidebar
2. Click "Add New Database User"
3. Choose "Password" authentication
4. Enter a username (e.g., "deepfake_user")
5. Enter a strong password (save this!)
6. Set user privileges to "Read and write to any database"
7. Click "Add User"

## Step 4: Configure Network Access

1. Go to "Network Access" in the left sidebar
2. Click "Add IP Address"
3. For development, click "Allow Access from Anywhere" (0.0.0.0/0)
   - **Note**: For production, add only your server's IP address
4. Click "Confirm"

## Step 5: Get Connection String

1. Go to "Database" in the left sidebar
2. Click "Connect" on your cluster
3. Choose "Connect your application"
4. Select "Python" and version "3.6 or later"
5. Copy the connection string
   - It will look like: `mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority`

## Step 6: Configure the Application

### Option 1: Environment Variable (Recommended)

Set the `MONGODB_URI` environment variable:

**Windows (PowerShell):**
```powershell
$env:MONGODB_URI="mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/deepfake_detection?retryWrites=true&w=majority"
```

**Windows (Command Prompt):**
```cmd
set MONGODB_URI=mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/deepfake_detection?retryWrites=true&w=majority
```

**macOS/Linux:**
```bash
export MONGODB_URI="mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/deepfake_detection?retryWrites=true&w=majority"
```

### Option 2: Edit config_mongodb.py

1. Open `config_mongodb.py`
2. Replace the connection string with your actual connection string
3. Replace `<username>` and `<password>` with your database user credentials
4. Make sure to include `/deepfake_detection` before the `?` in the connection string

**Example:**
```python
MONGODB_URI = "mongodb+srv://deepfake_user:YourPassword123@cluster0.xxxxx.mongodb.net/deepfake_detection?retryWrites=true&w=majority"
```

## Step 7: Test the Connection

1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Try uploading an image and analyzing it
3. If you see "✅ Test result saved to database!", the connection is working!

4. Access the Admin Dashboard:
   - Go to `http://localhost:8501/Admin`
   - Login with default credentials (admin/admin123)
   - Check if statistics are displayed

## Troubleshooting

### Connection Timeout
- Check your network access settings in MongoDB Atlas
- Ensure your IP address is whitelisted
- Try using "Allow Access from Anywhere" (0.0.0.0/0) for testing

### Authentication Failed
- Verify your username and password are correct
- Make sure you URL-encoded special characters in the password
- Check that the user has read/write permissions

### Database Not Found
- The database will be created automatically on first use
- Make sure the connection string includes the database name: `/deepfake_detection`

## Security Notes

⚠️ **Important Security Considerations:**

1. **Never commit your connection string to Git**
   - Use environment variables
   - Add `config_mongodb.py` to `.gitignore` if you modify it

2. **Change default admin credentials**
   - Edit `config_mongodb.py` or set environment variables:
     - `ADMIN_USERNAME`
     - `ADMIN_PASSWORD`

3. **Use strong passwords**
   - For both MongoDB user and admin account

4. **Restrict network access in production**
   - Only allow your server's IP address
   - Don't use 0.0.0.0/0 in production

## Next Steps

Once MongoDB is configured:
- Test the application with image uploads
- Check the Admin Dashboard for statistics
- Verify duplicate detection is working
- Monitor your MongoDB Atlas dashboard for usage

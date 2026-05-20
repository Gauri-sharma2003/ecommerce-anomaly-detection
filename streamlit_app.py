"""
Streamlit Cloud Entry Point
============================
This file ensures the app works on Streamlit Cloud deployment.
Deploy at: https://share.streamlit.io

Steps:
1. Push to GitHub
2. Go to share.streamlit.io
3. Connect your repo
4. Set main file path: src/streamlit_app.py
5. Deploy!
"""

import subprocess
import os

# Generate database if it doesn't exist (needed for Streamlit Cloud)
if not os.path.exists("ecom.db"):
    print("🔨 Loading real Kaggle dataset for first-time deployment...")
    subprocess.run(["python", "src/load_kaggle_data.py"], check=True)

# Run the actual app
exec(open("src/streamlit_app.py").read())

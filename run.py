"""Run the Streamlit app with settings from .env"""
import os
import subprocess
from dotenv import load_dotenv

load_dotenv()

port = os.getenv("STREAMLIT_SERVER_PORT", "8503")
subprocess.run([
    "streamlit", "run", "src/streamlit_app.py",
    "--server.port", port,
    "--server.headless", "true"
])

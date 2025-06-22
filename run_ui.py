#!/usr/bin/env python3
"""
NoCodeML UI Launcher

This script provides an easy way to start the NoCodeML user interface.
It handles environment setup and launches the Streamlit app.
"""

import subprocess
import sys
import os
from pathlib import Path
import time

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit',
        'pandas',
        'scikit-learn',
        'plotly'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies"""
    print("📦 Installing missing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def start_backend():
    """Start the backend server"""
    print("🚀 Starting NoCodeML backend server...")
    try:
        # Check if minimal_server.py exists
        if Path("minimal_server.py").exists():
            process = subprocess.Popen([
                sys.executable, "minimal_server.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Give the server a moment to start
            time.sleep(3)
            
            if process.poll() is None:  # Process is still running
                print("✅ Backend server started successfully!")
                return process
            else:
                print("⚠️ Backend server failed to start, but UI will work in offline mode")
                return None
        else:
            print("⚠️ Backend server not found, running in offline mode")
            return None
    except Exception as e:
        print(f"⚠️ Could not start backend server: {e}")
        print("Running in offline mode...")
        return None

def start_streamlit():
    """Start the Streamlit UI"""
    print("🌐 Starting NoCodeML Web Interface...")
    try:
        # Set Streamlit configuration
        env = os.environ.copy()
        env['STREAMLIT_SERVER_PORT'] = '8501'
        env['STREAMLIT_SERVER_ADDRESS'] = 'localhost'
        env['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ], env=env)
        
    except KeyboardInterrupt:
        print("\n👋 NoCodeML UI stopped by user")
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")

def print_banner():
    """Print welcome banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                        🤖 NoCodeML                           ║
║                No-Code Machine Learning Platform             ║
║                                                              ║
║              Making AI accessible to everyone!               ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_instructions():
    """Print usage instructions"""
    instructions = """
🚀 Starting NoCodeML Platform...

📋 What you can do:
  • Upload CSV/Excel datasets
  • Analyze data quality and get insights
  • Build ML models without coding
  • Make predictions with your models
  • Export results and models

💡 Tips:
  • Prepare your data with clear column headers
  • Ensure you have at least 50+ rows for good results
  • Check the Help section for tutorials

🌐 The web interface will open in your browser automatically.
   If it doesn't, go to: http://localhost:8501

📖 Need help? Check the 'Help & Tutorials' section in the app.
    """
    print(instructions)

def main():
    """Main launcher function"""
    print_banner()
    print_instructions()
    
    # Check if we're in the right directory
    if not Path("streamlit_app.py").exists():
        print("❌ Error: streamlit_app.py not found!")
        print("Please run this script from the NoCodeML project directory.")
        sys.exit(1)
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"⚠️ Missing dependencies: {', '.join(missing)}")
        response = input("Would you like to install them now? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            if not install_dependencies():
                print("❌ Failed to install dependencies. Please install manually:")
                print("pip install -r requirements.txt")
                sys.exit(1)
        else:
            print("❌ Dependencies required. Please install them and try again.")
            sys.exit(1)
    
    # Start backend (optional)
    backend_process = start_backend()
    
    try:
        # Start Streamlit UI
        start_streamlit()
    finally:
        # Clean up backend process if it was started
        if backend_process and backend_process.poll() is None:
            print("🔄 Stopping backend server...")
            backend_process.terminate()
            backend_process.wait()

if __name__ == "__main__":
    main()

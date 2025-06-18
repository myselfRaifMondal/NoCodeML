#!/usr/bin/env python3
"""
Development Setup Script for NoCodeML

This script sets up the development environment and performs initial checks.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("\n🐍 Checking Python version...")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    
    print("✅ Python version is compatible")
    return True

def setup_virtual_environment():
    """Set up virtual environment"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("\n📦 Virtual environment already exists")
        return True
    
    return run_command("python3 -m venv venv", "Creating virtual environment")

def install_dependencies():
    """Install required dependencies"""
    # Determine the correct pip command based on OS
    if os.name == 'nt':  # Windows
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Unix/MacOS
        pip_cmd = "venv/bin/pip"
    
    commands = [
        (f"{pip_cmd} install --upgrade pip", "Upgrading pip"),
        (f"{pip_cmd} install -r requirements.txt", "Installing dependencies")
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            return False
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "data/uploads",
        "data/samples", 
        "models/trained",
        "models/configs",
        "logs"
    ]
    
    print("\n📁 Creating directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {directory}")
    
    return True

def test_imports():
    """Test if key imports work"""
    print("\n🧪 Testing key imports...")
    
    test_imports = [
        "import pandas as pd",
        "import numpy as np", 
        "import sklearn",
        "import fastapi",
        "import uvicorn"
    ]
    
    for import_stmt in test_imports:
        try:
            exec(import_stmt)
            print(f"  ✅ {import_stmt}")
        except ImportError as e:
            print(f"  ❌ {import_stmt} - {e}")
            return False
    
    return True

def main():
    """Main setup function"""
    print("🚀 NoCodeML Development Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Setup virtual environment
    if not setup_virtual_environment():
        print("❌ Failed to create virtual environment")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print("❌ Failed to create directories")
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        print("❌ Some imports failed - please check your installation")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Activate the virtual environment:")
    if os.name == 'nt':
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("2. Start the development server:")
    print("   python -m uvicorn backend.main:app --reload")
    print("3. Open your browser to: http://localhost:8000")
    print("4. API documentation: http://localhost:8000/api/docs")

if __name__ == "__main__":
    main()


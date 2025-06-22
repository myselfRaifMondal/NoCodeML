#!/bin/bash
# NoCodeML UI Setup Script
# This script sets up the environment and installs dependencies for the NoCodeML UI

set -e  # Exit on any error

echo "🤖 NoCodeML UI Setup Script"
echo "============================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3.8 or higher and try again."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "🐍 Python version: $python_version"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "📦 Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "🚀 To start the NoCodeML UI:"
echo "   1. Run: source venv/bin/activate"
echo "   2. Run: python run_ui.py"
echo ""
echo "Or simply run: ./start_ui.sh"
echo ""
echo "📁 Sample datasets are available in the sample_data/ folder"
echo "📖 Check UI_GUIDE.md for detailed usage instructions"
echo ""
echo "🌐 The web interface will be available at: http://localhost:8501"

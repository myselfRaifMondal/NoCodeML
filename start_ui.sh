#!/bin/bash
# NoCodeML UI Start Script
# This script starts the NoCodeML user interface

echo "🤖 Starting NoCodeML User Interface..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found."
    echo "Please run setup_ui.sh first to set up the environment."
    exit 1
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Check if streamlit is installed
if ! python -c "import streamlit" &> /dev/null; then
    echo "❌ Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Start the UI
echo "🌐 Starting web interface..."
echo "📖 Check UI_GUIDE.md for usage instructions"
echo ""
python run_ui.py

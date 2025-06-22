# NoCodeML

A no-code machine learning platform that democratizes AI/ML model building for everyone.

## Vision

NoCodeML aims to eliminate the coding barrier in machine learning, allowing anyone to build robust models through an intuitive interface. The platform automatically analyzes datasets, recommends optimal ML approaches, handles hyperparameter tuning, and provides comprehensive model evaluation.

## Features

- 🔍 **Intelligent Dataset Analysis**: Automatic data profiling and quality assessment
- 🤖 **AutoML Pipeline**: Automated model selection and hyperparameter optimization
- 📊 **Multi-Algorithm Support**: Classical ML, Deep Learning, and Unsupervised Learning
- 🎯 **Smart Recommendations**: AI-powered suggestions for model types and preprocessing
- 📈 **Comprehensive Evaluation**: Detailed model performance metrics and explanations
- 🌐 **User-Friendly Interface**: Intuitive web interface requiring no coding skills

## Architecture

```
NoCodeML/
├── backend/          # FastAPI backend server
├── frontend/         # React web interface
├── core/            # Core ML pipeline and algorithms
├── data/            # Sample datasets and user uploads
├── models/          # Trained model storage
├── notebooks/       # Jupyter notebooks for experimentation
└── tests/           # Unit and integration tests
```

## Getting Started

### 🚀 Quick Start (User-Friendly Web Interface)

**For Non-Technical Users:**
```bash
# 1. Set up the environment (one-time setup)
./setup_ui.sh

# 2. Start the web interface
./start_ui.sh
```

The web interface will open automatically at `http://localhost:8501`

**Alternative Method:**
```bash
# Install dependencies
pip install -r requirements.txt

# Start the UI directly
python run_ui.py
```

### 🔧 Developer Setup (API Backend)

```bash
# Clone the repository
git clone https://github.com/yourusername/NoCodeML.git
cd NoCodeML

# Set up the environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the backend API
python minimal_server.py
```

## Development Status

🚧 **In Development** - This project is actively being built.

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## License

MIT License - see LICENSE file for details.

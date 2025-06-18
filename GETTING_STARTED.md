# Getting Started with NoCodeML

Welcome to NoCodeML! This guide will help you set up and start using the platform.

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for version control)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/NoCodeML.git
   cd NoCodeML
   ```

2. **Run the setup script:**
   ```bash
   python setup_dev.py
   ```

3. **Activate the virtual environment:**
   
   **On macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```
   
   **On Windows:**
   ```bash
   venv\Scripts\activate
   ```

4. **Start the server:**
   ```bash
   python -m uvicorn backend.main:app --reload
   ```

5. **Open your browser:**
   - Main interface: http://localhost:8000
   - API documentation: http://localhost:8000/api/docs

## 🔍 How It Works

### 1. Dataset Upload & Analysis
- Upload your dataset (CSV, Excel, JSON, or Parquet)
- Get instant analysis including:
  - Data quality assessment
  - Column type detection
  - Missing value analysis
  - ML problem type recommendations

### 2. Smart Recommendations
- Based on your data characteristics, NoCodeML suggests:
  - Best ML algorithms for your problem
  - Optimal hyperparameters
  - Required preprocessing steps
  - Expected performance metrics

### 3. Automated Model Training
- One-click model training
- Real-time progress tracking
- Automatic hyperparameter optimization
- Cross-validation and evaluation

### 4. Model Evaluation & Deployment
- Comprehensive performance metrics
- Feature importance analysis
- Model explainability
- Easy prediction API

## 🛠 API Usage Examples

### Upload a Dataset

```bash
curl -X POST "http://localhost:8000/api/dataset/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_dataset.csv"
```

### Get Dataset Analysis

```bash
curl -X GET "http://localhost:8000/api/dataset/{dataset_id}/info"
```

### Get ML Recommendations

```bash
curl -X POST "http://localhost:8000/api/analysis/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "problem_type": "classification",
    "target_column": "target",
    "feature_columns": ["feature1", "feature2"]
  }'
```

### Train a Model

```bash
curl -X POST "http://localhost:8000/api/model/train" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "your-dataset-id",
    "model_name": "My First Model",
    "algorithm": "Random Forest",
    "problem_type": "classification",
    "target_column": "target",
    "feature_columns": ["feature1", "feature2"],
    "hyperparameters": {},
    "preprocessing_config": {},
    "validation_config": {}
  }'
```

## 📊 Supported Problem Types

- **Classification**: Binary and multi-class classification
- **Regression**: Continuous value prediction
- **Clustering**: Unsupervised grouping
- **Anomaly Detection**: Outlier identification
- **Time Series**: Temporal data analysis
- **NLP**: Natural language processing
- **Computer Vision**: Image analysis (coming soon)

## 🔧 Supported Algorithms

### Classical ML
- Linear/Logistic Regression
- Random Forest
- Support Vector Machines
- Naive Bayes
- Decision Trees
- K-Nearest Neighbors

### Ensemble Methods
- XGBoost
- LightGBM
- AdaBoost
- Gradient Boosting

### Deep Learning
- Neural Networks
- Convolutional Neural Networks
- Recurrent Neural Networks

### Unsupervised
- K-Means Clustering
- DBSCAN
- Hierarchical Clustering
- Isolation Forest

## 📁 Project Structure

```
NoCodeML/
├── backend/              # FastAPI backend
│   ├── api/             # API routes
│   ├── models/          # Data models
│   └── services/        # Business logic
├── core/                # ML pipeline core
│   ├── analyzers/       # Dataset analysis
│   ├── automl/          # AutoML engine
│   ├── preprocessing/   # Data preprocessing
│   └── evaluation/      # Model evaluation
├── data/                # Dataset storage
│   ├── uploads/         # User uploads
│   └── samples/         # Sample datasets
├── models/              # Trained models
├── frontend/            # React frontend (coming soon)
└── tests/               # Test suite
```

## 🧪 Testing with Sample Data

We've included a sample Titanic dataset to test the platform:

1. **Upload the sample:**
   ```bash
   curl -X POST "http://localhost:8000/api/dataset/upload" \
     -F "file=@data/samples/titanic_sample.csv"
   ```

2. **Analyze the dataset:**
   - Check the response for dataset insights
   - Note the recommended problem types
   - Review data quality scores

3. **Train a model:**
   - Use "Survived" as the target column
   - Select features like "Pclass", "Sex", "Age", "Fare"
   - Choose classification as the problem type

## 🔍 Troubleshooting

### Common Issues

1. **Import errors:**
   - Make sure virtual environment is activated
   - Reinstall dependencies: `pip install -r requirements.txt`

2. **Server won't start:**
   - Check if port 8000 is available
   - Try a different port: `uvicorn backend.main:app --port 8001`

3. **File upload fails:**
   - Check file format (CSV, Excel, JSON, Parquet)
   - Ensure file size is under 100MB
   - Verify file is not corrupted

### Getting Help

- Check the API documentation at `/api/docs`
- Review the logs for detailed error messages
- Create an issue on GitHub for bugs or feature requests

## 🚧 Development Status

NoCodeML is actively under development. Current features:

✅ **Completed:**
- Dataset upload and analysis
- Intelligent data profiling
- ML algorithm recommendations
- Basic model training simulation
- REST API with comprehensive documentation

🚧 **In Progress:**
- Real AutoML pipeline implementation
- Advanced hyperparameter optimization
- Model explainability features
- Web-based frontend interface

📋 **Planned:**
- Computer vision support
- Advanced NLP capabilities
- Model deployment tools
- Collaborative features
- Enterprise integrations

## 🤝 Contributing

We welcome contributions! Please read our contributing guidelines and submit pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.


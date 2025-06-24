# 🤖 Fully Automated Machine Learning Pipeline

A comprehensive, zero-code machine learning solution that handles everything from data loading to model deployment. Just upload your data, specify the target column, and get a trained model ready for production use!

## ✨ Features

### 🚀 Fully Automated Process
- **Zero Configuration Required**: Just upload data and specify target column
- **Intelligent Data Loading**: Supports CSV, Excel, JSON, and URLs
- **Smart Data Cleaning**: Automatic handling of missing values, outliers, and data types
- **Feature Engineering**: Automated feature selection and transformation
- **Model Selection**: Tests multiple algorithms with hyperparameter optimization
- **Ready-to-Use Export**: Download trained model as pickle file

### 📊 Comprehensive Analysis
- **Exploratory Data Analysis (EDA)**: Automatic data profiling and insights
- **Data Visualization**: Generated charts for distributions, correlations, and patterns
- **Feature Importance**: Multiple methods to identify key predictors
- **Model Comparison**: Performance metrics across different algorithms
- **Progress Tracking**: Real-time updates on pipeline stages

### 🎯 Smart Machine Learning
- **Task Detection**: Automatically identifies classification vs regression
- **Multiple Algorithms**: Random Forest, XGBoost, Logistic Regression, SVM, and more
- **Hyperparameter Tuning**: Optuna-powered optimization for best performance
- **Cross-Validation**: Robust model evaluation with statistical confidence
- **Overfitting Prevention**: Smart feature selection and regularization

### 🖥️ Multiple Interfaces
- **Web Interface**: User-friendly Streamlit dashboard
- **Command Line**: Scriptable for automation and batch processing
- **Jupyter Compatible**: Import as Python module for custom workflows

## 🚀 Quick Start

### Option 1: One-Click Web Interface
```bash
# Install and launch web interface
python launch_automl.py --install-deps
python launch_automl.py --mode web
```
Then open http://localhost:8501 in your browser.

### Option 2: Command Line Interface
```bash
# Basic usage
python automated_ml_pipeline.py --input data.csv --target target_column

# With URL data source
python automated_ml_pipeline.py --input https://example.com/data.csv --target price

# Custom output directory
python automated_ml_pipeline.py --input data.xlsx --target category --output my_results
```

### Option 3: Quick Setup and Sample Data
```bash
# Complete setup with sample data
python launch_automl.py --mode setup
python launch_automl.py --mode sample
```

## 📋 Requirements

### System Requirements
- Python 3.8+ (Python 3.9+ recommended)
- 4GB+ RAM (8GB+ for large datasets)
- 2GB free disk space

### Dependencies
All dependencies are automatically installed with the launcher:
- pandas, numpy, scikit-learn
- xgboost, optuna (for ML)
- matplotlib, seaborn, plotly (for visualization)
- streamlit (for web interface)
- tqdm (for progress tracking)

## 📁 Project Structure

```
NoCodeML/
├── automated_ml_pipeline.py    # Core ML pipeline
├── automated_ml_web_app.py     # Streamlit web interface
├── launch_automl.py            # Easy launcher script
├── requirements.txt            # Dependencies
├── AUTOML_README.md           # This documentation
├── sample_employee_data.csv   # Generated sample data
└── automl_output/             # Default output directory
    ├── automl_model_*.pkl     # Trained model file
    ├── pipeline_summary.txt   # Execution summary
    └── visualizations/        # Generated charts
        ├── data_overview.png
        ├── correlations.png
        ├── distributions.png
        └── ...
```

## 🎯 How It Works

### 1. Data Loading & Exploration 📊
- Automatically detects file format (CSV, Excel, JSON)
- Supports local files and remote URLs
- Performs comprehensive EDA:
  - Data types and memory usage
  - Missing value patterns
  - Statistical summaries
  - Correlation analysis
  - Outlier detection

### 2. Data Cleaning & Preparation 🧹
- **Missing Values**: Smart imputation based on data distribution
  - Numerical: Mean (normal) or median (skewed)
  - Categorical: Mode or "Unknown"
  - Automatic column dropping if >50% missing
- **Outliers**: IQR-based detection and capping
- **Data Types**: Automatic conversion of numeric and datetime columns
- **Text Cleaning**: Standardization of categorical text
- **Duplicates**: Automatic removal

### 3. Feature Selection 🎯
- **Statistical Tests**: F-statistics for feature-target relationships
- **Tree-Based Importance**: Random Forest feature importance
- **Correlation Analysis**: Target correlation filtering
- **Ensemble Selection**: Combines multiple methods for robust feature ranking
- **Overfitting Prevention**: Optimal feature count selection

### 4. Data Transformation 🔧
- **Categorical Encoding**:
  - One-hot encoding for low cardinality (<= 10 categories)
  - Label encoding for high cardinality features
- **Numerical Scaling**:
  - StandardScaler for normal distributions
  - MinMaxScaler for skewed distributions
- **Automated Pipeline**: Consistent train/test transformations

### 5. Model Training & Selection 🤖
- **Multiple Algorithms**:
  - **Classification**: Random Forest, XGBoost, Logistic Regression, SVM, KNN, Naive Bayes
  - **Regression**: Random Forest, XGBoost, Linear Regression, SVR, KNN
- **Hyperparameter Optimization**: Optuna-powered tuning with 20+ trials per model
- **Cross-Validation**: 5-fold CV for robust performance estimation
- **Best Model Selection**: Automated based on CV scores

### 6. Model Export & Usage 📦
- **Pickle Export**: Complete model package with transformers
- **Usage Instructions**: Generated code examples
- **Reproducible**: Includes all preprocessing steps

## 🖼️ Web Interface Features

### Dashboard Overview
- **Drag & Drop Upload**: Easy file upload with format detection
- **URL Support**: Direct data loading from web sources
- **Live Preview**: Data exploration before processing
- **Target Selection**: Interactive column selection with task type detection

### Real-Time Progress
- **Stage Tracking**: Visual progress through 10 pipeline stages
- **Time Estimation**: Elapsed and remaining time estimates
- **Status Updates**: Detailed progress information

### Results Dashboard
- **EDA Results**: Interactive data analysis summary
- **Visualizations**: Generated charts with zoom and download options
- **Model Comparison**: Performance metrics across all tested models
- **Feature Analysis**: Importance scores from multiple methods
- **Model Download**: One-click model and results export

## 📊 Generated Outputs

### Model File (.pkl)
Complete model package containing:
```python
{
    'model': trained_model,                    # Best performing model
    'feature_names': selected_features,        # List of selected features
    'transformer': preprocessing_pipeline,     # Data transformation steps
    'model_name': 'RandomForest',             # Algorithm name
    'best_params': {...},                     # Optimized hyperparameters
    'is_classification': True,                # Task type
    'export_timestamp': '2024-06-23T...'     # Export time
}
```

### Visualizations
- **Data Overview**: Dataset statistics and type distribution
- **Missing Values**: Heatmap of missing data patterns
- **Correlations**: Feature correlation matrix
- **Distributions**: Histograms of numerical features
- **Categorical Analysis**: Value counts for categorical features
- **Outliers**: Box plots for outlier detection

### Summary Report
Text summary including:
- Execution time and performance metrics
- Data cleaning operations performed
- Feature selection results
- Model comparison and best model details

## 🔧 Advanced Usage

### Using the Trained Model
```python
import pickle
import pandas as pd

# Load the trained model
with open('automl_model_randomforest_20240623_143022.pkl', 'rb') as f:
    model_package = pickle.load(f)

# Extract components
model = model_package['model']
feature_names = model_package['feature_names']
transformer = model_package['transformer']

# Prepare new data
new_data = pd.read_csv('new_data.csv')

# Use only selected features
new_data_features = new_data[feature_names]

# Apply transformations
new_data_transformed, _ = transformer.transform_data(new_data_features)

# Make predictions
predictions = model.predict(new_data_transformed)
probabilities = model.predict_proba(new_data_transformed)  # For classification

print(f"Predictions: {predictions}")
```

### Custom Configuration
```python
from automated_ml_pipeline import AutoMLPipeline

# Create custom pipeline
pipeline = AutoMLPipeline(output_dir="custom_output")

# Run with custom settings
results = pipeline.run_pipeline(
    data_source="data.csv",
    target_column="target"
)

# Access detailed results
print(f"Best model: {results['best_model']['name']}")
print(f"Feature scores: {results['feature_scores']}")
print(f"Cleaning operations: {results['cleaning_log']}")
```

### Batch Processing
```bash
# Process multiple datasets
for file in *.csv; do
    python automated_ml_pipeline.py --input "$file" --target target --output "results_$file"
done
```

## 🎯 Supported Data Types

### Input Formats
- **CSV**: Comma-separated values (.csv)
- **Excel**: Excel workbooks (.xlsx, .xls)
- **JSON**: JSON format with tabular data
- **URLs**: Direct links to supported formats

### Data Types
- **Numerical**: Integers, floats (automatic scaling)
- **Categorical**: Text, categories (smart encoding)
- **Boolean**: True/false values
- **Datetime**: Automatic detection and conversion
- **Mixed Types**: Intelligent type inference

### Task Types
- **Binary Classification**: 2-class prediction problems
- **Multi-class Classification**: 3+ class prediction
- **Regression**: Continuous value prediction
- **Automatic Detection**: Based on target variable characteristics

## 🔍 Troubleshooting

### Common Issues

#### Dependencies Not Found
```bash
# Install all dependencies
python launch_automl.py --install-deps

# Or install manually
pip install -r requirements.txt
```

#### Memory Issues with Large Datasets
- Use data sampling for initial exploration
- Increase available RAM or use cloud instances
- Consider feature selection before full processing

#### Model Performance Issues
- Check data quality and feature relevance
- Ensure sufficient data for target classes
- Consider domain-specific feature engineering

#### Web Interface Won't Start
```bash
# Check Streamlit installation
python -m streamlit --version

# Launch with specific port
streamlit run automated_ml_web_app.py --server.port 8502
```

### Getting Help
1. Check the generated `pipeline_summary.txt` for execution details
2. Review error messages in the console output
3. Ensure data format matches expected structure
4. Verify target column exists in the dataset

## 🚀 Performance Tips

### For Better Results
- **Data Quality**: Clean, relevant data produces better models
- **Feature Engineering**: Domain knowledge can guide feature creation
- **Target Balance**: Ensure balanced classes for classification
- **Sample Size**: More data generally improves performance

### For Faster Processing
- **Feature Pre-selection**: Remove obviously irrelevant columns
- **Data Sampling**: Use subset for initial exploration
- **Resource Allocation**: Use more CPU cores and RAM when available

## 🔮 Future Enhancements

### Planned Features
- **Deep Learning Models**: TensorFlow/PyTorch integration
- **Time Series Support**: Forecasting and temporal analysis
- **Advanced Visualizations**: Interactive plots and dashboards
- **Model Interpretability**: SHAP and LIME explanations
- **API Deployment**: Automatic model serving endpoints
- **Distributed Processing**: Spark/Dask for big data

### Contributing
This project is part of the NoCodeML ecosystem. Contributions are welcome!

## 📄 License

This project is licensed under the MIT License - see the existing project license for details.

## 🙏 Acknowledgments

Built using excellent open-source libraries:
- scikit-learn for machine learning algorithms
- XGBoost for gradient boosting
- Optuna for hyperparameter optimization
- Streamlit for web interface
- Plotly for interactive visualizations

---

**Ready to turn your data into insights? Get started now!**

```bash
python launch_automl.py --mode web
```

🎉 **Happy AutoML!** 🤖

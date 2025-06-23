#!/usr/bin/env python3
"""
NoCodeML Streamlit Web Interface

A user-friendly web interface for the NoCodeML platform that allows non-technical users
to upload datasets, analyze data, and build ML models without coding.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import time
from pathlib import Path
import json
from typing import Dict, Any, List, Optional

# Configure Streamlit page
st.set_page_config(
    page_title="NoCodeML Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .error-box {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class NoCodeMLInterface:
    def __init__(self):
        self.api_base_url = "http://localhost:8000"
        self.session_state = st.session_state
        
        # Initialize session state
        if 'datasets' not in self.session_state:
            self.session_state.datasets = {}
        if 'current_dataset' not in self.session_state:
            self.session_state.current_dataset = None
        if 'analysis_results' not in self.session_state:
            self.session_state.analysis_results = None

    def render_header(self):
        """Render the main header"""
        st.markdown("""
        <div class="main-header">
            <h1>🤖 NoCodeML Platform</h1>
            <p>Build Machine Learning Models Without Writing Code</p>
        </div>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Render the sidebar navigation"""
        st.sidebar.title("🚀 Navigation")
        
        pages = {
            "🏠 Home": "home",
            "📂 Data Upload": "upload",
            "📊 Data Analysis": "analysis",
            "🧹 Data Preprocessing": "preprocessing",
            "🤖 Model Building": "modeling",
            "📈 Results & Export": "results",
            "❓ Help & Tutorials": "help"
        }
        
        selected_page = st.sidebar.radio("Go to:", list(pages.keys()))
        return pages[selected_page]

    def render_home_page(self):
        """Render the home page"""
        st.markdown("## Welcome to NoCodeML! 👋")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🔍 **Step 1: Upload Data**
            Upload your CSV or Excel files and get instant insights about your dataset.
            """)
            
        with col2:
            st.markdown("""
            ### 📊 **Step 2: Analyze Data**
            Explore your data with automatic visualizations and quality assessments.
            """)
            
        with col3:
            st.markdown("""
            ### 🤖 **Step 3: Build Models**
            Let AI select and train the best machine learning models for your data.
            """)

        # Show recent datasets if any
        if self.session_state.datasets:
            st.markdown("## 📁 Recent Datasets")
            for dataset_id, info in list(self.session_state.datasets.items())[-3:]:
                with st.expander(f"📊 {info.get('filename', 'Unknown')}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows", info.get('rows', 'N/A'))
                    with col2:
                        st.metric("Columns", info.get('columns', 'N/A'))
                    with col3:
                        st.metric("Quality Score", f"{info.get('data_quality_score', 'N/A')}%")

    def render_upload_page(self):
        """Render the data upload page"""
        st.markdown("## 📂 Upload Your Dataset")
        
        st.markdown("""
        ### Supported Formats
        - **CSV files** (.csv)
        - **Excel files** (.xlsx, .xls)
        
        ### Tips for Best Results
        - Ensure your data has column headers
        - Remove or clean obviously incorrect values
        - Keep file size under 200MB for faster processing
        """)
        
        uploaded_file = st.file_uploader(
            "Choose your dataset file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload a CSV or Excel file containing your dataset"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
            
            # Preview the data
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.markdown("### 👀 Data Preview")
                st.dataframe(df.head(10))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", len(df))
                with col2:
                    st.metric("Total Columns", len(df.columns))
                with col3:
                    st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
                
                if st.button("🚀 Analyze This Dataset", type="primary", key="analyze_dataset"):
                    self.process_uploaded_file(uploaded_file, df)
                    
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")

    def process_uploaded_file(self, uploaded_file, df):
        """Process and analyze the uploaded file"""
        with st.spinner("🔄 Analyzing your dataset..."):
            try:
                # Simulate the analysis (replace with actual API call)
                analysis = self.analyze_dataset_locally(df, uploaded_file.name)
                
                # Store in session state
                dataset_id = f"dataset_{int(time.time())}"
                self.session_state.datasets[dataset_id] = analysis
                self.session_state.current_dataset = dataset_id
                self.session_state.analysis_results = analysis
                
                st.success("✅ Dataset analyzed successfully!")
                st.balloons()
                
                # Show quick results
                st.markdown("### 📊 Quick Analysis Results")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Data Quality Score", f"{analysis['data_quality_score']}%")
                with col2:
                    st.metric("Missing Values", f"{analysis['missing_percentage']:.1f}%")
                with col3:
                    st.metric("Duplicate Rows", analysis['duplicate_rows'])
                with col4:
                    st.metric("Columns", analysis['columns'])
                
                st.info("👉 Go to 'Data Analysis' to see detailed insights!")
                
            except Exception as e:
                st.error(f"❌ Error analyzing dataset: {str(e)}")

    def analyze_dataset_locally(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Analyze dataset locally (fallback when API is not available)"""
        rows, columns = df.shape
        missing_values = df.isnull().sum().sum()
        missing_percentage = (missing_values / (rows * columns)) * 100
        
        # Column analysis
        column_info = []
        for col in df.columns:
            col_data = df[col]
            missing_count = col_data.isnull().sum()
            unique_count = col_data.nunique()
            
            # Determine data type
            if pd.api.types.is_numeric_dtype(col_data):
                data_type = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                data_type = "datetime"
            else:
                data_type = "categorical"
            
            column_info.append({
                "name": col,
                "data_type": data_type,
                "missing_count": int(missing_count),
                "missing_percentage": round((missing_count / rows) * 100, 2),
                "unique_count": int(unique_count),
                "sample_values": str(col_data.dropna().head(3).tolist())
            })
        
        # Generate suggestions
        suggestions = []
        if rows < 100:
            suggestions.append("Consider collecting more data for better model performance")
        if missing_percentage > 10:
            suggestions.append("High missing values detected - consider data cleaning")
        
        # Calculate quality score
        duplicate_count = df.duplicated().sum()
        quality_score = max(0, 100 - missing_percentage - (duplicate_count / rows * 10))
        
        return {
            "filename": filename,
            "rows": rows,
            "columns": columns,
            "column_info": column_info,
            "data_quality_score": round(quality_score, 1),
            "missing_values_total": int(missing_values),
            "missing_percentage": round(missing_percentage, 2),
            "duplicate_rows": int(duplicate_count),
            "suggested_problem_types": self.suggest_problem_types(df),
            "suggestions": suggestions,
            "warnings": [f"{duplicate_count} duplicate rows detected"] if duplicate_count > 0 else [],
            "upload_timestamp": datetime.now().isoformat(),
            "dataframe": df  # Store for visualization
        }

    def suggest_problem_types(self, df: pd.DataFrame) -> List[str]:
        """Suggest problem types based on data characteristics"""
        suggestions = []
        
        # Check for potential target columns
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            
            if pd.api.types.is_numeric_dtype(df[col]):
                if unique_ratio > 0.8:
                    suggestions.append("regression")
                else:
                    suggestions.append("classification")
            elif df[col].nunique() < 20:  # Categorical with few unique values
                suggestions.append("classification")
        
        return list(set(suggestions)) if suggestions else ["classification", "regression"]

    def render_analysis_page(self):
        """Render the data analysis page"""
        st.markdown("## 📊 Data Analysis & Insights")
        
        if not self.session_state.analysis_results:
            st.warning("⚠️ No dataset analyzed yet. Please go to 'Data Upload' first.")
            return
        
        analysis = self.session_state.analysis_results
        df = analysis.get('dataframe')
        
        if df is None:
            st.error("❌ Dataset not available for visualization.")
            return
        
        # Overview metrics
        st.markdown("### 📈 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Rows", analysis['rows'])
        with col2:
            st.metric("📋 Total Columns", analysis['columns'])
        with col3:
            st.metric("🎯 Quality Score", f"{analysis['data_quality_score']}%")
        with col4:
            st.metric("❓ Missing Data", f"{analysis['missing_percentage']:.1f}%")
        
        # Column details
        st.markdown("### 📋 Column Details")
        col_df = pd.DataFrame(analysis['column_info'])
        st.dataframe(col_df, use_container_width=True)
        
        # Visualizations
        st.markdown("### 📊 Data Visualizations")
        
        # Select columns for visualization
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        categorical_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
        
        tab1, tab2, tab3 = st.tabs(["📊 Distributions", "🔗 Correlations", "❓ Missing Data"])
        
        with tab1:
            if numeric_cols:
                selected_numeric = st.selectbox("Select numeric column:", numeric_cols)
                if selected_numeric:
                    fig = px.histogram(df, x=selected_numeric, title=f"Distribution of {selected_numeric}")
                    st.plotly_chart(fig, use_container_width=True)
            
            if categorical_cols:
                selected_categorical = st.selectbox("Select categorical column:", categorical_cols)
                if selected_categorical:
                    value_counts = df[selected_categorical].value_counts().head(10)
                    fig = px.bar(x=value_counts.index, y=value_counts.values, 
                               title=f"Top Values in {selected_categorical}")
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(corr_matrix, title="Correlation Matrix", color_continuous_scale="RdBu")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least 2 numeric columns for correlation analysis.")
        
        with tab3:
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                fig = px.bar(x=missing_data.index, y=missing_data.values, 
                           title="Missing Values by Column")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("🎉 No missing values detected!")
        
        # Suggestions and warnings
        if analysis.get('suggestions'):
            st.markdown("### 💡 Suggestions")
            for suggestion in analysis['suggestions']:
                st.info(f"💡 {suggestion}")
        
        if analysis.get('warnings'):
            st.markdown("### ⚠️ Warnings")
            for warning in analysis['warnings']:
                st.warning(f"⚠️ {warning}")

    def render_modeling_page(self):
        """Render the model building page"""
        st.markdown("## 🤖 Build Your ML Model")
        
        if not self.session_state.analysis_results:
            st.warning("⚠️ No dataset analyzed yet. Please upload and analyze a dataset first.")
            return
        
        analysis = self.session_state.analysis_results
        df = analysis.get('dataframe')
        
        st.markdown("### 🎯 Define Your Problem")
        
        # Problem type selection
        problem_types = analysis.get('suggested_problem_types', ['classification', 'regression'])
        problem_type = st.selectbox(
            "What type of problem are you solving?",
            problem_types,
            help="Classification: Predict categories (e.g., spam/not spam)\nRegression: Predict numbers (e.g., price, temperature)"
        )
        
        # Target variable selection
        target_column = st.selectbox(
            "Select the column you want to predict (target variable):",
            df.columns.tolist(),
            help="This is what your model will learn to predict"
        )
        
        # Feature selection
        available_features = [col for col in df.columns if col != target_column]
        selected_features = st.multiselect(
            "Select features (input variables) for your model:",
            available_features,
            default=available_features[:5] if len(available_features) > 5 else available_features,
            help="These are the columns your model will use to make predictions"
        )
        
        if not selected_features:
            st.warning("⚠️ Please select at least one feature.")
            return
        
        # Model configuration
        st.markdown("### ⚙️ Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider(
                "Test set size (% of data for testing):",
                min_value=10, max_value=40, value=20,
                help="Percentage of data to use for testing model performance"
            )
        
        with col2:
            auto_tune = st.checkbox(
                "Enable automatic hyperparameter tuning",
                value=True,
                help="Let the system find the best model settings automatically"
            )
        
        # Model building
        if st.button("🚀 Build Model", type="primary", key="build_model"):
            self.build_model(df, target_column, selected_features, problem_type, test_size, auto_tune)

    def build_model(self, df, target_column, selected_features, problem_type, test_size, auto_tune):
        """Build and train the ML model"""
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
        from sklearn.preprocessing import LabelEncoder
        
        with st.spinner("🔄 Building your model... This may take a few minutes."):
            try:
                # Prepare data
                X = df[selected_features].copy()
                y = df[target_column].copy()
                
                # Handle categorical variables
                label_encoders = {}
                for col in X.columns:
                    if X[col].dtype == 'object':
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].astype(str))
                        label_encoders[col] = le
                
                # Handle target variable if categorical
                if problem_type == 'classification' and y.dtype == 'object':
                    target_encoder = LabelEncoder()
                    y = target_encoder.fit_transform(y.astype(str))
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=42
                )
                
                # Train models
                models = {}
                if problem_type == 'classification':
                    models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42)
                    models['Logistic Regression'] = LogisticRegression(random_state=42, max_iter=1000)
                else:
                    models['Random Forest'] = RandomForestRegressor(n_estimators=100, random_state=42)
                    models['Linear Regression'] = LinearRegression()
                
                results = {}
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    
                    if problem_type == 'classification':
                        accuracy = accuracy_score(y_test, predictions)
                        results[name] = {'accuracy': accuracy, 'model': model, 'predictions': predictions}
                    else:
                        mse = mean_squared_error(y_test, predictions)
                        r2 = r2_score(y_test, predictions)
                        results[name] = {'mse': mse, 'r2': r2, 'model': model, 'predictions': predictions}
                
                # Store results in session state
                self.session_state.model_results = {
                    'problem_type': problem_type,
                    'target_column': target_column,
                    'features': selected_features,
                    'results': results,
                    'X_test': X_test,
                    'y_test': y_test,
                    'label_encoders': label_encoders
                }
                
                st.success("✅ Models trained successfully!")
                
                # Display results
                self.display_model_results(results, problem_type)
                
            except Exception as e:
                st.error(f"❌ Error building model: {str(e)}")

    def display_model_results(self, results, problem_type):
        """Display model results"""
        st.markdown("### 🎯 Model Performance")
        
        if problem_type == 'classification':
            performance_data = []
            for name, result in results.items():
                performance_data.append({
                    'Model': name,
                    'Accuracy': f"{result['accuracy']:.3f}",
                    'Accuracy %': f"{result['accuracy']*100:.1f}%"
                })
            
            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df, use_container_width=True)
            
            # Find best model
            best_model = max(results.keys(), key=lambda k: results[k]['accuracy'])
            st.success(f"🏆 Best Model: {best_model} (Accuracy: {results[best_model]['accuracy']*100:.1f}%)")
            
        else:  # regression
            performance_data = []
            for name, result in results.items():
                performance_data.append({
                    'Model': name,
                    'R² Score': f"{result['r2']:.3f}",
                    'Mean Squared Error': f"{result['mse']:.3f}"
                })
            
            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df, use_container_width=True)
            
            # Find best model
            best_model = max(results.keys(), key=lambda k: results[k]['r2'])
            st.success(f"🏆 Best Model: {best_model} (R² Score: {results[best_model]['r2']:.3f})")

    def render_results_page(self):
        """Render the results and export page"""
        st.markdown("## 📈 Results & Model Export")
        
        if not hasattr(self.session_state, 'model_results'):
            st.warning("⚠️ No trained models yet. Please build a model first.")
            return
        
        results = self.session_state.model_results
        
        st.markdown("### 🎯 Model Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Problem Type:** {results['problem_type'].title()}")
            st.info(f"**Target Column:** {results['target_column']}")
        
        with col2:
            st.info(f"**Features Used:** {len(results['features'])}")
            st.info(f"**Models Trained:** {len(results['results'])}")
        
        # Feature importance (for Random Forest)
        if 'Random Forest' in results['results']:
            st.markdown("### 🔍 Feature Importance")
            rf_model = results['results']['Random Forest']['model']
            
            if hasattr(rf_model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': results['features'],
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                           title="Feature Importance (Random Forest)")
                st.plotly_chart(fig, use_container_width=True)
        
        # Prediction interface
        st.markdown("### 🔮 Make Predictions")
        st.markdown("Enter values to get a prediction from your best model:")
        
        # Create input fields for each feature
        prediction_input = {}
        for feature in results['features']:
            prediction_input[feature] = st.number_input(
                f"Enter {feature}:",
                value=0.0,
                key=f"pred_{feature}"
            )
        
        if st.button("🎯 Make Prediction", key="make_prediction"):
            self.make_prediction(prediction_input, results)

    def make_prediction(self, input_data, results):
        """Make a prediction with the trained model"""
        try:
            # Get the best model
            if results['problem_type'] == 'classification':
                best_model_name = max(results['results'].keys(), 
                                    key=lambda k: results['results'][k]['accuracy'])
            else:
                best_model_name = max(results['results'].keys(), 
                                    key=lambda k: results['results'][k]['r2'])
            
            best_model = results['results'][best_model_name]['model']
            
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            
            # Apply label encoders if they exist
            if 'label_encoders' in results:
                for col, encoder in results['label_encoders'].items():
                    if col in input_df.columns:
                        # Handle unseen categories
                        try:
                            input_df[col] = encoder.transform(input_df[col].astype(str))
                        except ValueError:
                            st.warning(f"⚠️ Unknown category in {col}. Using default value.")
                            input_df[col] = 0
            
            # Make prediction
            prediction = best_model.predict(input_df)[0]
            
            st.success(f"🎯 **Prediction:** {prediction:.2f}")
            st.info(f"📊 **Model Used:** {best_model_name}")
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")

    def render_preprocessing_page(self):
        """Render the data preprocessing page with advanced cleaning features"""
        st.markdown("## 🧹 Data Preprocessing & Cleaning")
        
        if not self.session_state.analysis_results:
            st.warning("⚠️ No dataset analyzed yet. Please upload and analyze a dataset first.")
            return
        
        analysis = self.session_state.analysis_results
        df = analysis.get('dataframe')
        
        if df is None:
            st.error("❌ Dataset not available for preprocessing.")
            return
        
        # Import preprocessing module
        try:
            import sys
            from pathlib import Path
            project_root = Path(__file__).parent
            sys.path.append(str(project_root))
            from core.preprocessing.data_preprocessor import DataPreprocessor, PreprocessingResult
            from backend.models.schemas import ProblemType
        except ImportError:
            st.error("❌ Preprocessing module not available. Please ensure all dependencies are installed.")
            return
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        st.markdown("### 🔍 Current Dataset Status")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Rows", df.shape[0])
        with col2:
            st.metric("📋 Columns", df.shape[1])
        with col3:
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            st.metric("❓ Missing Data", f"{missing_pct:.1f}%")
        with col4:
            duplicates = df.duplicated().sum()
            st.metric("🔄 Duplicates", duplicates)
        
        # Data Quality Assessment
        st.markdown("### 🎯 Data Quality Assessment")
        
        with st.expander("🔍 Run Quality Assessment", expanded=True):
            if st.button("🚀 Assess Data Quality", type="primary", key="assess_data_quality"):
                with st.spinner("🔄 Assessing data quality..."):
                    try:
                        issues = preprocessor._assess_data_quality(df, None)
                        self.session_state.data_issues = issues
                        
                        if issues:
                            st.success(f"✅ Assessment complete! Found {len(issues)} issues to address.")
                            
                            # Group issues by severity
                            severity_groups = {'critical': [], 'high': [], 'medium': [], 'low': []}
                            for issue in issues:
                                severity_groups[issue.severity].append(issue)
                            
                            # Display issues by severity
                            for severity, color in [('critical', '🚨'), ('high', '⚠️'), ('medium', '💡'), ('low', 'ℹ️')]:
                                if severity_groups[severity]:
                                    st.markdown(f"#### {color} {severity.title()} Issues")
                                    for issue in severity_groups[severity]:
                                        if severity == 'critical':
                                            st.error(f"**{issue.issue_type.title()}:** {issue.description}")
                                        elif severity == 'high':
                                            st.warning(f"**{issue.issue_type.title()}:** {issue.description}")
                                        else:
                                            st.info(f"**{issue.issue_type.title()}:** {issue.description}")
                                        st.caption(f"💡 Solution: {issue.solution}")
                        else:
                            st.success("🎉 No data quality issues detected! Your dataset looks great.")
                            
                    except Exception as e:
                        st.error(f"❌ Error during assessment: {str(e)}")
        
        # Enhanced Preprocessing Option
        st.markdown("### 🚀 Enhanced Preprocessing")
        
        use_enhanced = st.checkbox(
            "Use Enhanced Preprocessing with Iterative Improvement",
            value=True,
            help="Enable advanced preprocessing with machine learning feedback and iterative improvement"
        )
        
        if use_enhanced:
            try:
                from ui.enhanced_preprocessing_ui import render_enhanced_preprocessing_page
                render_enhanced_preprocessing_page()
                return  # Exit early if enhanced preprocessing is used
            except ImportError as e:
                st.error(f"❌ Enhanced preprocessing import failed: {str(e)}")
                st.info("🔧 Attempting to fix the issue...")
                
                # Try to install missing dependencies
                with st.spinner("Installing missing dependencies..."):
                    try:
                        import subprocess
                        import sys
                        
                        # Install missing sklearn experimental features
                        result = subprocess.run(
                            [sys.executable, "-m", "pip", "install", "--upgrade", "scikit-learn"],
                            capture_output=True, text=True
                        )
                        
                        if result.returncode == 0:
                            st.success("✅ Dependencies updated successfully!")
                            st.info("🔄 Please refresh the page to use enhanced preprocessing.")
                        else:
                            st.warning("⚠️ Could not update dependencies automatically.")
                            
                    except Exception as install_error:
                        st.warning(f"⚠️ Auto-fix failed: {str(install_error)}")
                
                st.warning("⚠️ Enhanced preprocessing not available. Using advanced basic preprocessing with auto-correction.")
        
        # Advanced Basic Preprocessing Configuration
        st.markdown("### ⚙️ Advanced Preprocessing Configuration")
        st.info("💡 **Auto-Correction Enabled**: All issues will be automatically detected and fixed without user intervention.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_column = st.selectbox(
                "🎯 Select Target Column (optional):",
                ["None"] + df.columns.tolist(),
                help="Select the column you want to predict. This helps optimize preprocessing."
            )
            target_column = None if target_column == "None" else target_column
            
            problem_type = st.selectbox(
                "📊 Problem Type:",
                ["Auto-detect", "Classification", "Regression", "Clustering"],
                help="Type of machine learning problem"
            )
        
        with col2:
            auto_fix = st.checkbox(
                "🔧 Auto-fix Critical Issues",
                value=True,
                help="Automatically fix critical issues like removing columns with >70% missing values"
            )
            
            iterative_improvement = st.checkbox(
                "🔄 Enable Iterative Improvement",
                value=True,
                help="Run multiple preprocessing iterations until quality stabilizes"
            )
        
        # Advanced Settings
        with st.expander("🛠️ Advanced Preprocessing Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                missing_threshold = st.slider(
                    "Missing Data Threshold (% to remove column):",
                    min_value=30, max_value=90, value=70,
                    help="Columns with more than this % of missing values will be removed"
                )
                
                correlation_threshold = st.slider(
                    "Correlation Threshold (remove highly correlated features):",
                    min_value=0.8, max_value=0.99, value=0.95, step=0.01,
                    help="Remove features that are highly correlated with others"
                )
            
            with col2:
                outlier_method = st.selectbox(
                    "Outlier Handling Method:",
                    ["IQR Method", "Z-Score", "Remove", "Cap", "Transform"],
                    help="Method to handle outliers in numeric data"
                )
                
                encoding_strategy = st.selectbox(
                    "Categorical Encoding Strategy:",
                    ["Auto (Smart)", "One-Hot", "Label", "Target", "Frequency"],
                    help="How to encode categorical variables"
                )
        
        # Preprocessing Execution
        st.markdown("### 🚀 Execute Preprocessing")
        
        if st.button("🧹 Start Comprehensive Preprocessing", type="primary", key="start_preprocessing"):
            # Update preprocessor settings
            preprocessor.missing_threshold = missing_threshold / 100
            preprocessor.correlation_threshold = correlation_threshold
            
            # Convert problem type
            problem_type_map = {
                "Classification": ProblemType.CLASSIFICATION,
                "Regression": ProblemType.REGRESSION,
                "Clustering": ProblemType.CLUSTERING
            }
            selected_problem_type = problem_type_map.get(problem_type, None)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                if iterative_improvement:
                    status_text.text("🔄 Running iterative preprocessing...")
                    progress_bar.progress(20)
                    
                    processed_df, results_list = preprocessor.iterative_improvement(
                        df, target_column, selected_problem_type, max_iterations=3
                    )
                    progress_bar.progress(80)
                    
                    # Generate comprehensive report
                    report = preprocessor.generate_preprocessing_report(results_list)
                    
                else:
                    status_text.text("🔄 Running single-pass preprocessing...")
                    progress_bar.progress(30)
                    
                    processed_df, result = preprocessor.comprehensive_preprocessing(
                        df, target_column, selected_problem_type, auto_fix
                    )
                    progress_bar.progress(70)
                    
                    report = preprocessor.generate_preprocessing_report([result])
                
                progress_bar.progress(100)
                status_text.text("✅ Preprocessing completed!")
                
                # Store results
                self.session_state.processed_df = processed_df
                self.session_state.preprocessing_report = report
                
                # Display results
                self.display_preprocessing_results(df, processed_df, report, key_suffix="_current")
                
            except Exception as e:
                st.error(f"❌ Error during preprocessing: {str(e)}")
                st.exception(e)
        
        # Display previous results if available
        if hasattr(self.session_state, 'preprocessing_report'):
            st.markdown("### 📊 Previous Preprocessing Results")
            self.display_preprocessing_results(
                df, 
                self.session_state.processed_df, 
                self.session_state.preprocessing_report,
                key_suffix="_previous"
            )
    
    def display_preprocessing_results(self, original_df, processed_df, report, key_suffix=""):
        """Display comprehensive preprocessing results"""
        st.markdown("### 🎉 Preprocessing Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📊 Shape Change", 
                f"{processed_df.shape[0]}×{processed_df.shape[1]}",
                delta=f"{processed_df.shape[0] - original_df.shape[0]}, {processed_df.shape[1] - original_df.shape[1]}"
            )
        
        with col2:
            original_missing = (original_df.isnull().sum().sum() / (original_df.shape[0] * original_df.shape[1])) * 100
            processed_missing = (processed_df.isnull().sum().sum() / (processed_df.shape[0] * processed_df.shape[1])) * 100
            st.metric(
                "❓ Missing Data", 
                f"{processed_missing:.1f}%",
                delta=f"{processed_missing - original_missing:.1f}%"
            )
        
        with col3:
            st.metric(
                "🔄 Iterations", 
                report['summary']['iterations_performed']
            )
        
        with col4:
            st.metric(
                "📈 Quality Improvement", 
                f"{report['summary']['quality_improvement']:.1f}%"
            )
        
        # Detailed results in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Summary", "🔧 Transformations", "📊 Before/After", "💾 Export"])
        
        with tab1:
            st.markdown("#### 📋 Processing Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Dataset:**")
                st.write(f"• Rows: {report['summary']['original_shape'][0]:,}")
                st.write(f"• Columns: {report['summary']['original_shape'][1]}")
                
                st.markdown("**Issues Found:**")
                if report['issues_analysis']['issues_by_severity']:
                    for severity, count in report['issues_analysis']['issues_by_severity'].items():
                        emoji = {'critical': '🚨', 'high': '⚠️', 'medium': '💡', 'low': 'ℹ️'}[severity]
                        st.write(f"• {emoji} {severity.title()}: {count}")
                else:
                    st.write("• No issues detected")
            
            with col2:
                st.markdown("**Processed Dataset:**")
                st.write(f"• Rows: {report['summary']['final_shape'][0]:,}")
                st.write(f"• Columns: {report['summary']['final_shape'][1]}")
                
                st.markdown("**Transformations Applied:**")
                st.write(f"• Total: {report['summary']['total_transformations']}")
                st.write(f"• Auto-fixed issues: {report['issues_analysis']['auto_fixed_issues']}")
            
            if report.get('recommendations'):
                st.markdown("#### 💡 Recommendations")
                for rec in report['recommendations']:
                    st.info(f"💡 {rec}")
        
        with tab2:
            st.markdown("#### 🔧 Applied Transformations")
            
            if report['transformations']:
                for i, transformation in enumerate(report['transformations'], 1):
                    st.write(f"{i}. {transformation}")
            else:
                st.info("No transformations were applied.")
            
            if report.get('encoding_info'):
                st.markdown("#### 🏷️ Encoding Information")
                st.json(report['encoding_info'])
            
            if report.get('scaling_info'):
                st.markdown("#### 📏 Scaling Information")
                st.json(report['scaling_info'])
        
        with tab3:
            st.markdown("#### 📊 Before vs After Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Data (First 5 rows)**")
                st.dataframe(original_df.head(), use_container_width=True)
            
            with col2:
                st.markdown("**Processed Data (First 5 rows)**")
                st.dataframe(processed_df.head(), use_container_width=True)
            
            # Data type comparison
            st.markdown("#### 📋 Data Types Comparison")
            
            original_types = pd.DataFrame({
                'Column': original_df.columns,
                'Original Type': original_df.dtypes.astype(str),
                'Original Nulls': original_df.isnull().sum().values
            })
            
            if not processed_df.empty:
                # Only include columns that exist in both dataframes
                common_cols = list(set(original_df.columns) & set(processed_df.columns))
                
                processed_types = pd.DataFrame({
                    'Column': common_cols,
                    'Processed Type': processed_df[common_cols].dtypes.astype(str),
                    'Processed Nulls': processed_df[common_cols].isnull().sum().values
                })
                
                comparison = original_types.merge(processed_types, on='Column', how='outer')
                st.dataframe(comparison, use_container_width=True)
            
        with tab4:
            st.markdown("#### 💾 Export Processed Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Download Processed CSV", key=f"download_processed_csv{key_suffix}"):
                    csv = processed_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv,
                        file_name=f"processed_{analysis['filename'].replace('.', '_')}.csv",
                        mime="text/csv",
                        key=f"download_csv_button{key_suffix}"
                    )
            
            with col2:
                if st.button("📊 Use for Model Building", key=f"use_for_model_building{key_suffix}"):
                    # Update the analysis results with processed data
                    self.session_state.analysis_results['dataframe'] = processed_df
                    self.session_state.analysis_results['preprocessed'] = True
                    st.success("✅ Processed dataset is now ready for model building!")
                    st.info("👉 Go to 'Model Building' to train your ML models.")
            
            # Export preprocessing report
            st.markdown("#### 📄 Export Preprocessing Report")
            if st.button("📋 Download Preprocessing Report", key=f"download_preprocessing_report{key_suffix}"):
                report_json = json.dumps(report, indent=2, default=str)
                st.download_button(
                    label="💾 Download Report (JSON)",
                    data=report_json,
                    file_name=f"preprocessing_report_{analysis['filename'].replace('.', '_')}.json",
                    mime="application/json"
                )

    def render_help_page(self):
        """Render the help and tutorials page"""
        st.markdown("## ❓ Help & Tutorials")
        
        tab1, tab2, tab3 = st.tabs(["🚀 Getting Started", "📚 Tutorials", "❓ FAQ"])
        
        with tab1:
            st.markdown("""
            ### 🚀 Getting Started with NoCodeML
            
            #### Step 1: Upload Your Data
            - Go to the "Data Upload" page
            - Choose a CSV or Excel file
            - Preview your data and click "Analyze"
            
            #### Step 2: Explore Your Data
            - Visit the "Data Analysis" page
            - Review data quality metrics
            - Explore visualizations and insights
            
            #### Step 3: Build Your Model
            - Go to "Model Building"
            - Select your problem type (classification/regression)
            - Choose your target variable and features
            - Click "Build Model" and wait for training
            
            #### Step 4: Use Your Model
            - Check "Results & Export" for model performance
            - Make predictions with new data
            - Export your model for use elsewhere
            """)
        
        with tab2:
            st.markdown("""
            ### 📚 Tutorials
            
            #### 🎯 Classification Example: Predicting Customer Churn
            1. Upload customer data with features like age, usage, complaints
            2. Select "churn" as target variable (Yes/No)
            3. Choose relevant features like monthly_charges, tenure
            4. Build model and check accuracy
            
            #### 📈 Regression Example: Predicting House Prices
            1. Upload housing data with features like size, location, age
            2. Select "price" as target variable
            3. Choose features like square_feet, bedrooms, bathrooms
            4. Build model and check R² score
            
            #### 🔍 Data Quality Tips
            - Remove obviously wrong values (negative ages, etc.)
            - Handle missing values appropriately
            - Ensure consistent data formats
            - Include relevant features for your prediction goal
            """)
        
        with tab3:
            st.markdown("""
            ### ❓ Frequently Asked Questions
            
            **Q: What file formats are supported?**
            A: CSV (.csv) and Excel (.xlsx, .xls) files are supported.
            
            **Q: How much data do I need?**
            A: At minimum 50-100 rows, but 500+ rows will give better results.
            
            **Q: What's the difference between classification and regression?**
            A: Classification predicts categories (spam/not spam), regression predicts numbers (price, temperature).
            
            **Q: How do I know if my model is good?**
            A: For classification, look for accuracy >70%. For regression, R² score >0.7 is good.
            
            **Q: Can I use my model on new data?**
            A: Yes! Use the prediction interface in the Results page.
            
            **Q: What if my data has missing values?**
            A: The system will detect and warn you. Consider cleaning your data first.
            """)

def main():
    """Main application function"""
    # Initialize the interface
    interface = NoCodeMLInterface()
    
    # Render header
    interface.render_header()
    
    # Render sidebar and get selected page
    selected_page = interface.render_sidebar()
    
    # Render the selected page
    if selected_page == "home":
        interface.render_home_page()
    elif selected_page == "upload":
        interface.render_upload_page()
    elif selected_page == "analysis":
        interface.render_analysis_page()
    elif selected_page == "preprocessing":
        interface.render_preprocessing_page()
    elif selected_page == "modeling":
        interface.render_modeling_page()
    elif selected_page == "results":
        interface.render_results_page()
    elif selected_page == "help":
        interface.render_help_page()
    
    # Footer
    st.markdown("---")
    st.markdown("🤖 **NoCodeML Platform** - Making AI accessible to everyone!")

if __name__ == "__main__":
    main()

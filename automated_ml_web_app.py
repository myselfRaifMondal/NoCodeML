#!/usr/bin/env python3
"""
Automated ML Pipeline Web Interface
===================================

A Streamlit web application that provides a user-friendly interface
for the automated machine learning pipeline.

Features:
- Drag and drop file upload
- Real-time progress tracking
- Interactive visualizations
- Automatic model download
- Results dashboard

Usage:
    streamlit run automated_ml_web_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import io
import os
import sys
from pathlib import Path
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from datetime import datetime
import threading
import queue

# Import the main pipeline
from automated_ml_pipeline import AutoMLPipeline

# Configure Streamlit page
st.set_page_config(
    page_title="AutoML Pipeline",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_custom_css():
    """Inject custom CSS for better styling"""
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    
    .progress-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }
    
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 12px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 12px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .feature-highlight {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

def create_download_link(file_path: str, filename: str = None) -> str:
    """Create a download link for a file"""
    if not filename:
        filename = Path(file_path).name
    
    with open(file_path, "rb") as f:
        data = f.read()
    
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">📥 Download {filename}</a>'
    return href

def display_eda_results(eda_results: dict):
    """Display EDA results in an organized manner"""
    st.markdown("<h3 class='sub-header'>📊 Exploratory Data Analysis Results</h3>", unsafe_allow_html=True)
    
    # Basic Information
    col1, col2, col3, col4 = st.columns(4)
    
    basic_info = eda_results.get('basic_info', {})
    with col1:
        st.metric("Rows", f"{basic_info.get('shape', [0, 0])[0]:,}")
    with col2:
        st.metric("Columns", f"{basic_info.get('shape', [0, 0])[1]:,}")
    with col3:
        st.metric("Memory Usage", f"{basic_info.get('memory_usage', 0) / 1024 / 1024:.1f} MB")
    with col4:
        st.metric("Duplicate Rows", f"{basic_info.get('duplicate_rows', 0):,}")
    
    # Data Types
    st.subheader("Data Types Distribution")
    data_types = eda_results.get('data_types', {})
    
    col1, col2 = st.columns(2)
    with col1:
        if data_types:
            type_counts = {
                'Numeric': len(data_types.get('numeric', [])),
                'Categorical': len(data_types.get('categorical', [])),
                'DateTime': len(data_types.get('datetime', []))
            }
            fig = px.pie(values=list(type_counts.values()), names=list(type_counts.keys()),
                        title="Column Types Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Column Details:**")
        for dtype, cols in data_types.items():
            if cols:
                st.write(f"**{dtype.title()}:** {', '.join(cols[:5])}" + (f" (+{len(cols)-5} more)" if len(cols) > 5 else ""))
    
    # Missing Values
    missing_values = eda_results.get('missing_values', {})
    if missing_values.get('columns_with_missing'):
        st.subheader("Missing Values Analysis")
        missing_df = pd.DataFrame({
            'Column': missing_values['columns_with_missing'],
            'Missing Count': [missing_values['missing_counts'][col] for col in missing_values['columns_with_missing']],
            'Missing Percentage': [missing_values['missing_percentages'][col] for col in missing_values['columns_with_missing']]
        })
        
        fig = px.bar(missing_df, x='Column', y='Missing Percentage',
                    title="Missing Values by Column (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlations
    correlations = eda_results.get('correlations', {})
    if correlations.get('high_correlations'):
        st.subheader("High Correlations (>0.7)")
        high_corr_df = pd.DataFrame(correlations['high_correlations'])
        st.dataframe(high_corr_df, use_container_width=True)

def display_visualizations(visualizations: dict):
    """Display generated visualizations"""
    st.markdown("<h3 class='sub-header'>📈 Generated Visualizations</h3>", unsafe_allow_html=True)
    
    # Create tabs for different visualization types
    viz_tabs = st.tabs(["Data Overview", "Missing Values", "Correlations", "Distributions", "Categorical", "Outliers"])
    
    viz_names = ["data_overview", "missing_values", "correlations", "distributions", "categorical", "outliers"]
    
    for i, (tab, viz_name) in enumerate(zip(viz_tabs, viz_names)):
        with tab:
            viz_path = visualizations.get(viz_name)
            if viz_path and Path(viz_path).exists():
                st.image(viz_path, use_column_width=True)
            else:
                st.info(f"No {viz_name.replace('_', ' ')} visualization available")

def display_model_results(model_results: dict, best_model_info: dict):
    """Display model training results"""
    st.markdown("<h3 class='sub-header'>🎯 Model Training Results</h3>", unsafe_allow_html=True)
    
    # Best Model Summary
    st.success(f"🏆 Best Model: **{best_model_info['name']}**")
    
    # Model comparison
    if model_results:
        comparison_data = []
        for model_name, result in model_results.items():
            scores = result['scores']
            comparison_data.append({
                'Model': model_name,
                'CV Score': scores.get('cv_mean', 0),
                'CV Std': scores.get('cv_std', 0),
                'Train Score': scores.get('train_accuracy', scores.get('train_r2', 0)),
                'Test Score': scores.get('test_accuracy', scores.get('test_r2', 0)) if 'test_accuracy' in scores or 'test_r2' in scores else None
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Model performance chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='CV Score',
            x=comparison_df['Model'],
            y=comparison_df['CV Score'],
            error_y=dict(type='data', array=comparison_df['CV Std'])
        ))
        
        fig.update_layout(
            title="Model Performance Comparison (Cross-Validation)",
            xaxis_title="Models",
            yaxis_title="Score",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed scores table
        st.subheader("Detailed Model Scores")
        st.dataframe(comparison_df, use_container_width=True)

def display_feature_importance(feature_scores: dict, selected_features: list):
    """Display feature selection results"""
    st.markdown("<h3 class='sub-header'>🔍 Feature Selection Results</h3>", unsafe_allow_html=True)
    
    st.info(f"Selected {len(selected_features)} features from the original dataset")
    
    # Feature importance from different methods
    if feature_scores:
        tabs = st.tabs(list(feature_scores.keys()))
        
        for tab, (method, scores) in zip(tabs, feature_scores.items()):
            with tab:
                if scores:
                    # Sort features by score
                    sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:20]  # Top 20
                    
                    features_df = pd.DataFrame(sorted_features, columns=['Feature', 'Score'])
                    
                    fig = px.bar(features_df, x='Score', y='Feature', orientation='h',
                               title=f"Top 20 Features - {method.title()} Method")
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)

class StreamlitProgressTracker:
    """Custom progress tracker for Streamlit"""
    
    def __init__(self):
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.time_text = st.empty()
        self.current_stage = 0
        self.total_stages = 10
        self.start_time = time.time()
        
    def update_stage(self, stage_name: str):
        """Update the progress display"""
        stage_names = [
            "Data Loading", "Data Exploration", "Data Visualization",
            "Data Cleaning", "Feature Selection", "Data Transformation",
            "Model Selection", "Hyperparameter Tuning", "Model Training", "Model Export"
        ]
        
        if stage_name in stage_names:
            self.current_stage = stage_names.index(stage_name) + 1
        
        progress = self.current_stage / self.total_stages
        self.progress_bar.progress(progress)
        
        elapsed_time = time.time() - self.start_time
        estimated_total = elapsed_time / progress if progress > 0 else 0
        remaining_time = estimated_total - elapsed_time
        
        self.status_text.markdown(f"**Stage {self.current_stage}/{self.total_stages}: {stage_name}**")
        self.time_text.markdown(f"⏱️ Elapsed: {elapsed_time/60:.1f}m | Remaining: ~{remaining_time/60:.1f}m")

def run_pipeline_with_progress(data_source: str, target_column: str, output_dir: str, progress_queue: queue.Queue):
    """Run the pipeline in a separate thread with progress updates"""
    try:
        # Custom pipeline class that sends progress updates
        class StreamlitAutoMLPipeline(AutoMLPipeline):
            def __init__(self, output_dir: str, progress_queue: queue.Queue):
                super().__init__(output_dir)
                self.progress_queue = progress_queue
                
            def run_pipeline(self, data_source: str, target_column: str):
                # Override to send progress updates
                self.progress_queue.put(("stage", "Data Loading"))
                results = super().run_pipeline(data_source, target_column)
                self.progress_queue.put(("complete", results))
                return results
        
        pipeline = StreamlitAutoMLPipeline(output_dir, progress_queue)
        results = pipeline.run_pipeline(data_source, target_column)
        
    except Exception as e:
        progress_queue.put(("error", str(e)))

def main():
    """Main Streamlit application"""
    inject_custom_css()
    
    # Header
    st.markdown("<h1 class='main-header'>🤖 Automated ML Pipeline</h1>", unsafe_allow_html=True)
    
    # Feature highlights
    st.markdown("""
    <div class='feature-highlight'>
        <h3>🚀 Fully Automated Machine Learning</h3>
        <p>Upload your data, specify the target column, and let our AI handle everything else!</p>
        <ul>
            <li>📊 Automatic EDA and visualization</li>
            <li>🧹 Intelligent data cleaning</li>
            <li>🎯 Smart feature selection</li>
            <li>🔧 Hyperparameter optimization</li>
            <li>📦 Ready-to-use model export</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload your dataset",
            type=['csv', 'xlsx', 'xls'],
            help="Supported formats: CSV, Excel"
        )
        
        # URL input as alternative
        st.markdown("**Or provide a URL:**")
        data_url = st.text_input(
            "Data URL",
            placeholder="https://example.com/data.csv",
            help="Direct link to CSV, Excel, or JSON data"
        )
        
        # Output directory
        output_dir = st.text_input(
            "Output Directory",
            value="automl_output",
            help="Directory to save results and model"
        )
    
    # Main content area
    if uploaded_file is not None or data_url:
        # Handle data source
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            data_source = temp_path
        else:
            data_source = data_url
        
        # Load and preview data
        try:
            if data_source.startswith(('http://', 'https://')):
                preview_df = pd.read_csv(data_source, nrows=1000)  # Preview first 1000 rows
            else:
                if data_source.endswith(('.xlsx', '.xls')):
                    preview_df = pd.read_excel(data_source, nrows=1000)
                else:
                    preview_df = pd.read_csv(data_source, nrows=1000)
            
            st.success(f"✅ Data loaded successfully! Shape: {preview_df.shape}")
            
            # Data preview
            with st.expander("📋 Data Preview", expanded=True):
                st.dataframe(preview_df.head(10), use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Rows (preview)", f"{len(preview_df):,}")
                with col2:
                    st.metric("Columns", f"{len(preview_df.columns):,}")
            
            # Target column selection
            st.subheader("🎯 Select Target Column")
            target_column = st.selectbox(
                "Choose the column you want to predict:",
                options=preview_df.columns.tolist(),
                help="This is the column your model will learn to predict"
            )
            
            if target_column:
                # Show target column info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Unique Values", preview_df[target_column].nunique())
                with col2:
                    st.metric("Missing Values", preview_df[target_column].isnull().sum())
                with col3:
                    task_type = "Classification" if preview_df[target_column].nunique() <= 20 else "Regression"
                    st.metric("Task Type", task_type)
                
                # Target distribution
                if preview_df[target_column].nunique() <= 20:
                    fig = px.histogram(preview_df, x=target_column, title="Target Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = px.histogram(preview_df, x=target_column, title="Target Distribution", nbins=50)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Run pipeline button
                if st.button("🚀 Start Automated ML Pipeline", type="primary", use_container_width=True):
                    run_automated_pipeline(data_source, target_column, output_dir)
        
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
    
    else:
        # Welcome message
        st.markdown("""
        ## 👋 Welcome to the Automated ML Pipeline!
        
        To get started:
        1. **Upload your dataset** using the file uploader in the sidebar
        2. **Or provide a URL** to your data
        3. **Select your target column** (what you want to predict)
        4. **Click "Start Pipeline"** and let the AI do the rest!
        
        ### 📋 Supported Data Formats:
        - CSV files (.csv)
        - Excel files (.xlsx, .xls)
        - Direct URLs to data files
        
        ### 🎯 What the Pipeline Does:
        - **Loads and analyzes** your data automatically
        - **Cleans and prepares** data (handles missing values, outliers, etc.)
        - **Selects the best features** to reduce overfitting
        - **Trains multiple models** with hyperparameter tuning
        - **Exports the best model** as a downloadable pickle file
        
        The entire process is fully automated - just upload and wait for your trained model!
        """)

def run_automated_pipeline(data_source: str, target_column: str, output_dir: str):
    """Run the automated ML pipeline with Streamlit interface"""
    
    # Create progress indicators
    st.markdown("<div class='progress-container'>", unsafe_allow_html=True)
    st.markdown("### 🔄 Pipeline Progress")
    
    # Progress tracking
    progress_tracker = StreamlitProgressTracker()
    
    # Status container
    status_container = st.container()
    
    try:
        # Initialize and run pipeline
        pipeline = AutoMLPipeline(output_dir=output_dir)
        
        # Update progress for each stage
        stages = [
            "Data Loading", "Data Exploration", "Data Visualization",
            "Data Cleaning", "Feature Selection", "Data Transformation",
            "Model Selection", "Hyperparameter Tuning", "Model Training", "Model Export"
        ]
        
        start_time = time.time()
        
        # Run the pipeline (this is a simplified version - in practice, you'd need threading)
        with st.spinner("Running automated ML pipeline..."):
            results = pipeline.run_pipeline(data_source, target_column)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display results
        st.success("🎉 Pipeline completed successfully!")
        
        execution_time = results['execution_time']
        st.info(f"⏱️ Total execution time: {execution_time/60:.1f} minutes")
        
        # Results tabs
        result_tabs = st.tabs([
            "📊 EDA Results", 
            "📈 Visualizations", 
            "🎯 Model Results", 
            "🔍 Feature Selection",
            "📦 Download Model"
        ])
        
        with result_tabs[0]:
            display_eda_results(results['eda_results'])
        
        with result_tabs[1]:
            display_visualizations(results['visualizations'])
        
        with result_tabs[2]:
            display_model_results(results['model_results'], results['best_model'])
        
        with result_tabs[3]:
            display_feature_importance(results['feature_scores'], results['data_info']['selected_features'])
        
        with result_tabs[4]:
            st.markdown("### 📦 Download Your Trained Model")
            
            model_path = results['best_model']['path']
            if Path(model_path).exists():
                # Model info
                st.info(f"""
                **Model:** {results['best_model']['name']}  
                **Features:** {len(results['data_info']['selected_features'])}  
                **Export Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """)
                
                # Download button
                with open(model_path, "rb") as file:
                    st.download_button(
                        label="📥 Download Model (.pkl)",
                        data=file.read(),
                        file_name=Path(model_path).name,
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                
                # Usage instructions
                with st.expander("📋 How to Use Your Model"):
                    st.code("""
import pickle
import pandas as pd

# Load the model
with open('your_model.pkl', 'rb') as f:
    model_package = pickle.load(f)

model = model_package['model']
feature_names = model_package['feature_names']
transformer = model_package['transformer']

# Prepare new data (same format as training data)
new_data = pd.DataFrame(...)  # Your new data

# Select only the features used during training
new_data_features = new_data[feature_names]

# Transform the data (if needed)
if transformer:
    new_data_transformed, _ = transformer.transform_data(new_data_features)
else:
    new_data_transformed = new_data_features

# Make predictions
predictions = model.predict(new_data_transformed)
print(predictions)
                    """, language='python')
            
            # Additional files
            st.markdown("### 📁 Additional Files")
            
            # Summary file
            summary_path = Path(output_dir) / "pipeline_summary.txt"
            if summary_path.exists():
                with open(summary_path, "r") as f:
                    summary_content = f.read()
                
                with st.expander("📄 Pipeline Summary"):
                    st.text(summary_content)
                
                st.download_button(
                    label="📥 Download Summary",
                    data=summary_content,
                    file_name="pipeline_summary.txt",
                    mime="text/plain"
                )
            
            # Visualizations download
            viz_dir = Path(output_dir) / "visualizations"
            if viz_dir.exists():
                st.markdown("**Generated Visualizations:**")
                for viz_file in viz_dir.glob("*.png"):
                    st.markdown(f"- {viz_file.name}")
    
    except Exception as e:
        st.error(f"❌ Pipeline failed: {str(e)}")
        
        # Show error details in expandable section
        with st.expander("🔍 Error Details"):
            st.exception(e)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Fully Automated Machine Learning Pipeline
=========================================

This module provides a complete automated ML pipeline that handles:
- Data loading from files/URLs
- Exploratory Data Analysis (EDA)
- Automatic data visualization and analysis
- Data cleaning and preparation
- Feature selection and transformation
- Model training with hyperparameter tuning
- Model export as pickle files
- Progress tracking throughout the process

Usage:
    python automated_ml_pipeline.py --input data.csv --target target_column
"""

import os
import sys
import pickle
import warnings
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import time

# Core data science libraries
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Visualization libraries
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# AutoML and optimization
import optuna

# Progress tracking
from tqdm import tqdm

# Utilities
import requests
from urllib.parse import urlparse
import joblib

# Configure warnings and logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProgressTracker:
    """Track and display progress of ML pipeline stages"""
    
    def __init__(self, total_stages: int = 10):
        self.total_stages = total_stages
        self.current_stage = 0
        self.stage_names = [
            "Data Loading",
            "Data Exploration", 
            "Data Visualization",
            "Data Cleaning",
            "Feature Selection",
            "Data Transformation",
            "Model Selection",
            "Hyperparameter Tuning", 
            "Model Training",
            "Model Export"
        ]
        self.start_time = time.time()
        
    def update_stage(self, stage_name: str, progress: float = None):
        """Update current stage and progress"""
        if stage_name in self.stage_names:
            self.current_stage = self.stage_names.index(stage_name) + 1
        
        elapsed_time = time.time() - self.start_time
        if progress:
            estimated_total = elapsed_time / progress
            remaining_time = estimated_total - elapsed_time
        else:
            progress = self.current_stage / self.total_stages
            estimated_total = elapsed_time / progress if progress > 0 else 0
            remaining_time = estimated_total - elapsed_time
        
        print(f"\n{'='*50}")
        print(f"Stage {self.current_stage}/{self.total_stages}: {stage_name}")
        print(f"Progress: {progress*100:.1f}%")
        print(f"Elapsed: {elapsed_time/60:.1f} minutes")
        print(f"Estimated remaining: {remaining_time/60:.1f} minutes")
        print(f"{'='*50}")


class AutomatedDataLoader:
    """Automated data loading from various sources"""
    
    @staticmethod
    def load_data(source: str) -> pd.DataFrame:
        """Load data from file path or URL"""
        logger.info(f"Loading data from: {source}")
        
        # Check if source is URL
        if source.startswith(('http://', 'https://')):
            return AutomatedDataLoader._load_from_url(source)
        else:
            return AutomatedDataLoader._load_from_file(source)
    
    @staticmethod
    def _load_from_url(url: str) -> pd.DataFrame:
        """Load data from URL"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Determine file type from URL or content-type
            content_type = response.headers.get('content-type', '')
            if 'csv' in content_type or url.endswith('.csv'):
                return pd.read_csv(url)
            elif 'json' in content_type or url.endswith('.json'):
                return pd.read_json(url)
            elif url.endswith(('.xlsx', '.xls')):
                return pd.read_excel(url)
            else:
                # Try CSV as default
                return pd.read_csv(url)
                
        except Exception as e:
            logger.error(f"Failed to load data from URL: {e}")
            raise
    
    @staticmethod
    def _load_from_file(file_path: str) -> pd.DataFrame:
        """Load data from local file"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            if path.suffix.lower() == '.csv':
                return pd.read_csv(file_path)
            elif path.suffix.lower() in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            elif path.suffix.lower() == '.json':
                return pd.read_json(file_path)
            elif path.suffix.lower() == '.parquet':
                return pd.read_parquet(file_path)
            else:
                # Try CSV as default
                return pd.read_csv(file_path)
                
        except Exception as e:
            logger.error(f"Failed to load data from file: {e}")
            raise


class AutomatedEDA:
    """Automated Exploratory Data Analysis"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.analysis_results = {}
        
    def perform_eda(self) -> Dict[str, Any]:
        """Perform comprehensive EDA"""
        logger.info("Performing automated EDA...")
        
        self.analysis_results = {
            'basic_info': self._get_basic_info(),
            'data_types': self._analyze_data_types(),
            'missing_values': self._analyze_missing_values(),
            'statistical_summary': self._get_statistical_summary(),
            'correlations': self._analyze_correlations(),
            'outliers': self._detect_outliers(),
            'categorical_analysis': self._analyze_categorical_features(),
            'numerical_analysis': self._analyze_numerical_features()
        }
        
        return self.analysis_results
    
    def _get_basic_info(self) -> Dict[str, Any]:
        """Get basic dataset information"""
        return {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum(),
            'duplicate_rows': self.df.duplicated().sum()
        }
    
    def _analyze_data_types(self) -> Dict[str, List[str]]:
        """Categorize columns by data type"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = self.df.select_dtypes(include=['datetime64']).columns.tolist()
        
        return {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'datetime': datetime_cols
        }
    
    def _analyze_missing_values(self) -> Dict[str, Any]:
        """Analyze missing values pattern"""
        missing_counts = self.df.isnull().sum()
        missing_percentages = (missing_counts / len(self.df)) * 100
        
        return {
            'missing_counts': missing_counts.to_dict(),
            'missing_percentages': missing_percentages.to_dict(),
            'columns_with_missing': missing_counts[missing_counts > 0].index.tolist()
        }
    
    def _get_statistical_summary(self) -> Dict[str, Any]:
        """Get statistical summary of numerical columns"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return self.df[numeric_cols].describe().to_dict()
        return {}
    
    def _analyze_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between numerical features"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:  # High correlation threshold
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr_val
                        })
            
            return {
                'correlation_matrix': corr_matrix.to_dict(),
                'high_correlations': high_corr_pairs
            }
        return {}
    
    def _detect_outliers(self) -> Dict[str, List[int]]:
        """Detect outliers using IQR method"""
        outliers = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_indices = self.df[(self.df[col] < lower_bound) | 
                                    (self.df[col] > upper_bound)].index.tolist()
            outliers[col] = outlier_indices
        
        return outliers
    
    def _analyze_categorical_features(self) -> Dict[str, Any]:
        """Analyze categorical features"""
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        analysis = {}
        
        for col in categorical_cols:
            unique_values = self.df[col].nunique()
            value_counts = self.df[col].value_counts()
            
            analysis[col] = {
                'unique_count': unique_values,
                'unique_ratio': unique_values / len(self.df),
                'top_values': value_counts.head(10).to_dict(),
                'is_high_cardinality': unique_values > len(self.df) * 0.1
            }
        
        return analysis
    
    def _analyze_numerical_features(self) -> Dict[str, Any]:
        """Analyze numerical features"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        analysis = {}
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            if len(data) > 0:
                analysis[col] = {
                    'skewness': stats.skew(data),
                    'kurtosis': stats.kurtosis(data),
                    'is_normal': stats.normaltest(data)[1] > 0.05 if len(data) > 8 else False,
                    'zero_count': (data == 0).sum(),
                    'negative_count': (data < 0).sum()
                }
        
        return analysis


class AutomatedVisualizer:
    """Automated data visualization and analysis"""
    
    def __init__(self, df: pd.DataFrame, output_dir: str = "visualizations"):
        self.df = df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.visual_insights = {}
        
    def create_visualizations(self) -> Dict[str, str]:
        """Create comprehensive visualizations"""
        logger.info("Creating automated visualizations...")
        
        visualizations = {}
        
        # Data overview
        visualizations['data_overview'] = self._create_data_overview()
        
        # Missing values heatmap
        visualizations['missing_values'] = self._create_missing_values_heatmap()
        
        # Correlation heatmap
        visualizations['correlations'] = self._create_correlation_heatmap()
        
        # Distribution plots
        visualizations['distributions'] = self._create_distribution_plots()
        
        # Categorical analysis
        visualizations['categorical'] = self._create_categorical_plots()
        
        # Outlier detection plots
        visualizations['outliers'] = self._create_outlier_plots()
        
        # Feature importance (if target is available)
        # visualizations['feature_importance'] = self._create_feature_importance_plots()
        
        return visualizations
    
    def _create_data_overview(self) -> str:
        """Create data overview visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Dataset Overview', fontsize=16)
        
        # Data types
        dtype_counts = self.df.dtypes.value_counts()
        axes[0, 0].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Data Types Distribution')
        
        # Missing values
        missing_counts = self.df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        if len(missing_cols) > 0:
            axes[0, 1].bar(range(len(missing_cols)), missing_cols.values)
            axes[0, 1].set_xticks(range(len(missing_cols)))
            axes[0, 1].set_xticklabels(missing_cols.index, rotation=45)
            axes[0, 1].set_title('Missing Values by Column')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center')
            axes[0, 1].set_title('Missing Values')
        
        # Dataset shape info
        axes[1, 0].text(0.1, 0.8, f'Rows: {self.df.shape[0]:,}', fontsize=12)
        axes[1, 0].text(0.1, 0.6, f'Columns: {self.df.shape[1]:,}', fontsize=12)
        axes[1, 0].text(0.1, 0.4, f'Memory: {self.df.memory_usage(deep=True).sum()/1024/1024:.1f} MB', fontsize=12)
        axes[1, 0].text(0.1, 0.2, f'Duplicates: {self.df.duplicated().sum():,}', fontsize=12)
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].set_title('Dataset Statistics')
        axes[1, 0].axis('off')
        
        # Column types breakdown
        numeric_cols = len(self.df.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(self.df.select_dtypes(include=['object', 'category']).columns)
        datetime_cols = len(self.df.select_dtypes(include=['datetime64']).columns)
        
        categories = ['Numeric', 'Categorical', 'DateTime']
        counts = [numeric_cols, categorical_cols, datetime_cols]
        axes[1, 1].bar(categories, counts)
        axes[1, 1].set_title('Column Types')
        
        plt.tight_layout()
        output_path = self.output_dir / 'data_overview.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _create_missing_values_heatmap(self) -> str:
        """Create missing values heatmap"""
        plt.figure(figsize=(12, 8))
        missing_data = self.df.isnull()
        
        if missing_data.sum().sum() > 0:
            sns.heatmap(missing_data, yticklabels=False, cbar=True, cmap='viridis')
            plt.title('Missing Values Heatmap')
            plt.xlabel('Columns')
            plt.ylabel('Rows')
        else:
            plt.text(0.5, 0.5, 'No Missing Values in Dataset', 
                    ha='center', va='center', fontsize=16)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.title('Missing Values Analysis')
        
        output_path = self.output_dir / 'missing_values_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _create_correlation_heatmap(self) -> str:
        """Create correlation heatmap for numerical features"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 10))
            correlation_matrix = self.df[numeric_cols].corr()
            
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       fmt='.2f', square=True, linewidths=0.5)
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()
        else:
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, 'Insufficient numerical columns for correlation analysis', 
                    ha='center', va='center', fontsize=14)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.title('Correlation Analysis')
        
        output_path = self.output_dir / 'correlation_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _create_distribution_plots(self) -> str:
        """Create distribution plots for numerical features"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, 'No numerical columns found', 
                    ha='center', va='center', fontsize=14)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.title('Distribution Analysis')
        else:
            n_cols = min(4, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(numeric_cols):
                if i < len(axes):
                    self.df[col].hist(bins=30, ax=axes[i], alpha=0.7)
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
            
            # Hide unused subplots
            for i in range(len(numeric_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
        
        output_path = self.output_dir / 'distributions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _create_categorical_plots(self) -> str:
        """Create plots for categorical features"""
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns[:6]  # Limit to 6
        
        if len(categorical_cols) == 0:
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, 'No categorical columns found', 
                    ha='center', va='center', fontsize=14)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.title('Categorical Analysis')
        else:
            n_cols = min(3, len(categorical_cols))
            n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(categorical_cols):
                if i < len(axes):
                    value_counts = self.df[col].value_counts().head(10)
                    value_counts.plot(kind='bar', ax=axes[i])
                    axes[i].set_title(f'Top Values in {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Count')
                    axes[i].tick_params(axis='x', rotation=45)
            
            # Hide unused subplots
            for i in range(len(categorical_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
        
        output_path = self.output_dir / 'categorical_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _create_outlier_plots(self) -> str:
        """Create box plots to visualize outliers"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns[:6]  # Limit to 6
        
        if len(numeric_cols) == 0:
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, 'No numerical columns for outlier analysis', 
                    ha='center', va='center', fontsize=14)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.title('Outlier Analysis')
        else:
            n_cols = min(3, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(numeric_cols):
                if i < len(axes):
                    self.df.boxplot(column=col, ax=axes[i])
                    axes[i].set_title(f'Outliers in {col}')
            
            # Hide unused subplots
            for i in range(len(numeric_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
        
        output_path = self.output_dir / 'outlier_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)


class AutomatedDataCleaner:
    """Automated data cleaning and preparation"""
    
    def __init__(self, df: pd.DataFrame, target_column: str = None):
        self.df = df.copy()
        self.target_column = target_column
        self.cleaning_log = []
        
    def clean_data(self) -> pd.DataFrame:
        """Perform comprehensive data cleaning"""
        logger.info("Starting automated data cleaning...")
        
        # Remove duplicate rows
        self._remove_duplicates()
        
        # Handle missing values
        self._handle_missing_values()
        
        # Fix data types
        self._fix_data_types()
        
        # Handle outliers
        self._handle_outliers()
        
        # Clean text data
        self._clean_text_data()
        
        # Remove constant/near-constant features
        self._remove_constant_features()
        
        logger.info(f"Data cleaning completed. Applied {len(self.cleaning_log)} cleaning operations.")
        return self.df
    
    def get_cleaning_log(self) -> List[str]:
        """Get log of cleaning operations performed"""
        return self.cleaning_log
    
    def _remove_duplicates(self):
        """Remove duplicate rows"""
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        removed_rows = initial_rows - len(self.df)
        
        if removed_rows > 0:
            self.cleaning_log.append(f"Removed {removed_rows} duplicate rows")
    
    def _handle_missing_values(self):
        """Handle missing values with appropriate strategies"""
        missing_cols = self.df.columns[self.df.isnull().any()].tolist()
        
        for col in missing_cols:
            missing_count = self.df[col].isnull().sum()
            missing_ratio = missing_count / len(self.df)
            
            if missing_ratio > 0.5:
                # Drop columns with >50% missing values
                self.df = self.df.drop(columns=[col])
                self.cleaning_log.append(f"Dropped column '{col}' (>{missing_ratio:.1%} missing)")
            
            elif self.df[col].dtype in ['object', 'category']:
                # Fill categorical missing values with mode or 'Unknown'
                if self.df[col].mode().empty:
                    self.df[col] = self.df[col].fillna('Unknown')
                    self.cleaning_log.append(f"Filled missing values in '{col}' with 'Unknown'")
                else:
                    mode_value = self.df[col].mode()[0]
                    self.df[col] = self.df[col].fillna(mode_value)
                    self.cleaning_log.append(f"Filled missing values in '{col}' with mode: '{mode_value}'")
            
            elif self.df[col].dtype in [np.number]:
                # For numerical columns, choose strategy based on distribution
                if self.df[col].skew() > 1 or self.df[col].skew() < -1:
                    # Highly skewed - use median
                    median_value = self.df[col].median()
                    self.df[col] = self.df[col].fillna(median_value)
                    self.cleaning_log.append(f"Filled missing values in '{col}' with median: {median_value}")
                else:
                    # Normal distribution - use mean
                    mean_value = self.df[col].mean()
                    self.df[col] = self.df[col].fillna(mean_value)
                    self.cleaning_log.append(f"Filled missing values in '{col}' with mean: {mean_value:.2f}")
    
    def _fix_data_types(self):
        """Automatically fix data types"""
        for col in self.df.columns:
            # Try to convert object columns to numeric if possible
            if self.df[col].dtype == 'object':
                # Check if it's actually numeric
                try:
                    numeric_series = pd.to_numeric(self.df[col], errors='coerce')
                    if numeric_series.notna().sum() / len(self.df) > 0.8:  # 80% conversion success
                        self.df[col] = numeric_series
                        self.cleaning_log.append(f"Converted '{col}' from object to numeric")
                except:
                    pass
                
                # Check if it's datetime
                try:
                    datetime_series = pd.to_datetime(self.df[col], errors='coerce')
                    if datetime_series.notna().sum() / len(self.df) > 0.8:
                        self.df[col] = datetime_series
                        self.cleaning_log.append(f"Converted '{col}' from object to datetime")
                except:
                    pass
    
    def _handle_outliers(self):
        """Handle outliers in numerical columns"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outlier_count = outliers_mask.sum()
            
            if outlier_count > 0 and outlier_count < len(self.df) * 0.1:  # Less than 10% outliers
                # Cap outliers instead of removing them
                self.df.loc[self.df[col] < lower_bound, col] = lower_bound
                self.df.loc[self.df[col] > upper_bound, col] = upper_bound
                self.cleaning_log.append(f"Capped {outlier_count} outliers in '{col}'")
    
    def _clean_text_data(self):
        """Clean text data in categorical columns"""
        text_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in text_cols:
            if self.df[col].dtype == 'object':
                # Strip whitespace
                self.df[col] = self.df[col].astype(str).str.strip()
                
                # Convert to lowercase for consistency (if not already done)
                if self.df[col].str.islower().sum() < len(self.df) * 0.8:
                    self.df[col] = self.df[col].str.lower()
                    self.cleaning_log.append(f"Standardized text case in '{col}'")
    
    def _remove_constant_features(self):
        """Remove features with constant or near-constant values"""
        constant_cols = []
        
        for col in self.df.columns:
            # Skip target column
            if col == self.target_column:
                continue
                
            unique_ratio = self.df[col].nunique() / len(self.df)
            if unique_ratio < 0.01:  # Less than 1% unique values
                constant_cols.append(col)
        
        if constant_cols:
            self.df = self.df.drop(columns=constant_cols)
            self.cleaning_log.append(f"Removed {len(constant_cols)} near-constant columns: {constant_cols}")


class AutomatedFeatureSelector:
    """Automated feature selection to reduce overfitting and improve performance"""
    
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df
        self.target_column = target_column
        self.selected_features = []
        self.feature_scores = {}
        
    def select_features(self, max_features: int = None) -> List[str]:
        """Perform automated feature selection"""
        logger.info("Starting automated feature selection...")
        
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        
        # Prepare features for selection
        X_prepared = self._prepare_features_for_selection(X)
        
        # Determine if it's classification or regression
        is_classification = self._is_classification_task(y)
        
        # Apply different selection methods
        selected_features = []
        
        # 1. Statistical tests
        statistical_features = self._statistical_feature_selection(X_prepared, y, is_classification)
        selected_features.extend(statistical_features)
        
        # 2. Tree-based feature importance
        tree_features = self._tree_based_feature_selection(X_prepared, y, is_classification)
        selected_features.extend(tree_features)
        
        # 3. Correlation-based selection
        correlation_features = self._correlation_based_selection(X_prepared, y)
        selected_features.extend(correlation_features)
        
        # Combine and rank features
        feature_counts = {}
        for feature in selected_features:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        # Sort by frequency of selection across methods
        ranked_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Select top features
        if max_features is None:
            max_features = min(len(ranked_features), max(10, len(X.columns) // 2))
        
        self.selected_features = [feature for feature, _ in ranked_features[:max_features]]
        
        logger.info(f"Selected {len(self.selected_features)} features out of {len(X.columns)}")
        return self.selected_features
    
    def _prepare_features_for_selection(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for selection by encoding categorical variables"""
        X_prepared = X.copy()
        
        # Encode categorical variables
        categorical_cols = X_prepared.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if X_prepared[col].nunique() <= 10:  # Low cardinality - use one-hot encoding
                dummies = pd.get_dummies(X_prepared[col], prefix=col)
                X_prepared = pd.concat([X_prepared, dummies], axis=1)
                X_prepared = X_prepared.drop(columns=[col])
            else:  # High cardinality - use label encoding
                le = LabelEncoder()
                X_prepared[col] = le.fit_transform(X_prepared[col].astype(str))
        
        return X_prepared
    
    def _is_classification_task(self, y: pd.Series) -> bool:
        """Determine if the task is classification or regression"""
        if y.dtype == 'object' or y.nunique() <= 20:
            return True
        return False
    
    def _statistical_feature_selection(self, X: pd.DataFrame, y: pd.Series, is_classification: bool) -> List[str]:
        """Feature selection using statistical tests"""
        if is_classification:
            selector = SelectKBest(score_func=f_classif, k='all')
        else:
            selector = SelectKBest(score_func=f_regression, k='all')
        
        selector.fit(X, y)
        
        # Get feature scores
        feature_scores = dict(zip(X.columns, selector.scores_))
        self.feature_scores['statistical'] = feature_scores
        
        # Select top 50% features
        n_features = max(5, len(X.columns) // 2)
        top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:n_features]
        
        return [feature for feature, _ in top_features]
    
    def _tree_based_feature_selection(self, X: pd.DataFrame, y: pd.Series, is_classification: bool) -> List[str]:
        """Feature selection using tree-based feature importance"""
        if is_classification:
            model = RandomForestClassifier(n_estimators=50, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=50, random_state=42)
        
        model.fit(X, y)
        
        # Get feature importance
        feature_importance = dict(zip(X.columns, model.feature_importances_))
        self.feature_scores['tree_based'] = feature_importance
        
        # Select features with importance above threshold
        importance_threshold = np.mean(model.feature_importances_)
        important_features = [feature for feature, importance in feature_importance.items() 
                            if importance >= importance_threshold]
        
        return important_features
    
    def _correlation_based_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Feature selection based on correlation with target"""
        correlations = {}
        
        for col in X.columns:
            try:
                corr = abs(X[col].corr(y))
                if not np.isnan(corr):
                    correlations[col] = corr
            except:
                correlations[col] = 0
        
        self.feature_scores['correlation'] = correlations
        
        # Select features with correlation above threshold
        correlation_threshold = 0.1
        correlated_features = [feature for feature, corr in correlations.items() 
                             if corr >= correlation_threshold]
        
        return correlated_features


class AutomatedTransformer:
    """Automated data transformation and scaling"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.transformation_log = []
        
    def transform_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None, 
                      y_train: pd.Series = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply automated transformations to the data"""
        logger.info("Starting automated data transformation...")
        
        X_train_transformed = X_train.copy()
        X_test_transformed = X_test.copy() if X_test is not None else None
        
        # 1. Handle categorical variables
        X_train_transformed, X_test_transformed = self._encode_categorical_features(
            X_train_transformed, X_test_transformed)
        
        # 2. Scale numerical features
        X_train_transformed, X_test_transformed = self._scale_numerical_features(
            X_train_transformed, X_test_transformed)
        
        logger.info(f"Applied {len(self.transformation_log)} transformations")
        return X_train_transformed, X_test_transformed
    
    def _encode_categorical_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None):
        """Encode categorical features"""
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            unique_values = X_train[col].nunique()
            
            if unique_values <= 10:  # Low cardinality - use one-hot encoding
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                
                # Fit on training data
                encoded_train = encoder.fit_transform(X_train[[col]])
                feature_names = [f"{col}_{category}" for category in encoder.categories_[0]]
                
                # Create DataFrame for encoded features
                encoded_train_df = pd.DataFrame(encoded_train, columns=feature_names, index=X_train.index)
                
                # Replace original column
                X_train = pd.concat([X_train.drop(columns=[col]), encoded_train_df], axis=1)
                
                if X_test is not None:
                    encoded_test = encoder.transform(X_test[[col]])
                    encoded_test_df = pd.DataFrame(encoded_test, columns=feature_names, index=X_test.index)
                    X_test = pd.concat([X_test.drop(columns=[col]), encoded_test_df], axis=1)
                
                self.encoders[col] = encoder
                self.transformation_log.append(f"One-hot encoded '{col}' ({unique_values} categories)")
                
            else:  # High cardinality - use label encoding
                encoder = LabelEncoder()
                X_train[col] = encoder.fit_transform(X_train[col].astype(str))
                
                if X_test is not None:
                    # Handle unknown categories in test set
                    test_labels = []
                    for value in X_test[col].astype(str):
                        if value in encoder.classes_:
                            test_labels.append(encoder.transform([value])[0])
                        else:
                            test_labels.append(-1)  # Unknown category
                    X_test[col] = test_labels
                
                self.encoders[col] = encoder
                self.transformation_log.append(f"Label encoded '{col}' ({unique_values} categories)")
        
        return X_train, X_test
    
    def _scale_numerical_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None):
        """Scale numerical features"""
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            return X_train, X_test
        
        # Choose scaler based on data distribution
        scaler_choice = {}
        
        for col in numerical_cols:
            skewness = abs(X_train[col].skew())
            
            if skewness > 1:  # Highly skewed - use robust scaling
                scaler = MinMaxScaler()
                scaler_choice[col] = 'MinMax'
            else:  # Normal distribution - use standard scaling
                scaler = StandardScaler()
                scaler_choice[col] = 'Standard'
            
            # Fit and transform training data
            X_train[col] = scaler.fit_transform(X_train[[col]]).flatten()
            
            # Transform test data if provided
            if X_test is not None:
                X_test[col] = scaler.transform(X_test[[col]]).flatten()
            
            self.scalers[col] = scaler
        
        self.transformation_log.append(f"Scaled {len(numerical_cols)} numerical features")
        return X_train, X_test


class AutomatedModelTrainer:
    """Automated model selection, training, and hyperparameter tuning"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_params = None
        self.is_classification = None
        
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_test: pd.DataFrame = None, y_test: pd.Series = None) -> Dict[str, Any]:
        """Train multiple models and select the best one"""
        logger.info("Starting automated model training and selection...")
        
        # Determine task type
        self.is_classification = self._determine_task_type(y_train)
        
        # Get candidate models
        candidate_models = self._get_candidate_models()
        
        # Train and evaluate models
        results = {}
        
        for model_name, model in tqdm(candidate_models.items(), desc="Training models"):
            try:
                # Hyperparameter tuning using Optuna
                best_params = self._optimize_hyperparameters(model, X_train, y_train, model_name)
                
                # Train final model with best parameters
                final_model = model.set_params(**best_params)
                final_model.fit(X_train, y_train)
                
                # Evaluate model
                scores = self._evaluate_model(final_model, X_train, y_train, X_test, y_test)
                
                results[model_name] = {
                    'model': final_model,
                    'params': best_params,
                    'scores': scores
                }
                
                self.models[model_name] = final_model
                
            except Exception as e:
                logger.warning(f"Failed to train {model_name}: {e}")
                continue
        
        # Select best model
        self._select_best_model(results)
        
        logger.info(f"Best model: {self.best_model_name}")
        return results
    
    def _determine_task_type(self, y: pd.Series) -> bool:
        """Determine if task is classification or regression"""
        if y.dtype == 'object' or y.nunique() <= 20:
            return True  # Classification
        return False  # Regression
    
    def _get_candidate_models(self) -> Dict[str, Any]:
        """Get candidate models based on task type"""
        if self.is_classification:
            return {
                'RandomForest': RandomForestClassifier(random_state=42),
                'XGBoost': xgb.XGBClassifier(random_state=42),
                'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
                'KNeighbors': KNeighborsClassifier(),
                'DecisionTree': DecisionTreeClassifier(random_state=42),
                'NaiveBayes': GaussianNB()
            }
        else:
            return {
                'RandomForest': RandomForestRegressor(random_state=42),
                'XGBoost': xgb.XGBRegressor(random_state=42),
                'LinearRegression': LinearRegression(),
                'KNeighbors': KNeighborsRegressor(),
                'DecisionTree': DecisionTreeRegressor(random_state=42)
            }
    
    def _optimize_hyperparameters(self, model, X_train: pd.DataFrame, y_train: pd.Series, 
                                model_name: str, n_trials: int = 20) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""
        
        def objective(trial):
            params = self._suggest_hyperparameters(trial, model_name)
            temp_model = model.set_params(**params)
            
            # Use cross-validation for evaluation
            cv_scores = cross_val_score(temp_model, X_train, y_train, cv=3, 
                                      scoring='accuracy' if self.is_classification else 'r2')
            return cv_scores.mean()
        
        study = optuna.create_study(direction='maximize')
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        return study.best_params
    
    def _suggest_hyperparameters(self, trial, model_name: str) -> Dict[str, Any]:
        """Suggest hyperparameters for different models"""
        if model_name == 'RandomForest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
            }
        elif model_name == 'XGBoost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0)
            }
        elif model_name == 'LogisticRegression':
            return {
                'C': trial.suggest_float('C', 0.01, 100, log=True),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs'])
            }
        elif model_name == 'KNeighbors':
            return {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 20),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance'])
            }
        elif model_name == 'DecisionTree':
            return {
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
            }
        else:
            return {}
    
    def _evaluate_model(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                       X_test: pd.DataFrame = None, y_test: pd.Series = None) -> Dict[str, float]:
        """Evaluate model performance"""
        scores = {}
        
        # Training scores
        y_train_pred = model.predict(X_train)
        
        if self.is_classification:
            scores['train_accuracy'] = accuracy_score(y_train, y_train_pred)
            if len(np.unique(y_train)) == 2:  # Binary classification
                scores['train_precision'] = precision_score(y_train, y_train_pred, average='binary')
                scores['train_recall'] = recall_score(y_train, y_train_pred, average='binary')
                scores['train_f1'] = f1_score(y_train, y_train_pred, average='binary')
        else:
            scores['train_r2'] = r2_score(y_train, y_train_pred)
            scores['train_mse'] = mean_squared_error(y_train, y_train_pred)
            scores['train_mae'] = mean_absolute_error(y_train, y_train_pred)
        
        # Test scores (if test data is provided)
        if X_test is not None and y_test is not None:
            y_test_pred = model.predict(X_test)
            
            if self.is_classification:
                scores['test_accuracy'] = accuracy_score(y_test, y_test_pred)
                if len(np.unique(y_test)) == 2:  # Binary classification
                    scores['test_precision'] = precision_score(y_test, y_test_pred, average='binary')
                    scores['test_recall'] = recall_score(y_test, y_test_pred, average='binary')
                    scores['test_f1'] = f1_score(y_test, y_test_pred, average='binary')
            else:
                scores['test_r2'] = r2_score(y_test, y_test_pred)
                scores['test_mse'] = mean_squared_error(y_test, y_test_pred)
                scores['test_mae'] = mean_absolute_error(y_test, y_test_pred)
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                  scoring='accuracy' if self.is_classification else 'r2')
        scores['cv_mean'] = cv_scores.mean()
        scores['cv_std'] = cv_scores.std()
        
        return scores
    
    def _select_best_model(self, results: Dict[str, Any]):
        """Select the best model based on cross-validation scores"""
        best_score = -np.inf
        
        for model_name, result in results.items():
            cv_score = result['scores']['cv_mean']
            if cv_score > best_score:
                best_score = cv_score
                self.best_model = result['model']
                self.best_model_name = model_name
                self.best_params = result['params']


class AutoMLPipeline:
    """Main automated ML pipeline orchestrator"""
    
    def __init__(self, output_dir: str = "automl_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.progress_tracker = ProgressTracker()
        self.data_loader = AutomatedDataLoader()
        self.eda_analyzer = None
        self.visualizer = None
        self.data_cleaner = None
        self.feature_selector = None
        self.transformer = None
        self.model_trainer = None
        
        self.raw_data = None
        self.cleaned_data = None
        self.selected_features = None
        self.transformed_data = None
        self.results = None
        
    def run_pipeline(self, data_source: str, target_column: str) -> Dict[str, Any]:
        """Run the complete automated ML pipeline"""
        logger.info("Starting Automated ML Pipeline...")
        start_time = time.time()
        
        try:
            # Stage 1: Data Loading
            self.progress_tracker.update_stage("Data Loading")
            self.raw_data = self.data_loader.load_data(data_source)
            logger.info(f"Loaded data with shape: {self.raw_data.shape}")
            
            # Stage 2: Exploratory Data Analysis
            self.progress_tracker.update_stage("Data Exploration")
            self.eda_analyzer = AutomatedEDA(self.raw_data)
            eda_results = self.eda_analyzer.perform_eda()
            
            # Stage 3: Data Visualization
            self.progress_tracker.update_stage("Data Visualization")
            viz_dir = self.output_dir / "visualizations"
            self.visualizer = AutomatedVisualizer(self.raw_data, str(viz_dir))
            visualizations = self.visualizer.create_visualizations()
            
            # Stage 4: Data Cleaning
            self.progress_tracker.update_stage("Data Cleaning")
            self.data_cleaner = AutomatedDataCleaner(self.raw_data, target_column)
            self.cleaned_data = self.data_cleaner.clean_data()
            
            # Check if target column exists after cleaning
            if target_column not in self.cleaned_data.columns:
                raise ValueError(f"Target column '{target_column}' not found in cleaned data")
            
            # Stage 5: Feature Selection
            self.progress_tracker.update_stage("Feature Selection")
            self.feature_selector = AutomatedFeatureSelector(self.cleaned_data, target_column)
            self.selected_features = self.feature_selector.select_features()
            
            # Prepare data with selected features
            X = self.cleaned_data[self.selected_features]
            y = self.cleaned_data[target_column]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) <= 20 else None
            )
            
            # Stage 6: Data Transformation
            self.progress_tracker.update_stage("Data Transformation")
            self.transformer = AutomatedTransformer()
            X_train_transformed, X_test_transformed = self.transformer.transform_data(
                X_train, X_test, y_train
            )
            
            # Stage 7-9: Model Selection, Hyperparameter Tuning, and Training
            self.progress_tracker.update_stage("Model Selection")
            self.model_trainer = AutomatedModelTrainer()
            model_results = self.model_trainer.train_models(
                X_train_transformed, y_train, X_test_transformed, y_test
            )
            
            # Stage 10: Model Export
            self.progress_tracker.update_stage("Model Export")
            model_path = self._export_model()
            
            # Compile final results
            self.results = {
                'data_info': {
                    'original_shape': self.raw_data.shape,
                    'cleaned_shape': self.cleaned_data.shape,
                    'selected_features': self.selected_features,
                    'target_column': target_column
                },
                'eda_results': eda_results,
                'visualizations': visualizations,
                'cleaning_log': self.data_cleaner.get_cleaning_log(),
                'feature_scores': self.feature_selector.feature_scores,
                'transformation_log': self.transformer.transformation_log,
                'model_results': model_results,
                'best_model': {
                    'name': self.model_trainer.best_model_name,
                    'params': self.model_trainer.best_params,
                    'path': model_path
                },
                'execution_time': time.time() - start_time
            }
            
            # Save results summary
            self._save_results_summary()
            
            logger.info(f"Pipeline completed successfully in {self.results['execution_time']/60:.1f} minutes")
            return self.results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _export_model(self) -> str:
        """Export the best model as a pickle file"""
        if self.model_trainer.best_model is None:
            raise ValueError("No trained model available for export")
        
        # Create model package
        model_package = {
            'model': self.model_trainer.best_model,
            'feature_names': self.selected_features,
            'transformer': self.transformer,
            'model_name': self.model_trainer.best_model_name,
            'best_params': self.model_trainer.best_params,
            'is_classification': self.model_trainer.is_classification,
            'export_timestamp': datetime.now().isoformat()
        }
        
        # Save model
        model_filename = f"automl_model_{self.model_trainer.best_model_name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        model_path = self.output_dir / model_filename
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        logger.info(f"Model exported to: {model_path}")
        return str(model_path)
    
    def _save_results_summary(self):
        """Save a summary of results to a text file"""
        summary_path = self.output_dir / "pipeline_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("AUTOMATED ML PIPELINE SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Execution Time: {self.results['execution_time']/60:.1f} minutes\n")
            f.write(f"Data Shape (Original): {self.results['data_info']['original_shape']}\n")
            f.write(f"Data Shape (Cleaned): {self.results['data_info']['cleaned_shape']}\n")
            f.write(f"Selected Features: {len(self.results['data_info']['selected_features'])}\n")
            f.write(f"Target Column: {self.results['data_info']['target_column']}\n\n")
            
            f.write("BEST MODEL\n")
            f.write("-" * 20 + "\n")
            f.write(f"Model: {self.results['best_model']['name']}\n")
            f.write(f"Parameters: {self.results['best_model']['params']}\n")
            f.write(f"Model File: {self.results['best_model']['path']}\n\n")
            
            f.write("DATA CLEANING LOG\n")
            f.write("-" * 20 + "\n")
            for log_entry in self.results['cleaning_log']:
                f.write(f"- {log_entry}\n")
            
            f.write("\nTRANSFORMATION LOG\n")
            f.write("-" * 20 + "\n")
            for log_entry in self.results['transformation_log']:
                f.write(f"- {log_entry}\n")
        
        logger.info(f"Results summary saved to: {summary_path}")


def main():
    """Main function to run the automated ML pipeline"""
    parser = argparse.ArgumentParser(description="Automated ML Pipeline")
    parser.add_argument("--input", required=True, help="Input data file path or URL")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--output", default="automl_output", help="Output directory")
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = AutoMLPipeline(output_dir=args.output)
    
    try:
        results = pipeline.run_pipeline(args.input, args.target)
        
        print("\n" + "="*50)
        print("AUTOMATED ML PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Best Model: {results['best_model']['name']}")
        print(f"Model File: {results['best_model']['path']}")
        print(f"Execution Time: {results['execution_time']/60:.1f} minutes")
        print(f"Output Directory: {args.output}")
        print("="*50)
        
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

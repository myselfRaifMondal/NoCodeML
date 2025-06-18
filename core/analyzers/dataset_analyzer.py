"""
Intelligent Dataset Analyzer for NoCodeML

This module provides comprehensive dataset analysis capabilities including:
- Data type detection and profiling
- Quality assessment
- ML problem type recommendations
- Data preprocessing suggestions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import re
import json
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from backend.models.schemas import (
    DatasetInfo, ColumnInfo, DataType, ProblemType
)

class DatasetAnalyzer:
    """Intelligent dataset analyzer with automatic profiling capabilities"""
    
    def __init__(self):
        self.min_samples_for_ml = 100
        self.high_cardinality_threshold = 0.5
        self.text_length_threshold = 50
        
    async def analyze_dataset(self, file_path: str) -> DatasetInfo:
        """
        Perform comprehensive dataset analysis
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            DatasetInfo: Complete analysis results
        """
        try:
            # Load the dataset
            df = self._load_dataset(file_path)
            
            # Basic information
            filename = Path(file_path).name.split("_", 1)[-1]  # Remove dataset ID prefix
            size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            
            # Analyze columns
            column_info = []
            for col in df.columns:
                col_info = self._analyze_column(df, col)
                column_info.append(col_info)
            
            # Calculate data quality score
            quality_score = self._calculate_quality_score(df)
            
            # Recommend problem types
            recommended_types = self._recommend_problem_types(df, column_info)
            
            # Generate suggestions and warnings
            suggestions = self._generate_suggestions(df, column_info)
            warnings = self._generate_warnings(df, column_info)
            
            return DatasetInfo(
                filename=filename,
                size_mb=round(size_mb, 2),
                rows=len(df),
                columns=len(df.columns),
                column_info=column_info,
                data_quality_score=quality_score,
                recommended_problem_types=recommended_types,
                suggestions=suggestions,
                warnings=warnings,
                upload_timestamp=datetime.now()
            )
            
        except Exception as e:
            raise Exception(f"Error analyzing dataset: {str(e)}")
    
    def _load_dataset(self, file_path: str) -> pd.DataFrame:
        """Load dataset from various file formats"""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        try:
            if extension == '.csv':
                # Try different encodings and separators
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise Exception("Could not decode CSV file with standard encodings")
                    
            elif extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
                
            elif extension == '.json':
                df = pd.read_json(file_path)
                
            elif extension == '.parquet':
                df = pd.read_parquet(file_path)
                
            else:
                raise Exception(f"Unsupported file format: {extension}")
            
            # Clean column names
            df.columns = df.columns.astype(str).str.strip()
            
            return df
            
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")
    
    def _analyze_column(self, df: pd.DataFrame, column: str) -> ColumnInfo:
        """Analyze a single column and determine its characteristics"""
        col_data = df[column]
        
        # Basic statistics
        missing_count = col_data.isnull().sum()
        missing_percentage = (missing_count / len(df)) * 100
        unique_count = col_data.nunique()
        
        # Sample values (non-null, unique)
        sample_values = col_data.dropna().drop_duplicates().head(5).tolist()
        
        # Determine data type
        data_type = self._detect_data_type(col_data)
        
        # Column-specific statistics
        statistics = self._calculate_column_statistics(col_data, data_type)
        
        return ColumnInfo(
            name=column,
            data_type=data_type,
            missing_count=int(missing_count),
            missing_percentage=round(missing_percentage, 2),
            unique_count=int(unique_count),
            sample_values=sample_values,
            statistics=statistics
        )
    
    def _detect_data_type(self, series: pd.Series) -> DataType:
        \"\"\"Intelligently detect the data type of a column\"\"\"
        # Remove null values for analysis
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return DataType.CATEGORICAL
        
        # Check for datetime
        if self._is_datetime_column(non_null_series):
            return DataType.DATETIME
        
        # Check for numeric types
        if pd.api.types.is_numeric_dtype(series):
            # Check if it's actually binary (0/1, True/False)
            unique_vals = set(non_null_series.unique())
            if unique_vals.issubset({0, 1, True, False, '0', '1', 'True', 'False', 'true', 'false'}):
                return DataType.BINARY
            return DataType.NUMERIC
        
        # Check for text/string data
        if pd.api.types.is_object_dtype(series):
            # Check if binary categorical
            unique_vals = set(non_null_series.astype(str).str.lower().unique())
            if unique_vals.issubset({'0', '1', 'true', 'false', 'yes', 'no', 'y', 'n'}):
                return DataType.BINARY
            
            # Check average text length to distinguish text from categorical
            avg_length = non_null_series.astype(str).str.len().mean()
            if avg_length > self.text_length_threshold:
                return DataType.TEXT
            else:
                return DataType.CATEGORICAL
        
        return DataType.CATEGORICAL
    
    def _is_datetime_column(self, series: pd.Series) -> bool:
        \"\"\"Check if a column contains datetime data\"\"\"
        try:
            # Try to parse a sample of the data
            sample = series.head(min(100, len(series)))
            pd.to_datetime(sample, errors='raise')
            return True
        except:
            # Try common date patterns
            date_patterns = [
                r'\\d{4}-\\d{2}-\\d{2}',  # YYYY-MM-DD
                r'\\d{2}/\\d{2}/\\d{4}',  # MM/DD/YYYY
                r'\\d{2}-\\d{2}-\\d{4}',  # MM-DD-YYYY
            ]
            
            sample_str = series.astype(str).head(10)
            for pattern in date_patterns:
                if sample_str.str.match(pattern).any():
                    return True
            
            return False
    
    def _calculate_column_statistics(self, series: pd.Series, data_type: DataType) -> Dict[str, Any]:
        \"\"\"Calculate type-specific statistics for a column\"\"\"
        stats = {}
        non_null_series = series.dropna()
        
        if data_type == DataType.NUMERIC:
            stats.update({
                'mean': float(non_null_series.mean()) if len(non_null_series) > 0 else None,
                'median': float(non_null_series.median()) if len(non_null_series) > 0 else None,
                'std': float(non_null_series.std()) if len(non_null_series) > 0 else None,
                'min': float(non_null_series.min()) if len(non_null_series) > 0 else None,
                'max': float(non_null_series.max()) if len(non_null_series) > 0 else None,
                'skewness': float(non_null_series.skew()) if len(non_null_series) > 0 else None,
                'kurtosis': float(non_null_series.kurtosis()) if len(non_null_series) > 0 else None
            })
        
        elif data_type in [DataType.CATEGORICAL, DataType.BINARY]:
            value_counts = non_null_series.value_counts()
            stats.update({
                'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else None,
                'cardinality_ratio': non_null_series.nunique() / len(non_null_series) if len(non_null_series) > 0 else None,
                'top_categories': value_counts.head(5).to_dict()
            })
        
        elif data_type == DataType.TEXT:
            text_lengths = non_null_series.astype(str).str.len()
            stats.update({
                'avg_length': float(text_lengths.mean()) if len(text_lengths) > 0 else None,
                'min_length': int(text_lengths.min()) if len(text_lengths) > 0 else None,
                'max_length': int(text_lengths.max()) if len(text_lengths) > 0 else None,
                'contains_urls': bool(non_null_series.astype(str).str.contains(r'http[s]?://').any()),
                'contains_emails': bool(non_null_series.astype(str).str.contains(r'\\S+@\\S+').any())
            })
        
        return stats
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        \"\"\"Calculate overall data quality score (0-100)\"\"\"
        scores = []
        
        # Completeness score (based on missing values)
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        completeness_score = (1 - missing_ratio) * 100
        scores.append(completeness_score)
        
        # Consistency score (based on data types)
        consistency_score = 100  # Start with perfect score
        for col in df.columns:
            col_data = df[col]
            if pd.api.types.is_object_dtype(col_data):
                # Check for mixed types in object columns
                sample = col_data.dropna().head(100)
                if len(sample) > 0:
                    types = set(type(x).__name__ for x in sample)
                    if len(types) > 1:
                        consistency_score -= 10
        scores.append(max(0, consistency_score))
        
        # Uniqueness score (check for duplicates)
        duplicate_ratio = df.duplicated().sum() / len(df)
        uniqueness_score = (1 - duplicate_ratio) * 100
        scores.append(uniqueness_score)
        
        # Size adequacy score
        size_score = min(100, (len(df) / self.min_samples_for_ml) * 100)
        scores.append(size_score)
        
        return round(np.mean(scores), 1)
    
    def _recommend_problem_types(self, df: pd.DataFrame, column_info: List[ColumnInfo]) -> List[ProblemType]:
        \"\"\"Recommend suitable ML problem types based on dataset characteristics\"\"\"
        recommendations = []
        
        # Count column types
        numeric_cols = [col for col in column_info if col.data_type == DataType.NUMERIC]
        categorical_cols = [col for col in column_info if col.data_type == DataType.CATEGORICAL]
        binary_cols = [col for col in column_info if col.data_type == DataType.BINARY]
        text_cols = [col for col in column_info if col.data_type == DataType.TEXT]
        datetime_cols = [col for col in column_info if col.data_type == DataType.DATETIME]
        
        # Binary classification indicators
        if binary_cols:
            recommendations.append(ProblemType.CLASSIFICATION)
        
        # Multi-class classification indicators
        suitable_targets = [col for col in categorical_cols 
                          if col.unique_count <= 20 and col.unique_count >= 2]
        if suitable_targets:
            recommendations.append(ProblemType.CLASSIFICATION)
        
        # Regression indicators
        if numeric_cols and len(numeric_cols) >= 2:
            recommendations.append(ProblemType.REGRESSION)
        
        # Time series indicators
        if datetime_cols and numeric_cols:
            recommendations.append(ProblemType.TIME_SERIES)
        
        # Clustering (unsupervised) - always possible with numeric data
        if len(numeric_cols) >= 2:
            recommendations.append(ProblemType.CLUSTERING)
        
        # NLP indicators
        if text_cols:
            recommendations.append(ProblemType.NLP)
        
        # Anomaly detection - possible with numeric data
        if len(numeric_cols) >= 3:
            recommendations.append(ProblemType.ANOMALY_DETECTION)
        
        return list(set(recommendations)) if recommendations else [ProblemType.CLASSIFICATION]
    
    def _generate_suggestions(self, df: pd.DataFrame, column_info: List[ColumnInfo]) -> List[str]:
        \"\"\"Generate actionable suggestions for the dataset\"\"\"
        suggestions = []
        
        # Data size suggestions
        if len(df) < self.min_samples_for_ml:
            suggestions.append(f\"Consider collecting more data. Current size ({len(df)} rows) may be insufficient for robust ML models. Aim for at least {self.min_samples_for_ml} samples.\")
        
        # Missing data suggestions
        high_missing_cols = [col for col in column_info if col.missing_percentage > 30]
        if high_missing_cols:
            col_names = [col.name for col in high_missing_cols]
            suggestions.append(f\"Columns with high missing values detected: {', '.join(col_names)}. Consider imputation or removal.\")
        
        # High cardinality suggestions
        high_card_cols = [col for col in column_info 
                         if col.data_type == DataType.CATEGORICAL and 
                         col.unique_count / len(df) > self.high_cardinality_threshold]
        if high_card_cols:
            col_names = [col.name for col in high_card_cols]
            suggestions.append(f\"High cardinality categorical columns detected: {', '.join(col_names)}. Consider feature engineering or encoding strategies.\")
        
        # Target column suggestions
        binary_cols = [col for col in column_info if col.data_type == DataType.BINARY]
        if binary_cols:
            suggestions.append(f\"Potential target columns for classification: {', '.join([col.name for col in binary_cols])}\")
        
        # Feature engineering suggestions
        text_cols = [col for col in column_info if col.data_type == DataType.TEXT]
        if text_cols:
            suggestions.append(\"Text columns detected. Consider NLP techniques like TF-IDF, word embeddings, or sentiment analysis.\")
        
        datetime_cols = [col for col in column_info if col.data_type == DataType.DATETIME]
        if datetime_cols:
            suggestions.append(\"Datetime columns detected. Consider extracting features like year, month, day, hour, or day of week.\")
        
        return suggestions
    
    def _generate_warnings(self, df: pd.DataFrame, column_info: List[ColumnInfo]) -> List[str]:
        \"\"\"Generate warnings about potential data issues\"\"\"
        warnings = []
        
        # Duplicate rows warning
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            warnings.append(f\"{duplicate_count} duplicate rows detected. This may affect model performance.\")
        
        # Extremely high missing values
        critical_missing_cols = [col for col in column_info if col.missing_percentage > 70]
        if critical_missing_cols:
            col_names = [col.name for col in critical_missing_cols]
            warnings.append(f\"Columns with critical missing values (>70%): {', '.join(col_names)}. Consider removing these columns.\")
        
        # Single value columns
        single_value_cols = [col for col in column_info if col.unique_count <= 1]
        if single_value_cols:
            col_names = [col.name for col in single_value_cols]
            warnings.append(f\"Columns with single unique value detected: {', '.join(col_names)}. These provide no predictive value.\")
        
        # Memory usage warning
        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        if memory_usage_mb > 500:  # 500MB threshold
            warnings.append(f\"Large dataset detected ({memory_usage_mb:.1f}MB). Consider data sampling or optimization for faster processing.\")
        
        # Imbalanced data warning for potential classification
        binary_cols = [col for col in column_info if col.data_type == DataType.BINARY]
        for col in binary_cols:
            if col.statistics and 'top_categories' in col.statistics:
                values = list(col.statistics['top_categories'].values())
                if len(values) >= 2:
                    ratio = min(values) / max(values)
                    if ratio < 0.1:  # 10:1 ratio threshold
                        warnings.append(f\"Severe class imbalance detected in '{col.name}'. Consider balancing techniques.\")
        
        return warnings


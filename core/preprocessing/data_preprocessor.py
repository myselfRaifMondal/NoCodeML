"""
Advanced Data Preprocessing and Cleaning Module for NoCodeML

This module provides comprehensive data preprocessing, cleaning, and iterative
improvement capabilities to ensure high-quality datasets for ML model training.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy import stats
import warnings
from datetime import datetime
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from backend.models.schemas import DataType, ProblemType

# Import comprehensive error handler
sys.path.append(str(Path(__file__).parent.parent / 'utils'))
try:
    from comprehensive_error_handler import ComprehensiveErrorHandler
except ImportError:
    print("Warning: Could not import ComprehensiveErrorHandler")
    ComprehensiveErrorHandler = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIssue:
    """Represents a data quality issue that needs to be addressed"""
    
    def __init__(self, issue_type: str, severity: str, description: str, 
                 columns: List[str], solution: str, auto_fixable: bool = True):
        self.issue_type = issue_type
        self.severity = severity  # 'critical', 'high', 'medium', 'low'
        self.description = description
        self.columns = columns
        self.solution = solution
        self.auto_fixable = auto_fixable
        self.timestamp = datetime.now()

class PreprocessingResult:
    """Contains results of preprocessing operations"""
    
    def __init__(self):
        self.original_shape: Tuple[int, int] = (0, 0)
        self.processed_shape: Tuple[int, int] = (0, 0)
        self.issues_found: List[DataIssue] = []
        self.issues_fixed: List[DataIssue] = []
        self.transformations_applied: List[str] = []
        self.quality_improvement: float = 0.0
        self.preprocessing_log: List[str] = []
        self.feature_names: List[str] = []
        self.encoding_mappings: Dict[str, Any] = {}
        self.scaler_params: Dict[str, Any] = {}

class DataPreprocessor:
    """Advanced data preprocessing and cleaning engine"""
    
    def __init__(self):
        self.missing_threshold = 0.5  # Remove columns with >50% missing values
        self.correlation_threshold = 0.95  # Remove highly correlated features
        self.variance_threshold = 0.01  # Remove low variance features
        self.outlier_threshold = 3.0  # Z-score threshold for outlier detection
        
        # Initialize error handler if available
        self.error_handler = ComprehensiveErrorHandler() if ComprehensiveErrorHandler else None
        
    def comprehensive_preprocessing(self, df: pd.DataFrame, 
                                  target_column: Optional[str] = None,
                                  problem_type: Optional[ProblemType] = None,
                                  auto_fix: bool = True) -> Tuple[pd.DataFrame, PreprocessingResult]:
        """
        Perform comprehensive data preprocessing with iterative improvement
        
        Args:
            df: Input DataFrame
            target_column: Name of target column (if supervised learning)
            problem_type: Type of ML problem
            auto_fix: Whether to automatically fix detected issues
            
        Returns:
            Tuple of (processed_df, preprocessing_result)
        """
        result = PreprocessingResult()
        result.original_shape = df.shape
        
        # Create a copy to work with
        processed_df = df.copy()
        
        # Phase 1: Data Quality Assessment
        logger.info("Phase 1: Assessing data quality...")
        issues = self._assess_data_quality(processed_df, target_column)
        result.issues_found = issues
        
        # Phase 2: Critical Issue Resolution
        logger.info("Phase 2: Resolving critical issues...")
        if auto_fix:
            # Use comprehensive error handler if available
            if self.error_handler and target_column:
                try:
                    logger.info("Using comprehensive error handler for auto-fix...")
                    processed_df, target_series = self.error_handler.auto_fix_dataset(
                        processed_df, processed_df[target_column], problem_type
                    )
                    # Update the target column in the dataframe
                    processed_df[target_column] = target_series
                    result.transformations_applied.append("Applied comprehensive auto-fix")
                except Exception as e:
                    logger.warning(f"Comprehensive auto-fix failed: {e}. Using fallback fixes.")
                    processed_df, fixed_issues = self._fix_critical_issues(processed_df, issues)
                    result.issues_fixed.extend(fixed_issues)
            else:
                processed_df, fixed_issues = self._fix_critical_issues(processed_df, issues)
                result.issues_fixed.extend(fixed_issues)
        
        # Phase 3: Missing Data Handling
        logger.info("Phase 3: Handling missing data...")
        processed_df = self._handle_missing_data(processed_df, result)
        
        # Phase 4: Outlier Detection and Treatment
        logger.info("Phase 4: Detecting and treating outliers...")
        processed_df = self._handle_outliers(processed_df, result)
        
        # Phase 5: Data Type Optimization
        logger.info("Phase 5: Optimizing data types...")
        processed_df = self._optimize_data_types(processed_df, result)
        
        # Phase 6: Feature Engineering
        logger.info("Phase 6: Engineering features...")
        processed_df = self._engineer_features(processed_df, result)
        
        # Phase 7: Encoding Categorical Variables
        logger.info("Phase 7: Encoding categorical variables...")
        processed_df = self._encode_categorical_variables(processed_df, target_column, result)
        
        # Phase 8: Feature Scaling
        logger.info("Phase 8: Scaling features...")
        processed_df = self._scale_features(processed_df, target_column, result)
        
        # Phase 9: Feature Selection
        logger.info("Phase 9: Selecting optimal features...")
        if target_column and target_column in processed_df.columns:
            processed_df = self._select_features(processed_df, target_column, problem_type, result)
        
        # Phase 10: Final Quality Assessment
        logger.info("Phase 10: Final quality assessment...")
        result.processed_shape = processed_df.shape
        result.quality_improvement = self._calculate_quality_improvement(df, processed_df)
        
        return processed_df, result
    
    def _intelligent_missing_data_handling(self, df: pd.DataFrame, target_column: Optional[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Intelligently handle missing data with automatic strategy selection"""
        transformations = []
        processed_df = df.copy()
        
        for col in df.columns:
            if not df[col].isnull().any():
                continue
                
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            
            # Auto-remove columns with > 70% missing data
            if missing_pct > 70:
                processed_df = processed_df.drop(columns=[col])
                transformations.append(f"❌ Removed column '{col}' (73% missing data)")
                continue
            
            # Choose imputation strategy based on data type and missing percentage
            if df[col].dtype in ['int64', 'float64']:
                if missing_pct < 5:
                    processed_df[col].fillna(processed_df[col].median(), inplace=True)
                    transformations.append(f"📊 Median imputation for '{col}' ({missing_pct:.1f}% missing)")
                elif missing_pct < 20:
                    try:
                        imputer = KNNImputer(n_neighbors=min(5, len(df)//10))
                        processed_df[col] = imputer.fit_transform(processed_df[[col]]).flatten()
                        transformations.append(f"🎯 KNN imputation for '{col}' ({missing_pct:.1f}% missing)")
                    except:
                        processed_df[col].fillna(processed_df[col].median(), inplace=True)
                        transformations.append(f"📊 Fallback median imputation for '{col}'")
                else:
                    processed_df[col].fillna(processed_df[col].mean(), inplace=True)
                    transformations.append(f"📈 Mean imputation for '{col}' ({missing_pct:.1f}% missing)")
            else:
                # Categorical data
                if missing_pct < 10:
                    mode_val = processed_df[col].mode()
                    if len(mode_val) > 0:
                        processed_df[col].fillna(mode_val[0], inplace=True)
                        transformations.append(f"🎯 Mode imputation for '{col}' ({missing_pct:.1f}% missing)")
                else:
                    processed_df[col].fillna('Unknown', inplace=True)
                    transformations.append(f"❓ Unknown category imputation for '{col}' ({missing_pct:.1f}% missing)")
        
        return processed_df, transformations
    
    def _intelligent_outlier_handling(self, df: pd.DataFrame, target_column: Optional[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Intelligently detect and handle outliers"""
        transformations = []
        processed_df = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if target_column in numeric_cols:
            numeric_cols = numeric_cols.drop(target_column)
        
        for col in numeric_cols:
            if processed_df[col].notna().sum() < 10:
                continue
                
            # Detect outliers using IQR method
            Q1 = processed_df[col].quantile(0.25)
            Q3 = processed_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:
                continue
                
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (processed_df[col] < lower_bound) | (processed_df[col] > upper_bound)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                outlier_pct = (outlier_count / len(processed_df)) * 100
                
                if outlier_pct > 10:
                    # Too many outliers, use capping
                    processed_df[col] = processed_df[col].clip(lower_bound, upper_bound)
                    transformations.append(f"🔒 Capped outliers in '{col}' ({outlier_count} values, {outlier_pct:.1f}%)")
                elif outlier_pct > 5:
                    # Moderate outliers, use winsorization
                    processed_df[col] = processed_df[col].clip(
                        processed_df[col].quantile(0.05),
                        processed_df[col].quantile(0.95)
                    )
                    transformations.append(f"🎯 Winsorized outliers in '{col}' ({outlier_count} values, {outlier_pct:.1f}%)")
                else:
                    # Few outliers, keep them but note
                    transformations.append(f"📊 Detected {outlier_count} outliers in '{col}' (kept for analysis)")
        
        return processed_df, transformations
    
    def _intelligent_type_optimization(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Optimize data types for memory efficiency and accuracy"""
        transformations = []
        processed_df = df.copy()
        
        # Optimize numeric types
        for col in processed_df.select_dtypes(include=[np.number]).columns:
            original_dtype = processed_df[col].dtype
            
            # Check if integer column can be downcasted
            if 'int' in str(original_dtype):
                min_val = processed_df[col].min()
                max_val = processed_df[col].max()
                
                if min_val >= 0:  # Unsigned integer
                    if max_val <= 255:
                        processed_df[col] = processed_df[col].astype('uint8')
                        transformations.append(f"💾 Optimized '{col}': {original_dtype} → uint8")
                    elif max_val <= 65535:
                        processed_df[col] = processed_df[col].astype('uint16')
                        transformations.append(f"💾 Optimized '{col}': {original_dtype} → uint16")
                else:  # Signed integer
                    if min_val >= -128 and max_val <= 127:
                        processed_df[col] = processed_df[col].astype('int8')
                        transformations.append(f"💾 Optimized '{col}': {original_dtype} → int8")
                    elif min_val >= -32768 and max_val <= 32767:
                        processed_df[col] = processed_df[col].astype('int16')
                        transformations.append(f"💾 Optimized '{col}': {original_dtype} → int16")
            
            # Check if float can be downcasted
            elif 'float' in str(original_dtype) and original_dtype != 'float32':
                try:
                    float32_series = processed_df[col].astype('float32')
                    if np.allclose(processed_df[col], float32_series, equal_nan=True):
                        processed_df[col] = float32_series
                        transformations.append(f"💾 Optimized '{col}': {original_dtype} → float32")
                except:
                    pass
        
        # Optimize categorical columns
        for col in processed_df.select_dtypes(include=['object']).columns:
            unique_ratio = processed_df[col].nunique() / len(processed_df)
            
            # Convert low-cardinality string columns to category
            if unique_ratio < 0.5 and processed_df[col].nunique() < 1000:
                processed_df[col] = processed_df[col].astype('category')
                transformations.append(f"🏷️ Converted '{col}' to category ({processed_df[col].nunique()} unique values)")
        
        return processed_df, transformations
    
    def _intelligent_feature_engineering(self, df: pd.DataFrame, problem_type: Optional[ProblemType]) -> Tuple[pd.DataFrame, List[str]]:
        """Perform intelligent feature engineering"""
        transformations = []
        processed_df = df.copy()
        
        # Date/time feature engineering
        for col in processed_df.columns:
            if processed_df[col].dtype == 'object':
                # Try to parse as datetime
                try:
                    datetime_series = pd.to_datetime(processed_df[col], errors='coerce')
                    if datetime_series.notna().sum() > len(processed_df) * 0.8:  # >80% valid dates
                        processed_df[col] = datetime_series
                        
                        # Extract useful date features
                        processed_df[f'{col}_year'] = datetime_series.dt.year
                        processed_df[f'{col}_month'] = datetime_series.dt.month
                        processed_df[f'{col}_day'] = datetime_series.dt.day
                        processed_df[f'{col}_dayofweek'] = datetime_series.dt.dayofweek
                        
                        # Drop original datetime column
                        processed_df = processed_df.drop(columns=[col])
                        
                        transformations.append(f"📅 Converted '{col}' to datetime and extracted features")
                except:
                    pass
        
        # Numeric feature engineering
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
        
        # Create interaction features for small datasets
        if len(numeric_cols) > 1 and len(numeric_cols) < 10 and len(processed_df) < 10000:
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:i+3]:  # Limit to prevent explosion
                    try:
                        interaction_col = f'{col1}_x_{col2}'
                        processed_df[interaction_col] = processed_df[col1] * processed_df[col2]
                        transformations.append(f"✖️ Created interaction feature: {interaction_col}")
                    except:
                        pass
        
        return processed_df, transformations
    
    def _intelligent_categorical_encoding(self, df: pd.DataFrame, target_column: Optional[str], 
                                        problem_type: Optional[ProblemType]) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
        """Intelligently encode categorical variables"""
        transformations = []
        encoding_info = {}
        processed_df = df.copy()
        
        categorical_cols = processed_df.select_dtypes(include=['object', 'category']).columns
        if target_column in categorical_cols:
            categorical_cols = categorical_cols.drop(target_column)
        
        for col in categorical_cols:
            unique_count = processed_df[col].nunique()
            
            if unique_count == 1:
                # Single value column, remove it
                processed_df = processed_df.drop(columns=[col])
                transformations.append(f"❌ Removed single-value column '{col}'")
                continue
            
            if unique_count == 2:
                # Binary encoding
                le = LabelEncoder()
                processed_df[col] = le.fit_transform(processed_df[col].astype(str))
                encoding_info[col] = {'type': 'label', 'classes': le.classes_.tolist()}
                transformations.append(f"🏷️ Label encoded binary column '{col}'")
            
            elif unique_count <= 10:
                # One-hot encoding for low cardinality
                try:
                    dummies = pd.get_dummies(processed_df[col], prefix=col, dummy_na=False)
                    processed_df = processed_df.drop(columns=[col])
                    processed_df = pd.concat([processed_df, dummies], axis=1)
                    encoding_info[col] = {'type': 'onehot', 'columns': dummies.columns.tolist()}
                    transformations.append(f"🎆 One-hot encoded '{col}' ({unique_count} categories)")
                except:
                    # Fallback to label encoding
                    le = LabelEncoder()
                    processed_df[col] = le.fit_transform(processed_df[col].astype(str))
                    encoding_info[col] = {'type': 'label_fallback', 'classes': le.classes_.tolist()}
                    transformations.append(f"🏷️ Fallback label encoded '{col}'")
            
            else:
                # High cardinality - use frequency encoding or target encoding
                if unique_count > 100:
                    # Frequency encoding
                    freq_map = processed_df[col].value_counts().to_dict()
                    processed_df[f'{col}_frequency'] = processed_df[col].map(freq_map)
                    processed_df = processed_df.drop(columns=[col])
                    encoding_info[col] = {'type': 'frequency', 'map': freq_map}
                    transformations.append(f"📊 Frequency encoded high-cardinality column '{col}' ({unique_count} categories)")
                else:
                    # Label encoding for medium cardinality
                    le = LabelEncoder()
                    processed_df[col] = le.fit_transform(processed_df[col].astype(str))
                    encoding_info[col] = {'type': 'label', 'classes': le.classes_.tolist()}
                    transformations.append(f"🏷️ Label encoded '{col}' ({unique_count} categories)")
        
        return processed_df, encoding_info, transformations
    
    def _intelligent_feature_scaling(self, df: pd.DataFrame, target_column: Optional[str]) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
        """Intelligently scale features based on their characteristics"""
        transformations = []
        scaling_info = {}
        processed_df = df.copy()
        
        # Get numeric columns (excluding target if specified)
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
        if target_column in numeric_cols:
            numeric_cols = numeric_cols.drop(target_column)
        
        if len(numeric_cols) == 0:
            return processed_df, scaling_info, transformations
        
        # Analyze feature distributions to choose appropriate scaling
        for col in numeric_cols:
            if processed_df[col].std() == 0:
                continue  # Skip zero-variance features
            
            # Check if the feature has outliers
            Q1 = processed_df[col].quantile(0.25)
            Q3 = processed_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            has_outliers = False
            if IQR > 0:
                outliers = ((processed_df[col] < (Q1 - 1.5 * IQR)) | 
                           (processed_df[col] > (Q3 + 1.5 * IQR))).sum()
                has_outliers = outliers > len(processed_df) * 0.05
        
        # Apply appropriate scaling method
        try:
            if has_outliers:
                # Use robust scaling for features with outliers
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                scaled_data = scaler.fit_transform(processed_df[numeric_cols])
                scaling_method = "robust"
            else:
                # Use standard scaling for well-behaved features
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(processed_df[numeric_cols])
                scaling_method = "standard"
            
            # Update the dataframe
            for i, col in enumerate(numeric_cols):
                processed_df[col] = scaled_data[:, i]
            
            scaling_info = {
                'method': scaling_method,
                'columns': numeric_cols.tolist(),
                'scaler_params': {
                    'mean_': getattr(scaler, 'center_', getattr(scaler, 'mean_', None)),
                    'scale_': getattr(scaler, 'scale_', None)
                }
            }
            
            transformations.append(f"📀 Applied {scaling_method} scaling to {len(numeric_cols)} numeric features")
            
        except Exception as e:
            transformations.append(f"⚠️ Scaling failed: {str(e)}")
        
        return processed_df, scaling_info, transformations
    
    def _intelligent_feature_selection(self, df: pd.DataFrame, target_column: str, 
                                     problem_type: Optional[ProblemType]) -> Tuple[pd.DataFrame, List[str]]:
        """Intelligently select features based on their importance"""
        transformations = []
        processed_df = df.copy()
        
        if target_column not in processed_df.columns:
            transformations.append(f"⚠️ Target column '{target_column}' not found, skipping feature selection")
            return processed_df, transformations
        
        # Get feature columns
        feature_cols = [col for col in processed_df.columns if col != target_column]
        
        if len(feature_cols) < 2:
            transformations.append("📊 Too few features for selection")
            return processed_df, transformations
        
        try:
            X = processed_df[feature_cols]
            y = processed_df[target_column]
            
            # Remove features with zero variance
            zero_var_cols = [col for col in feature_cols if X[col].var() == 0]
            if zero_var_cols:
                processed_df = processed_df.drop(columns=zero_var_cols)
                transformations.append(f"❌ Removed {len(zero_var_cols)} zero-variance features")
                feature_cols = [col for col in feature_cols if col not in zero_var_cols]
                X = processed_df[feature_cols]
            
            # Remove highly correlated features
            if len(feature_cols) > 1:
                corr_matrix = X.corr().abs()
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                high_corr_pairs = [(col, row) for col in upper_tri.columns 
                                 for row in upper_tri.index if upper_tri.loc[row, col] > 0.95]
                
                cols_to_drop = set()
                for col1, col2 in high_corr_pairs:
                    cols_to_drop.add(col2)  # Drop the second column in each pair
                
                if cols_to_drop:
                    processed_df = processed_df.drop(columns=list(cols_to_drop))
                    transformations.append(f"🔗 Removed {len(cols_to_drop)} highly correlated features")
                    feature_cols = [col for col in feature_cols if col not in cols_to_drop]
            
            # If too many features remain, select top features
            if len(feature_cols) > 50:
                try:
                    if problem_type == ProblemType.CLASSIFICATION:
                        selector = SelectKBest(score_func=f_classif, k=min(50, len(feature_cols)))
                    else:
                        selector = SelectKBest(score_func=f_regression, k=min(50, len(feature_cols)))
                    
                    X_selected = selector.fit_transform(X[feature_cols], y)
                    selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
                    
                    # Keep only selected features
                    all_cols = selected_features + [target_column]
                    processed_df = processed_df[all_cols]
                    
                    transformations.append(f"🎯 Selected top {len(selected_features)} features using statistical tests")
                    
                except Exception as e:
                    transformations.append(f"⚠️ Feature selection failed: {str(e)}")
        
        except Exception as e:
            transformations.append(f"⚠️ Feature selection failed: {str(e)}")
        
        return processed_df, transformations
    
    def _final_quality_check_and_fix(self, df: pd.DataFrame, target_column: Optional[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Final quality check and emergency fixes"""
        transformations = []
        processed_df = df.copy()
        
        # Check for any remaining issues and fix them
        
        # 1. Remove any columns that are entirely NaN
        nan_cols = [col for col in processed_df.columns if processed_df[col].isnull().all()]
        if nan_cols:
            processed_df = processed_df.drop(columns=nan_cols)
            transformations.append(f"❌ Removed {len(nan_cols)} completely empty columns")
        
        # 2. Ensure no infinite values
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.isinf(processed_df[col]).any():
                processed_df[col] = processed_df[col].replace([np.inf, -np.inf], np.nan)
                processed_df[col].fillna(processed_df[col].median(), inplace=True)
                transformations.append(f"♾️ Fixed infinite values in '{col}'")
        
        # 3. Ensure minimum dataset size
        if len(processed_df) < 10:
            transformations.append("⚠️ Warning: Dataset has fewer than 10 rows after preprocessing")
        
        # 4. Ensure at least one feature column
        feature_cols = [col for col in processed_df.columns if col != target_column]
        if len(feature_cols) == 0:
            transformations.append("❌ Critical: No feature columns remaining after preprocessing")
        
        transformations.append(f"✅ Final dataset ready: {processed_df.shape[0]} rows, {processed_df.shape[1]} columns")
        
        return processed_df, transformations
    
    def _emergency_fallback_preprocessing(self, df: pd.DataFrame, target_column: Optional[str]) -> pd.DataFrame:
        """Emergency fallback preprocessing when main pipeline fails"""
        processed_df = df.copy()
        
        # Basic cleaning only
        # 1. Drop columns with >90% missing data
        for col in processed_df.columns:
            if processed_df[col].isnull().sum() / len(processed_df) > 0.9:
                processed_df = processed_df.drop(columns=[col])
        
        # 2. Simple imputation
        for col in processed_df.columns:
            if processed_df[col].isnull().any():
                if processed_df[col].dtype in ['int64', 'float64']:
                    processed_df[col].fillna(processed_df[col].median(), inplace=True)
                else:
                    processed_df[col].fillna('Unknown', inplace=True)
        
        # 3. Basic encoding
        for col in processed_df.select_dtypes(include=['object']).columns:
            if col != target_column:
                try:
                    le = LabelEncoder()
                    processed_df[col] = le.fit_transform(processed_df[col].astype(str))
                except:
                    processed_df = processed_df.drop(columns=[col])
        
        return processed_df
    
    def _generate_recommendations(self, original_df: pd.DataFrame, processed_df: pd.DataFrame, 
                                result: PreprocessingResult) -> List[str]:
        """Generate intelligent recommendations based on preprocessing results"""
        recommendations = []
        
        # Data quality recommendations
        original_missing = (original_df.isnull().sum().sum() / (original_df.shape[0] * original_df.shape[1])) * 100
        if original_missing > 20:
            recommendations.append("Consider collecting more complete data to improve model performance")
        
        # Feature recommendations
        feature_reduction = (original_df.shape[1] - processed_df.shape[1]) / original_df.shape[1] * 100
        if feature_reduction > 50:
            recommendations.append("Significant feature reduction occurred. Consider domain expertise for feature engineering")
        
        # Sample size recommendations
        if len(processed_df) < 100:
            recommendations.append("Small dataset size. Consider collecting more data or using regularization techniques")
        elif len(processed_df) < 1000:
            recommendations.append("Moderate dataset size. Ensemble methods or cross-validation recommended")
        
        # Model recommendations based on data characteristics
        numeric_ratio = len(processed_df.select_dtypes(include=[np.number]).columns) / processed_df.shape[1]
        if numeric_ratio < 0.3:
            recommendations.append("Dataset is mostly categorical. Consider tree-based models or neural networks")
        elif numeric_ratio > 0.8:
            recommendations.append("Dataset is mostly numeric. Linear models or SVM might work well")
        
        if not recommendations:
            recommendations.append("Dataset preprocessing completed successfully. Ready for model training!")
        
        return recommendations
    
    def _assess_data_quality(self, df: pd.DataFrame, target_column: Optional[str]) -> List[DataIssue]:
        """Assess data quality and identify issues"""
        issues = []
        
        # Check for missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        for col in missing_cols:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            if missing_pct > 70:
                severity = 'critical'
            elif missing_pct > 30:
                severity = 'high'
            elif missing_pct > 10:
                severity = 'medium'
            else:
                severity = 'low'
                
            issues.append(DataIssue(
                issue_type='missing_data',
                severity=severity,
                description=f"Column '{col}' has {missing_pct:.1f}% missing values",
                columns=[col],
                solution='Imputation or column removal'
            ))
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            issues.append(DataIssue(
                issue_type='duplicates',
                severity='medium',
                description=f"Found {duplicate_count} duplicate rows",
                columns=[],
                solution='Remove duplicate rows'
            ))
        
        # Check for single-value columns
        single_value_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if single_value_cols:
            issues.append(DataIssue(
                issue_type='no_variance',
                severity='high',
                description=f"Columns with single unique value: {single_value_cols}",
                columns=single_value_cols,
                solution='Remove zero-variance columns'
            ))
        
        # Check for high cardinality categorical columns
        for col in df.select_dtypes(include=['object']).columns:
            cardinality_ratio = df[col].nunique() / len(df)
            if cardinality_ratio > 0.5 and df[col].nunique() > 100:
                issues.append(DataIssue(
                    issue_type='high_cardinality',
                    severity='medium',
                    description=f"Column '{col}' has very high cardinality ({df[col].nunique()} unique values)",
                    columns=[col],
                    solution='Feature engineering or dimensionality reduction'
                ))
        
        # Check for outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != target_column:  # Don't flag target column outliers
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outlier_count = (z_scores > self.outlier_threshold).sum()
                if outlier_count > 0:
                    outlier_pct = (outlier_count / len(df)) * 100
                    severity = 'high' if outlier_pct > 5 else 'medium'
                    issues.append(DataIssue(
                        issue_type='outliers',
                        severity=severity,
                        description=f"Column '{col}' has {outlier_count} outliers ({outlier_pct:.1f}%)",
                        columns=[col],
                        solution='Outlier treatment (capping, transformation, or removal)'
                    ))
        
        # Check for highly correlated features
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > self.correlation_threshold:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            if high_corr_pairs:
                issues.append(DataIssue(
                    issue_type='high_correlation',
                    severity='medium',
                    description=f"Found {len(high_corr_pairs)} highly correlated feature pairs",
                    columns=[pair[1] for pair in high_corr_pairs],  # Remove second feature in each pair
                    solution='Remove redundant features'
                ))
        
        return issues
    
    def _fix_critical_issues(self, df: pd.DataFrame, issues: List[DataIssue]) -> Tuple[pd.DataFrame, List[DataIssue]]:
        """Fix critical data quality issues"""
        fixed_issues = []
        
        for issue in issues:
            if issue.severity == 'critical' and issue.auto_fixable:
                if issue.issue_type == 'missing_data':
                    # Remove columns with >70% missing values
                    for col in issue.columns:
                        if col in df.columns:
                            df = df.drop(columns=[col])
                            fixed_issues.append(issue)
                
                elif issue.issue_type == 'no_variance':
                    # Remove zero-variance columns
                    df = df.drop(columns=issue.columns)
                    fixed_issues.append(issue)
        
        # Remove duplicate rows
        duplicate_issues = [i for i in issues if i.issue_type == 'duplicates']
        if duplicate_issues:
            initial_shape = df.shape
            df = df.drop_duplicates()
            if df.shape[0] < initial_shape[0]:
                fixed_issues.extend(duplicate_issues)
        
        return df, fixed_issues
    
    def _handle_missing_data(self, df: pd.DataFrame, result: PreprocessingResult) -> pd.DataFrame:
        """Handle missing data with intelligent imputation strategies"""
        
        for col in df.columns:
            if df[col].isnull().any():
                missing_pct = (df[col].isnull().sum() / len(df)) * 100
                
                # Skip columns with too much missing data (should be handled in critical fixes)
                if missing_pct > 50:
                    continue
                
                if df[col].dtype in ['int64', 'float64']:
                    # Numeric columns
                    if missing_pct < 10:
                        # Use median for small amounts of missing data
                        df[col].fillna(df[col].median(), inplace=True)
                        result.transformations_applied.append(f"Median imputation for '{col}'")
                    else:
                        # Use KNN imputation for larger amounts
                        imputer = KNNImputer(n_neighbors=5)
                        df[[col]] = imputer.fit_transform(df[[col]])
                        result.transformations_applied.append(f"KNN imputation for '{col}'")
                
                elif df[col].dtype == 'object':
                    # Categorical columns
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col].fillna(mode_val[0], inplace=True)
                        result.transformations_applied.append(f"Mode imputation for '{col}'")
                    else:
                        df[col].fillna('Unknown', inplace=True)
                        result.transformations_applied.append(f"Unknown category imputation for '{col}'")
                
                elif df[col].dtype == 'bool':
                    # Boolean columns
                    df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else False, inplace=True)
                    result.transformations_applied.append(f"Mode imputation for boolean '{col}'")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, result: PreprocessingResult) -> pd.DataFrame:
        """Detect and handle outliers"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Calculate IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                outlier_pct = (outlier_count / len(df)) * 100
                
                if outlier_pct < 1:
                    # Remove outliers if very few
                    df = df[~outliers]
                    result.transformations_applied.append(f"Removed {outlier_count} outliers from '{col}'")
                elif outlier_pct < 5:
                    # Cap outliers
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    result.transformations_applied.append(f"Capped outliers in '{col}' to IQR bounds")
                else:
                    # Log transformation for heavily skewed data
                    if df[col].min() > 0:  # Only if all values are positive
                        skewness = df[col].skew()
                        if abs(skewness) > 2:
                            df[col] = np.log1p(df[col])
                            result.transformations_applied.append(f"Log transformation applied to '{col}' for skewness reduction")
        
        return df
    
    def _optimize_data_types(self, df: pd.DataFrame, result: PreprocessingResult) -> pd.DataFrame:
        """Optimize data types for memory efficiency and performance"""
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to more efficient types
                unique_vals = df[col].nunique()
                total_vals = len(df)
                
                # Convert to category if low cardinality
                if unique_vals / total_vals < 0.5:
                    df[col] = df[col].astype('category')
                    result.transformations_applied.append(f"Converted '{col}' to category type")
            
            elif df[col].dtype in ['int64', 'float64']:
                # Downcast numeric types
                if df[col].dtype == 'int64':
                    if df[col].min() >= 0:
                        if df[col].max() < 255:
                            df[col] = df[col].astype('uint8')
                        elif df[col].max() < 65535:
                            df[col] = df[col].astype('uint16')
                        elif df[col].max() < 4294967295:
                            df[col] = df[col].astype('uint32')
                    else:
                        if df[col].min() >= -128 and df[col].max() <= 127:
                            df[col] = df[col].astype('int8')
                        elif df[col].min() >= -32768 and df[col].max() <= 32767:
                            df[col] = df[col].astype('int16')
                        elif df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                            df[col] = df[col].astype('int32')
                
                elif df[col].dtype == 'float64':
                    # Try to convert to float32 if precision loss is minimal
                    float32_version = df[col].astype('float32')
                    if np.allclose(df[col].dropna(), float32_version.dropna(), rtol=1e-6):
                        df[col] = float32_version
                        result.transformations_applied.append(f"Downcasted '{col}' to float32")
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame, result: PreprocessingResult) -> pd.DataFrame:
        """Engineer new features from existing ones"""
        
        # Date/time feature engineering
        date_cols = df.select_dtypes(include=['datetime64']).columns
        for col in date_cols:
            if col in df.columns:
                # Extract date components
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                df[f'{col}_hour'] = df[col].dt.hour if hasattr(df[col].dt, 'hour') else 0
                
                result.transformations_applied.append(f"Extracted date features from '{col}'")
        
        # Polynomial features for numeric columns (limited to prevent explosion)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2 and len(numeric_cols) <= 5:  # Only for small number of numeric columns
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    # Create interaction term
                    interaction_name = f'{col1}_x_{col2}'
                    df[interaction_name] = df[col1] * df[col2]
                    result.transformations_applied.append(f"Created interaction feature '{interaction_name}'")
        
        # Binning for continuous variables with high cardinality
        for col in numeric_cols:
            if df[col].nunique() > 100:  # High cardinality numeric
                # Create binned version
                try:
                    df[f'{col}_binned'] = pd.qcut(df[col], q=10, labels=False, duplicates='drop')
                    result.transformations_applied.append(f"Created binned version of '{col}'")
                except ValueError:
                    # If qcut fails, use cut instead
                    try:
                        df[f'{col}_binned'] = pd.cut(df[col], bins=10, labels=False)
                        result.transformations_applied.append(f"Created binned version of '{col}' using equal-width bins")
                    except:
                        pass
        
        return df
    
    def _encode_categorical_variables(self, df: pd.DataFrame, target_column: Optional[str], 
                                    result: PreprocessingResult) -> pd.DataFrame:
        """Encode categorical variables using appropriate strategies"""
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_cols if col != target_column]
        
        for col in categorical_cols:
            unique_vals = df[col].nunique()
            
            if unique_vals == 2:
                # Binary encoding
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                result.encoding_mappings[col] = {
                    'type': 'label',
                    'classes': le.classes_.tolist()
                }
                result.transformations_applied.append(f"Label encoded binary column '{col}'")
            
            elif unique_vals <= 10:
                # One-hot encoding for low cardinality
                dummies = pd.get_dummies(df[col], prefix=col)
                df = df.drop(columns=[col])
                df = pd.concat([df, dummies], axis=1)
                result.encoding_mappings[col] = {
                    'type': 'onehot',
                    'columns': dummies.columns.tolist()
                }
                result.transformations_applied.append(f"One-hot encoded column '{col}'")
            
            else:
                # Target encoding or frequency encoding for high cardinality
                if target_column and target_column in df.columns:
                    # Target encoding (mean encoding)
                    target_mean = df.groupby(col)[target_column].mean()
                    df[f'{col}_target_encoded'] = df[col].map(target_mean)
                    df = df.drop(columns=[col])
                    result.transformations_applied.append(f"Target encoded high cardinality column '{col}'")
                else:
                    # Frequency encoding
                    freq_encoding = df[col].value_counts(normalize=True)
                    df[f'{col}_freq_encoded'] = df[col].map(freq_encoding)
                    df = df.drop(columns=[col])
                    result.transformations_applied.append(f"Frequency encoded high cardinality column '{col}'")
        
        return df
    
    def _scale_features(self, df: pd.DataFrame, target_column: Optional[str], 
                       result: PreprocessingResult) -> pd.DataFrame:
        """Scale numerical features"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_column]
        
        if len(numeric_cols) > 0:
            # Use StandardScaler for most cases
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            
            result.scaler_params = {
                'type': 'standard',
                'mean': scaler.mean_.tolist(),
                'scale': scaler.scale_.tolist(),
                'columns': numeric_cols
            }
            result.transformations_applied.append(f"Standard scaled {len(numeric_cols)} numeric columns")
        
        return df
    
    def _select_features(self, df: pd.DataFrame, target_column: str, 
                        problem_type: Optional[ProblemType], result: PreprocessingResult) -> pd.DataFrame:
        """Select the most relevant features"""
        
        if target_column not in df.columns:
            return df
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Skip if too few features
        if X.shape[1] <= 5:
            return df
        
        try:
            # Determine the appropriate scoring function
            if problem_type == ProblemType.REGRESSION:
                score_func = f_regression
            else:
                score_func = f_classif
            
            # Select top k features (at least 10 or 50% of features, whichever is smaller)
            k = min(max(10, X.shape[1] // 2), X.shape[1])
            
            selector = SelectKBest(score_func=score_func, k=k)
            X_selected = selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_features = X.columns[selector.get_support()].tolist()
            selected_features.append(target_column)  # Always keep target
            
            df_selected = df[selected_features]
            result.feature_names = selected_features
            result.transformations_applied.append(f"Selected top {len(selected_features)-1} features using statistical tests")
            
            return df_selected
            
        except Exception as e:
            logger.warning(f"Feature selection failed: {str(e)}")
            return df
    
    def _calculate_quality_improvement(self, original_df: pd.DataFrame, processed_df: pd.DataFrame) -> float:
        """Calculate the quality improvement percentage"""
        
        # Original quality metrics
        orig_missing_pct = (original_df.isnull().sum().sum() / (original_df.shape[0] * original_df.shape[1])) * 100
        orig_duplicates = original_df.duplicated().sum()
        
        # Processed quality metrics
        proc_missing_pct = (processed_df.isnull().sum().sum() / (processed_df.shape[0] * processed_df.shape[1])) * 100
        proc_duplicates = processed_df.duplicated().sum()
        
        # Calculate improvement
        missing_improvement = max(0, orig_missing_pct - proc_missing_pct)
        duplicate_improvement = max(0, orig_duplicates - proc_duplicates)
        
        # Weighted improvement score
        total_improvement = (missing_improvement * 0.7) + (duplicate_improvement / max(1, original_df.shape[0]) * 100 * 0.3)
        
        return round(total_improvement, 2)
    
    def iterative_improvement(self, df: pd.DataFrame, target_column: Optional[str],
                            problem_type: Optional[ProblemType], max_iterations: int = 3) -> Tuple[pd.DataFrame, List[PreprocessingResult]]:
        """
        Perform iterative preprocessing improvements until quality stabilizes
        
        Args:
            df: Input DataFrame
            target_column: Target column name
            problem_type: ML problem type
            max_iterations: Maximum number of improvement iterations
            
        Returns:
            Tuple of (best_df, list_of_results)
        """
        results = []
        current_df = df.copy()
        best_df = current_df.copy()
        best_quality = 0
        
        for iteration in range(max_iterations):
            logger.info(f"Iteration {iteration + 1}/{max_iterations}")
            
            # Perform preprocessing
            processed_df, result = self.comprehensive_preprocessing(
                current_df, target_column, problem_type, auto_fix=True
            )
            
            results.append(result)
            
            # Check if this iteration improved quality
            if result.quality_improvement > best_quality:
                best_quality = result.quality_improvement
                best_df = processed_df.copy()
                current_df = processed_df.copy()
            else:
                # Quality didn't improve, stop iterating
                logger.info(f"Quality stabilized after {iteration + 1} iterations")
                break
        
        return best_df, results
    
    def generate_preprocessing_report(self, results: List[PreprocessingResult]) -> Dict[str, Any]:
        """Generate a comprehensive preprocessing report"""
        
        if not results:
            return {}
        
        final_result = results[-1]
        
        report = {
            'summary': {
                'iterations_performed': len(results),
                'original_shape': final_result.original_shape,
                'final_shape': final_result.processed_shape,
                'quality_improvement': final_result.quality_improvement,
                'total_transformations': len(final_result.transformations_applied)
            },
            'issues_analysis': {
                'total_issues_found': len(final_result.issues_found),
                'issues_by_severity': {},
                'issues_by_type': {},
                'auto_fixed_issues': len(final_result.issues_fixed)
            },
            'transformations': final_result.transformations_applied,
            'encoding_info': final_result.encoding_mappings,
            'scaling_info': final_result.scaler_params,
            'recommendations': []
        }
        
        # Analyze issues by severity and type
        for issue in final_result.issues_found:
            severity = issue.severity
            issue_type = issue.issue_type
            
            if severity not in report['issues_analysis']['issues_by_severity']:
                report['issues_analysis']['issues_by_severity'][severity] = 0
            report['issues_analysis']['issues_by_severity'][severity] += 1
            
            if issue_type not in report['issues_analysis']['issues_by_type']:
                report['issues_analysis']['issues_by_type'][issue_type] = 0
            report['issues_analysis']['issues_by_type'][issue_type] += 1
        
        # Generate recommendations
        if final_result.quality_improvement < 10:
            report['recommendations'].append("Consider collecting more high-quality data")
        
        if final_result.processed_shape[1] < 5:
            report['recommendations'].append("Dataset has few features - consider feature engineering")
        
        if final_result.processed_shape[0] < 1000:
            report['recommendations'].append("Small dataset detected - consider data augmentation techniques")
        
        return report

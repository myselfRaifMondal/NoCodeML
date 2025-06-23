"""
Helper methods for the Enhanced Data Preprocessor
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class PreprocessorHelpers:
    """Helper methods for enhanced data preprocessing"""
    
    @staticmethod
    def assess_correlations(df: pd.DataFrame, target_column: Optional[str]) -> List:
        """Assess feature correlations"""
        from enhanced_preprocessor import DataIssue, IssueType, Severity
        
        issues = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if target_column in numeric_cols:
            numeric_cols = numeric_cols.drop(target_column)
        
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if corr_val > 0.95:
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr_val
                        })
            
            if high_corr_pairs:
                redundant_features = [pair['feature2'] for pair in high_corr_pairs]
                issues.append(DataIssue(
                    issue_type=IssueType.HIGH_CORRELATION,
                    severity=Severity.MEDIUM,
                    description=f"Found {len(high_corr_pairs)} highly correlated feature pairs",
                    columns=redundant_features,
                    solution="Remove redundant features to avoid multicollinearity",
                    confidence_score=0.9,
                    impact_on_model="May cause model instability and overfitting",
                    suggested_action="Remove one feature from each highly correlated pair"
                ))
        
        return issues
    
    @staticmethod
    def assess_variance(df: pd.DataFrame) -> List:
        """Assess feature variance"""
        from enhanced_preprocessor import DataIssue, IssueType, Severity
        
        issues = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        low_variance_cols = []
        for col in numeric_cols:
            if df[col].var() < 0.01:  # Very low variance threshold
                low_variance_cols.append(col)
        
        if low_variance_cols:
            issues.append(DataIssue(
                issue_type=IssueType.LOW_VARIANCE,
                severity=Severity.HIGH,
                description=f"Columns with very low variance: {low_variance_cols}",
                columns=low_variance_cols,
                solution="Remove low-variance features",
                confidence_score=1.0,
                impact_on_model="Low-variance features provide little predictive power",
                suggested_action="Remove these features to reduce noise"
            ))
        
        return issues
    
    @staticmethod
    def assess_cardinality(df: pd.DataFrame) -> List:
        """Assess categorical feature cardinality"""
        from enhanced_preprocessor import DataIssue, IssueType, Severity
        
        issues = []
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            cardinality_ratio = df[col].nunique() / len(df)
            unique_count = df[col].nunique()
            
            if cardinality_ratio > 0.5 and unique_count > 100:
                severity = Severity.HIGH if cardinality_ratio > 0.8 else Severity.MEDIUM
                
                issues.append(DataIssue(
                    issue_type=IssueType.HIGH_CARDINALITY,
                    severity=severity,
                    description=f"Column '{col}' has very high cardinality ({unique_count} unique values)",
                    columns=[col],
                    solution="Feature engineering, dimensionality reduction, or target encoding",
                    confidence_score=0.8,
                    impact_on_model="High cardinality may cause overfitting and memory issues",
                    suggested_action="Apply target encoding or frequency encoding"
                ))
        
        return issues
    
    @staticmethod
    def assess_data_leakage(df: pd.DataFrame, target_column: Optional[str]) -> List:
        """Detect potential data leakage"""
        from enhanced_preprocessor import DataIssue, IssueType, Severity
        
        issues = []
        
        if not target_column or target_column not in df.columns:
            return issues
        
        # Check for perfect correlations with target
        if df[target_column].dtype in ['int64', 'float64']:
            for col in df.columns:
                if col != target_column and df[col].dtype in ['int64', 'float64']:
                    try:
                        correlation = abs(df[col].corr(df[target_column]))
                        if correlation > 0.99:
                            issues.append(DataIssue(
                                issue_type=IssueType.DATA_LEAKAGE,
                                severity=Severity.CRITICAL,
                                description=f"Column '{col}' has suspiciously high correlation ({correlation:.3f}) with target",
                                columns=[col],
                                solution="Remove feature to prevent data leakage",
                                confidence_score=0.9,
                                impact_on_model="Will cause overly optimistic performance metrics",
                                suggested_action="Remove this feature immediately"
                            ))
                    except:
                        continue
        
        # Check for features that perfectly predict target categories
        if df[target_column].dtype == 'object' or df[target_column].nunique() < 20:
            for col in df.select_dtypes(include=['object', 'category']).columns:
                if col != target_column:
                    # Check if each unique value in col maps to only one target value
                    mapping = df.groupby(col)[target_column].nunique()
                    if (mapping == 1).all() and len(mapping) == df[target_column].nunique():
                        issues.append(DataIssue(
                            issue_type=IssueType.DATA_LEAKAGE,
                            severity=Severity.CRITICAL,
                            description=f"Column '{col}' perfectly predicts the target variable",
                            columns=[col],
                            solution="Remove feature to prevent data leakage",
                            confidence_score=0.95,
                            impact_on_model="Will cause perfect but invalid predictions",
                            suggested_action="Remove this feature immediately"
                        ))
        
        return issues
    
    @staticmethod
    def assess_distribution_skewness(df: pd.DataFrame) -> List:
        """Assess distribution skewness"""
        from enhanced_preprocessor import DataIssue, IssueType, Severity
        
        issues = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        highly_skewed_cols = []
        for col in numeric_cols:
            if df[col].notna().sum() > 10:  # Need sufficient data
                skewness = abs(df[col].skew())
                if skewness > 2:  # Highly skewed
                    highly_skewed_cols.append({
                        'column': col,
                        'skewness': skewness
                    })
        
        if highly_skewed_cols:
            col_names = [item['column'] for item in highly_skewed_cols]
            issues.append(DataIssue(
                issue_type=IssueType.SKEWED_DISTRIBUTION,
                severity=Severity.MEDIUM,
                description=f"Highly skewed distributions detected in: {col_names}",
                columns=col_names,
                solution="Apply log transformation, Box-Cox, or other normalization techniques",
                confidence_score=0.8,
                impact_on_model="Skewed distributions may affect model performance",
                suggested_action="Apply appropriate transformations to normalize distributions"
            ))
        
        return issues
    
    @staticmethod
    def assess_type_consistency(df: pd.DataFrame) -> List:
        """Assess data type consistency"""
        from enhanced_preprocessor import DataIssue, IssueType, Severity
        
        issues = []
        
        # Check for mixed types in object columns
        for col in df.select_dtypes(include=['object']).columns:
            sample_values = df[col].dropna().head(100)
            type_counts = {}
            
            for val in sample_values:
                val_type = type(val).__name__
                type_counts[val_type] = type_counts.get(val_type, 0) + 1
            
            if len(type_counts) > 1:
                issues.append(DataIssue(
                    issue_type=IssueType.INCONSISTENT_TYPES,
                    severity=Severity.MEDIUM,
                    description=f"Column '{col}' contains mixed data types: {list(type_counts.keys())}",
                    columns=[col],
                    solution="Convert to consistent data type or separate into multiple columns",
                    confidence_score=0.7,
                    impact_on_model="Mixed types may cause encoding issues",
                    suggested_action="Standardize data types in this column"
                ))
        
        return issues
    
    @staticmethod
    def assess_temporal_issues(df: pd.DataFrame) -> List:
        """Assess temporal data issues"""
        from enhanced_preprocessor import DataIssue, IssueType, Severity
        
        issues = []
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        
        for col in datetime_cols:
            # Check for future dates (suspicious)
            future_dates = df[col] > pd.Timestamp.now()
            if future_dates.any():
                future_count = future_dates.sum()
                issues.append(DataIssue(
                    issue_type=IssueType.TEMPORAL_ISSUES,
                    severity=Severity.MEDIUM,
                    description=f"Column '{col}' contains {future_count} future dates",
                    columns=[col],
                    solution="Validate and correct future dates or exclude them",
                    confidence_score=0.8,
                    impact_on_model="Future dates may indicate data quality issues",
                    suggested_action="Review and handle future dates appropriately"
                ))
            
            # Check for unrealistic historical dates
            very_old_dates = df[col] < pd.Timestamp('1900-01-01')
            if very_old_dates.any():
                old_count = very_old_dates.sum()
                issues.append(DataIssue(
                    issue_type=IssueType.TEMPORAL_ISSUES,
                    severity=Severity.LOW,
                    description=f"Column '{col}' contains {old_count} very old dates (before 1900)",
                    columns=[col],
                    solution="Validate historical dates or treat as missing values",
                    confidence_score=0.6,
                    impact_on_model="Very old dates may be data entry errors",
                    suggested_action="Review and validate historical dates"
                ))
        
        return issues
    
    @staticmethod
    def suggest_outlier_treatment(outlier_pct: float) -> str:
        """Suggest outlier treatment strategy"""
        if outlier_pct > 15:
            return "High outlier percentage - consider data transformation or robust scaling"
        elif outlier_pct > 5:
            return "Moderate outliers - apply IQR capping or Winsorization"
        else:
            return "Few outliers - consider removal or capping to IQR bounds"
    
    @staticmethod
    def resolve_critical_issues(df: pd.DataFrame, issues: List, user_preferences: Optional[Dict] = None) -> Tuple[pd.DataFrame, List]:
        """Resolve critical data quality issues"""
        fixed_issues = []
        
        for issue in issues:
            if issue.severity.value == 'critical' and issue.auto_fixable:
                if issue.issue_type.value == 'missing_data':
                    # Remove columns with >80% missing values
                    for col in issue.columns:
                        if col in df.columns:
                            missing_pct = (df[col].isnull().sum() / len(df)) * 100
                            if missing_pct > 80:
                                df = df.drop(columns=[col])
                                fixed_issues.append(issue)
                
                elif issue.issue_type.value == 'low_variance':
                    # Remove zero-variance columns
                    df = df.drop(columns=issue.columns, errors='ignore')
                    fixed_issues.append(issue)
                
                elif issue.issue_type.value == 'data_leakage':
                    # Remove data leakage columns
                    df = df.drop(columns=issue.columns, errors='ignore')
                    fixed_issues.append(issue)
        
        # Remove duplicate rows
        duplicate_issues = [i for i in issues if i.issue_type.value == 'duplicates']
        if duplicate_issues:
            initial_shape = df.shape
            df = df.drop_duplicates()
            if df.shape[0] < initial_shape[0]:
                fixed_issues.extend(duplicate_issues)
        
        return df, fixed_issues
    
    @staticmethod
    def advanced_outlier_handling(df: pd.DataFrame, target_column: Optional[str], result) -> pd.DataFrame:
        """Advanced outlier detection and treatment"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if target_column in numeric_cols:
            numeric_cols = numeric_cols.drop(target_column)
        
        for col in numeric_cols:
            if df[col].notna().sum() < 10:
                continue
            
            # Calculate outlier bounds using IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                outlier_pct = (outlier_count / len(df)) * 100
                
                if outlier_pct < 1:
                    # Remove outliers if very few
                    df = df[~outliers]
                    result.transformations_applied.append(f"Removed {outlier_count} outliers from '{col}'")
                elif outlier_pct < 5:
                    # Cap outliers using IQR bounds
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    result.transformations_applied.append(f"Capped outliers in '{col}' to IQR bounds")
                else:
                    # Apply log transformation for heavily skewed data
                    if df[col].min() > 0:
                        skewness = abs(df[col].skew())
                        if skewness > 2:
                            df[col] = np.log1p(df[col])
                            result.transformations_applied.append(f"Log transformation applied to '{col}' for skewness reduction")
        
        return df
    
    @staticmethod
    def enhance_feature_quality(df: pd.DataFrame, target_column: Optional[str], result) -> pd.DataFrame:
        """Enhance overall feature quality"""
        
        # Remove features with single unique value
        single_value_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if single_value_cols:
            df = df.drop(columns=single_value_cols)
            result.transformations_applied.append(f"Removed single-value columns: {single_value_cols}")
        
        # Remove highly correlated features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if target_column in numeric_cols:
            numeric_cols = numeric_cols.drop(target_column)
        
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()
            
            # Find highly correlated pairs
            to_remove = set()
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.95:
                        # Remove the feature with lower correlation to target
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        if target_column and target_column in df.columns:
                            corr1 = abs(df[col1].corr(df[target_column])) if df[col1].dtype in ['int64', 'float64'] else 0
                            corr2 = abs(df[col2].corr(df[target_column])) if df[col2].dtype in ['int64', 'float64'] else 0
                            to_remove.add(col2 if corr1 > corr2 else col1)
                        else:
                            to_remove.add(col2)  # Remove second feature by default
            
            if to_remove:
                df = df.drop(columns=list(to_remove))
                result.transformations_applied.append(f"Removed highly correlated features: {list(to_remove)}")
        
        return df
    
    @staticmethod
    def intelligent_type_optimization(df: pd.DataFrame, result) -> pd.DataFrame:
        """Intelligent data type optimization"""
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to more efficient types
                unique_vals = df[col].nunique()
                total_vals = len(df)
                
                # Convert to category if low cardinality
                if unique_vals / total_vals < 0.5 and unique_vals < 100:
                    df[col] = df[col].astype('category')
                    result.transformations_applied.append(f"Converted '{col}' to category type")
                
                # Try to convert to numeric if possible
                elif df[col].dropna().apply(lambda x: str(x).replace('.', '').replace('-', '').isdigit()).all():
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        result.transformations_applied.append(f"Converted '{col}' to numeric type")
                    except:
                        pass
            
            elif df[col].dtype in ['int64', 'float64']:
                # Downcast numeric types for memory efficiency
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
                    try:
                        float32_version = df[col].astype('float32')
                        if np.allclose(df[col].dropna(), float32_version.dropna(), rtol=1e-6):
                            df[col] = float32_version
                            result.transformations_applied.append(f"Downcasted '{col}' to float32")
                    except:
                        pass
        
        return df
    
    @staticmethod
    def calculate_comprehensive_metrics(original_df: pd.DataFrame, processed_df: pd.DataFrame, 
                                      target_column: Optional[str]):
        """Calculate comprehensive preprocessing metrics"""
        from enhanced_preprocessor import PreprocessingMetrics
        
        metrics = PreprocessingMetrics()
        
        # Data quality score
        orig_missing = (original_df.isnull().sum().sum() / (original_df.shape[0] * original_df.shape[1]))
        proc_missing = (processed_df.isnull().sum().sum() / (processed_df.shape[0] * processed_df.shape[1]))
        
        orig_duplicates = original_df.duplicated().sum() / len(original_df)
        proc_duplicates = processed_df.duplicated().sum() / len(processed_df)
        
        # Quality improvements
        missing_improvement = max(0, orig_missing - proc_missing)
        duplicate_improvement = max(0, orig_duplicates - proc_duplicates)
        
        metrics.data_quality_score = min(100, 100 - (proc_missing * 100) - (proc_duplicates * 10))
        metrics.missing_data_ratio = proc_missing
        metrics.duplicate_ratio = proc_duplicates
        
        # Feature quality score
        if len(processed_df.columns) > 0:
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                try:
                    corr_matrix = processed_df[numeric_cols].corr().abs()
                    # Penalize high correlations
                    high_corr_count = (corr_matrix > 0.9).sum().sum() - len(corr_matrix)
                    correlation_penalty = min(30, high_corr_count * 5)
                    metrics.feature_quality_score = max(0, 100 - correlation_penalty)
                except:
                    metrics.feature_quality_score = 80
            else:
                metrics.feature_quality_score = 80
        
        # Target balance score (for classification)
        if target_column and target_column in processed_df.columns:
            target_counts = processed_df[target_column].value_counts()
            if len(target_counts) > 1:
                balance_ratio = target_counts.min() / target_counts.max()
                metrics.target_balance_score = balance_ratio * 100
        
        # Memory usage
        metrics.memory_usage_mb = processed_df.memory_usage(deep=True).sum() / 1024 / 1024
        
        return metrics
    
    @staticmethod
    def calculate_quality_improvement(original_df: pd.DataFrame, processed_df: pd.DataFrame) -> float:
        """Calculate overall quality improvement percentage"""
        
        # Original quality metrics
        orig_missing_pct = (original_df.isnull().sum().sum() / (original_df.shape[0] * original_df.shape[1])) * 100
        orig_duplicates = original_df.duplicated().sum()
        
        # Processed quality metrics
        proc_missing_pct = (processed_df.isnull().sum().sum() / (processed_df.shape[0] * processed_df.shape[1])) * 100
        proc_duplicates = processed_df.duplicated().sum()
        
        # Calculate improvements
        missing_improvement = max(0, orig_missing_pct - proc_missing_pct)
        duplicate_improvement = max(0, orig_duplicates - proc_duplicates)
        
        # Feature count improvement (more features after engineering vs. removed bad features)
        feature_ratio = processed_df.shape[1] / max(1, original_df.shape[1])
        feature_score = min(20, max(0, (feature_ratio - 0.5) * 40))  # Optimal around 1.0-1.5x features
        
        # Weighted improvement score
        total_improvement = (missing_improvement * 0.4) + \
                          (duplicate_improvement / max(1, original_df.shape[0]) * 100 * 0.3) + \
                          (feature_score * 0.3)
        
        return round(min(100, total_improvement), 2)

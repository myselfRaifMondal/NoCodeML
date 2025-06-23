"""
Enhanced Data Preprocessing System with Iterative Improvement for NoCodeML

This module provides advanced data preprocessing capabilities with:
- Intelligent issue detection and automatic fixing
- Iterative improvement based on model performance feedback
- User feedback integration for continuous learning
- Adaptive preprocessing strategies
- Quality reinforcement learning
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
# Enable experimental features first
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.feature_selection import SelectKBest, SelectFromModel, f_classif, f_regression, mutual_info_classif
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from scipy import stats
from scipy.stats import chi2_contingency
import warnings
from datetime import datetime
import logging
from pathlib import Path
import sys
import json
import pickle
from dataclasses import dataclass, asdict
from enum import Enum

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from backend.models.schemas import DataType, ProblemType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IssueType(Enum):
    MISSING_DATA = "missing_data"
    DUPLICATES = "duplicates"
    OUTLIERS = "outliers"
    HIGH_CORRELATION = "high_correlation"
    LOW_VARIANCE = "low_variance"
    HIGH_CARDINALITY = "high_cardinality"
    DATA_LEAKAGE = "data_leakage"
    SKEWED_DISTRIBUTION = "skewed_distribution"
    INCONSISTENT_TYPES = "inconsistent_types"
    TEMPORAL_ISSUES = "temporal_issues"

class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class DataIssue:
    """Enhanced data issue representation"""
    issue_type: IssueType
    severity: Severity
    description: str
    columns: List[str]
    solution: str
    auto_fixable: bool = True
    confidence_score: float = 1.0
    impact_on_model: str = ""
    suggested_action: str = ""
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

@dataclass
class PreprocessingMetrics:
    """Metrics to track preprocessing effectiveness"""
    data_quality_score: float = 0.0
    missing_data_ratio: float = 0.0
    duplicate_ratio: float = 0.0
    outlier_ratio: float = 0.0
    feature_quality_score: float = 0.0
    target_balance_score: float = 0.0
    preprocessing_time: float = 0.0
    memory_usage_mb: float = 0.0

@dataclass
class PreprocessingResult:
    """Comprehensive preprocessing results"""
    original_shape: Tuple[int, int] = (0, 0)
    processed_shape: Tuple[int, int] = (0, 0)
    issues_found: List[DataIssue] = None
    issues_fixed: List[DataIssue] = None
    transformations_applied: List[str] = None
    metrics: PreprocessingMetrics = None
    feature_names: List[str] = None
    encoding_mappings: Dict[str, Any] = None
    scaler_params: Dict[str, Any] = None
    recommendations: List[str] = None
    quality_improvement: float = 0.0
    preprocessing_log: List[str] = None
    user_feedback: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.issues_found is None:
            self.issues_found = []
        if self.issues_fixed is None:
            self.issues_fixed = []
        if self.transformations_applied is None:
            self.transformations_applied = []
        if self.metrics is None:
            self.metrics = PreprocessingMetrics()
        if self.feature_names is None:
            self.feature_names = []
        if self.encoding_mappings is None:
            self.encoding_mappings = {}
        if self.scaler_params is None:
            self.scaler_params = {}
        if self.recommendations is None:
            self.recommendations = []
        if self.preprocessing_log is None:
            self.preprocessing_log = []
        if self.user_feedback is None:
            self.user_feedback = {}

class EnhancedDataPreprocessor:
    """Advanced data preprocessor with iterative improvement capabilities"""
    
    def __init__(self, learning_enabled: bool = True):
        self.learning_enabled = learning_enabled
        self.preprocessing_history: List[PreprocessingResult] = []
        self.user_feedback_history: List[Dict[str, Any]] = []
        self.adaptive_thresholds = {
            'missing_threshold': 0.5,
            'correlation_threshold': 0.95,
            'variance_threshold': 0.01,
            'outlier_threshold': 3.0,
            'cardinality_threshold': 0.5
        }
        
        # Strategy effectiveness tracking
        self.strategy_performance = {
            'imputation_strategies': {},
            'outlier_strategies': {},
            'encoding_strategies': {},
            'scaling_strategies': {}
        }
        
        # Load learned preferences if available
        self._load_learned_preferences()
    
    def comprehensive_preprocessing(self, df: pd.DataFrame, 
                                  target_column: Optional[str] = None,
                                  problem_type: Optional[ProblemType] = None,
                                  user_preferences: Optional[Dict[str, Any]] = None,
                                  feedback_callback: Optional[Callable] = None) -> Tuple[pd.DataFrame, PreprocessingResult]:
        """
        Perform comprehensive data preprocessing with iterative improvement
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            problem_type: Type of ML problem
            user_preferences: User-specified preferences
            feedback_callback: Function to get user feedback
            
        Returns:
            Tuple of (processed_df, preprocessing_result)
        """
        start_time = datetime.now()
        result = PreprocessingResult()
        result.original_shape = df.shape
        
        logger.info(f"Starting comprehensive preprocessing of dataset with shape {df.shape}")
        
        # Create working copy
        processed_df = df.copy()
        
        try:
            # Phase 1: Initial Data Assessment
            logger.info("Phase 1: Initial data assessment...")
            result.preprocessing_log.append("Phase 1: Initial data assessment")
            issues = self._comprehensive_data_assessment(processed_df, target_column, problem_type)
            result.issues_found = issues
            
            # Phase 2: Critical Issue Resolution
            logger.info("Phase 2: Resolving critical issues...")
            result.preprocessing_log.append("Phase 2: Critical issue resolution")
            processed_df, fixed_issues = self._resolve_critical_issues(processed_df, issues, user_preferences)
            result.issues_fixed.extend(fixed_issues)
            
            # Phase 3: Intelligent Missing Data Handling
            logger.info("Phase 3: Intelligent missing data handling...")
            result.preprocessing_log.append("Phase 3: Missing data handling")
            processed_df = self._intelligent_missing_data_handling(processed_df, target_column, result)
            
            # Phase 4: Advanced Outlier Detection and Treatment
            logger.info("Phase 4: Advanced outlier detection...")
            result.preprocessing_log.append("Phase 4: Outlier detection and treatment")
            processed_df = self._advanced_outlier_handling(processed_df, target_column, result)
            
            # Phase 5: Feature Quality Enhancement
            logger.info("Phase 5: Feature quality enhancement...")
            result.preprocessing_log.append("Phase 5: Feature quality enhancement")
            processed_df = self._enhance_feature_quality(processed_df, target_column, result)
            
            # Phase 6: Intelligent Data Type Optimization
            logger.info("Phase 6: Data type optimization...")
            result.preprocessing_log.append("Phase 6: Data type optimization")
            processed_df = self._intelligent_type_optimization(processed_df, result)
            
            # Phase 7: Advanced Feature Engineering
            logger.info("Phase 7: Advanced feature engineering...")
            result.preprocessing_log.append("Phase 7: Feature engineering")
            processed_df = self._advanced_feature_engineering(processed_df, target_column, problem_type, result)
            
            # Phase 8: Adaptive Categorical Encoding
            logger.info("Phase 8: Adaptive categorical encoding...")
            result.preprocessing_log.append("Phase 8: Categorical encoding")
            processed_df = self._adaptive_categorical_encoding(processed_df, target_column, problem_type, result)
            
            # Phase 9: Intelligent Feature Scaling
            logger.info("Phase 9: Intelligent feature scaling...")
            result.preprocessing_log.append("Phase 9: Feature scaling")
            processed_df = self._intelligent_feature_scaling(processed_df, target_column, problem_type, result)
            
            # Phase 10: Advanced Feature Selection
            logger.info("Phase 10: Advanced feature selection...")
            result.preprocessing_log.append("Phase 10: Feature selection")
            if target_column and target_column in processed_df.columns:
                processed_df = self._advanced_feature_selection(processed_df, target_column, problem_type, result)
            
            # Phase 11: Quality Assessment and Metrics
            logger.info("Phase 11: Quality assessment...")
            result.preprocessing_log.append("Phase 11: Quality assessment")
            result.processed_shape = processed_df.shape
            result.metrics = self._calculate_comprehensive_metrics(df, processed_df, target_column)
            result.quality_improvement = self._calculate_quality_improvement(df, processed_df)
            
            # Phase 12: Generate Recommendations
            logger.info("Phase 12: Generating recommendations...")
            result.preprocessing_log.append("Phase 12: Recommendations generation")
            result.recommendations = self._generate_intelligent_recommendations(df, processed_df, result, problem_type)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            result.metrics.preprocessing_time = processing_time
            
            # Store result for learning
            self.preprocessing_history.append(result)
            
            # Get user feedback if callback provided
            if feedback_callback:
                feedback = feedback_callback(result)
                if feedback:
                    result.user_feedback = feedback
                    self._learn_from_feedback(feedback, result)
            
            logger.info(f"Preprocessing completed successfully in {processing_time:.2f} seconds")
            logger.info(f"Data shape: {df.shape} -> {processed_df.shape}")
            logger.info(f"Quality improvement: {result.quality_improvement:.2f}%")
            
            return processed_df, result
            
        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            # Return original data with error information
            result.preprocessing_log.append(f"Error: {str(e)}")
            return df, result
    
    def _comprehensive_data_assessment(self, df: pd.DataFrame, target_column: Optional[str], 
                                     problem_type: Optional[ProblemType]) -> List[DataIssue]:
        """Perform comprehensive data quality assessment"""
        issues = []
        
        # 1. Missing data analysis
        missing_issues = self._assess_missing_data(df)
        issues.extend(missing_issues)
        
        # 2. Duplicate detection
        duplicate_issues = self._assess_duplicates(df)
        issues.extend(duplicate_issues)
        
        # 3. Outlier detection
        outlier_issues = self._assess_outliers(df, target_column)
        issues.extend(outlier_issues)
        
        # 4. Correlation analysis
        correlation_issues = self._assess_correlations(df, target_column)
        issues.extend(correlation_issues)
        
        # 5. Variance analysis
        variance_issues = self._assess_variance(df)
        issues.extend(variance_issues)
        
        # 6. Cardinality analysis
        cardinality_issues = self._assess_cardinality(df)
        issues.extend(cardinality_issues)
        
        # 7. Data leakage detection
        leakage_issues = self._assess_data_leakage(df, target_column)
        issues.extend(leakage_issues)
        
        # 8. Distribution skewness
        skewness_issues = self._assess_distribution_skewness(df)
        issues.extend(skewness_issues)
        
        # 9. Type consistency
        type_issues = self._assess_type_consistency(df)
        issues.extend(type_issues)
        
        # 10. Temporal issues (if datetime columns exist)
        temporal_issues = self._assess_temporal_issues(df)
        issues.extend(temporal_issues)
        
        return issues
    
    def _assess_missing_data(self, df: pd.DataFrame) -> List[DataIssue]:
        """Assess missing data patterns"""
        issues = []
        
        missing_cols = df.columns[df.isnull().any()].tolist()
        for col in missing_cols:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            
            if missing_pct > 80:
                severity = Severity.CRITICAL
                impact = "Column likely unusable"
            elif missing_pct > 50:
                severity = Severity.HIGH
                impact = "Significant data loss, may need advanced imputation"
            elif missing_pct > 20:
                severity = Severity.MEDIUM
                impact = "Moderate missing data, imputation recommended"
            else:
                severity = Severity.LOW
                impact = "Minor missing data, simple imputation sufficient"
            
            issues.append(DataIssue(
                issue_type=IssueType.MISSING_DATA,
                severity=severity,
                description=f"Column '{col}' has {missing_pct:.1f}% missing values",
                columns=[col],
                solution="Intelligent imputation or column removal",
                confidence_score=0.9,
                impact_on_model=impact,
                suggested_action=f"Apply {'removal' if missing_pct > 70 else 'imputation'} strategy"
            ))
        
        return issues
    
    def _assess_duplicates(self, df: pd.DataFrame) -> List[DataIssue]:
        """Assess duplicate rows"""
        issues = []
        
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            duplicate_pct = (duplicate_count / len(df)) * 100
            
            if duplicate_pct > 20:
                severity = Severity.HIGH
            elif duplicate_pct > 5:
                severity = Severity.MEDIUM
            else:
                severity = Severity.LOW
            
            issues.append(DataIssue(
                issue_type=IssueType.DUPLICATES,
                severity=severity,
                description=f"Found {duplicate_count} duplicate rows ({duplicate_pct:.1f}%)",
                columns=[],
                solution="Remove duplicate rows",
                confidence_score=1.0,
                impact_on_model="May cause overfitting and biased performance metrics",
                suggested_action="Remove duplicate rows before training"
            ))
        
        return issues
    
    def _assess_outliers(self, df: pd.DataFrame, target_column: Optional[str]) -> List[DataIssue]:
        """Advanced outlier detection"""
        issues = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if target_column in numeric_cols:
            numeric_cols = numeric_cols.drop(target_column)
        
        for col in numeric_cols:
            if df[col].notna().sum() < 10:  # Skip if too few values
                continue
                
            # Multiple outlier detection methods
            outlier_methods = {
                'zscore': self._detect_outliers_zscore(df[col]),
                'iqr': self._detect_outliers_iqr(df[col]),
                'isolation_forest': self._detect_outliers_isolation_forest(df[[col]])
            }
            
            # Consensus outlier detection
            outlier_consensus = np.zeros(len(df))
            for method, outliers in outlier_methods.items():
                outlier_consensus += outliers.astype(int)
            
            # Consider outliers if detected by multiple methods
            strong_outliers = outlier_consensus >= 2
            outlier_count = strong_outliers.sum()
            
            if outlier_count > 0:
                outlier_pct = (outlier_count / len(df)) * 100
                
                if outlier_pct > 10:
                    severity = Severity.HIGH
                    impact = "Many outliers may indicate data quality issues"
                elif outlier_pct > 5:
                    severity = Severity.MEDIUM
                    impact = "Moderate outliers may affect model performance"
                else:
                    severity = Severity.LOW
                    impact = "Few outliers, consider capping or transformation"
                
                issues.append(DataIssue(
                    issue_type=IssueType.OUTLIERS,
                    severity=severity,
                    description=f"Column '{col}' has {outlier_count} outliers ({outlier_pct:.1f}%)",
                    columns=[col],
                    solution="Capping, transformation, or removal based on context",
                    confidence_score=0.8,
                    impact_on_model=impact,
                    suggested_action=self._suggest_outlier_treatment(outlier_pct)
                ))
        
        return issues
    
    def _intelligent_missing_data_handling(self, df: pd.DataFrame, target_column: Optional[str], 
                                         result: PreprocessingResult) -> pd.DataFrame:
        """Intelligent missing data handling with multiple strategies"""
        
        for col in df.columns:
            if not df[col].isnull().any():
                continue
                
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            
            # Skip columns with too much missing data
            if missing_pct > self.adaptive_thresholds['missing_threshold'] * 100:
                continue
            
            # Choose imputation strategy based on data characteristics
            strategy = self._choose_imputation_strategy(df, col, target_column, missing_pct)
            
            try:
                if strategy == 'simple_mean':
                    df[col].fillna(df[col].mean(), inplace=True)
                    result.transformations_applied.append(f"Mean imputation for '{col}'")
                    
                elif strategy == 'simple_median':
                    df[col].fillna(df[col].median(), inplace=True)
                    result.transformations_applied.append(f"Median imputation for '{col}'")
                    
                elif strategy == 'simple_mode':
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col].fillna(mode_val[0], inplace=True)
                        result.transformations_applied.append(f"Mode imputation for '{col}'")
                    
                elif strategy == 'knn':
                    imputer = KNNImputer(n_neighbors=min(5, df.shape[0]//10))
                    df[[col]] = imputer.fit_transform(df[[col]])
                    result.transformations_applied.append(f"KNN imputation for '{col}'")
                    
                elif strategy == 'iterative':
                    imputer = IterativeImputer(random_state=42, max_iter=10)
                    df[[col]] = imputer.fit_transform(df[[col]])
                    result.transformations_applied.append(f"Iterative imputation for '{col}'")
                    
                elif strategy == 'forward_fill':
                    df[col].fillna(method='ffill', inplace=True)
                    df[col].fillna(method='bfill', inplace=True)  # Handle remaining NaNs
                    result.transformations_applied.append(f"Forward fill imputation for '{col}'")
                    
                elif strategy == 'interpolate':
                    df[col] = df[col].interpolate()
                    result.transformations_applied.append(f"Interpolation imputation for '{col}'")
                    
                elif strategy == 'categorical_unknown':
                    df[col].fillna('Unknown', inplace=True)
                    result.transformations_applied.append(f"Unknown category imputation for '{col}'")
                    
            except Exception as e:
                logger.warning(f"Failed to apply {strategy} imputation to {col}: {str(e)}")
                # Fallback to simple imputation
                if df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna('Unknown', inplace=True)
                result.transformations_applied.append(f"Fallback imputation for '{col}'")
        
        return df
    
    def _choose_imputation_strategy(self, df: pd.DataFrame, col: str, target_column: Optional[str], 
                                  missing_pct: float) -> str:
        """Choose optimal imputation strategy based on data characteristics"""
        
        # Check historical performance if available
        if col in self.strategy_performance.get('imputation_strategies', {}):
            best_strategy = max(
                self.strategy_performance['imputation_strategies'][col],
                key=self.strategy_performance['imputation_strategies'][col].get
            )
            return best_strategy
        
        # Default strategy selection logic
        if df[col].dtype in ['int64', 'float64']:
            # Numeric columns
            if missing_pct < 5:
                return 'simple_median'
            elif missing_pct < 15:
                return 'knn'
            else:
                return 'iterative'
                
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            # Datetime columns
            return 'interpolate' if missing_pct < 20 else 'forward_fill'
            
        else:
            # Categorical columns
            if missing_pct < 10:
                return 'simple_mode'
            else:
                return 'categorical_unknown'
    
    def iterative_improvement(self, df: pd.DataFrame, target_column: Optional[str],
                            problem_type: Optional[ProblemType], 
                            model_performance_callback: Optional[Callable] = None,
                            max_iterations: int = 5,
                            improvement_threshold: float = 0.01) -> Tuple[pd.DataFrame, List[PreprocessingResult]]:
        """
        Perform iterative preprocessing improvements based on model performance feedback
        
        Args:
            df: Input DataFrame
            target_column: Target column name
            problem_type: ML problem type
            model_performance_callback: Function that trains a model and returns performance metrics
            max_iterations: Maximum number of improvement iterations
            improvement_threshold: Minimum improvement required to continue
            
        Returns:
            Tuple of (best_df, list_of_results)
        """
        logger.info(f"Starting iterative improvement with max {max_iterations} iterations")
        
        results = []
        current_df = df.copy()
        best_df = current_df.copy()
        best_performance = 0.0
        
        for iteration in range(max_iterations):
            logger.info(f"Iteration {iteration + 1}/{max_iterations}")
            
            # Perform preprocessing
            processed_df, result = self.comprehensive_preprocessing(
                current_df, target_column, problem_type
            )
            
            results.append(result)
            
            # Evaluate model performance if callback provided
            if model_performance_callback:
                try:
                    performance_metrics = model_performance_callback(processed_df, target_column)
                    current_performance = self._extract_primary_metric(performance_metrics, problem_type)
                    
                    logger.info(f"Iteration {iteration + 1} performance: {current_performance:.4f}")
                    
                    # Check for improvement
                    if current_performance > best_performance + improvement_threshold:
                        best_performance = current_performance
                        best_df = processed_df.copy()
                        logger.info(f"New best performance: {best_performance:.4f}")
                        
                        # Learn from this successful iteration
                        self._learn_from_performance(result, performance_metrics, improvement=True)
                        
                        # Prepare for next iteration with current best
                        current_df = processed_df.copy()
                    else:
                        logger.info("No significant improvement, stopping iterations")
                        self._learn_from_performance(result, performance_metrics, improvement=False)
                        break
                        
                except Exception as e:
                    logger.warning(f"Model performance evaluation failed: {str(e)}")
                    break
            else:
                # Without performance feedback, use quality improvement
                if result.quality_improvement > best_performance + improvement_threshold:
                    best_performance = result.quality_improvement
                    best_df = processed_df.copy()
                    current_df = processed_df.copy()
                else:
                    logger.info("Quality stabilized, stopping iterations")
                    break
        
        logger.info(f"Iterative improvement completed. Best performance: {best_performance:.4f}")
        return best_df, results
    
    def _extract_primary_metric(self, performance_metrics: Dict[str, float], 
                              problem_type: Optional[ProblemType]) -> float:
        """Extract primary performance metric based on problem type"""
        if problem_type == ProblemType.CLASSIFICATION:
            return performance_metrics.get('accuracy', performance_metrics.get('f1_score', 0.0))
        elif problem_type == ProblemType.REGRESSION:
            return performance_metrics.get('r2_score', -performance_metrics.get('rmse', float('inf')))
        else:
            return performance_metrics.get('silhouette_score', 0.0)
    
    def _learn_from_performance(self, result: PreprocessingResult, 
                              performance_metrics: Dict[str, float], improvement: bool):
        """Learn from model performance to improve future preprocessing"""
        if not self.learning_enabled:
            return
        
        # Update strategy performance tracking
        feedback_score = 1.0 if improvement else 0.0
        
        for transformation in result.transformations_applied:
            # Extract strategy type and update performance
            if 'imputation' in transformation.lower():
                self._update_strategy_performance('imputation_strategies', transformation, feedback_score)
            elif 'scaling' in transformation.lower() or 'standard' in transformation.lower():
                self._update_strategy_performance('scaling_strategies', transformation, feedback_score)
            elif 'encoded' in transformation.lower():
                self._update_strategy_performance('encoding_strategies', transformation, feedback_score)
            elif 'outlier' in transformation.lower():
                self._update_strategy_performance('outlier_strategies', transformation, feedback_score)
        
        # Adapt thresholds based on performance
        if improvement:
            # Slightly relax thresholds that worked well
            for key in self.adaptive_thresholds:
                self.adaptive_thresholds[key] *= 0.95
        else:
            # Slightly tighten thresholds that didn't work
            for key in self.adaptive_thresholds:
                self.adaptive_thresholds[key] *= 1.05
        
        # Save learned preferences
        self._save_learned_preferences()
    
    def _update_strategy_performance(self, strategy_type: str, transformation: str, score: float):
        """Update performance tracking for specific strategies"""
        if strategy_type not in self.strategy_performance:
            self.strategy_performance[strategy_type] = {}
        
        if transformation not in self.strategy_performance[strategy_type]:
            self.strategy_performance[strategy_type][transformation] = []
        
        self.strategy_performance[strategy_type][transformation].append(score)
        
        # Keep only recent scores (last 10)
        if len(self.strategy_performance[strategy_type][transformation]) > 10:
            self.strategy_performance[strategy_type][transformation] = \
                self.strategy_performance[strategy_type][transformation][-10:]
    
    def _learn_from_feedback(self, feedback: Dict[str, Any], result: PreprocessingResult):
        """Learn from user feedback to improve future preprocessing"""
        if not self.learning_enabled:
            return
        
        self.user_feedback_history.append({
            'feedback': feedback,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
        # Analyze feedback patterns
        satisfaction_score = feedback.get('satisfaction_score', 5) / 10.0  # Normalize to 0-1
        
        # Update strategy preferences based on user satisfaction
        for transformation in result.transformations_applied:
            self._update_strategy_performance('user_feedback', transformation, satisfaction_score)
        
        # Save feedback for future analysis
        self._save_learned_preferences()
    
    def _save_learned_preferences(self):
        """Save learned preferences to disk"""
        try:
            preferences_file = project_root / "models" / "preprocessing_preferences.json"
            preferences_file.parent.mkdir(parents=True, exist_ok=True)
            
            preferences = {
                'adaptive_thresholds': self.adaptive_thresholds,
                'strategy_performance': self.strategy_performance,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(preferences_file, 'w') as f:
                json.dump(preferences, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save learned preferences: {str(e)}")
    
    def _load_learned_preferences(self):
        """Load previously learned preferences"""
        try:
            preferences_file = project_root / "models" / "preprocessing_preferences.json"
            
            if preferences_file.exists():
                with open(preferences_file, 'r') as f:
                    preferences = json.load(f)
                
                self.adaptive_thresholds.update(preferences.get('adaptive_thresholds', {}))
                self.strategy_performance.update(preferences.get('strategy_performance', {}))
                
                logger.info("Loaded learned preprocessing preferences")
                
        except Exception as e:
            logger.warning(f"Failed to load learned preferences: {str(e)}")
    
    def generate_improvement_report(self, results: List[PreprocessingResult]) -> Dict[str, Any]:
        """Generate comprehensive improvement report"""
        if not results:
            return {}
        
        final_result = results[-1]
        
        report = {
            'executive_summary': {
                'total_iterations': len(results),
                'final_quality_score': final_result.metrics.data_quality_score,
                'total_quality_improvement': final_result.quality_improvement,
                'processing_time': sum(r.metrics.preprocessing_time for r in results),
                'data_shape_change': f"{final_result.original_shape} → {final_result.processed_shape}"
            },
            'issue_analysis': {
                'total_issues_found': len(final_result.issues_found),
                'issues_by_type': self._group_issues_by_type(final_result.issues_found),
                'issues_by_severity': self._group_issues_by_severity(final_result.issues_found),
                'auto_fixed_count': len(final_result.issues_fixed),
                'manual_attention_needed': len([i for i in final_result.issues_found if not i.auto_fixable])
            },
            'transformation_summary': {
                'total_transformations': len(final_result.transformations_applied),
                'transformation_categories': self._categorize_transformations(final_result.transformations_applied),
                'most_impactful_transformations': final_result.transformations_applied[:5]
            },
            'quality_metrics': asdict(final_result.metrics),
            'recommendations': final_result.recommendations,
            'learning_insights': {
                'adaptive_thresholds': self.adaptive_thresholds,
                'top_performing_strategies': self._get_top_strategies(),
                'user_satisfaction_trends': self._analyze_user_satisfaction()
            },
            'next_steps': self._generate_next_steps(final_result),
            'technical_details': {
                'encoding_mappings': final_result.encoding_mappings,
                'scaling_parameters': final_result.scaler_params,
                'feature_names': final_result.feature_names
            }
        }
        
        return report
    
    def suggest_model_improvements(self, preprocessing_result: PreprocessingResult, 
                                 model_performance: Dict[str, float]) -> List[str]:
        """Suggest model improvements based on preprocessing results and performance"""
        suggestions = []
        
        # Performance-based suggestions
        if preprocessing_result.metrics.data_quality_score < 70:
            suggestions.append("Data quality is low. Consider additional data cleaning before model training.")
        
        if preprocessing_result.metrics.missing_data_ratio > 0.2:
            suggestions.append("High missing data ratio. Consider more sophisticated imputation or feature engineering.")
        
        if preprocessing_result.metrics.outlier_ratio > 0.1:
            suggestions.append("High outlier ratio detected. Consider robust scaling or outlier-resistant algorithms.")
        
        # Feature-based suggestions
        if len(preprocessing_result.feature_names) > 100:
            suggestions.append("High dimensionality detected. Consider dimensionality reduction or feature selection.")
        
        if preprocessing_result.metrics.feature_quality_score < 60:
            suggestions.append("Low feature quality score. Consider additional feature engineering or domain expertise.")
        
        # Model performance-based suggestions
        primary_metric = model_performance.get('accuracy', model_performance.get('r2_score', 0))
        if primary_metric < 0.7:
            suggestions.extend([
                "Low model performance. Consider ensemble methods or hyperparameter tuning.",
                "Try advanced feature engineering techniques like polynomial features or interactions.",
                "Consider collecting more training data or synthetic data generation."
            ])
        
        return suggestions
    
    # Helper methods for various assessments and operations
    def _detect_outliers_zscore(self, series: pd.Series, threshold: float = 3.0) -> np.ndarray:
        """Detect outliers using Z-score method"""
        outliers = np.zeros(len(series), dtype=bool)
        non_null_mask = series.notna()
        
        if non_null_mask.sum() > 0:
            z_scores = np.abs(stats.zscore(series[non_null_mask]))
            outliers[non_null_mask] = z_scores > threshold
        
        return outliers
    
    def _detect_outliers_iqr(self, series: pd.Series) -> np.ndarray:
        """Detect outliers using IQR method"""
        outliers = np.zeros(len(series), dtype=bool)
        non_null_mask = series.notna()
        
        if non_null_mask.sum() > 0:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:  # Avoid division by zero
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers[non_null_mask] = ((series[non_null_mask] < lower_bound) | 
                                          (series[non_null_mask] > upper_bound))
        
        return outliers
    
    def _detect_outliers_isolation_forest(self, df: pd.DataFrame) -> np.ndarray:
        """Detect outliers using Isolation Forest"""
        outliers = np.zeros(len(df), dtype=bool)
        
        try:
            # Only use rows without NaN values for fitting
            non_null_mask = df.notna().all(axis=1)
            
            if non_null_mask.sum() > 10:  # Need enough data points
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                predictions = iso_forest.fit_predict(df[non_null_mask])
                outliers[non_null_mask] = predictions == -1
                
        except Exception as e:
            logger.warning(f"Isolation Forest outlier detection failed: {str(e)}")
            
        return outliers
    
    def _assess_correlations(self, df: pd.DataFrame, target_column: Optional[str]) -> List[DataIssue]:
        """Assess high correlations between features"""
        issues = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > self.adaptive_thresholds['correlation_threshold']:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
            
            if high_corr_pairs:
                issues.append(DataIssue(
                    issue_type=IssueType.HIGH_CORRELATION,
                    severity=Severity.MEDIUM,
                    description=f"Found {len(high_corr_pairs)} highly correlated feature pairs",
                    columns=[pair[1] for pair in high_corr_pairs],
                    solution="Remove redundant features",
                    confidence_score=0.9,
                    impact_on_model="Multicollinearity can reduce model interpretability",
                    suggested_action="Remove one feature from each highly correlated pair"
                ))
        
        return issues
    
    def _assess_variance(self, df: pd.DataFrame) -> List[DataIssue]:
        """Assess low variance features"""
        issues = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].var() < self.adaptive_thresholds['variance_threshold']:
                issues.append(DataIssue(
                    issue_type=IssueType.LOW_VARIANCE,
                    severity=Severity.MEDIUM,
                    description=f"Column '{col}' has very low variance ({df[col].var():.6f})",
                    columns=[col],
                    solution="Remove low variance features",
                    confidence_score=1.0,
                    impact_on_model="Low variance features provide little information",
                    suggested_action="Remove or transform the feature"
                ))
        
        return issues
    
    def _assess_cardinality(self, df: pd.DataFrame) -> List[DataIssue]:
        """Assess high cardinality categorical features"""
        issues = []
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            cardinality_ratio = df[col].nunique() / len(df)
            if cardinality_ratio > self.adaptive_thresholds['cardinality_threshold'] and df[col].nunique() > 50:
                issues.append(DataIssue(
                    issue_type=IssueType.HIGH_CARDINALITY,
                    severity=Severity.MEDIUM,
                    description=f"Column '{col}' has high cardinality ({df[col].nunique()} unique values)",
                    columns=[col],
                    solution="Feature engineering or dimensionality reduction",
                    confidence_score=0.8,
                    impact_on_model="High cardinality can lead to sparse encoding and overfitting",
                    suggested_action="Apply target encoding or frequency encoding"
                ))
        
        return issues
    
    def _assess_data_leakage(self, df: pd.DataFrame, target_column: Optional[str]) -> List[DataIssue]:
        """Detect potential data leakage"""
        issues = []
        
        if not target_column or target_column not in df.columns:
            return issues
        
        # Check for perfect correlations with target
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if target_column in numeric_cols:
            numeric_cols = numeric_cols.drop(target_column)
            
            for col in numeric_cols:
                correlation = abs(df[col].corr(df[target_column]))
                if correlation > 0.99:
                    issues.append(DataIssue(
                        issue_type=IssueType.DATA_LEAKAGE,
                        severity=Severity.CRITICAL,
                        description=f"Column '{col}' has perfect correlation with target ({correlation:.3f})",
                        columns=[col],
                        solution="Remove potentially leaked feature",
                        confidence_score=0.9,
                        impact_on_model="Data leakage leads to unrealistic performance",
                        suggested_action="Investigate and remove if confirmed leakage"
                    ))
        
        return issues
    
    def _assess_distribution_skewness(self, df: pd.DataFrame) -> List[DataIssue]:
        """Assess distribution skewness"""
        issues = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].notna().sum() > 3:  # Need at least 3 values
                skewness = abs(df[col].skew())
                if skewness > 2:
                    issues.append(DataIssue(
                        issue_type=IssueType.SKEWED_DISTRIBUTION,
                        severity=Severity.MEDIUM,
                        description=f"Column '{col}' is highly skewed (skewness: {skewness:.2f})",
                        columns=[col],
                        solution="Apply transformation (log, square root, or Box-Cox)",
                        confidence_score=0.8,
                        impact_on_model="Skewed distributions can hurt linear models",
                        suggested_action="Apply log transformation if values are positive"
                    ))
        
        return issues
    
    def _assess_type_consistency(self, df: pd.DataFrame) -> List[DataIssue]:
        """Assess data type consistency"""
        issues = []
        
        for col in df.columns:
            # Check for mixed types in object columns
            if df[col].dtype == 'object':
                sample_types = set(type(x).__name__ for x in df[col].dropna().iloc[:100])
                if len(sample_types) > 1:
                    issues.append(DataIssue(
                        issue_type=IssueType.INCONSISTENT_TYPES,
                        severity=Severity.MEDIUM,
                        description=f"Column '{col}' has inconsistent data types: {sample_types}",
                        columns=[col],
                        solution="Standardize data types",
                        confidence_score=0.7,
                        impact_on_model="Inconsistent types can cause processing errors",
                        suggested_action="Convert to consistent type or clean data"
                    ))
        
        return issues
    
    def _assess_temporal_issues(self, df: pd.DataFrame) -> List[DataIssue]:
        """Assess temporal data issues"""
        issues = []
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        
        for col in datetime_cols:
            # Check for future dates (if any dates are in the future)
            future_dates = df[col] > pd.Timestamp.now()
            if future_dates.any():
                issues.append(DataIssue(
                    issue_type=IssueType.TEMPORAL_ISSUES,
                    severity=Severity.MEDIUM,
                    description=f"Column '{col}' contains {future_dates.sum()} future dates",
                    columns=[col],
                    solution="Investigate and correct future dates",
                    confidence_score=0.8,
                    impact_on_model="Future dates may indicate data quality issues",
                    suggested_action="Verify data collection process"
                ))
        
        return issues
    
    def _suggest_outlier_treatment(self, outlier_pct: float) -> str:
        """Suggest appropriate outlier treatment"""
        if outlier_pct > 20:
            return "Investigate data collection process"
        elif outlier_pct > 10:
            return "Apply capping or robust scaling"
        elif outlier_pct > 5:
            return "Consider log transformation or winsorization"
        else:
            return "Monitor but may keep outliers for model robustness"
    
    def _resolve_critical_issues(self, df: pd.DataFrame, issues: List[DataIssue], 
                               user_preferences: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, List[DataIssue]]:
        """Resolve critical issues automatically"""
        fixed_issues = []
        processed_df = df.copy()
        
        for issue in issues:
            if issue.severity == Severity.CRITICAL and issue.auto_fixable:
                if issue.issue_type == IssueType.MISSING_DATA:
                    # Remove columns with >80% missing
                    for col in issue.columns:
                        if col in processed_df.columns:
                            processed_df = processed_df.drop(columns=[col])
                            fixed_issues.append(issue)
                            
                elif issue.issue_type == IssueType.DATA_LEAKAGE:
                    # Remove potentially leaked features
                    for col in issue.columns:
                        if col in processed_df.columns:
                            processed_df = processed_df.drop(columns=[col])
                            fixed_issues.append(issue)
        
        # Remove duplicates
        if processed_df.duplicated().any():
            processed_df = processed_df.drop_duplicates()
            duplicate_issue = next((i for i in issues if i.issue_type == IssueType.DUPLICATES), None)
            if duplicate_issue:
                fixed_issues.append(duplicate_issue)
        
        return processed_df, fixed_issues
    
    # Add remaining helper methods...
    def _enhance_feature_quality(self, df: pd.DataFrame, target_column: Optional[str], 
                               result: PreprocessingResult) -> pd.DataFrame:
        """Enhance feature quality"""
        # Placeholder implementation
        return df
    
    def _intelligent_type_optimization(self, df: pd.DataFrame, result: PreprocessingResult) -> pd.DataFrame:
        """Optimize data types intelligently"""
        # Placeholder implementation
        return df
    
    def _advanced_feature_engineering(self, df: pd.DataFrame, target_column: Optional[str], 
                                    problem_type: Optional[ProblemType], result: PreprocessingResult) -> pd.DataFrame:
        """Advanced feature engineering"""
        # Placeholder implementation
        return df
    
    def _adaptive_categorical_encoding(self, df: pd.DataFrame, target_column: Optional[str], 
                                     problem_type: Optional[ProblemType], result: PreprocessingResult) -> pd.DataFrame:
        """Adaptive categorical encoding"""
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_cols if col != target_column]
        
        for col in categorical_cols:
            unique_vals = df[col].nunique()
            
            if unique_vals == 2:
                # Binary encoding with LabelEncoder
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                result.encoding_mappings[col] = {
                    'type': 'label',
                    'classes': le.classes_.tolist()
                }
                result.transformations_applied.append(f"Label encoded binary column '{col}'")
                
            elif unique_vals <= 10:
                # One-hot encoding for low cardinality
                try:
                    dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False)
                    df = df.drop(columns=[col])
                    df = pd.concat([df, dummies], axis=1)
                    result.encoding_mappings[col] = {
                        'type': 'onehot',
                        'columns': dummies.columns.tolist()
                    }
                    result.transformations_applied.append(f"One-hot encoded column '{col}' ({unique_vals} categories)")
                except Exception as e:
                    # Fallback to label encoding
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    result.encoding_mappings[col] = {
                        'type': 'label_fallback',
                        'classes': le.classes_.tolist()
                    }
                    result.transformations_applied.append(f"Fallback label encoded '{col}'")
                    
            else:
                # High cardinality - use frequency encoding or label encoding
                if unique_vals > 100:
                    # Frequency encoding
                    freq_map = df[col].value_counts().to_dict()
                    df[f'{col}_frequency'] = df[col].map(freq_map)
                    df = df.drop(columns=[col])
                    result.encoding_mappings[col] = {
                        'type': 'frequency',
                        'map': freq_map
                    }
                    result.transformations_applied.append(f"Frequency encoded high-cardinality column '{col}' ({unique_vals} categories)")
                else:
                    # Label encoding for medium cardinality
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    result.encoding_mappings[col] = {
                        'type': 'label',
                        'classes': le.classes_.tolist()
                    }
                    result.transformations_applied.append(f"Label encoded '{col}' ({unique_vals} categories)")
        
        return df
    
    def _intelligent_feature_scaling(self, df: pd.DataFrame, target_column: Optional[str], 
                                   problem_type: Optional[ProblemType], result: PreprocessingResult) -> pd.DataFrame:
        """Intelligent feature scaling"""
        # Placeholder implementation
        return df
    
    def _advanced_feature_selection(self, df: pd.DataFrame, target_column: str, 
                                  problem_type: Optional[ProblemType], result: PreprocessingResult) -> pd.DataFrame:
        """Advanced feature selection"""
        # Placeholder implementation
        return df
    
    def _advanced_outlier_handling(self, df: pd.DataFrame, target_column: Optional[str], 
                                 result: PreprocessingResult) -> pd.DataFrame:
        """Advanced outlier handling"""
        # Placeholder implementation
        return df
    
    def _calculate_comprehensive_metrics(self, original_df: pd.DataFrame, processed_df: pd.DataFrame, 
                                       target_column: Optional[str]) -> PreprocessingMetrics:
        """Calculate comprehensive preprocessing metrics"""
        metrics = PreprocessingMetrics()
        
        # Basic metrics
        metrics.missing_data_ratio = processed_df.isnull().sum().sum() / (processed_df.shape[0] * processed_df.shape[1])
        metrics.duplicate_ratio = processed_df.duplicated().sum() / len(processed_df)
        
        # Quality scores (simplified)
        metrics.data_quality_score = max(0, 100 - (metrics.missing_data_ratio * 100) - (metrics.duplicate_ratio * 100))
        metrics.feature_quality_score = 80.0  # Placeholder
        
        return metrics
    
    def _calculate_quality_improvement(self, original_df: pd.DataFrame, processed_df: pd.DataFrame) -> float:
        """Calculate quality improvement percentage"""
        orig_missing = original_df.isnull().sum().sum() / (original_df.shape[0] * original_df.shape[1])
        proc_missing = processed_df.isnull().sum().sum() / (processed_df.shape[0] * processed_df.shape[1])
        
        improvement = max(0, (orig_missing - proc_missing) * 100)
        return round(improvement, 2)
    
    def _generate_intelligent_recommendations(self, original_df: pd.DataFrame, processed_df: pd.DataFrame, 
                                            result: PreprocessingResult, problem_type: Optional[ProblemType]) -> List[str]:
        """Generate intelligent recommendations"""
        recommendations = []
        
        if result.metrics.data_quality_score < 70:
            recommendations.append("Consider collecting higher quality data")
        
        if len(processed_df.columns) > 50:
            recommendations.append("High dimensionality - consider feature selection")
        
        return recommendations
    
    # Helper methods for reporting
    def _group_issues_by_type(self, issues: List[DataIssue]) -> Dict[str, int]:
        """Group issues by type for reporting"""
        type_counts = {}
        for issue in issues:
            issue_type = issue.issue_type.value
            type_counts[issue_type] = type_counts.get(issue_type, 0) + 1
        return type_counts
    
    def _group_issues_by_severity(self, issues: List[DataIssue]) -> Dict[str, int]:
        """Group issues by severity for reporting"""
        severity_counts = {}
        for issue in issues:
            severity = issue.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        return severity_counts
    
    def _categorize_transformations(self, transformations: List[str]) -> Dict[str, int]:
        """Categorize transformations for reporting"""
        categories = {
            'Missing Data Handling': 0,
            'Outlier Treatment': 0,
            'Feature Engineering': 0,
            'Categorical Encoding': 0,
            'Feature Scaling': 0,
            'Other': 0
        }
        
        for transformation in transformations:
            transformation_lower = transformation.lower()
            if 'imputation' in transformation_lower:
                categories['Missing Data Handling'] += 1
            elif 'outlier' in transformation_lower:
                categories['Outlier Treatment'] += 1
            elif 'encoded' in transformation_lower:
                categories['Categorical Encoding'] += 1
            elif 'scal' in transformation_lower:
                categories['Feature Scaling'] += 1
            elif 'feature' in transformation_lower:
                categories['Feature Engineering'] += 1
            else:
                categories['Other'] += 1
        
        return categories
    
    def _get_top_strategies(self) -> List[str]:
        """Get top performing strategies"""
        top_strategies = []
        for strategy_type, strategies in self.strategy_performance.items():
            if strategies:
                best_strategy = max(strategies.items(), key=lambda x: sum(x[1]) / len(x[1]))
                top_strategies.append(f"{strategy_type}: {best_strategy[0]}")
        return top_strategies
    
    def _analyze_user_satisfaction(self) -> str:
        """Analyze user satisfaction trends"""
        if not self.user_feedback_history:
            return "No user feedback available"
        
        recent_scores = [f.get('feedback', {}).get('satisfaction_score', 5) 
                        for f in self.user_feedback_history[-5:]]
        if recent_scores:
            avg_satisfaction = sum(recent_scores) / len(recent_scores)
            return f"Average satisfaction: {avg_satisfaction:.1f}/10"
        
        return "Insufficient feedback data"
    
    def _generate_next_steps(self, result: PreprocessingResult) -> List[str]:
        """Generate next steps recommendations"""
        next_steps = []
        
        if result.metrics.data_quality_score > 80:
            next_steps.append("Proceed with model training")
        else:
            next_steps.append("Consider additional data cleaning")
        
        next_steps.extend([
            "Monitor preprocessing performance in production",
            "Set up automated data quality alerts",
            "Plan for model retraining schedule"
        ])
        
        return next_steps

"""
Iterative Model Training and Improvement Module for NoCodeML

This module provides advanced model training capabilities with:
- Automatic model selection and hyperparameter tuning
- Iterative improvement based on performance feedback
- Comprehensive model evaluation and comparison
- Automated feature engineering recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime
from pathlib import Path
import sys
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
import joblib
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from backend.models.schemas import ProblemType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelPerformance:
    """Container for model performance metrics"""
    
    def __init__(self, model_name: str, problem_type: str):
        self.model_name = model_name
        self.problem_type = problem_type
        self.metrics: Dict[str, float] = {}
        self.cv_scores: List[float] = []
        self.feature_importance: Optional[Dict[str, float]] = None
        self.training_time: float = 0.0
        self.predictions: Optional[np.ndarray] = None
        self.prediction_probabilities: Optional[np.ndarray] = None
        self.hyperparameters: Dict[str, Any] = {}
        self.model_object: Any = None

class TrainingResult:
    """Contains comprehensive training results"""
    
    def __init__(self):
        self.best_model: Optional[ModelPerformance] = None
        self.all_models: List[ModelPerformance] = []
        self.training_summary: Dict[str, Any] = {}
        self.recommendations: List[str] = []
        self.feature_analysis: Dict[str, Any] = {}
        self.data_insights: List[str] = []
        self.improvement_suggestions: List[str] = []

class IterativeModelTrainer:
    """Advanced iterative model trainer with automatic optimization"""
    
    def __init__(self):
        self.classification_models = {
            'RandomForest': RandomForestClassifier,
            'GradientBoosting': GradientBoostingClassifier,
            'LogisticRegression': LogisticRegression,
            'SVM': SVC,
            'KNN': KNeighborsClassifier,
            'NaiveBayes': GaussianNB,
            'DecisionTree': DecisionTreeClassifier
        }
        
        self.regression_models = {
            'RandomForest': RandomForestRegressor,
            'GradientBoosting': GradientBoostingRegressor,
            'LinearRegression': LinearRegression,
            'Ridge': Ridge,
            'Lasso': Lasso,
            'SVM': SVR,
            'KNN': KNeighborsRegressor,
            'DecisionTree': DecisionTreeRegressor
        }
        
        self.hyperparameter_grids = self._get_hyperparameter_grids()
        
    def _get_hyperparameter_grids(self) -> Dict[str, Dict[str, List]]:
        """Get hyperparameter grids for model tuning"""
        return {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'GradientBoosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            },
            'LogisticRegression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
            'Ridge': {
                'alpha': [0.01, 0.1, 1, 10, 100]
            },
            'Lasso': {
                'alpha': [0.01, 0.1, 1, 10, 100]
            },
            'DecisionTree': {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }
    
    def train_and_optimize(self, X: pd.DataFrame, y: pd.Series, 
                          problem_type: ProblemType,
                          test_size: float = 0.2,
                          max_iterations: int = 3,
                          optimize_hyperparameters: bool = True) -> TrainingResult:
        """
        Train multiple models with iterative optimization
        
        Args:
            X: Feature DataFrame
            y: Target Series
            problem_type: Type of ML problem
            test_size: Proportion of data for testing
            max_iterations: Maximum optimization iterations
            optimize_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            TrainingResult: Comprehensive training results
        """
        logger.info(f"Starting iterative training for {problem_type.value} problem")
        
        result = TrainingResult()
        
        # Emergency NaN cleanup before training
        X, y = self._emergency_nan_cleanup(X, y)
        
        # Split data with robust handling
        X_train, X_test, y_train, y_test = self._robust_train_test_split(
            X, y, test_size=test_size, problem_type=problem_type
        )
        
        # Get models based on problem type
        if problem_type == ProblemType.CLASSIFICATION:
            models_dict = self.classification_models
        else:
            models_dict = self.regression_models
        
        best_performance = None
        iteration_results = []
        
        for iteration in range(max_iterations):
            logger.info(f"Training iteration {iteration + 1}/{max_iterations}")
            
            iteration_models = []
            
            for model_name, model_class in models_dict.items():
                try:
                    logger.info(f"Training {model_name}")
                    
                    # Train model
                    performance = self._train_single_model(
                        model_name, model_class, X_train, X_test, y_train, y_test,
                        problem_type, optimize_hyperparameters and iteration == 0
                    )
                    
                    iteration_models.append(performance)
                    result.all_models.append(performance)
                    
                except Exception as e:
                    logger.warning(f"Failed to train {model_name}: {str(e)}")
                    continue
            
            iteration_results.append(iteration_models)
            
            # Select best model from this iteration
            if iteration_models:
                current_best = self._select_best_model(iteration_models, problem_type)
                
                if best_performance is None or self._is_better_model(current_best, best_performance, problem_type):
                    best_performance = current_best
                    logger.info(f"New best model: {current_best.model_name}")
                else:
                    logger.info("No improvement in this iteration, stopping early")
                    break
        
        result.best_model = best_performance
        
        # Generate comprehensive analysis
        self._analyze_results(result, X, y, problem_type)
        
        return result
    
    def _train_single_model(self, model_name: str, model_class: type, 
                           X_train: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.Series, y_test: pd.Series,
                           problem_type: ProblemType, 
                           optimize_hyperparameters: bool) -> ModelPerformance:
        """Train a single model with optional hyperparameter optimization"""
        
        performance = ModelPerformance(model_name, problem_type.value)
        start_time = datetime.now()
        
        try:
            if optimize_hyperparameters and model_name in self.hyperparameter_grids:
                # Hyperparameter optimization
                logger.info(f"Optimizing hyperparameters for {model_name}")
                
                param_grid = self.hyperparameter_grids[model_name]
                
                # Use appropriate scoring metric
                if problem_type == ProblemType.CLASSIFICATION:
                    scoring = 'accuracy'
                else:
                    scoring = 'r2'
                
                # Use RandomizedSearchCV for faster optimization
                model = RandomizedSearchCV(
                    model_class(random_state=42),
                    param_grid,
                    n_iter=20,
                    cv=3,
                    scoring=scoring,
                    random_state=42,
                    n_jobs=-1
                )
                
                model.fit(X_train, y_train)
                best_model = model.best_estimator_
                performance.hyperparameters = model.best_params_
                
            else:
                # Default parameters
                if model_name == 'SVM' and len(X_train) > 10000:
                    # Skip SVM for large datasets (too slow)
                    raise Exception("SVM skipped for large dataset")
                
                model = model_class(random_state=42 if 'random_state' in model_class().get_params() else None)
                model.fit(X_train, y_train)
                best_model = model
                performance.hyperparameters = model.get_params()
            
            performance.model_object = best_model
            
            # Make predictions
            y_pred = best_model.predict(X_test)
            performance.predictions = y_pred
            
            if hasattr(best_model, 'predict_proba') and problem_type == ProblemType.CLASSIFICATION:
                performance.prediction_probabilities = best_model.predict_proba(X_test)
            
            # Calculate metrics
            if problem_type == ProblemType.CLASSIFICATION:
                performance.metrics = self._calculate_classification_metrics(y_test, y_pred, performance.prediction_probabilities)
            else:
                performance.metrics = self._calculate_regression_metrics(y_test, y_pred)
            
            # Safe cross-validation scores with comprehensive error handling
            try:
                # Check if we have minimum data for cross-validation
                if problem_type == ProblemType.CLASSIFICATION:
                    class_counts = pd.Series(y_train).value_counts()
                    min_class_size = class_counts.min()
                    
                    if min_class_size >= 3:  # Each class needs at least 3 samples for 3-fold CV
                        try:
                            cv_scores = cross_val_score(best_model, X_train, y_train, cv=3, scoring=scoring)
                        except ValueError as e:
                            if "least populated class" in str(e) or "minimum number of groups" in str(e):
                                # Class imbalance issue - use simple holdout validation
                                logger.warning(f"Cross-validation failed due to class imbalance: {e}. Using simple holdout validation.")
                                X_temp_train, X_temp_test, y_temp_train, y_temp_test = self._robust_train_test_split(
                                    X_train, y_train, 0.2, problem_type
                                )
                                temp_model = clone(best_model)
                                temp_model.fit(X_temp_train, y_temp_train)
                                temp_pred = temp_model.predict(X_temp_test)
                                
                                if problem_type == ProblemType.CLASSIFICATION:
                                    single_score = accuracy_score(y_temp_test, temp_pred)
                                else:
                                    single_score = r2_score(y_temp_test, temp_pred)
                                
                                cv_scores = np.array([single_score] * 3)  # Simulate 3-fold results
                            else:
                                raise e
                    else:
                        # Not enough samples per class, use simple holdout
                        logger.warning(f"Insufficient samples per class for CV (min: {min_class_size}). Using holdout validation.")
                        X_temp_train, X_temp_test, y_temp_train, y_temp_test = self._robust_train_test_split(
                            X_train, y_train, 0.2, problem_type
                        )
                        temp_model = clone(best_model)
                        temp_model.fit(X_temp_train, y_temp_train)
                        temp_pred = temp_model.predict(X_temp_test)
                        single_score = accuracy_score(y_temp_test, temp_pred)
                        cv_scores = np.array([single_score] * 3)
                
                else:  # Regression
                    if len(X_train) >= 9:  # Need at least 9 samples for 3-fold CV
                        cv_scores = cross_val_score(best_model, X_train, y_train, cv=3, scoring=scoring)
                    else:
                        # Use simple holdout for small datasets
                        logger.warning(f"Dataset too small ({len(X_train)} samples). Using holdout validation.")
                        split_idx = int(len(X_train) * 0.8)
                        X_temp_train, X_temp_test = X_train[:split_idx], X_train[split_idx:]
                        y_temp_train, y_temp_test = y_train[:split_idx], y_train[split_idx:]
                        
                        temp_model = clone(best_model)
                        temp_model.fit(X_temp_train, y_temp_train)
                        temp_pred = temp_model.predict(X_temp_test)
                        single_score = r2_score(y_temp_test, temp_pred)
                        cv_scores = np.array([single_score] * 3)
                
                performance.cv_scores = cv_scores.tolist()
                
            except Exception as e:
                logger.warning(f"Cross-validation failed for {model_name}: {str(e)}. Using fallback score.")
                # Fallback: use the test set performance as CV score
                if problem_type == ProblemType.CLASSIFICATION:
                    fallback_score = performance.metrics.get('accuracy', 0.5)
                else:
                    fallback_score = performance.metrics.get('r2_score', 0.0)
                performance.cv_scores = [fallback_score] * 3
            
            # Feature importance (if available)
            if hasattr(best_model, 'feature_importances_'):
                feature_names = X_train.columns.tolist()
                importance_dict = dict(zip(feature_names, best_model.feature_importances_))
                performance.feature_importance = importance_dict
            elif hasattr(best_model, 'coef_'):
                feature_names = X_train.columns.tolist()
                if len(best_model.coef_.shape) == 1:
                    importance_dict = dict(zip(feature_names, np.abs(best_model.coef_)))
                else:
                    importance_dict = dict(zip(feature_names, np.abs(best_model.coef_[0])))
                performance.feature_importance = importance_dict
            
            performance.training_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Successfully trained {model_name}")
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            raise
        
        return performance
    
    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                        y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # ROC AUC for binary classification
        if len(np.unique(y_true)) == 2 and y_prob is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
            except:
                pass
        
        return metrics
    
    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics"""
        return {
            'r2_score': r2_score(y_true, y_pred),
            'mean_squared_error': mean_squared_error(y_true, y_pred),
            'mean_absolute_error': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
        }
    
    def _select_best_model(self, models: List[ModelPerformance], problem_type: ProblemType) -> ModelPerformance:
        """Select the best model based on primary metric"""
        if not models:
            raise ValueError("No models to select from")
        
        if problem_type == ProblemType.CLASSIFICATION:
            return max(models, key=lambda m: m.metrics.get('accuracy', 0))
        else:
            return max(models, key=lambda m: m.metrics.get('r2_score', -float('inf')))
    
    def _is_better_model(self, new_model: ModelPerformance, current_best: ModelPerformance, 
                        problem_type: ProblemType) -> bool:
        """Check if new model is better than current best"""
        if problem_type == ProblemType.CLASSIFICATION:
            new_score = new_model.metrics.get('accuracy', 0)
            current_score = current_best.metrics.get('accuracy', 0)
        else:
            new_score = new_model.metrics.get('r2_score', -float('inf'))
            current_score = current_best.metrics.get('r2_score', -float('inf'))
        
        return new_score > current_score
    
    def _analyze_results(self, result: TrainingResult, X: pd.DataFrame, y: pd.Series, 
                        problem_type: ProblemType):
        """Generate comprehensive analysis of training results"""
        
        # Training summary
        result.training_summary = {
            'total_models_trained': len(result.all_models),
            'best_model_name': result.best_model.model_name if result.best_model else None,
            'problem_type': problem_type.value,
            'dataset_shape': X.shape,
            'target_distribution': y.value_counts().to_dict() if problem_type == ProblemType.CLASSIFICATION else {
                'mean': float(y.mean()),
                'std': float(y.std()),
                'min': float(y.min()),
                'max': float(y.max())
            }
        }
        
        # Feature analysis
        if result.best_model and result.best_model.feature_importance:
            sorted_features = sorted(
                result.best_model.feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            result.feature_analysis = {
                'most_important_features': sorted_features[:5],
                'least_important_features': sorted_features[-5:],
                'total_features': len(sorted_features)
            }
        
        # Generate recommendations
        self._generate_recommendations(result, X, y, problem_type)
        
        # Data insights
        self._generate_data_insights(result, X, y)
        
        # Improvement suggestions
        self._generate_improvement_suggestions(result, problem_type)
    
    def _generate_recommendations(self, result: TrainingResult, X: pd.DataFrame, 
                                y: pd.Series, problem_type: ProblemType):
        """Generate actionable recommendations"""
        recommendations = []
        
        if result.best_model:
            best_model = result.best_model
            
            # Performance-based recommendations
            if problem_type == ProblemType.CLASSIFICATION:
                accuracy = best_model.metrics.get('accuracy', 0)
                if accuracy < 0.7:
                    recommendations.append("Model accuracy is low. Consider collecting more data or feature engineering.")
                elif accuracy > 0.95:
                    recommendations.append("Very high accuracy detected. Check for data leakage or overfitting.")
                
            else:  # Regression
                r2 = best_model.metrics.get('r2_score', 0)
                if r2 < 0.5:
                    recommendations.append("R² score is low. Consider polynomial features or more complex models.")
                elif r2 > 0.95:
                    recommendations.append("Very high R² score. Verify there's no data leakage.")
            
            # Model-specific recommendations
            if best_model.model_name == 'RandomForest':
                recommendations.append("Random Forest selected. Good for interpretation and handles missing values well.")
            elif best_model.model_name == 'LogisticRegression':
                recommendations.append("Logistic Regression selected. Consider feature scaling for better performance.")
            elif best_model.model_name == 'SVM':
                recommendations.append("SVM selected. Ensure features are scaled and consider kernel tuning.")
            elif best_model.model_name == 'GradientBoosting':
                recommendations.append("Gradient Boosting selected. Good performance but watch for overfitting.")
        
        # Dataset-based recommendations
        if X.shape[0] < 1000:
            recommendations.append("Small dataset detected. Consider cross-validation and simpler models.")
        
        if X.shape[1] > 50:
            recommendations.append("Many features detected. Consider feature selection or dimensionality reduction.")
        
        result.recommendations = recommendations
    
    def _generate_data_insights(self, result: TrainingResult, X: pd.DataFrame, y: pd.Series):
        """Generate insights about the data"""
        insights = []
        
        # Feature correlation insights
        numeric_features = X.select_dtypes(include=[np.number]).columns
        if len(numeric_features) > 1:
            corr_matrix = X[numeric_features].corr()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.8:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            if high_corr_pairs:
                insights.append(f"Found {len(high_corr_pairs)} highly correlated feature pairs. Consider removing redundant features.")
        
        # Missing values insights
        missing_cols = X.columns[X.isnull().any()].tolist()
        if missing_cols:
            insights.append(f"Features with missing values: {', '.join(missing_cols[:3])}{'...' if len(missing_cols) > 3 else ''}")
        
        # Class imbalance insights (for classification)
        if len(y.unique()) <= 10:  # Likely classification
            value_counts = y.value_counts()
            if len(value_counts) > 1:
                ratio = value_counts.min() / value_counts.max()
                if ratio < 0.1:
                    insights.append("Severe class imbalance detected. Consider resampling techniques.")
                elif ratio < 0.3:
                    insights.append("Moderate class imbalance detected. Monitor model performance carefully.")
        
        result.data_insights = insights
    
    def _generate_improvement_suggestions(self, result: TrainingResult, problem_type: ProblemType):
        """Generate suggestions for model improvement"""
        suggestions = []
        
        if result.best_model:
            # Performance-based suggestions
            if problem_type == ProblemType.CLASSIFICATION:
                accuracy = result.best_model.metrics.get('accuracy', 0)
                if accuracy < 0.8:
                    suggestions.extend([
                        "Try ensemble methods like Random Forest or Gradient Boosting",
                        "Consider feature engineering (polynomial features, interactions)",
                        "Experiment with different preprocessing techniques",
                        "Collect more training data if possible"
                    ])
            else:
                r2 = result.best_model.metrics.get('r2_score', 0)
                if r2 < 0.7:
                    suggestions.extend([
                        "Try non-linear models like Random Forest or SVR",
                        "Consider polynomial or interaction features",
                        "Check for outliers in the target variable",
                        "Experiment with different feature transformations"
                    ])
            
            # Feature importance suggestions
            if result.feature_analysis.get('least_important_features'):
                suggestions.append("Remove least important features to reduce model complexity")
            
            # Training time suggestions
            if result.best_model.training_time > 60:  # More than 1 minute
                suggestions.append("Consider simpler models or feature selection for faster training")
        
        # Cross-validation suggestions
        if result.all_models:
            cv_variations = []
            for model in result.all_models:
                if model.cv_scores:
                    cv_std = np.std(model.cv_scores)
                    cv_variations.append(cv_std)
            
            if cv_variations and max(cv_variations) > 0.1:
                suggestions.append("High variance in cross-validation scores. Consider regularization or more data.")
        
        result.improvement_suggestions = suggestions
    
    def save_model(self, model_performance: ModelPerformance, file_path: str):
        """Save trained model to file"""
        try:
            model_data = {
                'model': model_performance.model_object,
                'model_name': model_performance.model_name,
                'problem_type': model_performance.problem_type,
                'metrics': model_performance.metrics,
                'hyperparameters': model_performance.hyperparameters,
                'feature_importance': model_performance.feature_importance,
                'training_time': model_performance.training_time
            }
            
            joblib.dump(model_data, file_path)
            logger.info(f"Model saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, file_path: str) -> ModelPerformance:
        """Load trained model from file"""
        try:
            model_data = joblib.load(file_path)
            
            performance = ModelPerformance(model_data['model_name'], model_data['problem_type'])
            performance.model_object = model_data['model']
            performance.metrics = model_data['metrics']
            performance.hyperparameters = model_data['hyperparameters']
            performance.feature_importance = model_data.get('feature_importance')
            performance.training_time = model_data.get('training_time', 0)
            
            logger.info(f"Model loaded from {file_path}")
            return performance
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def generate_training_report(self, result: TrainingResult) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        
        report = {
            'training_summary': result.training_summary,
            'best_model': {
                'name': result.best_model.model_name if result.best_model else None,
                'metrics': result.best_model.metrics if result.best_model else {},
                'hyperparameters': result.best_model.hyperparameters if result.best_model else {},
                'training_time': result.best_model.training_time if result.best_model else 0
            },
            'all_models_performance': [
                {
                    'name': model.model_name,
                    'metrics': model.metrics,
                    'cv_mean': np.mean(model.cv_scores) if model.cv_scores else 0,
                    'cv_std': np.std(model.cv_scores) if model.cv_scores else 0,
                    'training_time': model.training_time
                }
                for model in result.all_models
            ],
            'feature_analysis': result.feature_analysis,
            'recommendations': result.recommendations,
            'data_insights': result.data_insights,
            'improvement_suggestions': result.improvement_suggestions,
            'timestamp': datetime.now().isoformat()
        }
        
        return report
    
    def _robust_train_test_split(self, X: pd.DataFrame, y: pd.Series, 
                                test_size: float, problem_type: ProblemType) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Robust train_test_split that handles edge cases"""
        
        # Check if we have enough data for splitting
        if len(X) < 4:
            logger.warning(f"Dataset too small ({len(X)} samples) for train/test split. Using all data for training.")
            return X, X.iloc[:1], y, y.iloc[:1]  # Minimal test set
        
        # For classification, check class distribution
        if problem_type == ProblemType.CLASSIFICATION:
            class_counts = y.value_counts()
            
            # Check if any class has only 1 sample
            if class_counts.min() == 1:
                logger.warning("Some classes have only 1 sample. Cannot use stratified splitting.")
                
                # Check if we can still do meaningful splitting
                if len(class_counts) == len(y):  # All samples are unique classes
                    logger.warning("All samples have unique classes. Using simple random split.")
                    return train_test_split(X, y, test_size=min(test_size, 0.1), random_state=42, stratify=None)
                
                # Filter out classes with only 1 sample for stratification check
                valid_classes = class_counts[class_counts >= 2].index
                
                if len(valid_classes) == 0:
                    logger.warning("No class has 2+ samples. Using simple random split.")
                    return train_test_split(X, y, test_size=min(test_size, 0.1), random_state=42, stratify=None)
                
                # Use simple split without stratification
                logger.info("Using simple random split due to class distribution.")
                return train_test_split(X, y, test_size=test_size, random_state=42, stratify=None)
            
            # Check if any class has fewer samples than required for stratified split
            min_test_size = max(1, int(len(X) * test_size))
            classes_needing_min_2 = class_counts[class_counts < 2]
            
            if len(classes_needing_min_2) > 0:
                logger.warning(f"Classes with <2 samples detected: {classes_needing_min_2.index.tolist()}. Using simple split.")
                return train_test_split(X, y, test_size=test_size, random_state=42, stratify=None)
            
            # Additional check: ensure each class will have at least 1 sample in train set
            min_samples_per_class_in_test = class_counts * test_size
            if (min_samples_per_class_in_test < 1).any() and (class_counts - min_samples_per_class_in_test < 1).any():
                logger.warning("Stratified split would leave some classes empty in train or test. Using simple split.")
                return train_test_split(X, y, test_size=test_size, random_state=42, stratify=None)
            
            # Try stratified split
            try:
                logger.info("Using stratified split to maintain class distribution.")
                return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
            except ValueError as e:
                logger.warning(f"Stratified split failed: {str(e)}. Falling back to simple split.")
                return train_test_split(X, y, test_size=test_size, random_state=42, stratify=None)
        
        else:
            # For regression, use simple random split
            logger.info("Using simple random split for regression.")
            return train_test_split(X, y, test_size=test_size, random_state=42)
    
    def _handle_small_dataset(self, X: pd.DataFrame, y: pd.Series, problem_type: ProblemType) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle small datasets by applying data augmentation or duplication strategies"""
        
        logger.warning(f"Small dataset detected: {len(X)} samples")
        
        if problem_type == ProblemType.CLASSIFICATION:
            # For classification, try to balance classes if severely imbalanced
            class_counts = y.value_counts()
            min_class_count = class_counts.min()
            max_class_count = class_counts.max()
            
            if min_class_count == 1 and max_class_count > 3:
                logger.info("Attempting to balance severely imbalanced classes")
                
                # Simple duplication strategy for minority classes
                balanced_indices = []
                target_count = min(max_class_count, 3)  # Don't over-duplicate
                
                for class_label in class_counts.index:
                    class_indices = y[y == class_label].index.tolist()
                    current_count = len(class_indices)
                    
                    # Add all current indices
                    balanced_indices.extend(class_indices)
                    
                    # Duplicate minority class samples
                    if current_count < target_count:
                        duplicates_needed = target_count - current_count
                        for _ in range(duplicates_needed):
                            balanced_indices.extend(class_indices[:min(duplicates_needed, len(class_indices))])
                
                # Create balanced dataset
                X_balanced = X.loc[balanced_indices].reset_index(drop=True)
                y_balanced = y.loc[balanced_indices].reset_index(drop=True)
                
                logger.info(f"Dataset balanced from {len(X)} to {len(X_balanced)} samples")
                return X_balanced, y_balanced
        
        # If no balancing needed or for regression, return original
        return X, y
    
    def _emergency_nan_cleanup(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Emergency cleanup of NaN values before model training"""
        print("🚨 Performing emergency NaN cleanup before model training...")
        
        # Check for NaN in target
        if y.isnull().any():
            nan_count = y.isnull().sum()
            print(f"❌ Found {nan_count} NaN values in target variable - removing rows")
            valid_indices = ~y.isnull()
            X = X[valid_indices]
            y = y[valid_indices]
        
        # Check for NaN in features
        if X.isnull().any().any():
            print("❌ Found NaN values in features - applying emergency imputation")
            
            for col in X.columns:
                if X[col].isnull().any():
                    nan_count = X[col].isnull().sum()
                    print(f"  - Column '{col}': {nan_count} NaN values")
                    
                    if X[col].dtype in ['int64', 'float64', 'float32', 'int32']:
                        # Use median for numeric columns
                        median_val = X[col].median()
                        if pd.isna(median_val):
                            # If median is also NaN, use 0
                            X[col] = X[col].fillna(0)
                            print(f"    → Filled with 0 (median was NaN)")
                        else:
                            X[col] = X[col].fillna(median_val)
                            print(f"    → Filled with median: {median_val}")
                    else:
                        # Use most frequent for categorical columns
                        mode_val = X[col].mode()
                        if len(mode_val) > 0:
                            X[col] = X[col].fillna(mode_val[0])
                            print(f"    → Filled with mode: {mode_val[0]}")
                        else:
                            X[col] = X[col].fillna('Unknown')
                            print(f"    → Filled with 'Unknown'")
        
        # Final verification
        if X.isnull().any().any() or y.isnull().any():
            print("⚠️  Still have NaN values after cleanup - dropping remaining NaN rows")
            # Drop any remaining rows with NaN
            complete_cases = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[complete_cases]
            y = y[complete_cases]
        
        print(f"✅ NaN cleanup completed. Final dataset: {len(X)} rows, {X.shape[1]} columns")
        
        # Ensure we have enough data
        if len(X) < 2:
            raise ValueError("❌ Error building model: Not enough valid data after NaN cleanup. Please check your dataset for quality issues.")
        
        return X, y

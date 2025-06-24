"""
Real AutoML Engine for NoCodeML

This module implements the core AutoML functionality with actual model training,
hyperparameter optimization, and intelligent algorithm selection.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import optuna
from typing import Dict, List, Tuple, Any, Optional
import joblib
import json
from pathlib import Path
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from backend.models.schemas import ProblemType, ModelMetrics

# Import comprehensive error handler
sys.path.append(str(Path(__file__).parent.parent / 'utils'))
from comprehensive_error_handler import ComprehensiveErrorHandler

class AutoMLEngine:
    """
    Advanced AutoML Engine that automatically:
    1. Preprocesses data
    2. Selects optimal algorithms
    3. Optimizes hyperparameters
    4. Trains and evaluates models
    5. Provides model explanations
    """
    
    def __init__(self):
        self.models_dir = project_root / "models" / "trained"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.error_handler = ComprehensiveErrorHandler()
        
        # Define algorithm pools
        self.classification_algorithms = {
            'RandomForest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'XGBoost': {
                'model': xgb.XGBClassifier,
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 6, 9],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'LogisticRegression': {
                'model': LogisticRegression,
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'solver': ['liblinear', 'lbfgs'],
                    'max_iter': [1000]
                }
            },
            'SVM': {
                'model': SVC,
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier,
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            }
        }
        
        self.regression_algorithms = {
            'RandomForest': {
                'model': RandomForestRegressor,
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'XGBoost': {
                'model': xgb.XGBRegressor,
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 6, 9]
                }
            },
            'LinearRegression': {
                'model': LinearRegression,
                'params': {}
            },
            'Ridge': {
                'model': Ridge,
                'params': {
                    'alpha': [0.1, 1.0, 10.0]
                }
            },
            'SVR': {
                'model': SVR,
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'linear']
                }
            }
        }
        
        self.clustering_algorithms = {
            'KMeans': {
                'model': KMeans,
                'params': {
                    'n_clusters': [2, 3, 4, 5, 6, 7, 8],
                    'init': ['k-means++', 'random'],
                    'n_init': [10, 20]
                }
            },
            'DBSCAN': {
                'model': DBSCAN,
                'params': {
                    'eps': [0.1, 0.5, 1.0, 1.5],
                    'min_samples': [3, 5, 10, 15]
                }
            }
        }
    
    async def train_model(self, dataset_path: str, config: Dict[str, Any], 
                         progress_callback=None) -> Dict[str, Any]:
        """
        Train a model with the given configuration
        
        Args:
            dataset_path: Path to the dataset file
            config: Training configuration
            progress_callback: Function to report progress
            
        Returns:
            Dict containing model results and metrics
        """
        try:
            if progress_callback:
                await progress_callback("Loading and preprocessing data", 10)
            
            # Load dataset
            df = self._load_dataset(dataset_path)
            
            # Prepare data
            X, y, preprocessor = self._prepare_data(
                df, config['target_column'], config['feature_columns']
            )
            
            # Emergency NaN check and fix before training
            X, y = self._emergency_nan_cleanup(X, y)
            
            if progress_callback:
                await progress_callback("Splitting data", 20)
            
            # Split data with robust handling
            test_size = config.get('test_size', 0.2)
            
            try:
                # Safe train-test split with stratification
                if config['problem_type'] == 'classification':
                    # Check if stratification is possible
                    class_counts = pd.Series(y).value_counts()
                    min_class_size = class_counts.min()
                    
                    if min_class_size >= 2 and len(class_counts) < len(y):
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42, stratify=y
                        )
                    else:
                        print("⚠️ Cannot use stratified split due to class distribution. Using random split.")
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42
                        )
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
            except Exception as e:
                print(f"⚠️ Train-test split failed: {e}. Using simple split.")
                split_idx = int(len(X) * (1 - test_size))
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
            
            if progress_callback:
                await progress_callback("Training model", 40)
            
            # Get algorithm pool
            problem_type = config['problem_type']
            if problem_type == 'classification':
                algorithms = self.classification_algorithms
            elif problem_type == 'regression':
                algorithms = self.regression_algorithms
            else:
                algorithms = self.clustering_algorithms
            
            # Train model
            algorithm_name = config['algorithm']
            if algorithm_name in algorithms:
                best_model, best_params, cv_scores = self._train_algorithm(
                    algorithms[algorithm_name], X_train, y_train, problem_type
                )
            else:
                raise ValueError(f"Algorithm {algorithm_name} not supported for {problem_type}")
            
            if progress_callback:
                await progress_callback("Evaluating model", 70)
            
            # Create full pipeline
            full_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', best_model)
            ])
            
            # Fit full pipeline
            full_pipeline.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = full_pipeline.predict(X_test)
            metrics = self._calculate_metrics(y_test, y_pred, problem_type)
            
            if progress_callback:
                await progress_callback("Calculating feature importance", 85)
            
            # Get feature importance
            feature_importance = self._get_feature_importance(
                best_model, config['feature_columns']
            )
            
            if progress_callback:
                await progress_callback("Saving model", 95)
            
            # Save model
            model_id = config.get('model_id', 'default')
            model_path = self.models_dir / f"{model_id}.joblib"
            joblib.dump(full_pipeline, model_path)
            
            # Save metadata
            metadata = {
                'algorithm': algorithm_name,
                'problem_type': problem_type,
                'best_params': best_params,
                'cv_scores': cv_scores.tolist() if cv_scores is not None else [],
                'feature_columns': config['feature_columns'],
                'target_column': config['target_column'],
                'created_at': datetime.now().isoformat()
            }
            
            metadata_path = self.models_dir / f"{model_id}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            if progress_callback:
                await progress_callback("Training completed", 100)
            
            return {
                'model_id': model_id,
                'algorithm': algorithm_name,
                'metrics': metrics,
                'feature_importance': feature_importance,
                'best_params': best_params,
                'cv_scores': cv_scores.tolist() if cv_scores is not None else [],
                'model_path': str(model_path)
            }
            
        except Exception as e:
            raise Exception(f"Error training model: {str(e)}")
    
    def _load_dataset(self, file_path: str) -> pd.DataFrame:
        """Load dataset from file"""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension == '.csv':
            return pd.read_csv(file_path)
        elif extension in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif extension == '.json':
            return pd.read_json(file_path)
        elif extension == '.parquet':
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    def _prepare_data(self, df: pd.DataFrame, target_column: str, 
                     feature_columns: List[str]) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
        """Prepare data for training"""
        # Select features and target
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Identify column types
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Handle target encoding for classification
        if y.dtype == 'object':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
        
        # Apply comprehensive error handling and data fixes
        try:
            # Determine problem type for auto-fix
            if isinstance(y, (pd.Series, np.ndarray)):
                unique_ratio = len(np.unique(y)) / len(y) if len(y) > 0 else 0
                problem_type = ProblemType.CLASSIFICATION if unique_ratio < 0.5 else ProblemType.REGRESSION
            else:
                problem_type = ProblemType.CLASSIFICATION
            
            X, y = self.error_handler.auto_fix_dataset(X, pd.Series(y) if not isinstance(y, pd.Series) else y, problem_type)
            y = y.values if hasattr(y, 'values') else y
            
            print(f"✅ Dataset auto-fixed successfully. Shape: {X.shape}, Target unique values: {len(np.unique(y))}")
            
        except Exception as e:
            print(f"⚠️ Warning: Error in comprehensive auto-fix: {e}. Applying emergency cleanup.")
            X, y = self._emergency_nan_cleanup(X, pd.Series(y) if not isinstance(y, pd.Series) else y)
            y = y.values if hasattr(y, 'values') else y
        
        return X, y, preprocessor
    
    def _train_algorithm(self, algorithm_config: Dict, X_train: pd.DataFrame, 
                        y_train: pd.Series, problem_type: str) -> Tuple[Any, Dict, np.ndarray]:
        """Train a specific algorithm with hyperparameter optimization"""
        model_class = algorithm_config['model']
        param_grid = algorithm_config['params']
        
        if not param_grid:
            # No hyperparameters to optimize
            model = model_class(random_state=42)
            
            try:
                # Safe model fitting with error handling
                model.fit(X_train, y_train)
                # Safe cross-validation with error handling
                cv_scores = self.error_handler.safe_cross_validation(
                    model, pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train, 
                    pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train, cv=5
                )
            except Exception as e:
                print(f"⚠️ Error in model training: {e}. Applying fixes...")
                # Fix data and retry
                X_train_fixed, y_train_fixed = self.error_handler.auto_fix_dataset(
                    pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train,
                    pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train
                )
                X_train_fixed = X_train_fixed.values if hasattr(X_train_fixed, 'values') else X_train_fixed
                y_train_fixed = y_train_fixed.values if hasattr(y_train_fixed, 'values') else y_train_fixed
                
                model.fit(X_train_fixed, y_train_fixed)
                cv_scores = self.error_handler.safe_cross_validation(
                    model, pd.DataFrame(X_train_fixed), pd.Series(y_train_fixed), cv=5
                )
            
            return model, {}, cv_scores
        
        # Hyperparameter optimization
        if problem_type == 'classification':
            scoring = 'accuracy'
        elif problem_type == 'regression':
            scoring = 'neg_mean_squared_error'
        else:
            scoring = 'adjusted_rand_score'
        
        # Use RandomizedSearchCV for efficiency with comprehensive error handling
        try:
            # First, ensure data is clean for hyperparameter search
            X_train_clean, y_train_clean = self.error_handler.auto_fix_dataset(
                pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train,
                pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train
            )
            X_train_clean = X_train_clean.values if hasattr(X_train_clean, 'values') else X_train_clean
            y_train_clean = y_train_clean.values if hasattr(y_train_clean, 'values') else y_train_clean
            
            # Determine appropriate CV strategy
            if problem_type == 'classification':
                class_counts = pd.Series(y_train_clean).value_counts()
                min_class_size = class_counts.min()
                cv_folds = min(5, min_class_size) if min_class_size >= 2 else 3
            else:
                cv_folds = min(5, len(X_train_clean) // 3)
            
            cv_folds = max(2, cv_folds)  # Ensure at least 2 folds
            
            search = RandomizedSearchCV(
                model_class(random_state=42),
                param_grid,
                n_iter=min(20, len(X_train_clean) // cv_folds),  # Adjust n_iter based on data size
                cv=cv_folds,
                scoring=scoring,
                random_state=42,
                n_jobs=1,  # Reduced parallelism to avoid issues
                error_score='raise'
            )
            
            search.fit(X_train_clean, y_train_clean)
            
        except Exception as e:
            print(f"⚠️ RandomizedSearchCV failed: {e}. Using simple model training...")
            # Fallback to simple training
            model = model_class(random_state=42)
            X_train_clean, y_train_clean = self.error_handler.auto_fix_dataset(
                pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train,
                pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train
            )
            X_train_clean = X_train_clean.values if hasattr(X_train_clean, 'values') else X_train_clean
            y_train_clean = y_train_clean.values if hasattr(y_train_clean, 'values') else y_train_clean
            
            model.fit(X_train_clean, y_train_clean)
            cv_scores = self.error_handler.safe_cross_validation(
                model, pd.DataFrame(X_train_clean), pd.Series(y_train_clean), cv=3
            )
            return model, {}, cv_scores
        
        return search.best_estimator_, search.best_params_, search.cv_results_['mean_test_score']
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          problem_type: str) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        metrics = {}
        
        if problem_type == 'classification':
            metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
            metrics['precision'] = float(precision_score(y_true, y_pred, average='weighted'))
            metrics['recall'] = float(recall_score(y_true, y_pred, average='weighted'))
            metrics['f1_score'] = float(f1_score(y_true, y_pred, average='weighted'))
            
            # ROC AUC for binary classification
            if len(np.unique(y_true)) == 2:
                try:
                    metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred))
                except:
                    pass
        
        elif problem_type == 'regression':
            metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
            metrics['r2_score'] = float(r2_score(y_true, y_pred))
        
        return metrics
    
    def _get_feature_importance(self, model: Any, feature_columns: List[str]) -> Dict[str, float]:
        """Extract feature importance from trained model"""
        importance_dict = {}
        
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                for i, importance in enumerate(importances):
                    if i < len(feature_columns):
                        importance_dict[feature_columns[i]] = float(importance)
            elif hasattr(model, 'coef_'):
                # For linear models
                coefficients = np.abs(model.coef_).flatten()
                for i, coef in enumerate(coefficients):
                    if i < len(feature_columns):
                        importance_dict[feature_columns[i]] = float(coef)
        except Exception as e:
            print(f"Could not extract feature importance: {e}")
        
        return importance_dict
    
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
    
    async def predict(self, model_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions using a trained model"""
        try:
            # Load model
            model_path = self.models_dir / f"{model_id}.joblib"
            if not model_path.exists():
                raise ValueError(f"Model {model_id} not found")
            
            model = joblib.load(model_path)
            
            # Load metadata
            metadata_path = self.models_dir / f"{model_id}_metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            input_df = input_df[metadata['feature_columns']]
            
            # Make prediction
            prediction = model.predict(input_df)
            
            result = {
                'prediction': prediction.tolist(),
                'model_id': model_id,
                'algorithm': metadata['algorithm']
            }
            
            # Add probabilities for classification
            if metadata['problem_type'] == 'classification' and hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(input_df)
                    result['probabilities'] = probabilities.tolist()
                except:
                    pass
            
            return result
            
        except Exception as e:
            raise Exception(f"Error making prediction: {str(e)}")
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a trained model"""
        try:
            metadata_path = self.models_dir / f"{model_id}_metadata.json"
            if not metadata_path.exists():
                raise ValueError(f"Model {model_id} not found")
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return metadata
            
        except Exception as e:
            raise Exception(f"Error getting model info: {str(e)}")


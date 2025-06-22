"""
Intelligent Recommendation Engine for NoCodeML

This module analyzes datasets and user requirements to recommend optimal
ML algorithms, preprocessing steps, and hyperparameters.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import sys
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from backend.models.schemas import (
    ProblemType, DataType, ModelRecommendation, AnalysisResult,
    RecommendationConfig
)
from core.analyzers.dataset_analyzer import DatasetAnalyzer

class RecommendationEngine:
    """
    Intelligent recommendation engine that suggests optimal ML approaches
    based on dataset characteristics and problem requirements.
    """
    
    def __init__(self):
        self.analyzer = DatasetAnalyzer()
        
        # Algorithm performance profiles based on data characteristics
        self.algorithm_profiles = {
            'classification': {
                'RandomForest': {
                    'best_for': ['mixed_features', 'medium_data', 'interpretability'],
                    'data_size_min': 100,
                    'data_size_optimal': 1000,
                    'handles_missing': True,
                    'handles_categorical': True,
                    'confidence_base': 0.85
                },
                'XGBoost': {
                    'best_for': ['structured_data', 'competitions', 'performance'],
                    'data_size_min': 500,
                    'data_size_optimal': 5000,
                    'handles_missing': True,
                    'handles_categorical': True,
                    'confidence_base': 0.90
                },
                'LogisticRegression': {
                    'best_for': ['linear_relationships', 'fast_training', 'interpretability'],
                    'data_size_min': 50,
                    'data_size_optimal': 1000,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'confidence_base': 0.75
                },
                'SVM': {
                    'best_for': ['high_dimensional', 'non_linear', 'small_data'],
                    'data_size_min': 100,
                    'data_size_optimal': 1000,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'confidence_base': 0.80
                },
                'NeuralNetwork': {
                    'best_for': ['complex_patterns', 'large_data', 'non_linear'],
                    'data_size_min': 1000,
                    'data_size_optimal': 10000,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'confidence_base': 0.85
                }
            },
            'regression': {
                'RandomForest': {
                    'best_for': ['mixed_features', 'non_linear', 'robust'],
                    'data_size_min': 100,
                    'data_size_optimal': 1000,
                    'handles_missing': True,
                    'handles_categorical': True,
                    'confidence_base': 0.85
                },
                'XGBoost': {
                    'best_for': ['structured_data', 'competitions', 'performance'],
                    'data_size_min': 500,
                    'data_size_optimal': 5000,
                    'handles_missing': True,
                    'handles_categorical': True,
                    'confidence_base': 0.90
                },
                'LinearRegression': {
                    'best_for': ['linear_relationships', 'interpretability', 'fast'],
                    'data_size_min': 50,
                    'data_size_optimal': 500,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'confidence_base': 0.70
                },
                'Ridge': {
                    'best_for': ['regularization', 'multicollinearity', 'stable'],
                    'data_size_min': 50,
                    'data_size_optimal': 1000,
                    'handles_missing': False,
                    'handles_categorical': False,
                    'confidence_base': 0.75
                }
            }
        }
    
    async def get_recommendations(self, dataset_id: str, config: RecommendationConfig) -> AnalysisResult:
        """
        Generate comprehensive ML recommendations for a dataset
        
        Args:
            dataset_id: ID of the uploaded dataset
            config: User configuration and preferences
            
        Returns:
            AnalysisResult: Complete analysis with model recommendations
        """
        try:
            # Find dataset file
            dataset_files = list((project_root / "data" / "uploads").glob(f"{dataset_id}_*"))
            if not dataset_files:
                raise ValueError(f"Dataset {dataset_id} not found")
            
            dataset_path = str(dataset_files[0])
            
            # Analyze dataset
            dataset_info = await self.analyzer.analyze_dataset(dataset_path)
            df = self.analyzer._load_dataset(dataset_path)
            
            # Get problem-specific recommendations
            recommendations = self._generate_algorithm_recommendations(
                df, dataset_info, config
            )
            
            # Determine preprocessing requirements
            preprocessing_required = self._analyze_preprocessing_needs(df, dataset_info)
            
            # Estimate training time
            estimated_time = self._estimate_training_time(df, recommendations)
            
            return AnalysisResult(
                dataset_id=dataset_id,
                problem_type=config.problem_type,
                target_column=config.target_column,
                feature_columns=config.feature_columns or self._suggest_feature_columns(df, config.target_column),
                recommended_models=recommendations,
                data_preprocessing_required=preprocessing_required,
                estimated_training_time=estimated_time,
                analysis_timestamp=datetime.now()
            )
            
        except Exception as e:
            raise Exception(f"Error generating recommendations: {str(e)}")
    
    def _generate_algorithm_recommendations(self, df: pd.DataFrame, dataset_info, 
                                          config: RecommendationConfig) -> List[ModelRecommendation]:
        """Generate algorithm recommendations based on data characteristics"""
        recommendations = []
        
        # Get algorithms for the problem type
        problem_type = config.problem_type.value
        if problem_type not in self.algorithm_profiles:
            return recommendations
        
        algorithms = self.algorithm_profiles[problem_type]
        
        # Analyze data characteristics
        data_characteristics = self._analyze_data_characteristics(df, dataset_info)
        
        # Score each algorithm
        algorithm_scores = {}
        for algorithm_name, profile in algorithms.items():
            score = self._calculate_algorithm_score(data_characteristics, profile)
            algorithm_scores[algorithm_name] = score
        
        # Sort by score and create recommendations
        sorted_algorithms = sorted(algorithm_scores.items(), key=lambda x: x[1], reverse=True)
        
        for algorithm_name, confidence_score in sorted_algorithms[:3]:  # Top 3 recommendations
            profile = algorithms[algorithm_name]
            
            # Generate hyperparameters
            hyperparameters = self._generate_hyperparameters(algorithm_name, data_characteristics)
            
            # Generate preprocessing steps
            preprocessing_steps = self._generate_preprocessing_steps(
                algorithm_name, data_characteristics, profile
            )
            
            # Estimate performance
            expected_performance = self._estimate_performance(
                algorithm_name, data_characteristics, problem_type
            )
            
            # Generate explanation
            explanation = self._generate_explanation(
                algorithm_name, data_characteristics, profile
            )
            
            # Determine algorithm type
            algorithm_type = self._get_algorithm_type(algorithm_name)
            
            recommendation = ModelRecommendation(
                algorithm=algorithm_name,
                algorithm_type=algorithm_type,
                confidence_score=confidence_score,
                expected_performance=expected_performance,
                hyperparameters=hyperparameters,
                preprocessing_steps=preprocessing_steps,
                explanation=explanation
            )
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _analyze_data_characteristics(self, df: pd.DataFrame, dataset_info) -> Dict[str, Any]:
        """Analyze key characteristics of the dataset"""
        characteristics = {
            'n_samples': len(df),
            'n_features': len(df.columns),
            'missing_ratio': df.isnull().sum().sum() / (df.shape[0] * df.shape[1]),
            'duplicate_ratio': df.duplicated().sum() / len(df),
            'categorical_ratio': len(df.select_dtypes(include=['object']).columns) / len(df.columns),
            'numeric_ratio': len(df.select_dtypes(include=[np.number]).columns) / len(df.columns),
            'data_quality_score': dataset_info.data_quality_score / 100,
            'has_missing_values': df.isnull().any().any(),
            'has_categorical': len(df.select_dtypes(include=['object']).columns) > 0,
            'has_high_cardinality': False,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        # Check for high cardinality categorical variables
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() > len(df) * 0.5:
                characteristics['has_high_cardinality'] = True
                break
        
        return characteristics
    
    def _calculate_algorithm_score(self, data_characteristics: Dict, profile: Dict) -> float:
        """Calculate confidence score for an algorithm based on data characteristics"""
        base_score = profile['confidence_base']
        score = base_score
        
        # Data size considerations
        n_samples = data_characteristics['n_samples']
        if n_samples < profile['data_size_min']:
            score *= 0.7  # Penalize if below minimum
        elif n_samples >= profile['data_size_optimal']:
            score *= 1.1  # Boost if optimal size
        
        # Missing value handling
        if data_characteristics['has_missing_values'] and not profile['handles_missing']:
            score *= 0.8
        
        # Categorical data handling
        if data_characteristics['has_categorical'] and not profile['handles_categorical']:
            score *= 0.85
        
        # Data quality considerations
        score *= data_characteristics['data_quality_score']
        
        # High cardinality penalty for some algorithms
        if data_characteristics['has_high_cardinality'] and profile.get('sensitive_to_cardinality', False):
            score *= 0.9
        
        return min(1.0, max(0.1, score))  # Clamp between 0.1 and 1.0
    
    def _generate_hyperparameters(self, algorithm_name: str, data_characteristics: Dict) -> Dict[str, Any]:
        """Generate optimized hyperparameters based on data characteristics"""
        n_samples = data_characteristics['n_samples']
        n_features = data_characteristics['n_features']
        
        if algorithm_name == 'RandomForest':
            n_estimators = min(200, max(50, n_samples // 10))
            max_depth = min(20, max(5, int(np.log2(n_features)) + 3))
            return {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': 5 if n_samples > 1000 else 2,
                'min_samples_leaf': 2 if n_samples > 1000 else 1
            }
        
        elif algorithm_name == 'XGBoost':
            return {
                'n_estimators': min(200, max(50, n_samples // 20)),
                'learning_rate': 0.1 if n_samples > 1000 else 0.2,
                'max_depth': min(8, max(3, int(np.log2(n_features)))),
                'subsample': 0.9 if n_samples > 1000 else 1.0
            }
        
        elif algorithm_name == 'LogisticRegression':
            return {
                'C': 1.0 if n_samples > 1000 else 10.0,
                'solver': 'lbfgs' if n_features < 100 else 'liblinear',
                'max_iter': 1000
            }
        
        elif algorithm_name == 'SVM':
            return {
                'C': 1.0,
                'kernel': 'rbf' if n_features < 100 else 'linear',
                'gamma': 'scale'
            }
        
        return {}
    
    def _generate_preprocessing_steps(self, algorithm_name: str, data_characteristics: Dict, 
                                    profile: Dict) -> List[str]:
        """Generate required preprocessing steps"""
        steps = []
        
        # Missing value handling
        if data_characteristics['has_missing_values']:
            if not profile['handles_missing']:
                steps.append('Handle Missing Values')
        
        # Categorical encoding
        if data_characteristics['has_categorical']:
            if not profile['handles_categorical']:
                if data_characteristics['has_high_cardinality']:
                    steps.append('Target Encoding')
                else:
                    steps.append('One-Hot Encoding')
        
        # Scaling for algorithms that need it
        if algorithm_name in ['LogisticRegression', 'SVM', 'NeuralNetwork']:
            steps.append('Feature Scaling')
        
        # Feature selection for high-dimensional data
        if data_characteristics['n_features'] > 100:
            steps.append('Feature Selection')
        
        # Outlier detection for sensitive algorithms
        if algorithm_name in ['LogisticRegression', 'SVM'] and data_characteristics['n_samples'] > 1000:
            steps.append('Outlier Detection')
        
        return steps
    
    def _estimate_performance(self, algorithm_name: str, data_characteristics: Dict, 
                            problem_type: str) -> Dict[str, float]:
        """Estimate expected model performance"""
        base_performance = {
            'classification': {
                'RandomForest': {'accuracy': 0.85, 'f1_score': 0.83},
                'XGBoost': {'accuracy': 0.87, 'f1_score': 0.85},
                'LogisticRegression': {'accuracy': 0.78, 'f1_score': 0.76},
                'SVM': {'accuracy': 0.82, 'f1_score': 0.80}
            },
            'regression': {
                'RandomForest': {'r2_score': 0.82, 'rmse': 0.25},
                'XGBoost': {'r2_score': 0.85, 'rmse': 0.22},
                'LinearRegression': {'r2_score': 0.75, 'rmse': 0.30},
                'Ridge': {'r2_score': 0.77, 'rmse': 0.28}
            }
        }
        
        if problem_type in base_performance and algorithm_name in base_performance[problem_type]:
            performance = base_performance[problem_type][algorithm_name].copy()
            
            # Adjust based on data quality
            quality_factor = data_characteristics['data_quality_score']
            for metric in performance:
                performance[metric] *= quality_factor
            
            return performance
        
        return {}
    
    def _generate_explanation(self, algorithm_name: str, data_characteristics: Dict, 
                            profile: Dict) -> str:
        """Generate human-readable explanation for algorithm choice"""
        explanations = {
            'RandomForest': f"Random Forest is recommended because it handles mixed data types well, requires minimal preprocessing, and provides good interpretability. With {data_characteristics['n_samples']} samples, it should perform robustly.",
            
            'XGBoost': f"XGBoost is highly recommended for structured data with {data_characteristics['n_samples']} samples. It often provides state-of-the-art performance and handles missing values automatically.",
            
            'LogisticRegression': f"Logistic Regression is suggested for its simplicity and interpretability. It works well when relationships are approximately linear and provides fast training.",
            
            'SVM': f"Support Vector Machine is recommended for this dataset size ({data_characteristics['n_samples']} samples). It can capture non-linear relationships and works well in high-dimensional spaces.",
            
            'LinearRegression': "Linear Regression is chosen for its simplicity and interpretability. It works best when the relationship between features and target is approximately linear.",
            
            'Ridge': "Ridge Regression is recommended to handle potential multicollinearity in your features while maintaining interpretability."
        }
        
        base_explanation = explanations.get(algorithm_name, f"{algorithm_name} is recommended based on your data characteristics.")
        
        # Add data-specific insights
        if data_characteristics['has_missing_values']:
            if profile['handles_missing']:
                base_explanation += " It naturally handles missing values in your dataset."
            else:
                base_explanation += " Note: Missing values will need to be handled during preprocessing."
        
        if data_characteristics['has_categorical']:
            if profile['handles_categorical']:
                base_explanation += " It can work directly with categorical features."
            else:
                base_explanation += " Categorical features will be encoded during preprocessing."
        
        return base_explanation
    
    def _get_algorithm_type(self, algorithm_name: str) -> str:
        """Determine the algorithm category"""
        algorithm_types = {
            'RandomForest': 'ensemble',
            'XGBoost': 'ensemble',
            'LogisticRegression': 'classical_ml',
            'LinearRegression': 'classical_ml',
            'Ridge': 'classical_ml',
            'SVM': 'classical_ml',
            'NeuralNetwork': 'deep_learning'
        }
        
        return algorithm_types.get(algorithm_name, 'classical_ml')
    
    def _suggest_feature_columns(self, df: pd.DataFrame, target_column: Optional[str]) -> List[str]:
        """Suggest feature columns if not provided"""
        all_columns = df.columns.tolist()
        
        if target_column and target_column in all_columns:
            feature_columns = [col for col in all_columns if col != target_column]
        else:
            feature_columns = all_columns
        
        # Remove columns that are likely not useful as features
        excluded_patterns = ['id', 'index', 'key', 'name', 'description']
        feature_columns = [
            col for col in feature_columns 
            if not any(pattern in col.lower() for pattern in excluded_patterns)
        ]
        
        return feature_columns
    
    def _analyze_preprocessing_needs(self, df: pd.DataFrame, dataset_info) -> bool:
        """Determine if preprocessing is required"""
        # Check for missing values
        if df.isnull().any().any():
            return True
        
        # Check for categorical variables
        if len(df.select_dtypes(include=['object']).columns) > 0:
            return True
        
        # Check for scale differences in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            ranges = []
            for col in numeric_cols:
                col_range = df[col].max() - df[col].min()
                if not np.isnan(col_range) and col_range > 0:
                    ranges.append(col_range)
            
            if len(ranges) > 1 and max(ranges) / min(ranges) > 100:
                return True
        
        return False
    
    def _estimate_training_time(self, df: pd.DataFrame, recommendations: List[ModelRecommendation]) -> int:
        """Estimate training time in minutes"""
        n_samples = len(df)
        n_features = len(df.columns)
        
        # Base time calculation
        base_time = (n_samples * n_features) / 10000  # minutes
        
        # Adjust for algorithm complexity
        if recommendations:
            primary_algorithm = recommendations[0].algorithm
            complexity_multipliers = {
                'LinearRegression': 0.5,
                'LogisticRegression': 0.7,
                'RandomForest': 1.0,
                'XGBoost': 1.5,
                'SVM': 2.0,
                'NeuralNetwork': 3.0
            }
            multiplier = complexity_multipliers.get(primary_algorithm, 1.0)
            base_time *= multiplier
        
        # Add preprocessing time
        base_time += 2  # 2 minutes for preprocessing
        
        # Add hyperparameter optimization time
        base_time += 3  # 3 minutes for hyperparameter tuning
        
        return max(1, int(base_time))  # Minimum 1 minute

